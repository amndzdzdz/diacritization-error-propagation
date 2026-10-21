"""Put QuranMB.v2 behind the s3prl manifest interface -- week-08 tasks 1 and 2.

Run **once**, on a login node, before any `sbatch`. Decodes
`safikhan/quran_mbv2_formatted` to 1,642 wavs inside the existing `corpus.path`
and emits **five** three-column manifests over that one wav directory, differing
only in column three. Those five files are the rows of plan section 1's 2x2:
`run/evaluate_quranmb.slurm` feeds one per inference pass.

## The column contract, which is where the correctness lives

    path        relative to `corpus.path`, as every other manifest is
    sentence    `annotation_phoneme_string` -- the human annotation A
    canonical   the variant: C_gold, or C_auto from one of four diacritizers

**`sentence` is the annotation, and that is not a leak.** It reads alarming at a
glance, so: `sentence` is the CTC target, which the expert writes out as
`{split}-ref.ark`, which is the thing we want predictions scored *against*. The
conditioned arm's model input is column three and nothing else
(`ctc_cond/data.py` encodes column two into `text` and column three into
`canonical`, and `MddModel.forward` receives only the latter). So the annotation
reaches the loss -- where it is already a free parameter of the evaluation, not
of the model -- and never the conditioner's keys.

The CTC loss and WER s3prl prints over column two are meaningless here (these
are cross-corpus clips and nothing is being trained); evaluation just shares a
code path with training. The numbers that matter are computed offline from the
`.ark` files.

## `C_gold` is leg A's, and this is the easiest thing in the week to get wrong

Week 6 carried two golds. `reference_phoneme_string` is leg B, the
published-F1 anchor, phonemised by the benchmark's uploader. Leg A is
`phonetize(normalize(recovered_text))`, phonemised by *our* phonetizer. Every
`C_auto` goes through our phonetizer, so only leg A is a like-for-like
comparison -- leg B as `C_gold` would bill our phonetizer's disagreement with
the uploader's (44 of 98 prompts of tajwid convention) to the diacritizers.

So `build_canonicals` is **imported** from `scripts/measure_label_path.py`
rather than re-derived. If the two ever disagreed, the 2x2 would be measuring
the phonetizer. The import is the guarantee, and the report's
`c_gold_vs_leg_b.prompts_identical` is the tripwire: at 98/98 the script has
somehow been handed leg B and refuses to write anything.

## Stereo, which a verbatim byte dump would have got silently wrong

96 of the 1,642 clips are two-channel (all of speaker `00015`, spanning 57
prompts); the other 1,546 are mono. All are 16 kHz 16-bit PCM. s3prl's loader
is `torchaudio.load(f)` followed by `wav.reshape(-1)`
(`ctc_cond/data.py:collect_audio_batch`), and on a (2, T) tensor that
*interleaves* the channels into a 2T-length signal -- not a downmix, a
corruption, at double the duration. It would have hit 5.8% of the evaluation
set in all seven passes and shown up only as a disappointing F1.

Both channels carry the same recitation (median per-channel RMS 1274 vs 1268,
median correlation 0.982, and no clip where one channel is near-silent), so the
mean is a sound downmix rather than a choice between two takes. It also matches
week 6: `quranmb_gate_inference.py` read `audio["array"]` through the HF `Audio`
feature, which downmixes to mono, so the organizers'-checkpoint predictions
this week's numbers are compared against were computed on mono too.

Decoding uses the standard library's `wave`, deliberately: `datasets` 5.x
decodes the `Audio` feature through torchcodec, which this environment does not
have and which `arabic_mdd.data.quranmb.load_ground_truth` already goes out of
its way to avoid. `Audio(decode=False)` hands over the raw RIFF bytes instead,
and every file is plain 16-bit PCM, so `wave` reads them with no backend and the
round trip is sample-exact for the 1,546 mono clips.

## Whitespace, and the week-7 artifact it explains

Column two is written through `normalize_transcript`, same as column three.
This is not cosmetic. s3prl's `WordTextEncoder.encode` splits the CTC target on
`split(" ")` -- a *single* space -- while `ctc_cond/data.py` splits the
canonical on bare `.split()`. A doubled space in column two therefore yields an
empty-string token, which `vocab_to_idx` maps to `<unk>`, which lands in
`{split}-ref.ark` and makes that utterance's reference differ from its canonical
for no reason. That is the mechanism behind week 7's three spurious
`informative_utterances` on dev (`insights/week-07.md`, third entry), which was
logged there as a padding round-trip artifact. Padding is not involved: the
evaluation batch size is 1, and `WordTextEncoder.decode` breaks at `<eos>`
before it can reach a pad. It is the two split rules.

QuranMB.v2 happens to need no normalization -- 0 of 1,642 annotation strings
and 0 of 1,642 reference strings contain non-single-space whitespace -- so this
week's `informative_utterances` is trustworthy either way. The call stays in
because that is a property of the current upload, not of the contract.

## Usage

    uv run python scripts/build_quranmb_manifests.py --data-root $DATA_DIR/CV-Ar

Idempotent: rerunning overwrites the wavs and the manifests with identical
bytes. `--skip-audio` reuses an existing dump, for when only the canonical
columns are being rebuilt.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

from arabic_mdd.data.holdout import CONDITIONED_COLUMNS, normalize_transcript, write_manifest
from arabic_mdd.data.phonetizer import Phonetizer
from arabic_mdd.data.quranmb import QURANMB_DATASET
from arabic_mdd.diacritizers import DIACRITIZERS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_label_path import build_canonicals, load_prompt_text  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP = REPO_ROOT / "run" / "quranmb_verse_map.json"
DEFAULT_VOCAB = REPO_ROOT / "scripts" / "baseline_reproduction" / "vocab" / "sws_arabic.txt"
DEFAULT_REPORT = REPO_ROOT / "run" / "quranmb_manifests.json"

#: Inside `corpus.path`, so `downstream_expert.corpus.path` does not have to
#: change between an arm's training config and its evaluation passes -- one
#: fewer value that can differ between a model and its measurement.
SUBDIR = "quranmb"

#: s3prl quotes its upstream stride (320 samples) in 16 kHz frames throughout,
#: so a clip at any other rate would be resampled by nothing and silently
#: evaluated at the wrong speed.
SAMPLE_RATE = 16_000

#: Week 5 measured 54 of 98 prompts phoneme-exact between our phonetizer and the
#: uploader's, which is leg A's fingerprint. Reported, not asserted: the refusal
#: below triggers only at the degenerate end, where C_gold *is* leg B.
LEG_A_EXPECTED_IDENTICAL = 54


def prompt_index(clip_id: str) -> str:
    """`MDDGroundTruth.prompt_index`, without building 1,642 dataclasses.

    One line, duplicated for that reason only; `arabic_mdd.data.quranmb` is the
    authority on why the prompt rather than the text is the grouping key.
    """
    return clip_id.split("_")[1][:5]


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def downmix_to_mono(raw: bytes) -> tuple[bytes, int, int]:
    """`(mono 16-bit PCM frames, channels in, frame count)` from a RIFF/WAVE blob.

    Refuses anything that is not 16 kHz 16-bit PCM rather than coercing it: a
    clip at 8 kHz would be run through an upstream that assumes 16 kHz, i.e.
    evaluated at half speed, and the only symptom would be a bad F1 on that
    clip. Mono input is returned untouched, so the 1,546 mono clips round-trip
    sample-exact.
    """
    with wave.open(io.BytesIO(raw)) as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        payload = handle.readframes(frames)
    if width != 2:
        raise ValueError(f"sample width is {width} bytes, expected 2 (16-bit PCM)")
    if rate != SAMPLE_RATE:
        raise ValueError(f"sample rate is {rate}, expected {SAMPLE_RATE}")
    if channels == 1:
        return payload, channels, frames
    if channels != 2:
        raise ValueError(f"{channels} channels; only mono and stereo are handled")
    # np.round is half-to-even. Stated because it is the one arbitrary choice
    # here, and because the alternative (truncation) would bias every downmixed
    # sample toward zero by up to half an LSB.
    stereo = np.frombuffer(payload, dtype="<i2").reshape(-1, 2)
    mono = np.round(stereo.mean(axis=1)).astype("<i2")
    return mono.tobytes(), channels, frames


def write_mono_wav(path: Path, payload: bytes) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(payload)


def unknown_tokens(strings: list[str], vocabulary: frozenset[str]) -> dict[str, int]:
    """`{token: occurrences}` for whitespace-separated tokens outside `vocabulary`.

    `arabic_mdd.data.canonical.count_unknown_tokens` answers this for a
    manifest's column three only, and here column **two** has to be checked as
    well: `run/phoneme_inventory_diff.json` cleared the four tools on
    Iqra_train dev, which is evidence about Iqra and not about Qur'anic text,
    and QuranMB's human annotations have never been checked against this
    vocabulary at all. The histogram rather than a bare count, so the report
    keeps the columns separable.
    """
    counts: dict[str, int] = {}
    for text in strings:
        for token in text.split():
            if token not in vocabulary:
                counts[token] = counts.get(token, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="the corpus root the manifests' paths are relative to (…/CV-Ar)",
    )
    parser.add_argument("--verse-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tools", nargs="*", default=list(DIACRITIZERS))
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="reuse an existing wav dump; only rebuild the manifests",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help=(
            "proceed despite out-of-vocabulary tokens. They become <unk>, which "
            "corrupts the conditioning input and the reference in the same pass -- "
            "record the reason in insights/ if you use this."
        ),
    )
    args = parser.parse_args()

    unregistered = [t for t in args.tools if t not in DIACRITIZERS]
    if unregistered:
        raise SystemExit(f"unknown tools {unregistered}; registered: {sorted(DIACRITIZERS)}")
    if not args.verse_map.exists():
        raise SystemExit(f"{args.verse_map} not found")

    wav_dir = args.data_root / SUBDIR / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # --- the five canonical columns, before any audio is touched ------------
    # Cheap and fatal-prone, so it runs first: a phonetizer drift or a tool
    # returning an empty string should fail in seconds rather than after a
    # 232 MB dump.
    prompt_text = load_prompt_text(args.verse_map)
    print("=== week-08 tasks 1-2: QuranMB.v2 behind the manifest interface ===")
    print(f"prompts  : {len(prompt_text)}")
    print(f"variants : gold + {len(args.tools)} diacritizers\n")
    canonicals, tool_meta, _normalized = build_canonicals(prompt_text, args.tools)
    variants = ["gold", *args.tools]
    columns = {
        variant: {p: " ".join(seq) for p, seq in canonicals[variant].items()}
        for variant in variants
    }

    # --- the audio and the annotation column -------------------------------
    import datasets

    rows = datasets.load_dataset(QURANMB_DATASET, split="train")
    rows = rows.cast_column("audio", datasets.Audio(decode=False))

    annotation: dict[str, str] = {}
    leg_b: dict[str, str] = {}
    relative: dict[str, str] = {}
    stereo: list[str] = []
    total_frames = 0
    missing_prompt: list[str] = []

    print("=== reading ids (--skip-audio) ===" if args.skip_audio else "=== dumping audio ===")
    for index, row in enumerate(rows):
        clip_id = str(row["id"])
        if prompt_index(clip_id) not in columns["gold"]:
            # A clip whose prompt the verse map does not carry has no canonical
            # under any variant, so it cannot be scored -- and must not be
            # written into a manifest with a guessed one.
            missing_prompt.append(clip_id)
            continue
        annotation[clip_id] = normalize_transcript(row["annotation_phoneme_string"])
        leg_b[clip_id] = normalize_transcript(row["reference_phoneme_string"])
        relative[clip_id] = f"{SUBDIR}/wav/{clip_id}.wav"
        target = args.data_root / relative[clip_id]
        if args.skip_audio:
            if not target.is_file():
                raise SystemExit(f"--skip-audio but {target} is missing; rerun without it")
            continue
        payload, channels, frames = downmix_to_mono(row["audio"]["bytes"])
        if channels == 2:
            stereo.append(clip_id)
        total_frames += frames
        write_mono_wav(target, payload)
        if index % 200 == 0:
            print(f"  {index}/{rows.num_rows}", flush=True)

    if missing_prompt:
        raise SystemExit(
            f"{len(missing_prompt)} clips read a prompt absent from {args.verse_map.name}, "
            f"e.g. {missing_prompt[:3]}. The verse map is supposed to cover all 98 prompts, "
            "so this is drift between the upload and week 5's recovery."
        )
    clips = sorted(annotation)
    if args.skip_audio:
        print(f"  {len(clips)} clips (audio reused)")
    else:
        print(
            f"  {len(clips)} clips, {len(stereo)} downmixed from stereo, "
            f"{total_frames / SAMPLE_RATE / 60:.1f} min of audio"
        )

    # --- the leg-A/leg-B tripwire ------------------------------------------
    # Per prompt, not per clip: C is constant within a prompt.
    leg_b_by_prompt: dict[str, str] = {}
    for clip_id, text in leg_b.items():
        leg_b_by_prompt.setdefault(prompt_index(clip_id), text)
    identical = sum(1 for p, text in leg_b_by_prompt.items() if columns["gold"][p] == text)
    print(
        f"\nC_gold (leg A) identical to reference_phoneme_string (leg B): "
        f"{identical}/{len(leg_b_by_prompt)} prompts "
        f"(week 5 measured {LEG_A_EXPECTED_IDENTICAL})"
    )
    if identical == len(leg_b_by_prompt):
        raise SystemExit(
            "C_gold matches leg B on every prompt, so it IS leg B. Every C_auto goes "
            "through our phonetizer, so leg B as C_gold would make each delta the sum "
            "of the diacritizer's error and our phonetizer's disagreement with the "
            "uploader's, and bill all of it to diacritization. See D9."
        )

    # --- task 2: the vocabulary check, before any GPU time -----------------
    vocabulary = frozenset(args.vocab.read_text(encoding="utf-8").split())
    oov = {"sentence (annotation A)": unknown_tokens([annotation[c] for c in clips], vocabulary)}
    for variant in variants:
        oov[f"canonical ({variant})"] = unknown_tokens(
            sorted(columns[variant].values()), vocabulary
        )
    print(f"\n=== task 2: every column against {args.vocab.name} ({len(vocabulary)} tokens) ===")
    for column, counts in oov.items():
        detail = "ok" if not counts else f"OOV {sum(counts.values())}x {sorted(counts)[:8]}"
        print(f"  {column:<28} {detail}")
    offenders = sorted(k for k, v in oov.items() if v)
    if offenders and not args.allow_unknown:
        print(f"\nERROR: out-of-vocabulary tokens in {offenders}.", file=sys.stderr)
        print(
            "The tokenizer maps these to <unk> *silently*, and here that would corrupt\n"
            "the model's conditioning input AND the reference in the same pass -- a\n"
            "degraded 2x2 whose only symptom is a disappointing |dF1|.\n"
            "run/phoneme_inventory_diff.json cleared the four tools on Iqra_train dev,\n"
            "which is evidence about Iqra and not about Qur'anic text.\n"
            "Pass --allow-unknown to proceed deliberately, and say why in insights/.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- the five manifests ------------------------------------------------
    print("\n=== manifests ===")
    written: dict[str, str] = {}
    for variant in variants:
        out = args.data_root / SUBDIR / f"quranmb.{variant}.tsv"
        write_manifest(
            out,
            list(CONDITIONED_COLUMNS),
            [[relative[c], annotation[c], columns[variant][prompt_index(c)]] for c in clips],
        )
        written[variant] = str(out)
        print(f"  {variant:<10} {out}")

    report = {
        "purpose": "docs/weeks/week-08.md tasks 1 and 2 — QuranMB.v2 as five s3prl manifests",
        "generated_by": "scripts/build_quranmb_manifests.py",
        "git_sha": _git_sha(),
        "dataset": QURANMB_DATASET,
        "data_root": str(args.data_root),
        "wav_dir": str(wav_dir),
        "manifests": written,
        "columns": {
            "path": f"{SUBDIR}/wav/<id>.wav, relative to corpus.path",
            "sentence": "annotation_phoneme_string (A) — the CTC target, hence {split}-ref.ark",
            "sentence_note": (
                "not a leak: ctc_cond/data.py encodes column two into `text` and column "
                "three into `canonical`, and MddModel.forward receives only the latter"
            ),
            "canonical": "the variant (C) — the only column that differs between the five",
            "whitespace": (
                "both text columns pass through normalize_transcript. s3prl's encode "
                "splits the CTC target on split(' ') while ctc_cond/data.py splits the "
                "canonical on .split(), so a doubled space in column two becomes an "
                "<unk> in {split}-ref.ark and a spurious informative utterance. "
                "QuranMB needs no normalization (0/1642 rows), so this is a guard."
            ),
        },
        "clips": len(clips),
        "prompts": len(prompt_text),
        "audio": {
            "sample_rate": SAMPLE_RATE,
            "channels_out": 1,
            "dumped": not args.skip_audio,
            "stereo_downmixed": None if args.skip_audio else len(stereo),
            "stereo_clip_ids": None if args.skip_audio else stereo,
            "minutes": None if args.skip_audio else total_frames / SAMPLE_RATE / 60,
            "decoder": "stdlib wave over Audio(decode=False) bytes; no torchcodec/torchaudio",
            "downmix_note": (
                "s3prl's collect_audio_batch does torchaudio.load(f).reshape(-1), which "
                "interleaves a (2, T) tensor into a 2T-length signal rather than "
                "downmixing it. Channels carry the same recitation (median RMS 1274 vs "
                "1268, median correlation 0.982), so the mean is a downmix and not a "
                "choice of take. Matches week 6, whose predictions came through the HF "
                "Audio feature's own mono downmix."
            ),
        },
        "c_gold": {
            "leg": "A",
            "rule": "phonetize(normalize(recovered_text)), U+0670 deleted — D9",
            "derivation": "imported from scripts/measure_label_path.py:build_canonicals",
            "phonetizer": Phonetizer("prescriptive").provenance(),
            "source": str(args.verse_map),
        },
        "c_gold_vs_leg_b": {
            "prompts_identical": identical,
            "prompts": len(leg_b_by_prompt),
            "week_5_measured": LEG_A_EXPECTED_IDENTICAL,
            "note": (
                "98/98 would mean C_gold IS leg B, which this script refuses: C_auto "
                "goes through our phonetizer, so leg B as C_gold would bill 44 prompts "
                "of tajwid disagreement to the diacritizers."
            ),
        },
        "vocabulary_check": {
            "vocab": str(args.vocab),
            "vocab_size": len(vocabulary),
            "out_of_vocabulary": oov,
            "allow_unknown": args.allow_unknown,
        },
        "tools": tool_meta,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.report}")
    print(
        "\nNext: run/evaluate_quranmb.slurm, one submission per (arm, variant) — "
        "seven passes. See docs/weeks/week-08.md task 4."
    )


if __name__ == "__main__":
    main()
