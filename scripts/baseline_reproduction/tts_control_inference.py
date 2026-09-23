"""Run a trained checkpoint over `IqraEval/Iqra_TTS`'s CLEAN (`original`) rows.

Stage A of the zero-base-rate control for `docs/msa-arm.md` §3.3.

**Why this exists.** The CV-Ar dev probe (`scripts/cv_dev_baserate.py`)
came back with a false-rejection rate of 0.0708 against QuranMB's 0.1241 --
*below* the reference, which reads as "no headroom for a mispronunciation
base rate". But that comparison is confounded: `dev-best.ckpt` was
model-selected on that very split (`docs/msa-arm.md` §5.2), so a lower
error rate there is expected whether or not speakers ever deviate. The
screen cannot distinguish "no mispronunciations" from "mispronunciations
masked by an in-domain model".

**What fixes it.** `Iqra_TTS` ships a `label` column with two values,
`original` and `augmented`. The `augmented` rows are the organizers'
*injected* mispronunciations (`phoneme_mis` != `phoneme_ref`). The
`original` rows are synthesized straight from `phoneme_ref`, so for them
`A == C` **by construction** -- a guaranteed zero base rate. Scoring those
with `A := C` is therefore not an assumption but an identity, and the
resulting FR rate is the model's pure error rate on MSA phonetics.

**The premise was checked before this was written, not assumed.** Over 800
sampled rows, after stripping `<sil>` (see DROP_TOKENS below):

    label=original : phoneme_mis == phoneme_ref in 500/500  (100.0%)
    label=augmented: phoneme_mis == phoneme_ref in   3/300  (  1.0%)

So `original` really is the clean half and `augmented` really is the
mispronounced half. Note the `<sil>` strip is load-bearing for that
identity: *unstripped*, `phoneme_mis` differs from `phoneme_ref` on every
`original` row, because only `phoneme_ref` carries the silence tokens.
Comparing them raw would have made the clean rows look 100% mispronounced.

Roughly a third of the 46,851 rows carry `label=original` (~15.6k), and
they are interspersed in blocks rather than contiguous, so the filter below
does real work.

**Read the result one-sidedly.** `Iqra_TTS` is part of the 131 h training
set, and TTS audio is acoustically cleaner than real speech. Both push this
control's FR rate *down*. So:

* control FR >= 0.0708 => decisive. Even with a guaranteed zero base rate
  and two advantages, the model errs at least as much as it disagrees on
  CV-Ar dev. No headroom; §3.3 option 2 is forced.
* control FR <  0.0708 => inconclusive. The gap may be real
  mispronunciation, or may just be real speech being harder than TTS.

Only the first outcome settles anything. Step 4's listening pilot remains
the unconfounded measurement either way.

Stage A writes prediction AND reference into one file so that Stage B never
has to re-fetch this 6 GB dataset.

Must run inside the S3PRL Python-3.8 venv, with a GPU. See
run/tts_control.slurm.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3prl_inference import S3PRLModel  # noqa: E402

DATASET = "IqraEval/Iqra_TTS"

# `Iqra_TTS`'s phoneme strings carry `<sil>`, which is NOT one of the 68
# tokens in sws_arabic.txt (`<` alone is -- it is the glottal stop). The CTC
# head cannot emit `<sil>`, so leaving it in the reference would score every
# silence as a guaranteed false rejection and inflate exactly the number
# this control exists to measure. `Iqra_train`'s `phoneme_ref` has no
# `<sil>`, which is why the CV-Ar probe needed no equivalent step.
DROP_TOKENS = {"<sil>", "sil", "<unk>"}


def clean_reference(phonemes: str) -> str:
    return " ".join(tok for tok in (phonemes or "").split() if tok not in DROP_TOKENS)


def _extract_prediction(raw_output: object) -> str:
    """Unwrap `S3PRLModel.__call__`'s one-element batch of hypotheses."""
    if isinstance(raw_output, (list, tuple)):
        if not raw_output:
            return ""
        return _extract_prediction(raw_output[0])
    return str(raw_output)


def run(
    ckpt: str,
    dict_path: str,
    output_json: str,
    split: str,
    label: str,
    limit: int | None,
    seed: int,
) -> None:
    import datasets

    model = S3PRLModel(ckpt, dict_path)
    ds = datasets.load_dataset(DATASET, split=split)
    print(f"{DATASET} {split}: {len(ds)} rows total", flush=True)

    # `Iqra_TTS` has no `id` column, so the row index is the identifier.
    # Capture it BEFORE filtering, so ids stay stable and traceable back to
    # the source row.
    ds = ds.map(lambda _row, idx: {"row_index": idx}, with_indices=True)
    ds = ds.filter(lambda row: row["label"] == label)
    print(f"  rows with label == {label!r}: {len(ds)}", flush=True)
    if len(ds) == 0:
        raise SystemExit(f"No rows with label == {label!r} -- check the column's values.")

    # Subsample RANDOMLY, never head-N. The dataset is ordered by speaker
    # (Amer, Husam, Callum, Jessica, Nabil, LJ, SASSC), so `select(range(N))`
    # would hand back one or two voices and measure their idiosyncratic error
    # rate rather than the model's. That would silently bias the one number
    # this control exists to produce.
    if limit and limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))
        print(f"  randomly subsampled to {len(ds)} rows (seed {seed})", flush=True)

    records: dict[str, dict[str, str]] = {}
    tmp_wav = Path("_tts_control_tmp.wav")
    for idx, row in enumerate(ds):
        example_id = str(row["row_index"])
        audio = row["audio"]
        wav_tensor = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
        torchaudio.save(str(tmp_wav), wav_tensor, audio["sampling_rate"])

        records[example_id] = {
            "prediction": _extract_prediction(model(str(tmp_wav))),
            # Stored cleaned, so Stage B scores exactly what was scored here.
            "phoneme_ref": clean_reference(row.get("phoneme_ref")),
            # Always stored, for BOTH labels. On `original` rows this equals
            # `phoneme_ref` and is redundant; on `augmented` rows it is `A`,
            # the sequence actually synthesized, which completes the
            # (C, A, P) triple the hierarchical metric needs. Capturing it
            # unconditionally means an `--label augmented` run needs no
            # second pass over this 6 GB dataset.
            "phoneme_mis": clean_reference(row.get("phoneme_mis")),
            "speaker": str(row.get("speaker") or ""),
        }

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(ds)}...", flush=True)

    tmp_wav.unlink(missing_ok=True)

    # Print the speaker mix: this control's whole value is that it measures
    # the MODEL, so a lopsided voice distribution is a caveat the reader of
    # the number needs to see.
    speakers = Counter(rec["speaker"] for rec in records.values())
    print("  speaker mix:", dict(speakers.most_common()))

    Path(output_json).write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Wrote {len(records)} records to {output_json}")
    for example_id, rec in list(records.items())[:3]:
        print(f"  {example_id}: ref={rec['phoneme_ref'][:60]!r}")
        print(f"     pred={rec['prediction'][:60]!r}")

    model.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path or URL to trained checkpoint.")
    parser.add_argument("--dict_path", required=True, help="Path or URL to vocab/dict file.")
    parser.add_argument("--output_json", default="tts_control_predictions.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--label", default="original", help="Which `label` value to keep.")
    parser.add_argument(
        "--limit",
        type=int,
        default=3000,
        help=(
            "Score a random subsample of this many rows (seeded). ~15.6k rows carry "
            "label=original, which overruns a sensible wall clock; 3000 gives a "
            "standard error near 0.002 on the FR rate, far finer than the 0.04 gap "
            "being tested. Pass 0 for all rows."
        ),
    )
    parser.add_argument("--seed", type=int, default=20261012, help="Subsampling seed.")
    args = parser.parse_args()
    run(
        args.ckpt,
        args.dict_path,
        args.output_json,
        args.split,
        args.label,
        args.limit or None,
        args.seed,
    )
