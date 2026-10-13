"""DER on genuinely gold text — week-05 exit criterion 2, first clause.

Every DER number this project has so far is an **agreement rate, not an
accuracy rate**. `run/msa_der.json` scores four diacritizers against
`Iqra_train`'s `tashkeel_sentence`, which is itself the output of a vowelizer;
it measures how much the tools agree with one more tool. The WikiNews
calibration (`scripts/calibrate_der.py`) is against real gold, but it is MSA
news text — the same domain the tools were built for, and not the domain this
paper's benchmark lives in.

This script closes that. The Qur'an has a gold diacritization in the strongest
sense available anywhere in Arabic NLP: the text is fixed, canonical, and its
vowelling is not an annotator's judgement. So a DER measured against it is an
accuracy rate, and `reference_is_gold: true` in the artifact is the field that
distinguishes it from `run/msa_der.json`.

## Two references, reported separately

1. **corpus** — all 6,236 verses of the Tanzil `simple` edition. Large enough
   that the rate is precise, and it is the whole domain rather than a sample of
   it.
2. **QuranMB subset** — the 98 `recovered_text` values from
   `run/quranmb_verse_map.json`: the text that actually carries this arm's
   labels, every span confirmed by ear (`run/quranmb_verse_map_verified.json`).
   4,442 characters, so ~150× smaller and correspondingly noisier.

They are not averaged together. The corpus number is the domain estimate; the
subset number is the one that applies to the benchmark, and the point of
reporting both is to find out whether the first can stand in for the second.

## Two conventions, reported side by side

`metrics.der` is ours and is argued for in its own docstring. `der_published`
is the convention CATT's Table 5 was computed in, and it reproduces all sixteen
of those figures exactly (`run/wikinews_der.json`, Leg A). Both are printed for
every tool, because prediction 1 in `insights/week-05.md` compares these rates
against published WikiNews figures and that comparison is only valid inside the
published convention.

## Dagger alef is a declared row, not a default

The Tanzil `simple` edition writes U+0670 (dagger alef) 3,218 times — 0.97% of
its letter positions — and **no tool under test ever emits one**, so it is a
guaranteed error wherever it appears. `der.SCORED_MARKS` includes it because it
spells a long vowel the reference writes; the published convention deletes it
outright. The difference was measured at 0.85 DER absolute on 60 verses while
planning, which is larger than the calibration tolerance, so:

- the **headline** keeps U+0670 scored, consistent with the MSA and WikiNews
  numbers so that all three are comparable;
- a **second row** reports the rate with U+0670 deleted from the reference,
  plus the raw count.

Deleting a mark from the reference text is a transform, not a new metric axis,
so `der.py` is untouched.

## Throughput

Measured before this was written, not assumed: batched over short verses,
`catt-ed` runs at 0.32 s/verse and `catt-eo` at 0.02 s/verse, so the full 6,236
verses is ~35 minutes for the slowest tool and no sampling fallback is needed.
The artifact records `sampled: false` so that this stays checkable.

Run with:
    uv run python scripts/measure_quranic_der.py --limit 60      # smoke
    uv run python scripts/measure_quranic_der.py --write
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from arabic_mdd.data.normalize import DAGGER_ALEF, strip_diacritics
from arabic_mdd.data.tanzil import EDITION_SHA256, TANZIL_EDITION, file_sha256, load_verses
from arabic_mdd.diacritizers import DIACRITIZERS
from arabic_mdd.diacritizers.chunking import diacritize_long
from arabic_mdd.metrics import der_published
from arabic_mdd.metrics.der import (
    CATT_WITH_CASE_ENDINGS,
    CATT_WITHOUT_CASE_ENDINGS,
    MARKED_ONLY_WITH_CASE_ENDINGS,
    MARKED_ONLY_WITHOUT_CASE_ENDINGS,
    score_corpus,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TANZIL = REPO_ROOT / "run" / "quran_simple.xml.local"
DEFAULT_MAP = REPO_ROOT / "run" / "quranmb_verse_map.json"
OUT = REPO_ROOT / "run" / "quranic_der.json"

VARIANTS = (
    CATT_WITH_CASE_ENDINGS,
    CATT_WITHOUT_CASE_ENDINGS,
    MARKED_ONLY_WITH_CASE_ENDINGS,
    MARKED_ONLY_WITHOUT_CASE_ENDINGS,
)

#: Pre-declared in `insights/week-05.md` before the run. A third instrument
#: disagreeing with the two that already agree is evidence the metric is wrong.
EXPECTED_ORDER = ("catt-eo", "catt-ed", "shakkala", "mishkal")


def _score_set(references: list[str], hypotheses: list[str]) -> dict:
    """Every rate for one reference/hypothesis set, in both conventions."""
    pairs = list(zip(references, hypotheses, strict=True))
    ours = {v.name: score_corpus(pairs, v) for v in VARIANTS}
    pub_ce = der_published.score_corpus(pairs, case_endings=True)
    pub_noce = der_published.score_corpus(pairs, case_endings=False)
    return {
        "published_convention": {
            "ce_der": pub_ce.der,
            "ce_wer": pub_ce.wer,
            "noce_der": pub_noce.der,
            "noce_wer": pub_noce.wer,
            "ce_positions": pub_ce.der_length,
        },
        "our_convention": {
            key: {
                "der": score.der,
                "wer": score.wer,
                "positions": score.positions,
                "position_errors": score.position_errors,
                "words": score.words,
                "word_errors": score.word_errors,
                "skeleton_mismatch": score.skeleton_mismatch,
            }
            for key, score in ours.items()
        },
    }


def _print_set(label: str, scored: dict) -> None:
    pub = scored["published_convention"]
    print(
        f"  {label:<22} published-convention DER {pub['ce_der']:7.3f} / "
        f"{pub['noce_der']:7.3f}   WER {pub['ce_wer']:7.3f} / {pub['noce_wer']:7.3f}"
    )
    for key, row in scored["our_convention"].items():
        print(
            f"  {'':<22} {key:<18} DER {row['der']:7.3f}  WER {row['wer']:7.3f}  "
            f"({row['position_errors']}/{row['positions']}, "
            f"skeleton mismatch {row['skeleton_mismatch']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tanzil", type=Path, default=DEFAULT_TANZIL)
    parser.add_argument("--verse-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--tools", nargs="*", default=list(DIACRITIZERS))
    parser.add_argument("--limit", type=int, default=0, help="first N verses only (smoke run)")
    parser.add_argument("--write", action="store_true", help=f"write {OUT.name}")
    args = parser.parse_args()

    if not args.tanzil.exists():
        raise SystemExit(
            f"{args.tanzil} not found. Tanzil ships a copyright block asking that it "
            "not be removed, so the text is not vendored: fetch it with "
            "`uv run python scripts/recover_quran_verses.py --download`."
        )

    digest = file_sha256(args.tanzil)
    if digest != EDITION_SHA256:
        raise SystemExit(
            f"{args.tanzil} has sha256 {digest}, not the pinned {EDITION_SHA256}. "
            "The verse map and every number below are tied to that edition; "
            "refusing to measure against a different one."
        )

    verses = load_verses(args.tanzil)
    corpus_refs = [v.text for v in verses]
    if args.limit:
        corpus_refs = corpus_refs[: args.limit]

    verse_map = json.loads(args.verse_map.read_text(encoding="utf-8"))
    subset_refs = [p["recovered_text"] for p in verse_map["prompts"]]

    dagger_count = sum(r.count(DAGGER_ALEF) for r in corpus_refs)
    subset_dagger = sum(r.count(DAGGER_ALEF) for r in subset_refs)

    sets = {
        "corpus": corpus_refs,
        "corpus_no_dagger_alef": [r.replace(DAGGER_ALEF, "") for r in corpus_refs],
        "quranmb_subset": subset_refs,
        "quranmb_subset_no_dagger_alef": [r.replace(DAGGER_ALEF, "") for r in subset_refs],
    }
    # One input per reference set: the undiacritized form. `strip_diacritics`
    # already removes U+0670, so the dagger-alef rows share their input with the
    # headline rows and the tools are handed byte-identical text either way.
    inputs = {key: [strip_diacritics(r) for r in refs] for key, refs in sets.items()}
    assert inputs["corpus"] == inputs["corpus_no_dagger_alef"]
    assert inputs["quranmb_subset"] == inputs["quranmb_subset_no_dagger_alef"]

    print("=== DER on gold Qur'anic text — exit criterion 2, first clause ===")
    print(f"edition   : {TANZIL_EDITION}, sha256 {digest}")
    print(f"corpus    : {len(corpus_refs)} verses, {sum(map(len, corpus_refs))} characters")
    print(
        f"subset    : {len(subset_refs)} QuranMB prompts, "
        f"{sum(map(len, subset_refs))} characters (route "
        f"{verse_map['route']}, every span confirmed by ear)"
    )
    print("reference : GOLD — canonical text, not a vowelizer's output")
    print(
        f"dagger    : U+0670 appears {dagger_count} times in the corpus, "
        f"{subset_dagger} in the subset"
    )
    print(f"order     : pre-declared {' < '.join(EXPECTED_ORDER)}\n")

    results: dict[str, dict] = {}
    for name in args.tools:
        tool = DIACRITIZERS[name]()
        print(f"=== {name} ===", flush=True)
        results[name] = {
            "provenance": tool.provenance(),
            "max_input_chars": tool.max_input_chars,
            "sets": {},
        }
        for key in ("corpus", "quranmb_subset"):
            # Timed per tool per set, and flushed, because job 224460 hit its
            # wall inside catt-ed and the log could not say whether it was
            # nearly done or nowhere near. A rate here turns the next timeout
            # into a measurement.
            started = time.monotonic()
            batch = diacritize_long(tool, inputs[key])
            elapsed = time.monotonic() - started
            print(
                f"  {'':<22} diacritized {len(batch.texts)} in {elapsed:.1f}s "
                f"({elapsed / max(len(batch.texts), 1):.3f} s/item)",
                flush=True,
            )
            for suffix in ("", "_no_dagger_alef"):
                scored = _score_set(sets[key + suffix], batch.texts)
                scored["references"] = len(sets[key + suffix])
                scored["chunking"] = batch.report()
                scored["diacritize_seconds"] = elapsed
                results[name]["sets"][key + suffix] = scored
                _print_set(key + suffix, scored)
            print(
                f"  {'':<22} chunking: {batch.chunked_items} of {len(batch.texts)} split, "
                f"{batch.total_seams} seams (limit {tool.max_input_chars})",
                flush=True,
            )
        print()

    print("=== pre-declared sanity check: tool ordering ===")
    ranked = [
        (n, results[n]["sets"]["corpus"]["published_convention"]["ce_der"])
        for n in EXPECTED_ORDER
        if n in results
    ]
    for name, der in ranked:
        print(f"  {name:<10} {der:7.3f}")
    observed = [n for n, _d in sorted(ranked, key=lambda r: r[1])]
    if observed == [n for n, _d in ranked]:
        print("  HOLDS: the ordering is the one two other instruments already agree on.")
    else:
        print(
            f"  BROKEN: observed {' < '.join(observed)}. Two unrelated instruments "
            "already agree on the pre-declared order, so this is evidence against "
            "the metric, not against the tools. Do not report the rates until it "
            "is explained."
        )

    if args.write:
        if args.limit or sorted(args.tools) != sorted(DIACRITIZERS):
            raise SystemExit(
                "refusing to write a partial artifact: --write needs all tools and no "
                "--limit, or run/quranic_der.json would claim more than was measured."
            )
        OUT.write_text(
            json.dumps(
                {
                    "purpose": "docs/weeks/week-05.md exit criterion 2, first clause",
                    "generated_by": "scripts/measure_quranic_der.py",
                    "reference_is_gold": True,
                    "reference_note": (
                        "the canonical Qur'anic text — unlike run/msa_der.json, whose "
                        "reference is itself a vowelizer's output, so these are accuracy "
                        "rates rather than agreement rates"
                    ),
                    "edition": {
                        "name": TANZIL_EDITION,
                        "sha256": digest,
                        "verses": len(verses),
                    },
                    "subset": {
                        "source": str(args.verse_map.relative_to(REPO_ROOT)),
                        "route": verse_map["route"],
                        "route_note": verse_map["route_note"],
                        "verified_by": "run/quranmb_verse_map_verified.json — human audio",
                        "prompts": len(subset_refs),
                        "characters": sum(map(len, subset_refs)),
                    },
                    "input": "strip_diacritics(reference)",
                    "sampled": False,
                    "sampling_note": (
                        "the full corpus was measured for every tool; catt-ed's "
                        "0.32 s/verse batched made the sampling fallback unnecessary"
                    ),
                    "dagger_alef": {
                        "codepoint": "U+0670",
                        "corpus_occurrences": dagger_count,
                        "subset_occurrences": subset_dagger,
                        "headline_row": "corpus / quranmb_subset — U+0670 scored",
                        "declared_second_row": (
                            "*_no_dagger_alef — U+0670 deleted from the reference; "
                            "the input is byte-identical either way, since "
                            "strip_diacritics already removes it"
                        ),
                        "why": (
                            "no tool under test emits U+0670, so it is a guaranteed "
                            "error wherever the reference writes it; our convention "
                            "scores it and the published convention deletes it, and "
                            "the gap exceeds the calibration tolerance"
                        ),
                    },
                    "conventions": {
                        "ours": "arabic_mdd.metrics.der — four named variants",
                        "published": der_published.convention(),
                        "calibration": "run/wikinews_der.json Leg A — all 16 Table-5 figures exact",
                    },
                    "expected_order": list(EXPECTED_ORDER),
                    "observed_order": observed,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
