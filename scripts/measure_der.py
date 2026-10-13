"""RQ1: DER/WER per diacritizer on the MSA deployment corpus — week-05 tasks 3 and 8.

Measures `der(reference, tool(strip(reference)))` for every tool in the
registry, in all four `der.Variant` combinations, and prints the week-5 task 8
decision rule's inputs.

**What the reference is, and what the number therefore means.** On
`Iqra_train` the only diacritized column available before the MSA block is
annotated is `tashkeel_sentence`, which is the organizers' *vowelizer's
output, not gold* — week 4 measured it wrong on 43.6% of pilot utterances. So
this is **agreement with the organizers' pipeline**, not accuracy, and
labelling it accuracy would invert the paper's own finding. RQ1-MSA proper is
computed against `C_gold` when the block is annotated (D5).

The input is the reference with marks stripped, not the dataset's `sentence`
column. Two reasons: a fifth of `sentence` arrives already vowelized (§4.1),
and stripping the reference is what guarantees the tool is scored on the same
word sequence it was asked about, so a skeleton mismatch means the tool
edited the text rather than that the two columns disagree.

Run with:
    uv run python scripts/measure_der.py --split dev
    uv run python scripts/measure_der.py --split dev --tools catt-eo catt-ed
    uv run python scripts/measure_der.py --split dev --limit 200   # smoke run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arabic_mdd.data.iqra_train import load_iqra_train
from arabic_mdd.data.normalize import strip_diacritics
from arabic_mdd.diacritizers import DIACRITIZERS
from arabic_mdd.metrics.der import (
    CATT_WITH_CASE_ENDINGS,
    CATT_WITHOUT_CASE_ENDINGS,
    MARKED_ONLY_WITH_CASE_ENDINGS,
    MARKED_ONLY_WITHOUT_CASE_ENDINGS,
    score_corpus,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "run" / "msa_der.json"

VARIANTS = (
    CATT_WITH_CASE_ENDINGS,
    CATT_WITHOUT_CASE_ENDINGS,
    MARKED_ONLY_WITH_CASE_ENDINGS,
    MARKED_ONLY_WITHOUT_CASE_ENDINGS,
)

#: Week-5 task 8's rule, pre-committed before the measurement: ED is chosen
#: only if it leads EO by more than this on `ce_all`; otherwise EO.
ED_MARGIN_REQUIRED = 0.3

#: Week 4's per-utterance phoneme conservation tail (>10% short), the
#: pre-registered tie-breaker. Printed next to the DER so the tie-break is
#: auditable rather than recalled.
CONSERVATION_TAIL = {"catt-eo": 0.08, "catt-ed": 1.04, "shakkala": 1.39, "mishkal": 5.14}

BATCH = 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--tools", nargs="*", default=list(DIACRITIZERS))
    parser.add_argument("--limit", type=int, default=0, help="first N rows only (smoke run)")
    parser.add_argument("--write", action="store_true", help=f"write {OUT.name}")
    args = parser.parse_args()

    examples = load_iqra_train(split=args.split)
    examples = [e for e in examples if e.tashkeel_sentence.strip()]
    if args.limit:
        examples = examples[: args.limit]

    references = [e.tashkeel_sentence for e in examples]
    inputs = [strip_diacritics(r) for r in references]
    print(f"corpus: Iqra_train {args.split}, {len(references)} rows with a reference")
    print("reference: tashkeel_sentence = the organizers' VOWELIZER OUTPUT, not gold.")
    print("           these are agreement rates, not accuracy rates.\n")

    results: dict[str, dict] = {}
    for name in args.tools:
        tool = DIACRITIZERS[name]()
        outputs: list[str] = []
        for start in range(0, len(inputs), BATCH):
            outputs.extend(tool.diacritize_batch(inputs[start : start + BATCH]))

        pairs = list(zip(references, outputs, strict=True))
        scores = {v.name: score_corpus(pairs, v) for v in VARIANTS}
        results[name] = {
            "provenance": tool.provenance(),
            "rows": len(pairs),
            "variants": {
                key: {
                    "der": score.der,
                    "wer": score.wer,
                    "positions": score.positions,
                    "position_errors": score.position_errors,
                    "words": score.words,
                    "word_errors": score.word_errors,
                    "skeleton_mismatch": score.skeleton_mismatch,
                }
                for key, score in scores.items()
            },
        }

        headline = scores[CATT_WITH_CASE_ENDINGS.name]
        print(f"=== {name} ===")
        for key, score in scores.items():
            print(
                f"  {key:<18} DER {score.der:6.3f}  WER {score.wer:6.3f}  "
                f"({score.position_errors}/{score.positions} positions)"
            )
        print(
            f"  skeleton mismatch : {headline.skeleton_mismatch}/{len(pairs)} "
            f"= {100 * headline.skeleton_mismatch / len(pairs):.2f}% (excluded from the rates)"
        )
        print(f"  week-4 conservation tail >10% short: {CONSERVATION_TAIL.get(name, '?')}%\n")

    _report_variant_choice(results)

    if args.write:
        OUT.write_text(
            json.dumps(
                {
                    "source": "IqraEval/Iqra_train",
                    "split": args.split,
                    "reference_column": "tashkeel_sentence",
                    "reference_is_gold": False,
                    "reference_note": (
                        "organizers' vowelizer output; week 4 measured it wrong on 43.6% of "
                        "pilot utterances. These are agreement rates, not accuracy rates."
                    ),
                    "input": "strip_diacritics(tashkeel_sentence)",
                    "decision_rule": (
                        "week-05 task 8: choose on ce_all DER; ED only if it leads EO by "
                        f"more than {ED_MARGIN_REQUIRED} absolute, else EO"
                    ),
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {OUT.relative_to(REPO_ROOT)}")


def _report_variant_choice(results: dict[str, dict]) -> None:
    """Apply task 8's rule. Printed, not decided silently."""
    key = CATT_WITH_CASE_ENDINGS.name
    if not {"catt-eo", "catt-ed"} <= results.keys():
        return

    eo = results["catt-eo"]["variants"][key]["der"]
    ed = results["catt-ed"]["variants"][key]["der"]
    print("=== task 8: CATT variant, by the rule fixed before the measurement ===")
    print(f"  catt-eo {key} DER : {eo:.3f}")
    print(f"  catt-ed {key} DER : {ed:.3f}")
    print(f"  ED leads by       : {eo - ed:+.3f} (needs > {ED_MARGIN_REQUIRED} to win)")
    chosen = "catt-ed" if (eo - ed) > ED_MARGIN_REQUIRED else "catt-eo"
    print(f"  CHOSEN            : {chosen}")
    if chosen == "catt-eo" and ed < eo:
        print("  (ED is ahead but inside the margin, so the rule keeps EO.)")


if __name__ == "__main__":
    main()
