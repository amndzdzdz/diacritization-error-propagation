"""Did the text-dependent arm read the audio, or echo the canonical?

Week-07 task 7. `arabic_mdd.models.collapse` is the judgment; this is the
plumbing that feeds it the three `.ark` files
`scripts/baseline_reproduction/ctc_cond/expert.py` writes during evaluation:

    {split}-hyp.ark      the prediction
    {split}-ref.ark      the CTC target -- what the speaker actually said
    {split}-canon.ark    the canonical -- what they were supposed to say

Called from `run/train_mdd_arm.slurm` after `-m evaluate`, for **both** arms.
The prompt-free arm's copy rate, measured against a canonical it never saw, is
the null distribution the conditioned arm's number has to be read against: a
conditioned `copy_rate` of 0.30 means nothing until you know whether the
ablated arm scores 0.05 or 0.28 on the same dev set.

    uv run python scripts/report_collapse.py \
        --expdir run/s3prl_checkout/s3prl/xlsr300m_mdd_conditioned --split dev

**What this run can and cannot tell you.** The dev split is held-out
`Iqra_train`, where the CTC target *is* `phoneme_ref` for every row -- so
canonical and target never disagree, `informative_utterances` is 0 and
`canonical_preference` is NaN by construction. On that split `copy_rate` also
degenerates: "prediction == canonical" and "prediction is correct" are the same
event. Both numbers are still worth printing (a conditioned `copy_rate` far
above the ablated arm's is a signal even here, and a NaN that appears where it
should is evidence the join is right), but neither is the measurement.

The decisive measurement needs utterances where the speaker demonstrably said
something other than the canonical, and it needs no new infrastructure: that is
**QuranMB.v2**, whose human annotation disagrees with the canonical on ~12.4% of
phonemes, and whose `(canonical, human_annotation, prediction)` triple is
already exactly what `arabic_mdd.metrics.hierarchical.evaluate_utterance` and
`collapse_report` both consume. Wiring it is week 8's evaluation, not another
dataloader here.

Rows are joined on the utterance id in column one rather than on line order.
The expert writes all three files in a single `zip`, so the order does agree --
but a misjoin here would silently compare one utterance's prediction against
another's canonical, which is the same class of bug the dataloader's
length-sort had to avoid, and the key is free.

## `--null`: D12's rule, executed rather than recited

Week 8 settled what the collapse number is allowed to do
(`docs/weeks/week-08.md`, D12), and the rule is a *difference* from the ablated
arm on a minimum sample -- because week 7 showed the statistic has no absolute
scale: both the conditioned arm and the `use_canonical=False` arm, which cannot
causally prefer a canonical, reported `canonical_preference = 1.0000` on dev.

`--null <the ablated arm's report.json>` makes the verdict a computation. D12's
whole justification is that the rule predates the number; a rule applied by hand
after reading two JSON files is a rule that *could* have been chosen afterwards,
and nothing in the artifact would show the difference. So the threshold and the
minimum sample are constants in this file, and the artifact records which branch
fired.

    uv run python scripts/report_collapse.py \
        --expdir run/quranmb_eval/ablated/gold --split quranmb \
        --output run/quranmb_collapse_ablated.json
    uv run python scripts/report_collapse.py \
        --expdir run/quranmb_eval/conditioned/gold --split quranmb \
        --null run/quranmb_collapse_ablated.json \
        --output run/quranmb_collapse_conditioned.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arabic_mdd.models.collapse import (
    D12_MIN_INFORMATIVE,
    D12_THRESHOLD,
    CollapseReport,
    collapse_report,
    d12_verdict,
)

#: What each branch of D12 licenses the paper to say. The arithmetic lives in
#: `arabic_mdd.models.collapse.d12_verdict` under test; only the prose is here,
#: because prose is what this script exists to print.
D12_PROSE = {
    "underpowered": (
        "underpowered — report both arms' numbers with the informative count stated and no verdict"
    ),
    "fires": (
        "FIRES — the bottom-left cell of plan section 1's 2x2 becomes an upper "
        "bound, not an estimate, and the prompt-free scoring column carries RQ2. "
        "Mitigation retrains stay refused per D11."
    ),
    "does_not_fire": "does not fire — the input-path cell is reported as an estimate",
}


def apply_d12(report: CollapseReport, null: dict[str, float | int]) -> dict[str, object]:
    """D12's branch, chosen by the numbers rather than by the reader.

    `null` is the ablated arm's `as_dict()`, read back off disk from that arm's
    own evaluation pass.
    """
    verdict = d12_verdict(report, CollapseReport.from_dict(null))
    return {
        "decision": "D12 — docs/weeks/week-08.md",
        "threshold": D12_THRESHOLD,
        "min_informative_utterances": D12_MIN_INFORMATIVE,
        "null_arm": null.get("expdir"),
        "conditioned_canonical_preference": report.canonical_preference,
        "ablated_canonical_preference": null["canonical_preference"],
        "delta": verdict.delta,
        "informative_utterances_min": verdict.informative_utterances,
        "powered": verdict.powered,
        "falsifier_fires": verdict.fires,
        "branch": verdict.verdict,
        "verdict": D12_PROSE[verdict.verdict],
        "if_it_fires_wording": (
            "for the text-dependent arm the input-path effect is bounded above by X; a "
            "copy-rate diagnostic indicates the arm partially restates reference "
            "corruption, so X is not attributable to degraded conditioning alone"
        ),
    }


def read_ark(path: Path) -> dict[str, list[str]]:
    """`{utterance_id: tokens}` from an `id tok tok tok` file.

    A line with no tokens after the id is a legitimately empty hypothesis --
    CTC emitting nothing but blanks -- so it maps to `[]` rather than being
    skipped, which would shrink the hypothesis set and break the join.
    """
    rows: dict[str, list[str]] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        key = parts[0]
        if key in rows:
            raise ValueError(f"{path}:{lineno}: duplicate utterance id {key!r}")
        rows[key] = parts[1].split() if len(parts) > 1 else []
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expdir", type=Path, required=True, help="the s3prl expdir holding the .ark files"
    )
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output", type=Path, help="where to write the JSON report")
    parser.add_argument(
        "--null",
        type=Path,
        help=(
            "a report.json from the ablated (use_canonical=False) arm on the SAME "
            "manifest. Supplying it applies D12's pre-registered rule."
        ),
    )
    args = parser.parse_args()

    paths = {name: args.expdir / f"{args.split}-{name}.ark" for name in ("hyp", "ref", "canon")}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        print(f"ERROR: missing {', '.join(missing)}.", file=sys.stderr)
        print(
            "`{split}-canon.ark` is written only by downstream/ctc_cond, so a run of\n"
            "s3prl's own `-d ctc` will not have one. The hyp/ref pair alone cannot\n"
            "answer this question: without the canonical there is no copy to detect.",
            file=sys.stderr,
        )
        sys.exit(1)

    tables = {name: read_ark(path) for name, path in paths.items()}
    keys = sorted(tables["hyp"])
    for name, table in tables.items():
        if set(table) != set(tables["hyp"]):
            print(
                f"ERROR: {paths[name].name} covers {len(table)} utterances, "
                f"{paths['hyp'].name} covers {len(tables['hyp'])}. "
                "These should be the same set, written in one pass.",
                file=sys.stderr,
            )
            sys.exit(1)

    report = collapse_report(
        [tables["hyp"][k] for k in keys],
        [tables["canon"][k] for k in keys],
        [tables["ref"][k] for k in keys],
    )
    payload = {
        "expdir": str(args.expdir),
        "split": args.split,
        **report.as_dict(),
    }

    if args.null:
        null = json.loads(args.null.read_text(encoding="utf-8"))
        if null.get("split") != args.split:
            raise SystemExit(
                f"{args.null.name} was produced on split {null.get('split')!r} and this "
                f"run is on {args.split!r}. D12 is a difference between two arms on the "
                "same clips; across splits it is not a difference."
            )
        payload["d12"] = apply_d12(report, null)

    print(f"utterances:              {report.utterances}")
    print(f"copy_rate:               {report.copy_rate:.4f}")
    print(f"informative_utterances:  {report.informative_utterances}")
    print(f"canonical_preference:    {report.canonical_preference:.4f}")
    if "d12" in payload:
        d12 = payload["d12"]
        print("\n=== D12 ===")
        print(f"  ablated canonical_preference: {d12['ablated_canonical_preference']}")
        print(f"  delta:                        {d12['delta']:+.4f} (threshold > {D12_THRESHOLD})")
        print(
            f"  informative (min of arms):    {d12['informative_utterances_min']} "
            f"(need >= {D12_MIN_INFORMATIVE})"
        )
        print(f"  {d12['verdict']}")
    if report.informative_utterances == 0:
        print(
            "\nNOTE: no utterance here has canonical != target, so canonical_preference\n"
            "is undefined (NaN, not 0.0 -- 0.0 would read as 'no collapse detected').\n"
            "On a held-out Iqra_train split that is EXPECTED, not a problem: every row\n"
            "there has phoneme_ref as its CTC target, so there is nothing for the model\n"
            "to resolve one way or the other. copy_rate is correspondingly degenerate\n"
            "on this split -- it coincides with exact-match accuracy.\n"
            "The discriminating run is against QuranMB.v2 in week 8, where the human\n"
            "annotation and the canonical disagree on ~12.4% of phonemes."
        )

    if args.output:
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
