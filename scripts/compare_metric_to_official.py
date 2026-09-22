"""Cross-check this project's MDD metric against the organizers' own scorer.

Step 1 of the week-3 gate debugging order (docs/weeks/week-03.md: "check the
metric implementation against the official repo's output on the same
predictions, to rule out a scoring bug").

`src/arabic_mdd/metrics/hierarchical.py` was re-derived from the official
algorithm rather than copied, and was only ever validated against
hand-constructed cases (insights/week-02.md). This runs it and the official
`mdd_eval/` scorer over identical predictions and ground truth and prints
them side by side, so a scoring bug can be ruled in or out on real data.

The official scorer needs only pandas/numpy/stdlib, both already available
in this project's env, so unlike the S3PRL inference stage this runs under
plain `uv run` -- no Python-3.8 venv needed.

Run with:
    uv run python scripts/compare_metric_to_official.py run/quranmb_predictions.json

Note the official scorer's alignment (`mdd_eval/metric.py::Align`) is a
hand-rolled Needleman-Wunsch whose tie-breaking among equal-scoring paths is
implementation-specific and not unique. Small divergences in the raw
TA/TR/FA/FR tallies are therefore expected and tolerable; a large F1 gap is
not, and would mean the port is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from arabic_mdd.data.quranmb import MDDGroundTruth, load_ground_truth, score_predictions
from arabic_mdd.metrics.hierarchical import compute_metrics

MDD_EVAL_DIR = Path(__file__).resolve().parent / "baseline_reproduction" / "mdd_eval"


def _official_metrics(
    predictions: dict[str, str], ground_truth: dict[str, MDDGroundTruth]
) -> dict[str, float]:
    """Score with the vendored official `mdd_eval/`, returning its metrics dict.

    Two stages, matching how the organizers' leaderboard drives it:
    `align_data.evaluate_from_dfs` writes three alignment detail files, then
    `ins_del_cor_sub_analysis.analyze_alignment` re-parses them into metrics.
    """
    import pandas as pd

    # `align_data` does `from metric import ...`, so its own directory has to
    # be importable by plain module name.
    sys.path.insert(0, str(MDD_EVAL_DIR))
    from align_data import evaluate_from_dfs
    from ins_del_cor_sub_analysis import analyze_alignment

    shared_ids = [i for i in predictions if i in ground_truth]
    truth_df = pd.DataFrame(
        {
            "ID": shared_ids,
            "Reference_phn": [" ".join(ground_truth[i].canonical) for i in shared_ids],
            "Annotation_phn": [" ".join(ground_truth[i].human_annotation) for i in shared_ids],
        }
    )
    pred_df = pd.DataFrame({"ID": shared_ids, "Prediction": [predictions[i] for i in shared_ids]})

    with tempfile.TemporaryDirectory() as tmp:
        evaluate_from_dfs(truth_df, pred_df, output_dir=tmp, print_output=False)
        return analyze_alignment(tmp)


def main(predictions_path: str) -> None:
    predictions: dict[str, str] = json.loads(Path(predictions_path).read_text())
    ground_truth = load_ground_truth()

    ours = compute_metrics(score_predictions(predictions, ground_truth))
    official = _official_metrics(predictions, ground_truth)
    if "Error" in official:
        raise SystemExit(f"Official scorer failed: {official['Error']}")

    # The official scorer reports rates only, never raw counts. Its "TA" is a
    # true-acceptance *rate* (ta/(ta+fr)), i.e. 1 - FRR -- not our `counts.ta`.
    rows = [
        ("F1", ours.f1, official["F1-score"]),
        ("Precision", ours.precision, official["Precision"]),
        ("Recall", ours.recall, official["Recall"]),
        ("TA rate", 1.0 - ours.frr, official["TA"]),
        ("FRR", ours.frr, official["FRR"]),
        ("FAR", ours.far, official["FAR"]),
        ("DER", ours.der, official["DER"]),
        ("DetAcc", ours.detection_accuracy, official["DetAcc"]),
    ]

    print(f"scored {sum(1 for i in predictions if i in ground_truth)} utterances\n")
    print(f"{'metric':<12}{'ours':>10}{'official':>10}{'delta':>10}")
    print("-" * 42)
    worst = 0.0
    for name, mine, theirs in rows:
        delta = mine - theirs
        worst = max(worst, abs(delta))
        print(f"{name:<12}{mine:>10.4f}{theirs:>10.4f}{delta:>+10.4f}")

    print(f"\nour raw counts: {ours.counts}")
    print(f"largest absolute delta: {worst:.4f}")
    print(
        "VERDICT: metric implementations agree -- scoring ruled out as the cause"
        if worst < 0.01
        else "VERDICT: implementations DISAGREE -- investigate the port before "
        "trusting any gate number"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_json", nargs="?", default="run/quranmb_predictions.json")
    args = parser.parse_args()
    main(args.predictions_json)
