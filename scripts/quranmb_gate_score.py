"""Score S3PRL predictions on QuranMB.v2 against the week-3 gate.

Stage B of the two-stage gate-check pipeline (see
scripts/baseline_reproduction/README.md Step 6, and docs/weeks/week-03.md).
Reads the `{id: predicted_phoneme_string}` JSON produced by
scripts/baseline_reproduction/quranmb_gate_inference.py (Stage A, run
inside the S3PRL Python-3.8 venv on the cluster -- needs a GPU and network
access to `safikhan/quran_mbv2_formatted`) and scores it with this
project's own hierarchical MDD metric.

Run with:
    uv run python scripts/quranmb_gate_score.py path/to/quranmb_predictions.json

Gate criterion (docs/weeks/week-03.md): F1 ≈ 0.4414 +/- 0.02.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arabic_mdd.data.quranmb import load_ground_truth, score_predictions
from arabic_mdd.metrics.hierarchical import compute_metrics

GATE_F1 = 0.4414
GATE_TOLERANCE = 0.02


def main(predictions_path: str) -> None:
    predictions: dict[str, str] = json.loads(Path(predictions_path).read_text())
    ground_truth = load_ground_truth()

    matched = sum(1 for example_id in predictions if example_id in ground_truth)
    print(
        f"{matched}/{len(predictions)} predictions matched a ground-truth id "
        f"(ground truth has {len(ground_truth)} utterances total)"
    )
    if matched == 0:
        raise SystemExit(
            "No predictions matched any ground-truth id -- check that Stage A's "
            "`id` values line up with `safikhan/quran_mbv2_formatted`'s `id` column."
        )

    counts = score_predictions(predictions, ground_truth)
    metrics = compute_metrics(counts)

    lo, hi = GATE_F1 - GATE_TOLERANCE, GATE_F1 + GATE_TOLERANCE
    passed = lo <= metrics.f1 <= hi
    print(f"counts: {counts}")
    print(f"precision: {metrics.precision:.4f}  recall: {metrics.recall:.4f}")
    print(f"FAR: {metrics.far:.4f}  FRR: {metrics.frr:.4f}  DER: {metrics.der:.4f}")
    print(f"detection accuracy: {metrics.detection_accuracy:.4f}")
    print(f"F1: {metrics.f1:.4f}  (gate: {GATE_F1} +/- {GATE_TOLERANCE} -> [{lo:.4f}, {hi:.4f}])")
    print("GATE: PASS" if passed else "GATE: FAIL (or not yet -- check training completion)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_json")
    args = parser.parse_args()
    main(args.predictions_json)
