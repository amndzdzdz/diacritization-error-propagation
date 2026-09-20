"""Tests for the hierarchical MDD metric (`arabic_mdd.metrics.hierarchical`).

Each case hand-derives the expected Needleman-Wunsch alignment and the
resulting TA/TR/FA/FR/Correct-Diagnosis/Error-Diagnosis classification for
a small phoneme sequence, independently of the implementation, before
asserting against it. See the module docstring in `hierarchical.py` for
the classification rules being checked here.
"""

import math

from arabic_mdd.metrics.hierarchical import (
    MDDCounts,
    aggregate,
    compute_metrics,
    evaluate,
    evaluate_utterance,
)


def test_all_correct_is_true_acceptance() -> None:
    counts = evaluate_utterance(["b", "a"], ["b", "a"], ["b", "a"])
    assert counts == MDDCounts(ta=2)


def test_false_rejection_when_prediction_flags_correct_pronunciation() -> None:
    # Speaker said "ba" correctly (annotation == canonical), but the
    # system predicted "bu" -- a false alarm on a genuinely correct vowel.
    counts = evaluate_utterance(["b", "a"], ["b", "a"], ["b", "u"])
    assert counts == MDDCounts(ta=1, fr=1)


def test_correctly_diagnosed_substitution() -> None:
    # Speaker actually substituted a->u; the system predicts exactly what
    # was said, so the error is both detected (TR) and correctly
    # diagnosed (CD).
    counts = evaluate_utterance(["b", "a"], ["b", "u"], ["b", "u"])
    assert counts == MDDCounts(ta=1, tr=1, correct_diagnosis=1)

    metrics = compute_metrics(counts)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.far == 0.0
    assert metrics.frr == 0.0
    assert metrics.der == 0.0
    assert metrics.detection_accuracy == 1.0


def test_wrongly_diagnosed_substitution() -> None:
    # Speaker substituted a->u; the system detects an error but predicts
    # the wrong replacement ("i" instead of "u") -- detected (TR) but
    # wrongly diagnosed (ED).
    counts = evaluate_utterance(["b", "a"], ["b", "u"], ["b", "i"])
    assert counts == MDDCounts(ta=1, tr=1, error_diagnosis=1)

    metrics = compute_metrics(counts)
    assert metrics.der == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0


def test_missed_substitution_is_false_acceptance() -> None:
    # Speaker substituted a->u, but the system predicts the canonical
    # phoneme "a", missing the error entirely.
    counts = evaluate_utterance(["b", "a"], ["b", "u"], ["b", "a"])
    assert counts == MDDCounts(ta=1, fa=1)

    metrics = compute_metrics(counts)
    assert metrics.recall == 0.0
    assert math.isnan(metrics.precision)  # no TR, no FR in this utterance


def test_correctly_detected_deletion() -> None:
    # Speaker dropped the "a" in canonical "bat" -> "bt"; the system
    # correctly predicts the deletion.
    counts = evaluate_utterance(["b", "a", "t"], ["b", "t"], ["b", "t"])
    assert counts == MDDCounts(ta=2, tr=1, correct_diagnosis=1)

    metrics = compute_metrics(counts)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.detection_accuracy == 1.0


def test_missed_deletion_is_false_acceptance() -> None:
    # Speaker dropped the "a", but the system predicts the full canonical
    # sequence, missing the deletion.
    counts = evaluate_utterance(["b", "a", "t"], ["b", "t"], ["b", "a", "t"])
    assert counts == MDDCounts(ta=2, fa=1)


def test_correctly_diagnosed_insertion() -> None:
    # Speaker inserted an extra "a" not present in canonical "bt"; the
    # system correctly predicts the inserted phoneme.
    counts = evaluate_utterance(["b", "t"], ["b", "a", "t"], ["b", "a", "t"])
    assert counts == MDDCounts(ta=2, tr=1, correct_diagnosis=1)


def test_multi_error_utterance() -> None:
    # One real, correctly-diagnosed substitution (a->u at position 1),
    # plus a spurious extra error the system hallucinates on a phoneme
    # that was actually pronounced correctly (canon/annotation both "t"
    # at position 2, system predicts "x") -- a mix of TR/CD and FR in one
    # utterance.
    counts = evaluate_utterance(["b", "a", "t"], ["b", "u", "t"], ["b", "u", "x"])
    assert counts == MDDCounts(ta=1, tr=1, fr=1, correct_diagnosis=1)

    metrics = compute_metrics(counts)
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert math.isclose(metrics.f1, 2 / 3, rel_tol=1e-9)
    assert metrics.far == 0.0
    assert metrics.frr == 0.5
    assert metrics.der == 0.0
    assert math.isclose(metrics.detection_accuracy, 2 / 3, rel_tol=1e-9)


def test_aggregate_pools_counts_across_utterances() -> None:
    utterance_1 = evaluate_utterance(["b", "u"], ["b", "u"], ["b", "u"])  # TA=2
    utterance_2 = evaluate_utterance(["b", "a"], ["b", "u"], ["b", "u"])  # TA=1, TR=1, CD=1
    total = aggregate([utterance_1, utterance_2])
    assert total == MDDCounts(ta=3, tr=1, correct_diagnosis=1)


def test_evaluate_convenience_wrapper_matches_manual_pipeline() -> None:
    assert evaluate(["b", "a"], ["b", "u"], ["b", "u"]) == compute_metrics(
        evaluate_utterance(["b", "a"], ["b", "u"], ["b", "u"])
    )
