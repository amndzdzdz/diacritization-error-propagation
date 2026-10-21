"""The collapse diagnostic has to distinguish a canonical echo from a detector."""

from __future__ import annotations

import math

import pytest

from arabic_mdd.models.collapse import (
    D12_MIN_INFORMATIVE,
    D12_THRESHOLD,
    CollapseReport,
    canonical_copy_rate,
    collapse_report,
    d12_verdict,
    token_edit_distance,
)

CANON = ["b", "i", "s", "m"]
SAID = ["b", "i", "z", "m"]  # the speaker substituted s -> z


def test_edit_distance_basics() -> None:
    assert token_edit_distance(CANON, CANON) == 0
    assert token_edit_distance(CANON, SAID) == 1
    assert token_edit_distance(CANON, []) == 4
    assert token_edit_distance([], SAID) == 4
    assert token_edit_distance([], []) == 0


def test_edit_distance_is_symmetric() -> None:
    assert token_edit_distance(CANON, SAID) == token_edit_distance(SAID, CANON)


def test_edit_distance_counts_insertions_and_deletions() -> None:
    assert token_edit_distance(["b", "i"], ["b", "i", "s"]) == 1
    assert token_edit_distance(["b", "i", "s"], ["b", "s"]) == 1


def test_copy_rate_is_one_when_every_prediction_echoes_the_canonical() -> None:
    assert canonical_copy_rate([CANON, CANON], [CANON, CANON]) == 1.0


def test_copy_rate_is_zero_when_nothing_echoes() -> None:
    assert canonical_copy_rate([SAID, SAID], [CANON, CANON]) == 0.0


def test_copy_rate_counts_partial_echoes() -> None:
    assert canonical_copy_rate([CANON, SAID], [CANON, CANON]) == 0.5


def test_copy_rate_rejects_ragged_input() -> None:
    with pytest.raises(ValueError, match="canonicals"):
        canonical_copy_rate([CANON], [CANON, CANON])


def test_copy_rate_rejects_an_empty_dev_set() -> None:
    with pytest.raises(ValueError, match="empty dev set"):
        canonical_copy_rate([], [])


def test_a_collapsed_arm_is_flagged() -> None:
    # The failure this module exists for: prediction == canonical everywhere,
    # including on rows where the speaker demonstrably said something else.
    report = collapse_report([CANON, CANON], [CANON, CANON], [SAID, SAID])
    assert report.copy_rate == 1.0
    assert report.informative_utterances == 2
    assert report.canonical_preference == 1.0


def test_a_working_arm_is_not_flagged() -> None:
    # Prediction follows the audio, not the canonical, on the disagreements.
    report = collapse_report([SAID, SAID], [CANON, CANON], [SAID, SAID])
    assert report.copy_rate == 0.0
    assert report.canonical_preference == 0.0


def test_agreeing_utterances_are_excluded_from_the_preference() -> None:
    # Where canonical == target there is nothing to resolve, so such rows
    # cannot say whether the model read or copied.
    report = collapse_report(
        [CANON, CANON],
        [CANON, CANON],
        [CANON, SAID],  # first row agrees, second disagrees
    )
    assert report.informative_utterances == 1
    assert report.canonical_preference == 1.0
    # ...but an echo is still an echo, so copy_rate counts both.
    assert report.copy_rate == 1.0


def test_a_tie_does_not_count_as_preferring_the_canonical() -> None:
    # Conservative on purpose: ties must not let us claim collapse.
    # "b i x m" is one substitution from both "b i s m" and "b i z m".
    tied = ["b", "i", "x", "m"]
    report = collapse_report([tied], [CANON], [SAID])
    assert report.informative_utterances == 1
    assert report.canonical_preference == 0.0


def test_preference_is_nan_when_nothing_is_informative() -> None:
    # A dev set with no canonical/target disagreements cannot measure this.
    # Reporting 0.0 would read as "no collapse detected", which is a lie.
    report = collapse_report([CANON], [CANON], [CANON])
    assert report.informative_utterances == 0
    assert math.isnan(report.canonical_preference)


def test_report_rejects_ragged_input() -> None:
    with pytest.raises(ValueError, match="ragged"):
        collapse_report([CANON], [CANON], [CANON, SAID])


def test_report_serializes_for_a_json_artifact() -> None:
    report = collapse_report([CANON], [CANON], [SAID])
    assert report.as_dict() == {
        "utterances": 1,
        "copy_rate": 1.0,
        "informative_utterances": 1,
        "canonical_preference": 1.0,
    }


def test_report_round_trips_through_its_dict() -> None:
    # The ablated arm reaches D12 as JSON written by a different process.
    report = collapse_report([CANON, SAID], [CANON, CANON], [SAID, SAID])
    assert CollapseReport.from_dict(report.as_dict()) == report


def _report(preference: float, informative: int) -> CollapseReport:
    """A report with only the two fields D12 reads set meaningfully."""
    return CollapseReport(
        utterances=informative,
        copy_rate=0.0,
        informative_utterances=informative,
        canonical_preference=preference,
    )


def test_d12_fires_on_a_powered_difference_above_the_threshold() -> None:
    verdict = d12_verdict(_report(0.90, 1642), _report(0.60, 1642))
    assert verdict.delta == pytest.approx(0.30)
    assert verdict.powered
    assert verdict.fires
    assert verdict.verdict == "fires"


def test_d12_does_not_fire_on_a_small_difference() -> None:
    verdict = d12_verdict(_report(0.70, 1642), _report(0.60, 1642))
    assert verdict.powered
    assert not verdict.fires
    assert verdict.verdict == "does_not_fire"


def test_d12_threshold_is_exclusive() -> None:
    # Exactly at the threshold is not "more than" the threshold. Subtracting
    # from zero so the delta is the threshold bit-for-bit -- `0.60 + 0.15` is
    # 0.15000000000000002 away from 0.60 and would fire on rounding alone.
    verdict = d12_verdict(_report(D12_THRESHOLD, 1642), _report(0.0, 1642))
    assert verdict.delta == D12_THRESHOLD
    assert not verdict.fires


def test_d12_power_test_beats_a_firing_delta() -> None:
    # Week 7's exact trap: a huge delta over 3 informative utterances is noise,
    # and the absolute form of this rule "fired" there on a model that cannot
    # consume a canonical at all.
    verdict = d12_verdict(_report(1.0, 3), _report(0.0, 3))
    assert verdict.delta == pytest.approx(1.0)
    assert not verdict.powered
    assert not verdict.fires
    assert verdict.verdict == "underpowered"


def test_d12_power_test_takes_the_smaller_arm() -> None:
    # Disagreeing counts mean the two passes did not see the same clips; the
    # comparison must get weaker, not stronger.
    verdict = d12_verdict(_report(0.9, 10_000), _report(0.1, 7))
    assert verdict.informative_utterances == 7
    assert not verdict.powered


def test_d12_is_powered_exactly_at_the_minimum() -> None:
    verdict = d12_verdict(_report(0.9, D12_MIN_INFORMATIVE), _report(0.1, D12_MIN_INFORMATIVE))
    assert verdict.powered
    assert verdict.fires


def test_d12_cannot_fire_on_an_undefined_preference() -> None:
    # No informative utterances means no measurement, and NaN must not become a
    # verdict by sneaking through the comparison.
    verdict = d12_verdict(_report(float("nan"), 0), _report(float("nan"), 0))
    assert math.isnan(verdict.delta)
    assert not verdict.fires
    assert verdict.verdict == "underpowered"


def test_d12_direction_is_not_symmetric() -> None:
    # Swapping the arms asks a different question, and the rule must not answer
    # the same way: an off-by-one in the argument order would be invisible if it
    # did.
    conditioned, ablated = _report(0.90, 1642), _report(0.60, 1642)
    assert d12_verdict(conditioned, ablated).fires
    assert not d12_verdict(ablated, conditioned).fires


def test_d12_serializes_for_a_json_artifact() -> None:
    verdict = d12_verdict(_report(0.90, 1642), _report(0.60, 1642))
    assert verdict.as_dict() == {
        "delta": pytest.approx(0.30),
        "informative_utterances": 1642,
        "powered": True,
        "fires": True,
        "verdict": "fires",
    }
