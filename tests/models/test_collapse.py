"""The collapse diagnostic has to distinguish a canonical echo from a detector."""

from __future__ import annotations

import math

import pytest

from arabic_mdd.models.collapse import (
    canonical_copy_rate,
    collapse_report,
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
