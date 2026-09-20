"""Loaders for the QuranMB.v2 benchmark (Qur'anic arm, gold-by-convention).

`IqraEval/QuranMB.v2` on its own ships only `ID`/`audio` -- no phoneme
labels. The canonical reference and human annotation
(what the reciter actually said) live in the separate, gated
`IqraEval/IqraEval_Test_GT` dataset, which would otherwise need to be
joined on `ID` (this is the concrete answer to week 1's open question of
whether QuranMB.v2 test labels are public or leaderboard-held: they're
held separately, access-gated, not bundled with the audio).

`safikhan/quran_mbv2_formatted` is a public HF dataset that already
performs that join (audio + `Reference_phn` + `Annotation_phn`, plus a
recovered Arabic reference text), so it is used here directly instead of
loading and joining the two source datasets ourselves.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from arabic_mdd.data.phonemes import parse_phoneme_sequence
from arabic_mdd.metrics.hierarchical import MDDCounts, aggregate, evaluate_utterance

QURANMB_DATASET = "safikhan/quran_mbv2_formatted"


@dataclass(frozen=True)
class MDDGroundTruth:
    """One QuranMB.v2 test utterance's canonical reference + human annotation."""

    id: str
    canonical: list[str]
    human_annotation: list[str]
    reference_arabic: str
    match_type: str


def parse_ground_truth_rows(rows: Iterable[Mapping[str, object]]) -> dict[str, MDDGroundTruth]:
    """Normalize raw `quran_mbv2_formatted` rows into `id -> MDDGroundTruth`."""
    result: dict[str, MDDGroundTruth] = {}
    for row in rows:
        example_id = row["id"]
        result[example_id] = MDDGroundTruth(  # type: ignore
            id=example_id,  # type: ignore
            canonical=parse_phoneme_sequence(row["reference_phoneme_string"]),  # type: ignore
            human_annotation=parse_phoneme_sequence(row["annotation_phoneme_string"]),  # type: ignore
            reference_arabic=row["reference_arabic_string"],  # type: ignore
            match_type=row["match_type"],  # type: ignore
        )
    return result


def load_ground_truth() -> dict[str, MDDGroundTruth]:
    """Load and normalize the full QuranMB.v2 test ground truth (network call).

    Public dataset, no HF auth required. Not covered by unit tests (see
    `tests/data/test_quranmb.py`); the pure `parse_ground_truth_rows`
    function above is what's tested.
    """
    import datasets

    ds = datasets.load_dataset(QURANMB_DATASET, split="train")
    return parse_ground_truth_rows(ds)  # type: ignore


def load_audio():
    """Load QuranMB.v2 test audio + ids for cluster-side inference.

    Uses the same combined dataset as `load_ground_truth` (it already
    carries `audio`), so cluster inference and scoring both read from one
    source instead of joining two separately-gated datasets.
    """
    import datasets

    return datasets.load_dataset(QURANMB_DATASET, split="train")


def score_predictions(
    predictions: Mapping[str, str],
    ground_truth: Mapping[str, MDDGroundTruth],
) -> MDDCounts:
    """Score a system's predictions against QuranMB.v2 ground truth.

    `predictions` maps utterance id to a whitespace-separated predicted
    phoneme string (the cluster job's expected output format). Utterances
    without ground truth are silently skipped, mirroring the official
    repo's `evaluate_from_dfs`, which inner-joins on id. Pool the returned
    `MDDCounts` across all utterances before calling
    `arabic_mdd.metrics.hierarchical.compute_metrics` -- this is what the
    week-3 gate is scored with.
    """
    per_utterance = []
    for example_id, predicted in predictions.items():
        truth = ground_truth.get(example_id)
        if truth is None:
            continue
        per_utterance.append(
            evaluate_utterance(
                truth.canonical, truth.human_annotation, parse_phoneme_sequence(predicted)
            )
        )
    return aggregate(per_utterance)
