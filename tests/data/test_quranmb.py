from arabic_mdd.data.quranmb import MDDGroundTruth, parse_ground_truth_rows, score_predictions
from arabic_mdd.metrics.hierarchical import MDDCounts


def _row(
    example_id: str,
    reference: str,
    annotation: str,
    reference_arabic: str = "بَ",
    match_type: str = "exact",
) -> dict:
    return {
        "id": example_id,
        "reference_phoneme_string": reference,
        "annotation_phoneme_string": annotation,
        "reference_arabic_string": reference_arabic,
        "match_type": match_type,
    }


def test_parse_ground_truth_rows_normalizes_phonemes_and_keys_by_id() -> None:
    rows = [
        _row("q1", "b a", "b a"),
        _row("q2", "b a t", "b t"),
    ]
    result = parse_ground_truth_rows(rows)

    assert set(result) == {"q1", "q2"}
    assert result["q1"] == MDDGroundTruth(
        id="q1",
        canonical=["b", "a"],
        human_annotation=["b", "a"],
        reference_arabic="بَ",
        match_type="exact",
    )
    assert result["q2"].canonical == ["b", "a", "t"]
    assert result["q2"].human_annotation == ["b", "t"]


def test_score_predictions_matches_by_id_and_pools_counts() -> None:
    ground_truth = parse_ground_truth_rows(
        [
            _row("q1", "b a", "b a"),  # correct pronunciation
            _row("q2", "b a", "b u"),  # substitution error
        ]
    )
    predictions = {
        "q1": "b a",  # system agrees: TA
        "q2": "b u",  # system correctly diagnoses the substitution: TR + CD
    }

    counts = score_predictions(predictions, ground_truth)

    # q1 ("b a" / "b a" / "b a"): both phonemes correctly pronounced and
    # correctly predicted -> TA=2. q2 ("b a" / "b u" / "b u"): "b" is TA,
    # "a"->"u" is a correctly-diagnosed substitution -> TA=1, TR=1, CD=1.
    assert counts == MDDCounts(ta=3, tr=1, correct_diagnosis=1)


def test_score_predictions_skips_ids_without_ground_truth() -> None:
    ground_truth = parse_ground_truth_rows([_row("q1", "b a", "b a")])
    predictions = {"q1": "b a", "unknown_id": "x y z"}

    counts = score_predictions(predictions, ground_truth)

    assert counts == MDDCounts(ta=2)
