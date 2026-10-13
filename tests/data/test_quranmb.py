from arabic_mdd.data.quranmb import MDDGroundTruth, parse_ground_truth_rows, score_predictions
from arabic_mdd.metrics.hierarchical import MDDCounts


def _row(
    example_id: str,
    reference: str,
    annotation: str,
    reference_arabic: str = "بَ",
    match_type: str = "exact",
    match_distance: int = 0,
) -> dict:
    return {
        "id": example_id,
        "reference_phoneme_string": reference,
        "annotation_phoneme_string": annotation,
        "reference_arabic_string": reference_arabic,
        "match_type": match_type,
        "match_distance": match_distance,
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
        match_distance=0,
    )
    assert result["q2"].canonical == ["b", "a", "t"]
    assert result["q2"].human_annotation == ["b", "t"]


def test_match_distance_is_exposed() -> None:
    """The column that proves `reference_arabic` is circular — it must be
    readable from the loader, not only from a week-4 document."""
    result = parse_ground_truth_rows([_row("q1", "b a", "b a", match_distance=3)])
    assert result["q1"].match_distance == 3


def test_prompt_index_groups_takes_and_speakers() -> None:
    """`id` is `speaker_promptindex[take]`. One prompt read by two speakers,
    with a retake, is one group."""
    rows = [
        _row("00000_00042", "b a", "b a"),
        _row("00005_00042", "b a", "b a"),
        _row("00005_00042472", "b a", "b a"),
    ]
    result = parse_ground_truth_rows(rows)
    assert {gt.prompt_index for gt in result.values()} == {"00042"}


def test_prompt_index_separates_the_pausal_pair() -> None:
    """Prompts 9 and 42 share their Arabic string but not their phoneme string
    (`wasaTanaa` vs `wasATAn`). Keyed on text they merge; keyed on prompt index
    they stay two, which is the distinction the Qur'anic arm needs."""
    shared_arabic = "وَكَذَٰلِكَ جَعَلْنَاكُمْ أُمَّةً وَسَطًا"
    rows = [
        _row("00001_00009", "w a s a T A n aa", "w a s a T A n aa", shared_arabic, "fuzzy", 1),
        _row("00001_00042", "w a s A T A n", "w a s A T A n", shared_arabic, "fuzzy", 1),
    ]
    result = parse_ground_truth_rows(rows)

    assert len({gt.reference_arabic for gt in result.values()}) == 1
    assert {gt.prompt_index for gt in result.values()} == {"00009", "00042"}


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
