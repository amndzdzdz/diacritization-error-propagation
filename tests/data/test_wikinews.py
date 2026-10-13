from pathlib import Path

from arabic_mdd.data.wikinews import Segment, Word, read_segments, skeleton_mismatches

HEADER = "dediac\toriginal_gold\tmax_gold\tmax_gold_alt"


def write_tsv(tmp_path: Path, rows: list[str], name: str = "wikinews.tsv") -> Path:
    path = tmp_path / name
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def test_blank_and_article_marker_rows_both_break_a_segment(tmp_path: Path) -> None:
    """A headline and the body after it are not one sentence."""
    tsv = write_tsv(
        tmp_path,
        [
            "كتب\tكَتَبَ\tكَتَبَ\t",
            "",
            "درس\tدَرَسَ\tدَرَسَ\t",
            "#\t\t\t",
            "قرأ\tقَرَأَ\tقَرَأَ\t",
        ],
    )
    segments = read_segments(tsv)
    assert [s.blank_text for s in segments] == ["كتب", "درس", "قرأ"]
    assert [s.index for s in segments] == [0, 1, 2]


def test_segments_without_arabic_are_dropped(tmp_path: Path) -> None:
    tsv = write_tsv(tmp_path, ["Reuters\tReuters\tReuters\t", "", "كتب\tكَتَبَ\tكَتَبَ\t"])
    assert len(read_segments(tsv)) == 1


def test_non_arabic_words_are_kept_inside_a_segment(tmp_path: Path) -> None:
    """They are part of the text a tool is handed, so they stay in blank/gold."""
    tsv = write_tsv(tmp_path, ["كتب\tكَتَبَ\tكَتَبَ\t", "Reuters\tReuters\tReuters\t"])
    segment = read_segments(tsv)[0]
    assert segment.blank_text == "كتب Reuters"
    assert len(segment.words) == 2
    assert len(segment.arabic_words) == 1


def test_missing_trailing_columns_are_tolerated(tmp_path: Path) -> None:
    tsv = write_tsv(tmp_path, ["كتب\tكَتَبَ"])
    word = read_segments(tsv)[0].words[0]
    assert word.max_gold == ""
    assert word.max_gold_alt == ""


def test_bucket_b_is_the_word_final_mark(tmp_path: Path) -> None:
    """The same positional i'rab proxy `metrics.der` uses, not a parse."""
    segment = Segment(index=0, words=[Word("كتب", "كَتَبَ", "كَتَبَ", "")])
    assert segment.bucket_counts() == (2, 1)


def test_sukun_counts_as_bucket_b_but_a_bare_final_letter_counts_in_neither() -> None:
    """Buckets count *marked* positions, so an unmarked i'rab slot is not one."""
    assert Segment(index=0, words=[Word("كتب", "كَتَبْ", "كَتَبْ", "")]).bucket_counts() == (2, 1)
    assert Segment(index=0, words=[Word("كتب", "كَتَب", "كَتَب", "")]).bucket_counts() == (2, 0)


def test_skeleton_mismatch_is_detected(tmp_path: Path) -> None:
    """Zero is a precondition for both consumers; see `skeleton_mismatches`."""
    good = write_tsv(tmp_path, ["كتب\tكَتَبَ\tكَتَبَ\t"], name="good.tsv")
    assert skeleton_mismatches(read_segments(good)) == 0

    bad = write_tsv(tmp_path, ["كتب\tدَرَسَ\tدَرَسَ\t"], name="bad.tsv")
    assert skeleton_mismatches(read_segments(bad)) == 1
