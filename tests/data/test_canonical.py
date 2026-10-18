"""Where column three comes from, and the one case that must never be guessed."""

from __future__ import annotations

import pytest

from arabic_mdd.data.canonical import (
    build_canonical_rows,
    canonical_relative_path,
    count_unknown_tokens,
    is_iqra_train_path,
)

REF = "w a l aa t a b kh a s uu"
MIS = "w a l aa t a b k a s uu"  # the synthesized mispronunciation: kh -> k


def _no_files(_: str) -> None:
    """A filesystem with no canonicals in it."""
    return None


# --- which half of the training set is this row from? ----------------------


def test_numeric_id_is_an_iqra_train_row() -> None:
    assert is_iqra_train_path("train/wav/audio_12345.wav")


def test_speaker_named_row_is_not_an_iqra_train_row() -> None:
    """The merged Iqra_TTS form. Its `sentence` is phoneme_mis, not the canonical."""
    assert not is_iqra_train_path("train/wav/audio_Amer_17.wav")


def test_unparseable_path_is_not_an_iqra_train_row() -> None:
    assert not is_iqra_train_path("train/wav/00123.flac")


def test_canonical_path_mirrors_the_transcript_substitution() -> None:
    assert (
        canonical_relative_path("train/wav/audio_Amer_17.wav")
        == "train/canonicals/canonical_Amer_17.txt"
    )


def test_canonical_path_refuses_an_underivable_name() -> None:
    with pytest.raises(ValueError, match="audio_<id>.wav"):
        canonical_relative_path("train/wav/00123.flac")


# --- the build ------------------------------------------------------------


def test_iqra_train_row_takes_its_canonical_from_the_sentence() -> None:
    """`download_hugg_data.py` writes phoneme_ref as the transcript, so they coincide."""
    build = build_canonical_rows([["train/wav/audio_1.wav", REF]], _no_files)
    assert build.rows == [["train/wav/audio_1.wav", REF, REF]]
    assert (build.from_sentence, build.from_file) == (1, 0)


def test_iqra_train_row_never_touches_the_filesystem() -> None:
    def explode(_: str) -> str:
        raise AssertionError("a numeric-id row has no business reading a canonical file")

    build = build_canonical_rows([["train/wav/audio_1.wav", REF]], explode)
    assert build.rows[0][2] == REF


def test_tts_row_takes_its_canonical_from_the_file_not_the_sentence() -> None:
    # The row that makes the whole arm meaningful: target != canonical.
    build = build_canonical_rows(
        [["train/wav/audio_Amer_17.wav", MIS]],
        lambda path: REF if path == "train/canonicals/canonical_Amer_17.txt" else None,
    )
    assert build.rows == [["train/wav/audio_Amer_17.wav", MIS, REF]]
    assert (build.from_sentence, build.from_file) == (0, 1)


def test_tts_row_with_no_canonical_is_dropped_and_reported_not_filled_in() -> None:
    """The silent failure this module exists to prevent.

    Defaulting to `sentence` here would set canonical == phoneme_mis on exactly
    the 52 h that punish canonical-copying, leaving a manifest that trains and
    converges with the anti-collapse signal deleted.
    """
    build = build_canonical_rows([["train/wav/audio_Amer_17.wav", MIS]], _no_files)
    assert build.rows == []
    assert build.missing_canonical == ["train/wav/audio_Amer_17.wav"]


def test_a_blank_canonical_file_counts_as_missing() -> None:
    # An empty canonical becomes an all-masked attention row, which returns
    # NaN rather than raising.
    build = build_canonical_rows([["train/wav/audio_Amer_17.wav", MIS]], lambda _: "   \n")
    assert build.missing_canonical == ["train/wav/audio_Amer_17.wav"]


def test_canonical_whitespace_is_collapsed() -> None:
    build = build_canonical_rows([["train/wav/audio_Amer_17.wav", MIS]], lambda _: f"  {REF}\n")
    assert build.rows[0][2] == REF


def test_build_is_idempotent_on_a_three_column_manifest() -> None:
    rows = [["train/wav/audio_Amer_17.wav", MIS, REF]]
    build = build_canonical_rows(rows, _no_files)
    assert build.rows == rows
    assert build.passed_through == 1
    # Counted apart, so a rerun's report cannot be read as a fresh build's.
    assert (build.from_sentence, build.from_file) == (0, 0)


def test_a_mixed_manifest_reports_both_halves() -> None:
    rows = [
        ["train/wav/audio_1.wav", REF],
        ["train/wav/audio_Amer_17.wav", MIS],
        ["train/wav/audio_2.wav", REF],
    ]
    build = build_canonical_rows(rows, lambda _: REF)
    report = build.report()
    assert report["rows"] == 3
    assert report["canonical_from_sentence"] == 2
    assert report["canonical_from_file"] == 1
    assert report["missing_canonical"] == 0


def test_unparseable_path_raises_rather_than_being_skipped() -> None:
    with pytest.raises(ValueError, match="audio_<id>.wav"):
        build_canonical_rows([["train/wav/00123.flac", REF]], _no_files)


# --- the vocabulary pass --------------------------------------------------


def test_unknown_tokens_are_counted_with_examples() -> None:
    vocabulary = frozenset(REF.split())
    rows = [["a.wav", REF, f"<sil> {REF} <sil>"]]
    count, examples = count_unknown_tokens(rows, vocabulary)
    assert (count, examples) == (2, ["<sil>"])


def test_a_clean_canonical_has_no_unknown_tokens() -> None:
    count, examples = count_unknown_tokens([["a.wav", REF, REF]], frozenset(REF.split()))
    assert (count, examples) == (0, [])


def test_the_target_column_is_not_checked() -> None:
    # Only the canonical is validated: `sentence` is the baseline's own CTC
    # target and changing what we accept there would change the comparison.
    count, _ = count_unknown_tokens([["a.wav", "zzz qqq", REF]], frozenset(REF.split()))
    assert count == 0
