"""The manifest filter, including the two failure modes that are silent in production."""

from __future__ import annotations

from pathlib import Path

import pytest

from arabic_mdd.data.holdout import (
    COLUMNS,
    Holdout,
    filter_rows,
    load_holdout,
    normalize_transcript,
    read_manifest,
    utterance_id,
    write_manifest,
)

# A phoneme ref as the TSV actually carries it: the transcript column is
# `phoneme_ref`, not Arabic text (`csv_to_tsv_with_transcripts.py`).
REF = "w a l aa t a b kh a s uu"
OTHER = "b i s m i l l a a h i"


def test_utterance_id_reads_the_iqra_train_form() -> None:
    assert utterance_id("train/wav/audio_12345.wav") == "12345"


def test_utterance_id_reads_the_tts_form_too() -> None:
    """It parses, but a speaker-named id can never match an `Iqra_train` id."""
    assert utterance_id("train/wav/audio_Amer_17.wav") == "Amer_17"


def test_utterance_id_is_none_for_an_unexpected_path() -> None:
    assert utterance_id("train/wav/00123.flac") is None


def test_normalize_transcript_collapses_whitespace_only() -> None:
    assert normalize_transcript("  w a   l aa\t") == "w a l aa"


def test_normalize_transcript_keeps_vowels() -> None:
    """Unlike `skeleton`: on a phoneme string, vowels are the distinction that matters."""
    assert normalize_transcript("b a") != normalize_transcript("b i")


def test_filter_drops_by_id() -> None:
    rows = [("train/wav/audio_1.wav", REF), ("train/wav/audio_2.wav", OTHER)]
    result = filter_rows(rows, Holdout(ids=frozenset({"1"}), phoneme_refs=frozenset()))
    assert [path for path, _ in result.kept] == ["train/wav/audio_2.wav"]
    assert len(result.dropped_by_id) == 1
    assert result.rows_in == 2


def test_filter_reaches_merged_tts_rows_by_phoneme_ref() -> None:
    """The hole an id-only holdout leaves: a synthesis of held-out text.

    `prepare_baseline.sh` step 4 merges Iqra_TTS into the train directory, and
    its filenames carry a speaker name rather than an `Iqra_train` id — so no
    id filter can reach it.
    """
    rows = [("train/wav/audio_Amer_17.wav", REF), ("train/wav/audio_Amer_18.wav", OTHER)]
    holdout = Holdout(ids=frozenset({"1"}), phoneme_refs=frozenset({REF}))
    result = filter_rows(rows, holdout)
    assert [path for path, _ in result.kept] == ["train/wav/audio_Amer_18.wav"]
    assert [path for path, _ in result.dropped_by_phoneme_ref] == ["train/wav/audio_Amer_17.wav"]
    assert not result.dropped_by_id


def test_filter_counts_an_id_match_once_not_twice() -> None:
    """A held-out row matches *both* keys; the report must not double-count it."""
    rows = [("train/wav/audio_1.wav", REF)]
    holdout = Holdout(ids=frozenset({"1"}), phoneme_refs=frozenset({REF}))
    report = filter_rows(rows, holdout).report()
    assert report["dropped_total"] == 1
    assert (report["dropped_by_id"], report["dropped_by_phoneme_ref"]) == (1, 0)


def test_filter_records_unparseable_paths_without_dropping_them() -> None:
    rows = [("train/wav/weird.wav", OTHER)]
    result = filter_rows(rows, Holdout(ids=frozenset({"1"}), phoneme_refs=frozenset()))
    assert result.unparseable_paths == ["train/wav/weird.wav"]
    assert len(result.kept) == 1


def test_filter_refuses_an_empty_holdout() -> None:
    """The silent failure this module exists to prevent: filtering nothing."""
    with pytest.raises(ValueError, match="refusing to 'filter' nothing"):
        filter_rows([("train/wav/audio_1.wav", REF)], Holdout(frozenset(), frozenset()))


def test_load_holdout_shares_phoneme_refs_across_splits() -> None:
    payload = {
        "holdout_train_ids": ["1", "2"],
        "holdout_dev_ids": ["9"],
        "holdout_phoneme_refs": [REF],
    }
    train = load_holdout(payload, split="train")
    dev = load_holdout(payload, split="dev")
    assert train.ids == frozenset({"1", "2"})
    assert dev.ids == frozenset({"9"})
    assert train.phoneme_refs == dev.phoneme_refs == frozenset({REF})


def test_load_holdout_rejects_a_foreign_payload() -> None:
    with pytest.raises(KeyError, match="holdout_train_ids missing"):
        load_holdout({"pool_ids": []}, split="train")


def test_manifest_roundtrips(tmp_path) -> None:
    path = tmp_path / "train.tsv"
    rows = [["train/wav/audio_1.wav", REF], ["train/wav/audio_2.wav", OTHER]]
    write_manifest(path, list(COLUMNS), rows)
    header, read_back = read_manifest(path)
    assert tuple(header) == COLUMNS
    assert read_back == rows


def test_read_manifest_rejects_an_unexpected_header(tmp_path) -> None:
    path = tmp_path / "train.tsv"
    path.write_text("file_path\ttranscript\na\tb\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header is"):
        read_manifest(path)


def test_write_manifest_refuses_to_write_nothing(tmp_path) -> None:
    """If every row matched, the keys are wrong — say so now, not in the dataloader."""
    with pytest.raises(ValueError, match="empty manifest"):
        write_manifest(tmp_path / "train.tsv", list(COLUMNS), [])


# --- the text-dependent arm's three-column manifest ------------------------


def test_read_manifest_accepts_the_canonical_column(tmp_path: Path) -> None:
    path = tmp_path / "train.tsv"
    path.write_text(
        f"path\tsentence\tcanonical\ntrain/wav/audio_1.wav\t{REF}\t{OTHER}\n",
        encoding="utf-8",
    )
    header, rows = read_manifest(path)
    assert header == ["path", "sentence", "canonical"]
    assert rows == [["train/wav/audio_1.wav", REF, OTHER]]


def test_read_manifest_rejects_a_differently_named_third_column(tmp_path: Path) -> None:
    # s3prl's dataset reads row[0] and row[1] and ignores the rest, so a
    # mislabelled third column would reach a conditioned run unnoticed.
    path = tmp_path / "train.tsv"
    path.write_text(f"path\tsentence\tphoneme_mis\ta.wav\t{REF}\t{OTHER}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header is"):
        read_manifest(path)


def test_read_manifest_rejects_a_short_row(tmp_path: Path) -> None:
    # A missing canonical becomes an empty attention row, which returns NaN
    # rather than raising once it reaches the conditioner.
    path = tmp_path / "train.tsv"
    path.write_text(
        f"path\tsentence\tcanonical\na.wav\t{REF}\t{OTHER}\nb.wav\t{REF}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="2 fields but the header declares 3"):
        read_manifest(path)


def test_filter_preserves_the_canonical_column(tmp_path: Path) -> None:
    # The regression this change exists to prevent: truncating rows to
    # (path, sentence) during filtering would strip the canonical and leave the
    # conditioned arm with nothing to condition on.
    rows = [
        ["train/wav/audio_keep.wav", OTHER, "c a n o n"],
        ["train/wav/audio_00015.wav", REF, "c a n o n"],
    ]
    result = filter_rows(rows, Holdout(ids=frozenset({"00015"}), phoneme_refs=frozenset()))
    assert result.kept == [["train/wav/audio_keep.wav", OTHER, "c a n o n"]]
    assert result.dropped_by_id == [["train/wav/audio_00015.wav", REF, "c a n o n"]]


def test_filter_does_not_key_on_the_canonical_column() -> None:
    # The canonical duplicates `sentence` for every Iqra_train row. Keying on
    # it as well would drop the same row twice and corrupt the audit counts.
    rows = [["train/wav/audio_keep.wav", OTHER, REF]]
    result = filter_rows(rows, Holdout(ids=frozenset(), phoneme_refs=frozenset({REF})))
    assert result.kept == rows
    assert result.report()["dropped_total"] == 0


def test_report_still_lists_dropped_paths_with_three_columns() -> None:
    rows = [["train/wav/audio_00015.wav", REF, REF]]
    result = filter_rows(rows, Holdout(ids=frozenset({"00015"}), phoneme_refs=frozenset()))
    assert result.report()["dropped_paths_by_id"] == ["train/wav/audio_00015.wav"]
