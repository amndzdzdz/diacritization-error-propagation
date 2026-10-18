"""The output vocabulary has to stay index-compatible with the baseline."""

from __future__ import annotations

from pathlib import Path

import pytest

from arabic_mdd.data.phonetizer import SWS_ARABIC_INVENTORY
from arabic_mdd.models.vocab import RESERVED_TOKENS, PhonemeVocab, load_vocab

BASELINE_VOCAB = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "baseline_reproduction"
    / "vocab"
    / "sws_arabic.txt"
)


@pytest.fixture
def vocab() -> PhonemeVocab:
    return load_vocab(BASELINE_VOCAB)


def test_reserved_tokens_occupy_the_first_three_indices(vocab: PhonemeVocab) -> None:
    assert vocab.tokens[:3] == RESERVED_TOKENS
    assert (vocab.pad_idx, vocab.eos_idx, vocab.unk_idx) == (0, 1, 2)


def test_blank_is_pad_not_a_fourth_symbol(vocab: PhonemeVocab) -> None:
    # s3prl's expert builds nn.CTCLoss(blank=tokenizer.pad_idx). If these ever
    # diverge, every trained model silently learns a different blank class.
    assert vocab.blank_idx == vocab.pad_idx


def test_size_is_68_phonemes_plus_3_reserved(vocab: PhonemeVocab) -> None:
    assert len(vocab.phonemes) == 68
    assert vocab.size == 71


def test_phonemes_match_the_phonetizer_inventory(vocab: PhonemeVocab) -> None:
    # Same tokens as the set the phonetizer validates against; this module only
    # adds the ordering that the set cannot carry.
    assert set(vocab.phonemes) == set(SWS_ARABIC_INVENTORY)


def test_file_order_is_preserved_not_sorted(vocab: PhonemeVocab) -> None:
    # The file is not alphabetical. Sorting it would renumber all 68 classes
    # and make our predictions a permutation of the baseline's.
    assert vocab.phonemes[:4] == ("bb", "x", "yy", "s")
    assert vocab.phonemes != tuple(sorted(vocab.phonemes))
    assert vocab.to_idx("bb") == 3


def test_encode_appends_eos(vocab: PhonemeVocab) -> None:
    encoded = vocab.encode(["bb", "x"])
    assert encoded == [vocab.to_idx("bb"), vocab.to_idx("x"), vocab.eos_idx]


def test_encode_can_omit_eos(vocab: PhonemeVocab) -> None:
    assert vocab.encode(["bb"], append_eos=False) == [vocab.to_idx("bb")]


def test_unknown_token_maps_to_unk_rather_than_raising(vocab: PhonemeVocab) -> None:
    assert vocab.to_idx("not-a-phoneme") == vocab.unk_idx


def test_decode_stops_at_eos(vocab: PhonemeVocab) -> None:
    ids = [vocab.to_idx("bb"), vocab.eos_idx, vocab.to_idx("x")]
    assert vocab.decode(ids) == ["bb"]


def test_decode_drops_blanks(vocab: PhonemeVocab) -> None:
    ids = [vocab.pad_idx, vocab.to_idx("bb"), vocab.pad_idx]
    assert vocab.decode(ids) == ["bb"]


def test_decode_collapses_repeats_only_when_asked(vocab: PhonemeVocab) -> None:
    bb = vocab.to_idx("bb")
    assert vocab.decode([bb, bb]) == ["bb", "bb"]
    assert vocab.decode([bb, bb], collapse_repeats=True) == ["bb"]


def test_blank_separated_repeat_survives_collapsing(vocab: PhonemeVocab) -> None:
    # The whole point of the CTC blank: a geminate spelled as two emissions
    # must not collapse to one. Checking the *preceding index* rather than the
    # preceding emitted token is what makes this work.
    bb = vocab.to_idx("bb")
    assert vocab.decode([bb, vocab.pad_idx, bb], collapse_repeats=True) == ["bb", "bb"]


def test_round_trip_through_encode_and_decode(vocab: PhonemeVocab) -> None:
    phonemes = ["b", "i", "s", "m", "i", "ll", "aa", "h", "i"]
    assert vocab.decode(vocab.encode(phonemes)) == phonemes


def test_rejects_a_vocabulary_without_the_reserved_prefix() -> None:
    with pytest.raises(ValueError, match="must start with"):
        PhonemeVocab(tokens=("bb", "x"))


def test_rejects_duplicate_entries() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        PhonemeVocab.from_phonemes(["bb", "x", "bb"])


def test_loader_rejects_two_tokens_on_one_line(tmp_path: Path) -> None:
    # Not pedantry: index assignment is positional, so one split line shifts
    # every class after it.
    path = tmp_path / "vocab.txt"
    path.write_text("bb\nx yy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one token per line"):
        load_vocab(path)


def test_loader_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "vocab.txt"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no tokens"):
        load_vocab(path)


def test_loader_skips_a_trailing_blank_line(tmp_path: Path) -> None:
    path = tmp_path / "vocab.txt"
    path.write_text("bb\nx\n\n", encoding="utf-8")
    assert load_vocab(path).phonemes == ("bb", "x")
