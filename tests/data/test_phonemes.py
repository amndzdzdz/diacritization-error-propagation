from arabic_mdd.data.phonemes import parse_phoneme_sequence


def test_splits_on_whitespace() -> None:
    assert parse_phoneme_sequence("b a t") == ["b", "a", "t"]


def test_collapses_repeated_whitespace() -> None:
    assert parse_phoneme_sequence("b   a\tt") == ["b", "a", "t"]


def test_empty_string_is_empty_sequence() -> None:
    assert parse_phoneme_sequence("") == []
