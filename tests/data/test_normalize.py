from arabic_mdd.data.normalize import (
    DIGITS,
    LATIN,
    NO_MSA_EQUIVALENT,
    NON_PHONEMIC_MARK,
    PRESENTATION_FORM,
    PUNCTUATION,
    SCRIPT_VARIANT,
    SUPPORTED,
    UNSUPPORTED,
    normalize,
    unsupported_characters,
)
from arabic_mdd.data.phonetizer import Phonetizer


def test_supported_is_derived_from_the_vendored_map() -> None:
    """36 letters + 8 combining marks. Hand-listing this would let it drift."""
    assert len(SUPPORTED) == 44
    assert "\u0628" in SUPPORTED  # beh
    assert "\u0651" in SUPPORTED  # shadda
    assert "\u0671" not in SUPPORTED  # alef wasla — the measured gap


def test_clean_arabic_passes_through_unchanged() -> None:
    text = "وَلَا تَبْخَسُوا النَّاسَ أَشْيَاءَهُمْ"

    result = normalize(text)

    assert result.text == text
    assert result.flags == frozenset()
    assert result.fair


def test_punctuation_becomes_a_space_rather_than_disappearing() -> None:
    """Deleting it merges the neighbouring words, which changes the phonemes.

    Not a theoretical worry: across a word boundary the phonetizer applies
    vowel-length and glide rules, so `فِي الشَّمْسِ` phonetizes to 6 tokens
    spaced and 8 merged.
    """
    result = normalize("فِي—الشَّمْسِ")

    assert result.flags == frozenset({PUNCTUATION})
    assert result.text == "فِي الشَّمْسِ"

    phonetizer = Phonetizer(convention="pausal")
    assert phonetizer.phonetize(result.text) == ["f", "ii", "$$", "a", "m", "s"]
    assert phonetizer.phonetize("فِيالشَّمْسِ") == ["f", "i", "y", "aa", "$$", "a", "m", "s"]


def test_persian_lookalikes_are_mapped_to_their_arabic_equivalents() -> None:
    result = normalize("ی ک ھ")

    assert SCRIPT_VARIANT in result.flags
    assert result.text == "ي ك ه"
    # Same letter, different orthographic tradition — the row stays usable.
    assert result.fair


def test_letters_with_no_msa_equivalent_are_dropped_and_the_row_flagged() -> None:
    result = normalize("چ")

    assert NO_MSA_EQUIVALENT in result.flags
    assert result.dropped == ("چ",)
    assert not result.fair


def test_presentation_forms_are_folded() -> None:
    result = normalize("\ufefb")  # lam-alef ligature

    assert PRESENTATION_FORM in result.flags
    assert result.text == "لا"


def test_presentation_form_flag_ignores_combining_mark_reordering() -> None:
    """Regression: NFKC reorders `<shadda><harakah>` on ~24.5% of CV rows.

    Flagging "NFKC changed something" reported 6% of utterances as containing
    presentation forms when the whole 71,391-row corpus holds six such
    characters.
    """
    result = normalize("إِنَّ")  # contains shadda + harakah

    assert PRESENTATION_FORM not in result.flags


def test_latin_is_flagged_unfair_because_it_phonetizes_into_garbage() -> None:
    """`Memorial` transliterates as Arabic and invents five phonemes."""
    assert Phonetizer(convention="pausal").phonetize("Memorial") == ["m", "r", "i", "a", "l"]

    result = normalize("ظهرت في Memorial Address")

    assert LATIN in result.flags
    assert not result.fair
    assert not any("a" <= ch.lower() <= "z" for ch in result.text)


def test_digits_are_flagged_unfair() -> None:
    result = normalize("سنة 1998")

    assert DIGITS in result.flags
    assert not result.fair


def test_dagger_alef_survives_normalization() -> None:
    """It is `phonetizer.normalize`'s business, and it is live in 3.8% of rows."""
    assert normalize("مُوسَىٰ").text == "مُوسَىٰ"


def test_unsupported_characters_reports_what_would_vanish() -> None:
    assert unsupported_characters("مُوسَىٰ") == {}
    assert unsupported_characters("ٱلْحَمْدُ") == {"\u0671": 1}
    assert unsupported_characters("نعم☭") == {"☭": 1}


def test_normalization_is_idempotent() -> None:
    text = "ظهرت ی چ — Memorial \ufefb سنة 1998"

    once = normalize(text).text

    assert normalize(once).text == once


def test_quranic_annotation_marks_are_dropped_as_non_phonemic() -> None:
    """U+06D6 etc. are nonspacing marks with no phonemic content.

    Dropping them is validated, not assumed: 82 occur in `tashkeel_sentence`,
    which round-tripped at 99.85% with the phonetizer discarding them. They
    must not be reported as unexpected characters.
    """
    result = normalize("نَعَمْ\u06d6")

    assert NON_PHONEMIC_MARK in result.flags
    assert UNSUPPORTED not in result.flags
    assert result.fair
    assert result.text == "نَعَمْ"


def test_genuinely_unexpected_characters_are_still_flagged() -> None:
    result = normalize("نعم\u262d")  # HAMMER AND SICKLE

    assert UNSUPPORTED in result.flags
    assert result.dropped == ("\u262d",)
