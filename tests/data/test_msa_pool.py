import pytest

from arabic_mdd.data.msa_pool import (
    DIGITS,
    DUPLICATE,
    LATIN,
    NO_ARABIC,
    NO_TASHKEEL,
    PARTIALLY_VOWELIZED,
    PRE_VOWELIZED,
    SHORT,
    WORD_COUNT_MISMATCH,
    build_pool,
    classify,
    diacritization_rate,
    draw_block,
    skeleton,
    strip_diacritics,
)

BARE = "ولا تبخسوا الناس أشياءهم"
VOWELIZED = "وَلَا تَبْخَسُوا النَّاسَ أَشْيَاءَهُمْ"


def test_diacritization_rate_separates_bare_from_vowelized() -> None:
    assert diacritization_rate(BARE) == 0.0
    assert diacritization_rate(VOWELIZED) > 0.5


def test_diacritization_rate_is_none_without_arabic_letters() -> None:
    """Distinguishes "no marks" from "nothing to mark" — they filter differently."""
    assert diacritization_rate("Memorial Address 1998") is None


def test_strip_diacritics_and_skeleton() -> None:
    assert strip_diacritics(VOWELIZED) == BARE
    # The vowelizer's own normalization is punctuation removal, so the dedup
    # key has to ignore punctuation too.
    assert skeleton("هل تعذر ؟") == skeleton("هل تعذر")


def test_pre_vowelized_rows_are_excluded_but_partial_ones_are_kept() -> None:
    assert PRE_VOWELIZED in classify("a", VOWELIZED, VOWELIZED).flags
    assert not classify("a", VOWELIZED, VOWELIZED).eligible

    partial = classify("b", "وَلَا تَبْخَسُوا الناس أشياءهم", VOWELIZED)
    assert partial.flags == frozenset({PARTIALLY_VOWELIZED})
    assert partial.eligible


def test_punctuation_alone_is_not_a_word_count_mismatch() -> None:
    """Regression: raw whitespace tokens flag 8.4% of dev, all of it punctuation.

    The vowelizer strips punctuation by design (`docs/msa-arm.md` §4.1), so
    counting raw tokens scores that normalization as a dropped word. Only
    tokens carrying an Arabic letter count.
    """
    assert WORD_COUNT_MISMATCH not in classify("a", "هل تعذر لحظه ؟", "هَلْ تَعَذَّرَ لَحَظَهُ").flags


def test_dropped_words_are_flagged_not_excluded() -> None:
    """A vowelizer that drops a word is a real tool error — the arm's subject."""
    candidate = classify("a", BARE, "ۚ وَلَا تَبْخَسُوا")

    assert WORD_COUNT_MISMATCH in candidate.flags
    assert candidate.eligible


def test_missing_tashkeel_is_excluded() -> None:
    assert NO_TASHKEEL in classify("a", BARE, "   ").flags
    assert not classify("a", BARE, "   ").eligible


def test_no_arabic_is_excluded() -> None:
    assert not classify("a", "Memorial Address", "Memorial Address").eligible
    assert NO_ARABIC in classify("a", "Memorial Address", "Memorial Address").flags


def test_short_latin_and_digit_rows_are_kept_and_flagged() -> None:
    short = classify("a", "هل تعذر", "هَلْ تَعَذَّرَ")
    assert short.flags == frozenset({SHORT})
    assert short.eligible

    mixed = classify("b", "ظهرت الأغنية في Memorial Address 1998", VOWELIZED)
    assert {LATIN, DIGITS} <= mixed.flags
    assert mixed.eligible


def test_build_pool_dedups_by_skeleton() -> None:
    pool = build_pool(
        [
            ("u1", BARE, VOWELIZED),
            ("u2", BARE + " ؟", VOWELIZED),
            ("u3", "قال ربنا الله وحده", "قَالَ رَبُّنَا اللَّهُ وَحْدَهُ"),
        ]
    )

    assert len(pool.eligible) == 2
    assert pool.exclusion_counts()[DUPLICATE] == 1


def test_dedup_keeps_the_eligible_member_not_the_lowest_id() -> None:
    """Regression: dev's duplicate groups are mostly bare/pre-vowelized pairs.

    Keeping the lowest id discarded the usable member of 16 of 33 mixed groups,
    shrinking the pool for no reason. Eligibility outranks id order.
    """
    pool = build_pool([("u1", VOWELIZED, VOWELIZED), ("u2", BARE, VOWELIZED)])

    assert [c.id for c in pool.eligible] == ["u2"]


def test_draw_block_is_reproducible_and_seed_sensitive() -> None:
    pool = build_pool([(f"u{i:03d}", f"{BARE} {i}", VOWELIZED) for i in range(100)])

    first = [c.id for c in draw_block(pool, 10, seed=1)]
    assert first == [c.id for c in draw_block(pool, 10, seed=1)]
    assert first != [c.id for c in draw_block(pool, 10, seed=2)]


def test_draw_block_ignores_row_order() -> None:
    """The block must not change because the split was re-downloaded."""
    rows = [(f"u{i:03d}", f"{BARE} {i}", VOWELIZED) for i in range(100)]

    forward = [c.id for c in draw_block(build_pool(rows), 10, seed=1)]
    backward = [c.id for c in draw_block(build_pool(reversed(rows)), 10, seed=1)]

    assert forward == backward


def test_draw_block_never_returns_an_ineligible_row() -> None:
    rows = [(f"u{i:03d}", f"{BARE} {i}", VOWELIZED) for i in range(20)]
    rows += [(f"v{i:03d}", f"{VOWELIZED} {i}", VOWELIZED) for i in range(20)]

    assert all(c.eligible for c in draw_block(build_pool(rows), 20, seed=1))


def test_draw_block_refuses_to_overdraw() -> None:
    pool = build_pool([("u1", BARE, VOWELIZED)])

    with pytest.raises(ValueError, match="only 1 are eligible"):
        draw_block(pool, 2, seed=1)
