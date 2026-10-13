"""Tests for `arabic_mdd.metrics.der`.

The interesting cases are all definitional rather than algorithmic: which
positions land in the denominator, and what counts as the same diacritic.
"""

from __future__ import annotations

import pytest

from arabic_mdd.data.normalize import strip_diacritics
from arabic_mdd.metrics.der import (
    CATT_WITH_CASE_ENDINGS,
    CATT_WITHOUT_CASE_ENDINGS,
    MARKED_ONLY_WITH_CASE_ENDINGS,
    MARKED_ONLY_WITHOUT_CASE_ENDINGS,
    Score,
    Variant,
    arabic_words,
    marked_letters,
    score_corpus,
    score_utterance,
)

FATHA = "\u064e"
DAMMA = "\u064f"
KASRA = "\u0650"
SHADDA = "\u0651"
SUKUN = "\u0652"
DAGGER = "\u0670"


def test_variant_names_are_distinct_and_stable() -> None:
    """A reported number must be able to say which variant it is."""
    names = {
        CATT_WITH_CASE_ENDINGS.name,
        CATT_WITHOUT_CASE_ENDINGS.name,
        MARKED_ONLY_WITH_CASE_ENDINGS.name,
        MARKED_ONLY_WITHOUT_CASE_ENDINGS.name,
    }
    assert names == {"ce_all", "noce_all", "ce_markedonly", "noce_markedonly"}


def test_variant_has_no_default() -> None:
    with pytest.raises(TypeError):
        Variant()  # type: ignore[call-arg]


def test_identical_text_scores_zero() -> None:
    word = f"\u0643\u062a{FATHA}\u0628{FATHA}"
    score = score_utterance(word, word, CATT_WITH_CASE_ENDINGS)
    assert score.der == 0.0
    assert score.wer == 0.0
    assert score.skeleton_mismatch == 0
    assert score.positions == 3


def test_every_letter_is_in_the_all_denominator() -> None:
    """Three letters, two marked in the reference: axis 2 = all letters."""
    reference = f"\u0643{FATHA}\u062a{FATHA}\u0628"
    score = score_utterance(reference, reference, CATT_WITH_CASE_ENDINGS)
    assert score.positions == 3


def test_marked_only_denominator_excludes_bare_reference_letters() -> None:
    reference = f"\u0643{FATHA}\u062a{FATHA}\u0628"
    score = score_utterance(reference, reference, MARKED_ONLY_WITH_CASE_ENDINGS)
    assert score.positions == 2


def test_marked_only_forgives_an_omission_the_reference_also_makes() -> None:
    """The axis-2 choice is not cosmetic: the same pair of strings gives a
    different error count under each denominator."""
    reference = f"\u0643{FATHA}\u062a\u0628"
    hypothesis = f"\u0643{FATHA}\u062a{DAMMA}\u0628"

    lenient = score_utterance(reference, hypothesis, MARKED_ONLY_WITH_CASE_ENDINGS)
    strict = score_utterance(reference, hypothesis, CATT_WITH_CASE_ENDINGS)

    assert lenient.position_errors == 0
    assert strict.position_errors == 1


def test_without_case_endings_drops_the_final_letter() -> None:
    reference = f"\u0643{FATHA}\u062a{FATHA}\u0628{DAMMA}"
    hypothesis = f"\u0643{FATHA}\u062a{FATHA}\u0628{KASRA}"

    with_ce = score_utterance(reference, hypothesis, CATT_WITH_CASE_ENDINGS)
    without_ce = score_utterance(reference, hypothesis, CATT_WITHOUT_CASE_ENDINGS)

    assert (with_ce.positions, with_ce.position_errors) == (3, 1)
    assert (without_ce.positions, without_ce.position_errors) == (2, 0)


def test_without_case_endings_drops_the_position_even_when_unmarked() -> None:
    """The literature's definition is positional, not syntactic — the docstring
    says so, and this pins it so the proxy cannot drift into a real i'rab test."""
    reference = f"\u0643{FATHA}\u062a{FATHA}\u0628"
    hypothesis = f"\u0643{FATHA}\u062a{FATHA}\u0628{DAMMA}"

    without_ce = score_utterance(reference, hypothesis, CATT_WITHOUT_CASE_ENDINGS)
    assert without_ce.positions == 2
    assert without_ce.position_errors == 0


def test_shadda_harakah_order_is_not_an_error() -> None:
    """NFC reorders `<shadda><harakah>` on ~24.5% of rows; scoring mark strings
    instead of mark sets would read that as a diacritization error."""
    reference = f"\u0643{SHADDA}{FATHA}\u062a"
    hypothesis = f"\u0643{FATHA}{SHADDA}\u062a"
    assert score_utterance(reference, hypothesis, CATT_WITH_CASE_ENDINGS).position_errors == 0


def test_missing_shadda_is_an_error() -> None:
    reference = f"\u0643{SHADDA}{FATHA}\u062a"
    hypothesis = f"\u0643{FATHA}\u062a"
    assert score_utterance(reference, hypothesis, CATT_WITH_CASE_ENDINGS).position_errors == 1


def test_dagger_alef_is_scored() -> None:
    """Present in the WikiNews reference, and it spells a long vowel."""
    reference = f"\u0647{FATHA}{DAGGER}\u0630{KASRA}"
    hypothesis = f"\u0647{FATHA}\u0630{KASRA}"
    assert score_utterance(reference, hypothesis, CATT_WITH_CASE_ENDINGS).position_errors == 1


def test_quranic_annotation_marks_are_not_scored() -> None:
    """U+06D6-U+06ED are recitation editorial marks; no tool emits them, so
    charging every tool for their absence would be a constant penalty."""
    reference = f"\u0643{FATHA}\u062a{FATHA}\u06d6"
    hypothesis = f"\u0643{FATHA}\u062a{FATHA}"
    score = score_utterance(reference, hypothesis, CATT_WITH_CASE_ENDINGS)
    assert score.skeleton_mismatch == 0
    assert score.position_errors == 0


def test_skeleton_mismatch_is_refused_not_aligned() -> None:
    """Mishkal merges words (week 4, U+0001); such a row must not be aligned
    around and must not silently leave the denominator."""
    score = score_utterance(
        f"\u0630{FATHA}\u0647{FATHA}\u0628{FATHA} \u062b{FATHA}\u0645{FATHA}",
        f"\u0630{FATHA}\u0647{FATHA}\u0628{FATHA}\u062b{FATHA}\u0645{FATHA}",
        CATT_WITH_CASE_ENDINGS,
    )
    assert score.skeleton_mismatch == 1
    assert score.positions == 0
    assert score.utterances == 1


def test_a_dropped_word_is_a_skeleton_mismatch() -> None:
    score = score_utterance(
        f"\u0643{FATHA}\u062a{FATHA} \u0628{FATHA}\u0628{FATHA}",
        f"\u0643{FATHA}\u062a{FATHA}",
        CATT_WITH_CASE_ENDINGS,
    )
    assert score.skeleton_mismatch == 1


def test_punctuation_difference_is_not_a_skeleton_mismatch() -> None:
    """A tool that spaces punctuation differently has not edited the letters."""
    score = score_utterance(
        f"\u0646{FATHA}\u0639{FATHA}\u0645\u060c\u0646{FATHA}\u0639{FATHA}\u0645",
        f"\u0646{FATHA}\u0639{FATHA}\u0645 \u060c \u0646{FATHA}\u0639{FATHA}\u0645",
        CATT_WITH_CASE_ENDINGS,
    )
    assert score.skeleton_mismatch == 0
    assert score.words == 2


def test_wer_counts_words_not_positions() -> None:
    reference = f"\u0643{FATHA}\u062a{FATHA} \u0628{FATHA}\u0628{FATHA}"
    hypothesis = f"\u0643{DAMMA}\u062a{DAMMA} \u0628{FATHA}\u0628{FATHA}"
    score = score_utterance(reference, hypothesis, CATT_WITH_CASE_ENDINGS)
    assert score.words == 2
    assert score.word_errors == 1
    assert score.wer == 50.0
    assert score.position_errors == 2
    assert score.der == 50.0


def test_single_letter_word_has_no_positions_without_case_endings() -> None:
    score = score_utterance(f"\u0648{FATHA}", f"\u0648{DAMMA}", CATT_WITHOUT_CASE_ENDINGS)
    assert score.positions == 0
    assert score.words == 0
    assert score.der == 0.0


def test_corpus_is_micro_averaged_over_positions() -> None:
    """Published DER is a position rate, not a mean of per-utterance rates. A
    long clean utterance and a short dirty one must not weigh the same."""
    long_clean = (f"\u0643{FATHA}\u062a{FATHA}\u0628{FATHA}\u0645{FATHA}", None)
    short_dirty = (f"\u0643{FATHA}", f"\u0643{DAMMA}")

    pairs = [
        (long_clean[0], long_clean[0]),
        (short_dirty[0], short_dirty[1]),
    ]
    score = score_corpus(pairs, CATT_WITH_CASE_ENDINGS)
    assert score.positions == 5
    assert score.position_errors == 1
    assert score.der == pytest.approx(20.0)
    assert score.utterances == 2


def test_scores_pool_additively() -> None:
    a = score_utterance(f"\u0643{FATHA}", f"\u0643{DAMMA}", CATT_WITH_CASE_ENDINGS)
    b = score_utterance(f"\u062a{FATHA}", f"\u062a{FATHA}", CATT_WITH_CASE_ENDINGS)
    pooled = a + b
    assert pooled.positions == 2
    assert pooled.position_errors == 1
    assert pooled.utterances == 2


def test_pooling_across_variants_is_refused() -> None:
    a = score_utterance(f"\u0643{FATHA}", f"\u0643{FATHA}", CATT_WITH_CASE_ENDINGS)
    b = score_utterance(f"\u0643{FATHA}", f"\u0643{FATHA}", CATT_WITHOUT_CASE_ENDINGS)
    with pytest.raises(ValueError, match="cannot pool"):
        _ = a + b


def test_empty_corpus_has_no_rate() -> None:
    score = score_corpus([], CATT_WITH_CASE_ENDINGS)
    assert score.der == 0.0
    assert score.wer == 0.0
    assert score.positions == 0


def test_marked_letters_groups_marks_onto_the_preceding_letter() -> None:
    pairs = marked_letters(f"\u0643{SHADDA}{FATHA}\u062a{SUKUN}")
    assert [letter for letter, _ in pairs] == ["\u0643", "\u062a"]
    assert pairs[0][1] == frozenset({SHADDA, FATHA})
    assert pairs[1][1] == frozenset({SUKUN})


@pytest.mark.parametrize(
    "word",
    [
        f"\u0643{SHADDA}{FATHA}\u062a{SUKUN}",
        f"\u0647{FATHA}{DAGGER}\u0630{KASRA}",
        f"\u0643{FATHA}\u062a{FATHA}\u06d6",
        f"\u0643\u0640{FATHA}\u062a",
        "\u0643\u0653\u062a",
    ],
)
def test_marked_letters_agrees_with_strip_diacritics(word: str) -> None:
    """The invariant `score_utterance`'s positional comparison rests on: the
    letters here are exactly the skeleton the mismatch check compares. If these
    two ever disagree, index `i` stops meaning the same letter on both sides and
    the DER is quietly measuring the wrong positions."""
    letters = "".join(letter for letter, _ in marked_letters(word))
    assert letters == strip_diacritics(word)


def test_marked_letters_drops_a_leading_orphan_mark() -> None:
    assert marked_letters(f"{FATHA}\u0643") == [("\u0643", frozenset())]


def test_arabic_words_ignores_latin_and_digit_tokens() -> None:
    assert arabic_words(f"\u0643{FATHA}\u062a Memorial 2014") == [f"\u0643{FATHA}\u062a"]


def test_score_is_frozen() -> None:
    score = score_utterance(f"\u0643{FATHA}", f"\u0643{FATHA}", CATT_WITH_CASE_ENDINGS)
    assert isinstance(score, Score)
    with pytest.raises(AttributeError):
        score.positions = 99  # type: ignore[misc]
