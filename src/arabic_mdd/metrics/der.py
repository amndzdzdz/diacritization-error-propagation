"""Diacritization Error Rate (DER) and diacritization Word Error Rate (WER).

RQ1's instrument: how wrong is an automatic diacritizer's output, measured
against a diacritized reference. Used for the EO-vs-ED choice (week-5 task 8),
for RQ1 on both arms, and — once the MSA block is annotated — for D5's
`tashkeel_sentence`-vs-`C_gold` comparison.

**This module exists because DER is not one number.** Published figures differ
on two axes, and a reported DER that does not name its variant cannot be
compared with anything. Plan §2 flags exactly this when it puts CATT's 5.43
next to KSAA-2026's 10.56. So the variant is a required argument here, not a
default: see `Variant` and the four module-level constants that name the
combinations.

Axis 1 — **case endings.** Arabic marks syntactic case on a word's last
letter, which is the hardest position (it needs syntax, not just lexical
lookup) and the one a speech application often does not care about. The
literature's "without case endings" means *excluding each word's final
letter*, which is a position-based proxy for a syntactic notion: it also
excludes word-final letters that bear no case at all. We adopt the proxy
because it is what the published numbers use, and name the field
`case_endings` rather than `irab` so the approximation stays visible.

Axis 2 — **the denominator.** Some papers score every Arabic letter; others
score only letters the reference actually marks. The second is a much smaller
denominator (fully-vowelized MSA leaves long vowels and the article's lam
bare, so `msa_pool.diacritization_rate` lands around 0.7-0.9), and it forgives
a tool for omitting a mark the reference also omits. The same predictions give
materially different numbers.

**Alignment is refused, not approximated.** A diacritizer is supposed to add
marks to text without touching its letters, so reference and hypothesis should
share a consonantal skeleton exactly, and position *i* then means the same
letter in both. Where they do not, the tool has edited the text and no
principled letter-to-letter correspondence remains. Rather than invent an
alignment, those utterances are counted in `skeleton_mismatch` and excluded
from the rate, which is reported next to it.

That is not a formality. Week 4 measured Mishkal merging words via a U+0001
artifact, and per-utterance phoneme conservation failing by >10% on 5.14% of
its rows. A DER that quietly aligned around such output would report Mishkal
as *better* than it is, by deleting its worst failures from the denominator.
`skeleton_mismatch` is therefore a result in its own right, and a non-zero one
is reported with the DER, never instead of it.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from arabic_mdd.data.normalize import (
    DAGGER_ALEF,
    DIACRITICS,
    TATWEEL,
    is_arabic_letter,
    strip_diacritics,
)

#: Marks a diacritizer is held responsible for predicting: harakat, tanwin,
#: shadda and sukun (U+064B-U+0652), plus dagger alef.
#:
#: Narrower than `normalize.DIACRITICS`, deliberately. That set answers "what
#: is not part of the skeleton", so it includes the Qur'anic recitation
#: annotations (U+06D6-U+06ED) and the hamza/madda marks (U+0653-U+0655). Those
#: are editorial or orthographic, no tool under test emits them, and scoring
#: their absence as an error would charge every tool for them on the Qur'anic
#: arm.
#:
#: Dagger alef is *in*: the WikiNews reference writes it (37 occurrences in the
#: Darwish gold, 163 in the maximal gold) and it spells a long vowel, so
#: excluding it would silently excuse a real omission.
SCORED_MARKS: frozenset[str] = frozenset({chr(c) for c in range(0x064B, 0x0653)} | {DAGGER_ALEF})


@dataclass(frozen=True)
class Variant:
    """Which DER definition. Both fields are required — there is no default."""

    case_endings: bool
    """Score each word's final letter. False = the literature's "without case
    endings", which drops that position whether or not it bears case."""

    score_unmarked_reference: bool
    """Count reference letters carrying no mark. False = score only positions
    the reference marks, shrinking the denominator and forgiving agreed-on
    omissions."""

    @property
    def name(self) -> str:
        return (
            f"{'ce' if self.case_endings else 'noce'}"
            f"_{'all' if self.score_unmarked_reference else 'markedonly'}"
        )


#: The variant CATT's Table 5 reports (arXiv 2407.03236v3): every letter in the
#: denominator, word-final letters included. EO 5.425 / ED 5.963 on WikiNews.
CATT_WITH_CASE_ENDINGS = Variant(case_endings=True, score_unmarked_reference=True)

#: CATT's "without case endings" column. EO 3.105 / ED 3.631.
CATT_WITHOUT_CASE_ENDINGS = Variant(case_endings=False, score_unmarked_reference=True)

#: The stricter denominator, both ways. Not a published target; carried so a
#: sensitivity row can show what the axis-2 choice is worth.
MARKED_ONLY_WITH_CASE_ENDINGS = Variant(case_endings=True, score_unmarked_reference=False)
MARKED_ONLY_WITHOUT_CASE_ENDINGS = Variant(case_endings=False, score_unmarked_reference=False)


@dataclass(frozen=True)
class Score:
    """Counts first, rates derived. Pooled by `+`, so corpus DER is
    micro-averaged over positions rather than averaged over utterances —
    the former is what published DER means."""

    variant: Variant
    positions: int
    position_errors: int
    words: int
    word_errors: int
    utterances: int
    skeleton_mismatch: int

    @property
    def der(self) -> float:
        """Percent of scored letter positions whose marks differ. 0.0 if empty."""
        return 100.0 * self.position_errors / self.positions if self.positions else 0.0

    @property
    def wer(self) -> float:
        """Percent of words with at least one wrong scored position."""
        return 100.0 * self.word_errors / self.words if self.words else 0.0

    def __add__(self, other: Score) -> Score:
        if self.variant != other.variant:
            raise ValueError(f"cannot pool {self.variant.name} with {other.variant.name}")
        return Score(
            variant=self.variant,
            positions=self.positions + other.positions,
            position_errors=self.position_errors + other.position_errors,
            words=self.words + other.words,
            word_errors=self.word_errors + other.word_errors,
            utterances=self.utterances + other.utterances,
            skeleton_mismatch=self.skeleton_mismatch + other.skeleton_mismatch,
        )


def marked_letters(word: str) -> list[tuple[str, frozenset[str]]]:
    """Split a word into `(letter, marks)` pairs in letter order.

    Marks are a *set*, so `<shadda><fatha>` and `<fatha><shadda>` compare
    equal. They encode the same thing and differ only by Unicode canonical
    ordering, which `normalize`'s docstring warns NFC reorders on ~24.5% of
    rows — comparing mark strings would score that reordering as a
    diacritization error.

    Marks with no preceding letter (a leading mark, which is malformed) are
    dropped rather than raised on: a diacritizer emitting one is a tool bug to
    be measured, and it will surface as a skeleton mismatch or as wrong marks
    elsewhere in the word.

    A character that `strip_diacritics` removes but `SCORED_MARKS` does not
    cover — a Qur'anic recitation annotation, a hamza/madda mark, tatweel — is
    ignored entirely: neither a letter nor a scored position. The letter
    sequence this returns is therefore exactly `strip_diacritics(word)`, which
    is what makes the skeleton check in `score_utterance` a valid guarantee
    that index *i* refers to the same letter on both sides. Letting such a
    character count as a letter here instead would desynchronize the two
    sequences whenever only one side wrote it.
    """
    pairs: list[tuple[str, set[str]]] = []
    for ch in word:
        if ch in SCORED_MARKS:
            if pairs:
                pairs[-1][1].add(ch)
            continue
        if ch in DIACRITICS or ch == TATWEEL:
            continue
        pairs.append((ch, set()))
    return [(letter, frozenset(marks)) for letter, marks in pairs]


def arabic_words(text: str) -> list[str]:
    """Whitespace tokens containing an Arabic letter, with marks preserved.

    Punctuation becomes a space before splitting, so a reference writing
    `نعم،نعم` and a tool writing `نعم ، نعم` yield the same two words instead
    of registering as a skeleton mismatch. NFC composes letter+hamza sequences
    so that letter comparison is not defeated by encoding choice; it is safe
    here because this runs on text, never on phonetizer output (see the
    ordering constraint in `normalize`).
    """
    folded = "".join(
        " " if unicodedata.category(ch).startswith("P") else ch
        for ch in unicodedata.normalize("NFC", text)
    )
    return [token for token in folded.split() if any(is_arabic_letter(ch) for ch in token)]


def score_utterance(reference: str, hypothesis: str, variant: Variant) -> Score:
    """Score one hypothesis against one reference.

    Returns a `Score` with `skeleton_mismatch=1` and no positions when the two
    do not share a consonantal skeleton word for word — see the module
    docstring on why that is refused rather than aligned.
    """
    ref_words = arabic_words(reference)
    hyp_words = arabic_words(hypothesis)

    if [strip_diacritics(w) for w in ref_words] != [strip_diacritics(w) for w in hyp_words]:
        return Score(
            variant=variant,
            positions=0,
            position_errors=0,
            words=0,
            word_errors=0,
            utterances=1,
            skeleton_mismatch=1,
        )

    positions = position_errors = words = word_errors = 0
    for ref_word, hyp_word in zip(ref_words, hyp_words, strict=True):
        ref_pairs = marked_letters(ref_word)
        hyp_pairs = marked_letters(hyp_word)
        scored = len(ref_pairs) if variant.case_endings else len(ref_pairs) - 1

        word_positions = word_errs = 0
        for index in range(max(scored, 0)):
            ref_marks = ref_pairs[index][1]
            if not variant.score_unmarked_reference and not ref_marks:
                continue
            word_positions += 1
            if ref_marks != hyp_pairs[index][1]:
                word_errs += 1

        positions += word_positions
        position_errors += word_errs
        if word_positions:
            words += 1
            word_errors += 1 if word_errs else 0

    return Score(
        variant=variant,
        positions=positions,
        position_errors=position_errors,
        words=words,
        word_errors=word_errors,
        utterances=1,
        skeleton_mismatch=0,
    )


def score_corpus(pairs: Iterable[tuple[str, str]], variant: Variant) -> Score:
    """Micro-average `(reference, hypothesis)` pairs into one `Score`."""
    total = Score(
        variant=variant,
        positions=0,
        position_errors=0,
        words=0,
        word_errors=0,
        utterances=0,
        skeleton_mismatch=0,
    )
    for reference, hypothesis in pairs:
        total = total + score_utterance(reference, hypothesis, variant)
    return total
