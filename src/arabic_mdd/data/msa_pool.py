"""The MSA arm's annotation sampling frame — which utterances are eligible, and the draw.

`docs/msa-arm.md` §3.3b D4. The arm hand-annotates `C_gold` (correct
diacritization) for a block of `Iqra_train` utterances. Two things about that
have to be decided *before* a human starts, because neither is reversible
afterwards without discarding work:

* **Which rows are eligible.** `sentence` is not the bare undiacritized input
  the arm originally assumed (§4.1): a fifth of the dev split arrives already
  vowelized. Annotating those would measure the uploader's vowelizer rather
  than ours.
* **Which eligible rows are drawn.** A block chosen by scrolling is a block
  whose selection cannot be reproduced or audited, and the pilot already had
  to declare one mid-stream deviation (§3.3a). The draw here is seeded.

The filter is deliberately **thin**. Only three predicates exclude a row;
everything else is kept and *flagged*, so that the flag can appear as a
breakdown row in the results instead of as a silent deletion. That asymmetry
is the lesson from §3.3a: a word-count mismatch between `sentence` and
`tashkeel_sentence` means the vowelizer dropped or merged a word, which is a
genuine tool error and exactly what the arm is measuring. Filtering those out
would delete real errors from the denominator and bias the headline *down*.

Thresholds are reused from §4.1 (0.50 "heavily diacritized", 0.05
"effectively undiacritized") rather than invented here, so the pool is
comparable with the corpus-level numbers already reported.
"""

from __future__ import annotations

import random
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

from arabic_mdd.data.normalize import (
    DIACRITICS,
    TATWEEL,
    is_arabic_letter,
    skeleton,
    strip_diacritics,
)

# Re-exported: this module is a sampling frame, and the three names above are
# the normalization contract's (§4.1), promoted into `normalize` in week 5 so
# that RQ1's stripping step and `metrics.der` share one implementation with the
# draw. Imported here rather than redefined so the committed block stays a
# function of the same bytes it was drawn with.
__all__ = [
    "BARE_RATE",
    "DIACRITICS",
    "EXCLUDING",
    "MIN_WORDS",
    "PRE_VOWELIZED_RATE",
    "TATWEEL",
    "Candidate",
    "Pool",
    "build_pool",
    "classify",
    "diacritization_rate",
    "draw_block",
    "is_arabic_letter",
    "skeleton",
    "strip_diacritics",
]

# §4.1's bands. A row at or above `PRE_VOWELIZED_RATE` is not usable as
# undiacritized input; below `BARE_RATE` it is effectively bare, which is what
# the arm wants. The gap between them is the awkward middle: real text with a
# few marks already on it, kept because excluding it would quietly restrict the
# arm to the easiest orthography.
PRE_VOWELIZED_RATE = 0.50
BARE_RATE = 0.05

# Below this, an utterance has too few words for a case-ending judgment to be
# interesting (i'rab is syntactic). Kept and flagged, not excluded.
MIN_WORDS = 3

# --- flags ----------------------------------------------------------------
NO_ARABIC = "no_arabic"
NO_TASHKEEL = "no_tashkeel"
PRE_VOWELIZED = "pre_vowelized"
DUPLICATE = "duplicate"

PARTIALLY_VOWELIZED = "partially_vowelized"
WORD_COUNT_MISMATCH = "word_count_mismatch"
SHORT = "short"
LATIN = "latin"
DIGITS = "digits"

#: Flags that remove a row from the pool. Everything else is kept and reported.
EXCLUDING: frozenset[str] = frozenset({NO_ARABIC, NO_TASHKEEL, PRE_VOWELIZED, DUPLICATE})


def diacritization_rate(text: str) -> float | None:
    """Fraction of Arabic letters carrying at least one diacritic.

    `None` when the text has no Arabic letters at all. A fully diacritized
    string does not reach 1.0: long vowels (alef/waw/ya as madd) and the
    definite article's lam are conventionally bare, so well-formed
    fully-vowelized MSA lands around 0.7–0.9. This is the same definition that
    produced §4.1's corpus-level rates.
    """
    letters = 0
    marked = 0
    chars = list(text)
    for i, ch in enumerate(chars):
        if not is_arabic_letter(ch):
            continue
        letters += 1
        if i + 1 < len(chars) and chars[i + 1] in DIACRITICS:
            marked += 1
    return marked / letters if letters else None


def _arabic_words(text: str) -> list[str]:
    """Whitespace tokens containing at least one Arabic letter.

    Tokens without one are punctuation or Latin runs. The vowelizer removes
    punctuation as part of its own normalization (§4.1), so counting raw
    whitespace tokens would score that normalization as a dropped word.
    """
    return [token for token in text.split() if any(is_arabic_letter(ch) for ch in token)]


def _has_latin(text: str) -> bool:
    return any("LATIN" in unicodedata.name(ch, "") for ch in text)


def _has_digits(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


@dataclass(frozen=True)
class Candidate:
    """One row of the sampling frame, with its measured rate and flags."""

    id: str
    sentence: str
    tashkeel_sentence: str
    rate: float | None
    flags: frozenset[str] = field(default_factory=frozenset)

    @property
    def eligible(self) -> bool:
        return not (self.flags & EXCLUDING)


def classify(example_id: str, sentence: str, tashkeel_sentence: str) -> Candidate:
    """Measure one row and attach its flags, ignoring duplication.

    `DUPLICATE` is not decided here: it depends on the other rows, and on the
    flags this function assigns (`build_pool` keeps whichever member of a
    duplicate group is eligible).
    """
    rate = diacritization_rate(sentence)
    flags: set[str] = set()

    if rate is None:
        flags.add(NO_ARABIC)
    elif rate >= PRE_VOWELIZED_RATE:
        flags.add(PRE_VOWELIZED)
    elif rate >= BARE_RATE:
        flags.add(PARTIALLY_VOWELIZED)

    # `tashkeel_sentence` is the vowelizer output the arm compares `C_gold`
    # against (D5). Without it the row carries no measurement.
    if not tashkeel_sentence.strip():
        flags.add(NO_TASHKEEL)
    elif len(_arabic_words(sentence)) != len(_arabic_words(tashkeel_sentence)):
        flags.add(WORD_COUNT_MISMATCH)

    if len(_arabic_words(sentence)) < MIN_WORDS:
        flags.add(SHORT)
    if _has_latin(sentence):
        flags.add(LATIN)
    if _has_digits(sentence):
        flags.add(DIGITS)

    return Candidate(
        id=example_id,
        sentence=sentence,
        tashkeel_sentence=tashkeel_sentence,
        rate=rate,
        flags=frozenset(flags),
    )


@dataclass(frozen=True)
class Pool:
    """The eligible rows plus the audit trail of what was dropped and why."""

    candidates: tuple[Candidate, ...]
    total: int

    @property
    def eligible(self) -> tuple[Candidate, ...]:
        return tuple(c for c in self.candidates if c.eligible)

    def flag_counts(self) -> dict[str, int]:
        """How many *eligible* rows carry each non-excluding flag."""
        counts = dict.fromkeys((PARTIALLY_VOWELIZED, WORD_COUNT_MISMATCH, SHORT, LATIN, DIGITS), 0)
        for candidate in self.eligible:
            for flag in candidate.flags:
                if flag in counts:
                    counts[flag] += 1
        return counts

    def exclusion_counts(self) -> dict[str, int]:
        """How many rows each excluding flag removed. A row can trip several."""
        counts = dict.fromkeys(sorted(EXCLUDING), 0)
        for candidate in self.candidates:
            for flag in candidate.flags & EXCLUDING:
                counts[flag] += 1
        return counts


def build_pool(rows: Iterable[tuple[str, str, str]]) -> Pool:
    """Classify `(id, sentence, tashkeel_sentence)` rows into a sampling frame.

    Deduplication is by consonantal skeleton. The sibling Qur'anic corpus
    collapses 1,642 rows onto 98 prompt indices and only 96 distinct Arabic
    strings, so this is a measured hazard rather than a precaution — and on
    `Iqra_train` dev the duplicate
    groups are overwhelmingly *mixed*: the same verse once bare and once
    pre-vowelized. Keeping the lowest id would therefore discard the usable
    member of the pair half the time, so an otherwise-eligible row wins over an
    ineligible one, and id order only breaks the remaining ties.
    """
    graded = [classify(*row) for row in sorted(rows, key=lambda row: row[0])]

    groups: dict[str, list[Candidate]] = {}
    for candidate in graded:
        groups.setdefault(skeleton(candidate.sentence), []).append(candidate)

    keep = {
        min(group, key=lambda c: (bool(c.flags & EXCLUDING), c.id)).id for group in groups.values()
    }
    candidates = tuple(
        c
        if c.id in keep
        else Candidate(c.id, c.sentence, c.tashkeel_sentence, c.rate, c.flags | {DUPLICATE})
        for c in graded
    )
    return Pool(candidates=candidates, total=len(candidates))


def skeleton_overlap(
    rows: Iterable[tuple[str, str]], *, against: Iterable[str]
) -> dict[str, list[str]]:
    """Which `(id, sentence)` rows share a consonantal skeleton with `against`.

    The cross-split counterpart of `build_pool`'s dedup, and the same key, for
    the same reason: `Iqra_train`'s train and dev splits are *id*-disjoint and
    *raw-string*-disjoint, yet 28.0% of the annotated block's sentences also
    occur in train — once bare in dev, once pre-vowelized in train. Only the
    skeleton sees it (`docs/msa-arm.md` §5.2).

    Returns shared skeleton -> ids of the matching `rows`, so a caller can both
    count the contamination and drop exactly those rows. Ids come out in input
    order; a skeleton absent from `against` is absent from the result.
    """
    keys = {skeleton(text) for text in against}
    found: dict[str, list[str]] = {}
    for row_id, sentence in rows:
        key = skeleton(sentence)
        if key in keys:
            found.setdefault(key, []).append(row_id)
    return found


def draw_block(pool: Pool, size: int, *, seed: int) -> list[Candidate]:
    """Draw `size` eligible rows without replacement, reproducibly.

    The draw is a function of (`seed`, the set of eligible ids) alone: the pool
    is re-sorted by id first, so adding an unrelated column or re-downloading
    the split cannot silently change who gets annotated.
    """
    eligible = sorted(pool.eligible, key=lambda c: c.id)
    if size > len(eligible):
        raise ValueError(f"asked for {size} rows but only {len(eligible)} are eligible")
    return random.Random(seed).sample(eligible, size)
