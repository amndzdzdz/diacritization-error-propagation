"""The WikiNews gold diacritization set — the external calibration corpus.

`CAMeL-Lab/wild_diacritics` `data/WikiNewsMax.tsv`: one Arabic word per row,
with three gold columns beside the undiacritized surface form. It arrived for
the annotator qualification test (`scripts/build_qualification_test.py`) and
earned a place in `src/` when the DER calibration became its second consumer —
the published CATT/Shakkala/Mishkal WikiNews figures are what tells us our DER
definition matches the literature's, so this file is an instrument and not
just a data source.

**`original_gold` is the reference, not `max_gold`.** They differ on 3.72% of
Arabic words. `original_gold` is the Darwish et al. 2017 reference the
published machine DER figures were measured on; calibrating against `max_gold`
would fail for a reason that has nothing to do with our metric.
`max_gold_alt` carries a second acceptable reading where the annotators
recorded one, and is used by the qualification scorer, not by DER.

Not vendored: CC-BY-SA-4.0, and share-alike is viral over derivatives. The
file stays local and consumers record its sha256.
"""

from __future__ import annotations

import csv
import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from arabic_mdd.data.normalize import strip_diacritics
from arabic_mdd.metrics.der import marked_letters

#: A row whose first cell is this marks an article boundary in the source file.
ARTICLE_MARKER = "#"


@dataclass(frozen=True)
class Word:
    dediac: str
    original_gold: str
    max_gold: str
    max_gold_alt: str

    @property
    def is_arabic(self) -> bool:
        return any("\u0621" <= c <= "\u064a" for c in self.dediac)


@dataclass(frozen=True)
class Segment:
    index: int
    """Position in the source file, so a drawn segment can be found again."""

    words: list[Word]

    @property
    def arabic_words(self) -> list[Word]:
        return [w for w in self.words if w.is_arabic]

    @property
    def gold_text(self) -> str:
        return " ".join(w.original_gold for w in self.words)

    @property
    def blank_text(self) -> str:
        return " ".join(w.dediac for w in self.words)

    def bucket_counts(self) -> tuple[int, int]:
        """(bucket_A, bucket_B) *marked* positions in `original_gold`.

        Bucket B is the mark on a word's final letter — the positional proxy
        for i'rab, the same one `metrics.der` uses. Bucket A is every other
        marked position. Counts are pre-pausal-filter;
        `docs/annotator-qualification.md` §8 applies that at scoring time,
        which can only reduce bucket B.
        """
        a = b = 0
        for word in self.arabic_words:
            pairs = marked_letters(word.original_gold)
            for i, (_letter, marks) in enumerate(pairs):
                if not marks:
                    continue
                if i == len(pairs) - 1:
                    b += 1
                else:
                    a += 1
        return a, b


def read_segments(tsv: Path) -> list[Segment]:
    """Segment the word-per-row TSV on blank rows.

    Article-marker rows (`#`) also break a segment: a headline and the body
    that follows it are not one sentence.
    """
    with tsv.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))[1:]

    segments: list[Segment] = []
    current: list[Word] = []

    def flush() -> None:
        if current and any(w.is_arabic for w in current):
            segments.append(Segment(index=len(segments), words=list(current)))
        current.clear()

    for row in rows:
        cells = (row + ["", "", "", ""])[:4]
        if not cells[0].strip() or cells[0].strip() == ARTICLE_MARKER:
            flush()
            continue
        current.append(
            Word(
                dediac=unicodedata.normalize("NFC", cells[0]),
                original_gold=unicodedata.normalize("NFC", cells[1]),
                max_gold=unicodedata.normalize("NFC", cells[2]),
                max_gold_alt=unicodedata.normalize("NFC", cells[3]),
            )
        )
    flush()
    return segments


def skeleton_mismatches(segments: list[Segment]) -> int:
    """Words where `strip_diacritics(original_gold)` != the file's own `dediac`.

    Must be zero, and both consumers treat a non-zero count as fatal rather
    than as a warning:

    * for the qualification packet, the blank text handed to an annotator would
      not be the text the gold was written for, so §8 step 1's skeleton check
      would fire on our preprocessing instead of on the annotator;
    * for the DER calibration, the input fed to the tools would not be the
      undiacritized form of the reference, so we would be scoring different
      text than the published figures were measured on.
    """
    return sum(
        1
        for segment in segments
        for w in segment.words
        if strip_diacritics(w.original_gold) != w.dediac
    )


def file_sha256(path: Path) -> str:
    """Digest the source file; it is not vendored, so consumers pin it instead."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
