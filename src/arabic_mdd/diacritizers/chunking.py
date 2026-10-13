"""Running a diacritizer on input longer than it accepts.

Three of the four tools have a hard input ceiling and **fail** at it rather
than truncating (`Diacritizer.max_input_chars` records each one, measured).
This never fired through week 4 because `Iqra_train` and QuranMB utterances
are short; it fires immediately on the WikiNews calibration corpus, where 2 of
558 segments exceed CATT's 1,024 characters.

Dropping those 2 segments was the cheaper option and is the wrong one: the
calibration's whole purpose is to reproduce a published figure on the corpus
it was published on, so a corpus we quietly shrank cannot do that job.

**Chunking lives here, not in the wrappers**, for the same reason
`base.Diacritizer` refuses to normalize its input: it changes what the tool is
asked to predict — a chunk boundary removes the context a model would
otherwise condition on, and the prediction at the seam can differ — so it has
to be a visible harness stage that a measurement reports, not a hidden tool
property that a measurement cannot see. `ChunkedBatch.seams` exists so every
artifact can carry the count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from arabic_mdd.diacritizers.base import Diacritizer


@dataclass(frozen=True)
class ChunkedBatch:
    """Diacritized texts, plus how much splitting it took to get them."""

    texts: list[str]

    seams: list[int]
    """Per input, the number of joins in its output. 0 means it went through whole."""

    @property
    def chunked_items(self) -> int:
        return sum(1 for s in self.seams if s)

    @property
    def total_seams(self) -> int:
        return sum(self.seams)

    def report(self) -> dict[str, int]:
        """The two numbers an artifact records next to a rate."""
        return {"chunked_items": self.chunked_items, "seams": self.total_seams}


def split_on_words(text: str, limit: int) -> list[str]:
    """Greedily pack `text`'s whitespace-separated words into chunks <= `limit`.

    Splitting inside a word is never done: it would hand the tool a fragment
    that is not Arabic morphology, and the diacritics it assigns to a fragment
    are not the ones it would assign to the word. A single word over the limit
    is therefore an error rather than a hard split.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in text.split():
        if len(word) > limit:
            raise ValueError(
                f"word of {len(word)} characters exceeds the {limit}-character "
                f"limit on its own; splitting inside a word would change what "
                f"the tool is asked to diacritize"
            )
        extra = len(word) + (1 if current else 0)
        if length + extra > limit:
            chunks.append(" ".join(current))
            current, length = [word], len(word)
        else:
            current.append(word)
            length += extra
    if current:
        chunks.append(" ".join(current))
    return chunks


def diacritize_long(tool: Diacritizer, texts: Sequence[str]) -> ChunkedBatch:
    """`tool.diacritize_batch`, with over-long inputs split and rejoined.

    Inputs within the limit are passed through untouched, so the output is
    byte-identical to calling the tool directly. All chunks go to the tool in
    one call, so a batching tool still batches.
    """
    limit = tool.max_input_chars
    if limit is None:
        return ChunkedBatch(texts=tool.diacritize_batch(texts), seams=[0] * len(texts))

    pieces: list[list[str]] = [
        [text] if len(text) <= limit else split_on_words(text, limit) for text in texts
    ]
    flat = [chunk for piece in pieces for chunk in piece]
    outputs = tool.diacritize_batch(flat)
    if len(outputs) != len(flat):
        raise RuntimeError(f"{tool.name} returned {len(outputs)} outputs for {len(flat)} chunks")

    rejoined: list[str] = []
    cursor = 0
    for piece in pieces:
        rejoined.append(" ".join(outputs[cursor : cursor + len(piece)]))
        cursor += len(piece)
    return ChunkedBatch(texts=rejoined, seams=[len(piece) - 1 for piece in pieces])
