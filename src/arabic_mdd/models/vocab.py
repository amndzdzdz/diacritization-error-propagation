"""The CTC output vocabulary, index-compatible with the Iqra'Eval baseline.

Our text-dependent arm is our own code, but its output indices are *not* free
to differ from the baseline's. Two reasons:

* The scorer in the leaderboard HF Space consumes phoneme strings, so the
  mapping from index to phoneme has to be the same one the baseline decodes
  with, or our predictions are a permutation of the right answer.
* The week-3 reproduction and the task-6 XLS-R run both come out of s3prl with
  s3prl's own tokenizer. Comparing their F1 against ours only means something
  if "class 37" denotes the same phoneme in all three.

So this module is a deliberate re-implementation of s3prl's
`downstream/ctc/text.py::WordTextEncoder` at the pinned commit, under
`sws.yaml`'s `text.mode: 'word'`, rather than a fresh design. The conventions
copied from it, each of which would silently shift every index if changed:

* The three reserved tokens come **first**, in the order
  ``<pad>``, ``<eos>``, ``<unk>`` -- so `pad_idx == 0`, `eos_idx == 1`,
  `unk_idx == 2` and the phonemes start at 3.
* **`<pad>` is also the CTC blank.** `expert.py` constructs
  `nn.CTCLoss(blank=self.tokenizer.pad_idx, ...)`. There is no separate blank
  symbol; index 0 does both jobs.
* The phoneme order is the **file order** of `vocab/sws_arabic.txt`, which is
  not sorted and not grouped -- it begins `bb x yy s r k n ...`. Sorting it
  would be tidier and would renumber all 68 classes.
* `encode` **appends `<eos>` to the target sequence.** This is unusual for
  CTC -- the model is trained to emit an end symbol it does not need -- but it
  is what the baseline trained with, and the decode side strips it, so the
  behaviour is self-consistent. Mirrored rather than corrected, because the
  comparison is the point; see `append_eos` if that ever has to be revisited.

`arabic_mdd.data.phonetizer.SWS_ARABIC_INVENTORY` holds the same 68 tokens as
a `frozenset`, which is the right type for membership tests and the wrong one
here: a set has no order, and the order *is* the index assignment.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "RESERVED_TOKENS",
    "PhonemeVocab",
    "load_vocab",
]

# Order matters; see the module docstring.
RESERVED_TOKENS: tuple[str, ...] = ("<pad>", "<eos>", "<unk>")


@dataclass(frozen=True)
class PhonemeVocab:
    """An ordered phoneme vocabulary with s3prl's reserved-token prefix.

    `tokens` is the full list including the reserved prefix, so a token's
    position *is* its class index and `len(tokens)` is the CTC head's output
    width.
    """

    tokens: tuple[str, ...]
    # Derived, and excluded from equality so two vocabs compare by their token
    # list alone.
    _index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.tokens[: len(RESERVED_TOKENS)] != RESERVED_TOKENS:
            raise ValueError(
                f"vocabulary must start with {RESERVED_TOKENS!r}; "
                f"got {self.tokens[: len(RESERVED_TOKENS)]!r}. These three "
                "positions fix pad/eos/unk, and pad is also the CTC blank."
            )
        duplicates = sorted({t for t in self.tokens if self.tokens.count(t) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate vocabulary entries {duplicates!r}: a token would get "
                "two indices and only the first would ever be decoded."
            )
        object.__setattr__(self, "_index", {t: i for i, t in enumerate(self.tokens)})

    @classmethod
    def from_phonemes(cls, phonemes: Sequence[str]) -> PhonemeVocab:
        """Build from the 68 phonemes alone, in the order given."""
        return cls(tokens=RESERVED_TOKENS + tuple(phonemes))

    @property
    def size(self) -> int:
        """The CTC head's output width, blank included."""
        return len(self.tokens)

    @property
    def pad_idx(self) -> int:
        return 0

    @property
    def eos_idx(self) -> int:
        return 1

    @property
    def unk_idx(self) -> int:
        return 2

    @property
    def blank_idx(self) -> int:
        """The CTC blank, which is `pad_idx` -- not a fourth reserved symbol."""
        return self.pad_idx

    @property
    def phonemes(self) -> tuple[str, ...]:
        """The vocabulary without the reserved prefix, still in index order."""
        return self.tokens[len(RESERVED_TOKENS) :]

    def to_idx(self, token: str) -> int:
        """Index of `token`, or `unk_idx` if it is out of vocabulary.

        Mirrors s3prl's `vocab_to_idx`, which also maps silently rather than
        raising. Silent is the right behaviour at training time -- one stray
        token should not kill a 48-hour job -- but it does mean a systematically
        mis-tokenized manifest degrades quietly, so callers that can afford a
        pre-pass should count `unk` themselves.
        """
        return self._index.get(token, self.unk_idx)

    def encode(self, phonemes: Sequence[str], *, append_eos: bool = True) -> list[int]:
        """Phoneme tokens -> CTC target indices.

        `append_eos` defaults to True to match the baseline. Pass False only if
        the resulting checkpoint is never compared against an s3prl-trained one.
        """
        ids = [self.to_idx(token) for token in phonemes]
        if append_eos:
            ids.append(self.eos_idx)
        return ids

    def decode(self, ids: Iterable[int], *, collapse_repeats: bool = False) -> list[str]:
        """CTC output indices -> phoneme tokens.

        Faithful to s3prl's `WordTextEncoder.decode`: stop at the first `<eos>`,
        drop blanks, and -- when `collapse_repeats` is set, which is what greedy
        CTC decoding needs -- drop a token that repeats the *immediately
        preceding index*. Checking the preceding index rather than the preceding
        emitted token is what makes `[p, blank, p]` correctly decode to two
        `p`s instead of one.

        Returns a list rather than s3prl's space-joined string, because every
        consumer in this repo (`metrics.hierarchical`,
        `data.phonemes.parse_phoneme_sequence`) wants the token sequence.
        """
        ids = list(ids)
        out: list[str] = []
        for position, idx in enumerate(ids):
            if idx == self.eos_idx:
                break
            if idx == self.pad_idx:
                continue
            if collapse_repeats and position > 0 and idx == ids[position - 1]:
                continue
            out.append(self.tokens[idx])
        return out


def load_vocab(path: str | Path) -> PhonemeVocab:
    """Read a `sws_arabic.txt`-style one-token-per-line file, preserving order.

    Deliberately not `str.split()` over the whole file, which is what
    `phonetizer.load_inventory` does: that is fine for building a set but it
    would also silently accept two tokens on one line, which here would shift
    every subsequent index by one.
    """
    text = Path(path).read_text(encoding="utf-8")
    phonemes: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip("\r\n")
        if not line:
            continue
        if len(line.split()) != 1:
            raise ValueError(
                f"{path}:{lineno}: expected exactly one token per line, got {line!r}. "
                "Index assignment is positional, so a split line renumbers the rest."
            )
        phonemes.append(line)
    if not phonemes:
        raise ValueError(f"{path}: no tokens found; a vocabulary of blanks alone is not usable.")
    return PhonemeVocab.from_phonemes(phonemes)
