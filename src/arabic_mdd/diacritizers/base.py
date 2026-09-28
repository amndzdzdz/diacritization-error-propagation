"""Common interface for the diacritization tools under study.

Plan §3 wants the MDD degradation plotted as a *curve* against diacritizer
quality, not as a single before/after pair, so the weak tools matter as much
as CATT does and every tool has to be callable — and recorded — the same way.

Two deliberate non-features:

* **Wrappers do not normalize their input.** Punctuation, Latin script,
  digits and U+0670 are governed by the normalization contract
  (`docs/msa-arm.md` §4.1), which is a separate pinned pipeline step. Burying
  it inside a wrapper would make the tools incomparable to each other and
  make the contract invisible in the write-up.
* **Wrappers do not phonetize.** They return diacritized Arabic text; the
  text→phoneme step is `arabic_mdd.data.phonemes` (week 4 task 3).

The one liberty taken with a tool's output is stripping leading/trailing
whitespace, because at least one tool (Mishkal) adds a leading space that is
an artifact of its own tokenizer rather than a diacritization decision.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Diacritizer(ABC):
    """A diacritization tool, callable on a batch of undiacritized strings."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short stable label for tables, plots and result columns."""

    @abstractmethod
    def provenance(self) -> dict[str, str]:
        """Exact tool identity and version, for the paper's methods section.

        `docs/msa-arm.md` §5.1: anything that differs between arms and is not
        deliberate is a confound. Diacritizer versions are recorded for the
        same reason the upstream checkpoint sha is.
        """

    @abstractmethod
    def diacritize_batch(self, texts: Sequence[str]) -> list[str]:
        """Diacritize each input, returning one output per input, in order."""

    def diacritize(self, text: str) -> str:
        """Diacritize a single string."""
        return self.diacritize_batch([text])[0]
