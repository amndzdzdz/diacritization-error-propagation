"""Mishkal (pypi.org/project/mishkal), the rule-based weak point of the tool set.

Plan §3 allows "Mishkal or Farasa" here. Mishkal is chosen: it is pure
Python, installs cleanly on this project's 3.12 environment, and needs no
JVM or jar downloads, where `farasapy` shells out to Java. RQ3 also states
a Mishkal rank prediction in advance, so this is the tool that prediction is
about.

One quirk of the tool leaks into its output and is corrected here.
`TashkeelClass.tashkeel` replaces sentence-final punctuation with U+0001,
its internal sentence separator, and does so *including the following
space*: "ذهب الولد. ثم عاد" comes back as "ذَهَبَ الْوَلَدُ\\x01ثُمَّ عَادٌ",
which whitespace-splits into three words instead of four. Left alone that
silently corrupts word alignment downstream, so U+0001 is restored to a
space rather than deleted.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import version
from typing import Any

from arabic_mdd.diacritizers.base import Diacritizer

SENTENCE_SEPARATOR = "\x01"


class MishkalDiacritizer(Diacritizer):
    """Mishkal's `TashkeelClass`, one utterance at a time (it has no batch API)."""

    def __init__(self, vocalizer: Any = None) -> None:
        self._vocalizer = vocalizer

    @property
    def name(self) -> str:
        return "mishkal"

    def provenance(self) -> dict[str, str]:
        return {"tool": "mishkal", "package": "mishkal", "version": version("mishkal")}

    @property
    def vocalizer(self) -> Any:
        """Mishkal's `TashkeelClass`, constructed on first use.

        Deferred because importing `mishkal` loads several Arabic morphology
        databases and emits a wall of `SyntaxWarning`s from its dependencies.
        """
        if self._vocalizer is None:
            from mishkal.tashkeel import TashkeelClass

            self._vocalizer = TashkeelClass()
        return self._vocalizer

    def diacritize_batch(self, texts: Sequence[str]) -> list[str]:
        return [self._diacritize_one(text) for text in texts]

    def _diacritize_one(self, text: str) -> str:
        output = self.vocalizer.tashkeel(text)
        return " ".join(output.replace(SENTENCE_SEPARATOR, " ").split())
