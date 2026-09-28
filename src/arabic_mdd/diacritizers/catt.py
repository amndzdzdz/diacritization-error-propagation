"""CATT (github.com/abjadai/catt), the strong modern system in the tool set.

Both published variants are wrapped: encoder-only (`eo`) and encoder-decoder
(`ed`). ED is the more accurate one per CATT's own numbers, and RQ4's
token-probability work (week 4 task 2) has to be confirmed on it rather than
assumed from EO, so neither variant may be dropped here.

The `catt_tashkeel` PyPI package is ONNX-only and downloads its model
archives on first construction (EO 72 MB, ED 86 MB), so construction is
deferred until the first call — importing this module must stay free.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import version
from typing import Any

from arabic_mdd.diacritizers.base import Diacritizer

VARIANTS = ("eo", "ed")


class CattDiacritizer(Diacritizer):
    """CATT, in its encoder-only or encoder-decoder variant."""

    def __init__(self, variant: str = "eo", model: Any = None, batch_size: int = 16) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown CATT variant {variant!r}; expected one of {VARIANTS}")
        self.variant = variant
        self.batch_size = batch_size
        self._model = model

    @property
    def name(self) -> str:
        return f"catt-{self.variant}"

    def provenance(self) -> dict[str, str]:
        return {
            "tool": "catt",
            "variant": self.variant,
            "package": "catt-tashkeel",
            "version": version("catt-tashkeel"),
        }

    @property
    def model(self) -> Any:
        """The underlying `catt_tashkeel` model, constructed on first use."""
        if self._model is None:
            import catt_tashkeel

            cls = (
                catt_tashkeel.CATTEncoderOnly
                if self.variant == "eo"
                else catt_tashkeel.CATTEncoderDecoder
            )
            self._model = cls()
        return self._model

    def diacritize_batch(self, texts: Sequence[str]) -> list[str]:
        if not texts:
            return []
        outputs = self.model.do_tashkeel_batch(
            list(texts), batch_size=self.batch_size, verbose=False
        )
        return [out.strip() for out in outputs]
