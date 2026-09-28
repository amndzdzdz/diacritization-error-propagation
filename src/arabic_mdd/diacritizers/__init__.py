"""Wrappers around the diacritization tools under study.

Plan §3 fixes the tool set: CATT (EO and ED) as the strong modern system,
Shakkala and Mishkal as the weaker points, chosen because CATT's Table 5
spans 5.4%-19.6% CE DER across them — enough spread to plot MDD degradation
as a curve against diacritizer quality. Farasa is the alternative to Mishkal
and is not wrapped: Mishkal is pure Python where `farasapy` shells out to
Java, and RQ3's pre-registered rank prediction is about Mishkal.

`DIACRITIZERS` maps each tool's `name` to a zero-argument factory, so the
RQ1 sweep can iterate the set without knowing what is behind each one.
"""

from collections.abc import Callable

from arabic_mdd.diacritizers.base import Diacritizer
from arabic_mdd.diacritizers.catt import CattDiacritizer
from arabic_mdd.diacritizers.mishkal import MishkalDiacritizer
from arabic_mdd.diacritizers.shakkala import ShakkalaDiacritizer

DIACRITIZERS: dict[str, Callable[[], Diacritizer]] = {
    "catt-eo": lambda: CattDiacritizer(variant="eo"),
    "catt-ed": lambda: CattDiacritizer(variant="ed"),
    "shakkala": ShakkalaDiacritizer,
    "mishkal": MishkalDiacritizer,
}


def build(name: str) -> Diacritizer:
    """Construct the diacritizer registered under `name`."""
    if name not in DIACRITIZERS:
        raise ValueError(f"unknown diacritizer {name!r}; expected one of {sorted(DIACRITIZERS)}")
    return DIACRITIZERS[name]()


__all__ = [
    "DIACRITIZERS",
    "CattDiacritizer",
    "Diacritizer",
    "MishkalDiacritizer",
    "ShakkalaDiacritizer",
    "build",
]
