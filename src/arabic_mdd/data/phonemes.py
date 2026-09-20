"""Phoneme-sequence parsing shared by the data loaders and the hierarchical metric."""

from __future__ import annotations


def parse_phoneme_sequence(text: str) -> list[str]:
    """Split a whitespace-separated phoneme string into tokens.

    Matches the tokenization convention used throughout the official
    Iqra'Eval baseline's data/scoring code (plain `str.split()`), which
    `arabic_mdd.metrics.hierarchical` expects as input: a `Sequence[str]`
    of phoneme tokens, one per position.
    """
    return text.split()
