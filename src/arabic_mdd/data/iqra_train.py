"""Loaders for the mHuBERT baseline's training data.

`IqraEval/Iqra_train` (Common Voice Arabic, 79h) + `IqraEval/Iqra_TTS`
(TTS-synthesized, 52h) together form the baseline's 131h combined training
set (`docs/mdd-paper-project-plan-v2.md`, Sources). Both are public HF
datasets. The actual multi-hour training run happens on the user's cluster
via S3PRL (see `scripts/baseline_reproduction/`), not in this sandbox --
these loaders exist so this project's own code has a normalized way to
read the same data, not to reimplement the training pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from arabic_mdd.data.phonemes import parse_phoneme_sequence

IQRA_TRAIN_DATASET = "IqraEval/Iqra_train"
IQRA_TTS_DATASET = "IqraEval/Iqra_TTS"


@dataclass(frozen=True)
class TrainingExample:
    """One training utterance's id, canonical phoneme target, and sentence text."""

    id: str
    canonical: list[str]
    sentence: str


def parse_iqra_train_rows(rows: Iterable[Mapping[str, str]]) -> list[TrainingExample]:
    """Normalize raw `Iqra_train` rows (`id`, `phoneme_ref`, `sentence`, ...)."""
    return [
        TrainingExample(
            id=row["id"],
            canonical=parse_phoneme_sequence(row["phoneme_ref"]),
            sentence=row["sentence"],
        )
        for row in rows
    ]


def parse_iqra_tts_rows(rows: Iterable[Mapping[str, str]]) -> list[TrainingExample]:
    """Normalize raw `Iqra_TTS` rows.

    `Iqra_TTS` uses different column names than `Iqra_train`
    (`sentence_ref` instead of `sentence`) and ships no utterance id, so
    one is synthesized from row position.
    """
    return [
        TrainingExample(
            id=f"iqra_tts_{i}",
            canonical=parse_phoneme_sequence(row["phoneme_ref"]),
            sentence=row["sentence_ref"],
        )
        for i, row in enumerate(rows)
    ]


def load_iqra_train(split: str = "train") -> list[TrainingExample]:
    """Load and normalize `Iqra_train` (network call). `split`: "train" or "dev"."""
    import datasets

    ds = datasets.load_dataset(IQRA_TRAIN_DATASET, split=split)
    return parse_iqra_train_rows(ds)


def load_iqra_tts() -> list[TrainingExample]:
    """Load and normalize `Iqra_TTS` (network call)."""
    import datasets

    ds = datasets.load_dataset(IQRA_TTS_DATASET, split="train")
    return parse_iqra_tts_rows(ds)
