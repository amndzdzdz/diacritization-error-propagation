"""MDD model architectures (prompt-free and canonical-sequence-conditioned baselines).

Both arms are one class, `MddModel`, switched by `MddConfig.use_canonical`.
See `arabic_mdd.models.mdd` for why that is a requirement and not a convenience.
"""

from arabic_mdd.models.conditioning import CanonicalEncoder, CrossAttentionConditioner
from arabic_mdd.models.mdd import (
    Batch,
    MddConfig,
    MddModel,
    RecurrentHead,
    collate_batch,
    ctc_loss,
    greedy_indices,
)
from arabic_mdd.models.vocab import RESERVED_TOKENS, PhonemeVocab, load_vocab

__all__ = [
    "RESERVED_TOKENS",
    "Batch",
    "CanonicalEncoder",
    "CrossAttentionConditioner",
    "MddConfig",
    "MddModel",
    "PhonemeVocab",
    "RecurrentHead",
    "collate_batch",
    "ctc_loss",
    "greedy_indices",
    "load_vocab",
]
