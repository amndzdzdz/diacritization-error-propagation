"""Canonical-sequence conditioning: the single block that separates the arms.

Plan §3 requires the prompt-free and text-dependent arms to be "two of your own
systems differing in exactly one property". This module is that property. Every
other part of the downstream -- projector, recurrent head, CTC loss, optimizer,
step count, seed -- is shared, and `MddConfig.use_canonical` decides whether
what is here runs.

**The mechanism.** A single cross-attention block: the acoustic frames are the
queries, an embedding of the canonical phoneme sequence supplies keys and
values, and the attended context is added back onto the frames. CTC is
untouched, so the loss is identical across arms and the two F1 numbers are
comparable without an argument about what they measure.

Three alternatives were considered in `docs/weeks/week-07.md` task 7 and
rejected: pooling the canonical sequence to one vector (too weak to produce the
conditioning effect the design is built to measure), a seq2seq decoder (changes
the loss, so the arms stop being comparable), and frame-level concatenation
(needs a monotonic alignment we would have to estimate, which puts a new
failure mode *inside* the thing being measured).

**The zero-initialized gate, which is the part worth understanding.** The
residual's output projection starts at exactly zero, so at step 0 the block
adds exactly zero and the conditioned arm computes the *bitwise identical*
function to the ablated arm. That is not cosmetic. It means:

* the two arms start from one point, so a difference in their learning curves
  is attributable to the conditioning and not to a different initialization
  landing in a different basin; and
* `tests/models/test_conditioning.py` can assert the identity exactly rather
  than within a tolerance, which turns "the ablation is faithful" from a claim
  into a test.

The known cost, documented because it is a training dynamic and not a bug: at
step 0 the gradient reaching the attention weights is zero (it arrives through
the gate, which is zero), so the attention itself only starts learning once the
gate has moved off zero -- which happens on the first step, since the gate's
own gradient is the outer product of a nonzero context with a nonzero upstream
gradient. The block "unlocks" after one step; it does not stay stuck.

Normalization is pre-norm on the query path only. A post-residual `LayerNorm`
would be the more conventional arrangement and would destroy the identity
above, since `LayerNorm(x) != x`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = [
    "CanonicalEncoder",
    "CrossAttentionConditioner",
    "sinusoidal_positions",
]


def sinusoidal_positions(length: int, dim: int, *, device=None, dtype=None) -> Tensor:
    """The standard fixed sinusoidal position table, shape `(length, dim)`.

    Computed per call rather than cached in a buffer sized to some
    `max_canonical_len`. It is cheap next to the attention it feeds, and it
    removes a configuration knob whose only purpose would be to raise on a long
    verse -- which is exactly the input a Qur'anic corpus supplies.

    Fixed rather than learned so that a canonical sequence longer than anything
    in training is still positioned sensibly at evaluation time.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even for sinusoidal positions, got {dim}")
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    index = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    frequency = torch.exp(-math.log(10000.0) * index / dim)
    table = torch.zeros(length, dim, device=device, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency)
    return table.to(dtype) if dtype is not None else table


class CanonicalEncoder(nn.Module):
    """Embed a canonical phoneme-index sequence into keys/values.

    Shares the CTC head's vocabulary and index convention (see
    `arabic_mdd.models.vocab`), so `padding_idx` is the same index that serves
    as the CTC blank. That overlap is harmless -- the two uses never meet -- but
    it is why the padding index is passed in rather than assumed.
    """

    def __init__(self, vocab_size: int, dim: int, *, padding_idx: int = 0) -> None:
        super().__init__()
        self.dim = dim
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.norm = nn.LayerNorm(dim)

    def forward(self, canonical_ids: Tensor) -> Tensor:
        """`(batch, canonical_len)` indices -> `(batch, canonical_len, dim)`."""
        embedded = self.embedding(canonical_ids)
        positions = sinusoidal_positions(
            canonical_ids.size(1),
            self.dim,
            device=embedded.device,
            dtype=embedded.dtype,
        )
        return self.norm(embedded + positions.unsqueeze(0))


class CrossAttentionConditioner(nn.Module):
    """Add canonical context onto acoustic frames, as a zero-gated residual.

    Queries are frames, keys/values are the encoded canonical sequence. Because
    cross-attention treats each query position independently, a frame's output
    depends only on that frame and the canonical sequence -- never on other
    frames. So no query-side mask is needed for correctness: padded frame
    positions produce values that the packed recurrent head and the CTC input
    lengths both discard, and they cannot contaminate a valid position.

    The canonical side *does* need masking, and gets it from `canonical_lens`.
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        *,
        num_heads: int = 8,
        dropout: float = 0.0,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.encoder = CanonicalEncoder(vocab_size, dim, padding_idx=padding_idx)
        self.query_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(dim, dim)
        self.reset_gate()

    def reset_gate(self) -> None:
        """Zero the output projection, making the whole block an exact no-op.

        Called from `__init__`. Exposed because a checkpoint-loading path that
        wants to re-ablate a trained conditioned model needs it, and because a
        test that asserts the identity should be able to restore it.
        """
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        frames: Tensor,
        canonical_ids: Tensor,
        canonical_lens: Tensor,
    ) -> Tensor:
        """`(batch, frames, dim)` -> `(batch, frames, dim)`.

        `canonical_lens` gives the unpadded length of each canonical sequence.
        A length of zero would mask out an entire attention row, which makes
        `MultiheadAttention` return NaN rather than raise, so it is rejected
        here: a canonical sequence is never legitimately empty in this task,
        and a NaN surfacing 40 hours into a run is the expensive way to find
        out.
        """
        if canonical_ids.size(0) != frames.size(0):
            raise ValueError(
                f"batch mismatch: {frames.size(0)} frame rows vs "
                f"{canonical_ids.size(0)} canonical rows"
            )
        if torch.any(canonical_lens < 1):
            raise ValueError(
                "canonical_lens contains a zero: an all-masked attention row "
                "returns NaN instead of raising. Drop the row upstream instead."
            )
        if int(canonical_lens.max()) > canonical_ids.size(1):
            raise ValueError(
                f"canonical_lens max {int(canonical_lens.max())} exceeds the padded "
                f"canonical width {canonical_ids.size(1)}"
            )

        keys = self.encoder(canonical_ids)
        # True marks a position to ignore, which is the opposite of the
        # convention used for lengths everywhere else in this repo.
        positions = torch.arange(canonical_ids.size(1), device=canonical_ids.device)
        key_padding_mask = positions.unsqueeze(0) >= canonical_lens.unsqueeze(1).to(
            positions.device
        )
        context, _ = self.attention(
            self.query_norm(frames),
            keys,
            keys,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return frames + self.gate(context)
