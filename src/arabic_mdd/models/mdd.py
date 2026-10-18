"""The MDD downstream: one code path, two arms.

`MddConfig.use_canonical` is the only switch. With it off this is the
prompt-free arm; with it on, the cross-attention block from
`arabic_mdd.models.conditioning` runs between the projector and the recurrent
head and the model reads the canonical phoneme sequence. Nothing else differs:
same parameters, same initialization, same loss, same optimizer surface.

Two design points that are easy to get wrong and are load-bearing here.

**The conditioner is always constructed, even in the ablated arm.** It would be
tidier to build it only when `use_canonical` is set. It would also silently
break the comparison: `nn.MultiheadAttention` and `nn.Embedding` draw from the
global RNG at construction, so a model that skips them consumes less randomness
and its *recurrent head* ends up with different weights under the same seed.
The two arms would then differ in two properties -- the conditioning and the
initialization -- and plan §3's "exactly one property" claim would be false in
a way no result would reveal. So both arms construct all three submodules, and
the ablated arm simply never routes activations through one of them. Its
parameters receive no gradient and sit unused in the checkpoint, which costs
about 25 MB of disk and buys a defensible comparison.
`tests/models/test_mdd.py` asserts the head weights are identical across arms.

**The recurrent head is a narrow port of s3prl's `RNNs`, not a reuse and not a
reimplementation of all of it.** The baseline reproduction deliberately runs
s3prl verbatim, so why port at all: because `src/` cannot import from a
Python-3.8 checkout that only exists on the cluster, and an injectable-head
abstraction would leave the arms untestable in the sandbox. The port is kept
honest by scope rather than by faith -- under `sws.yaml` the configuration is
`total_rate: 320` against a 320-frame upstream rate and `sample_rate: [1, 1]`,
so **every downsampling branch in s3prl's head is dead code**, and this port
omits it entirely and raises if a configuration would have needed it. What
remains is two bidirectional LSTM layers with dropout and a projected tanh, in
s3prl's operation order, which is small enough to read against the original
side by side.

**This module must stay importable under Python 3.8.** The dependency runs the
other way as well: `scripts/baseline_reproduction/ctc_cond/expert.py` imports
`MddConfig`/`MddModel` from here *inside* s3prl's 3.8 venv, so that the arms
the cluster trains are the arms `tests/models/` tests, rather than a second
implementation that merely resembles them. The practical cost is small and
specific -- no `zip(..., strict=True)`, no `match`, no runtime subscripting of
builtins outside annotations (`from __future__ import annotations` handles the
annotations) -- and it is recorded in `pyproject.toml`'s per-file B905 ignore.
Nothing enforces it automatically; `prepare_baseline.sh` fails loudly on import
if it is broken, which is the cheap end of the trade and the reason this
paragraph exists.

**The consequence for the 2×2, stated rather than buried.** Because the arms
come from this module, the 2×2's prompt-free row is *this* model with
`use_canonical=False` -- a third training run, not task 6's s3prl XLS-R run.
Task 6's run keeps its own job: it is the "strong prompt-free" modern-system
anchor for the architecture-independence claim, and it is the control that says
whether this port is sound, since a port that is wrong will show up as our
ablated arm underperforming s3prl's at the same config. The extra run is
budget, so the scheduling decision belongs in `docs/weeks/week-07.md` task 7
and not here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from arabic_mdd.models.conditioning import CrossAttentionConditioner

__all__ = [
    "MddConfig",
    "MddModel",
    "RecurrentHead",
    "collate_batch",
    "ctc_loss",
    "greedy_indices",
]


@dataclass(frozen=True)
class MddConfig:
    """Hyperparameters, defaulted to the organizers' `sws.yaml` where it has an opinion.

    Defaults match `scripts/baseline_reproduction/config/sws.yaml` field for
    field, so a difference between this model and the baseline is a difference
    in architecture rather than an accident of tuning. The conditioner's own
    hyperparameters have no counterpart there and are set here.
    """

    vocab_size: int
    upstream_dim: int = 1024
    project_dim: int = 1024

    # sws.yaml -> downstream_expert.model.RNNs
    rnn_dims: tuple[int, ...] = (1024, 1024)
    rnn_dropout: tuple[float, ...] = (0.2, 0.2)
    rnn_layer_norm: tuple[bool, ...] = (False, False)
    rnn_proj: tuple[bool, ...] = (True, True)
    rnn_module: str = "LSTM"
    bidirectional: bool = True
    # total_rate / upstream_rate. Anything but 1 means s3prl would have
    # downsampled, which this port does not implement.
    sample_rate: int = 1

    # sws.yaml -> downstream_expert.model.zero_infinity
    zero_infinity: bool = True

    # The arm switch, and the conditioner's own settings.
    use_canonical: bool = False
    conditioner_heads: int = 8
    conditioner_dropout: float = 0.0
    padding_idx: int = 0

    def __post_init__(self) -> None:
        widths = {
            "rnn_dropout": len(self.rnn_dropout),
            "rnn_layer_norm": len(self.rnn_layer_norm),
            "rnn_proj": len(self.rnn_proj),
        }
        expected = len(self.rnn_dims)
        mismatched = {name: n for name, n in widths.items() if n != expected}
        if mismatched:
            raise ValueError(
                f"per-layer lists must all have {expected} entries to match "
                f"rnn_dims; got {mismatched}"
            )
        if self.sample_rate != 1:
            raise ValueError(
                f"sample_rate={self.sample_rate} would require s3prl's time-downsampling "
                "branches, which this port deliberately omits (see the module docstring). "
                "sws.yaml sets total_rate 320 against a 320 upstream rate, giving 1."
            )


class RecurrentHead(nn.Module):
    """s3prl's `RNNs` at `sws.yaml`'s settings, minus the dead downsampling paths."""

    def __init__(self, config: MddConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        width = config.project_dim
        for dim, dropout, layer_norm, proj in zip(
            config.rnn_dims,
            config.rnn_dropout,
            config.rnn_layer_norm,
            config.rnn_proj,
        ):
            out_dim = 2 * dim if config.bidirectional else dim
            self.layers.append(
                getattr(nn, config.rnn_module.upper())(
                    width,
                    dim,
                    bidirectional=config.bidirectional,
                    num_layers=1,
                    batch_first=True,
                )
            )
            # `None` entries keep the module lists index-aligned with
            # `self.layers`, so the forward pass can zip them without
            # bookkeeping. They are not parameters, so they do not touch the
            # RNG or the checkpoint.
            self.norms.append(nn.LayerNorm(out_dim) if layer_norm else None)
            self.dropouts.append(nn.Dropout(p=dropout) if dropout > 0 else None)
            self.projections.append(nn.Linear(out_dim, out_dim) if proj else None)
            width = out_dim

        self.output = nn.Linear(width, config.vocab_size)

    def forward(self, x: Tensor, x_lens: Tensor) -> tuple[Tensor, Tensor]:
        """`(batch, frames, project_dim)` -> logits `(batch, frames, vocab)` and lengths."""
        # Equal lengths by construction: `__init__` appends to all four lists
        # once per layer, using `None` for the absent entries.
        for layer, norm, dropout, projection in zip(
            self.layers, self.norms, self.dropouts, self.projections
        ):
            # `pack_padded_sequence` requires lengths on the CPU regardless of
            # where the tensors live.
            packed = pack_padded_sequence(x, x_lens.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = layer(packed)
            x, x_lens = pad_packed_sequence(packed_out, batch_first=True)
            if norm is not None:
                x = norm(x)
            if dropout is not None:
                x = dropout(x)
            if projection is not None:
                x = torch.tanh(projection(x))
        return self.output(x), x_lens


class MddModel(nn.Module):
    """Projector -> optional canonical conditioning -> recurrent head -> CTC logits."""

    def __init__(self, config: MddConfig) -> None:
        super().__init__()
        self.config = config
        # Construction order fixes RNG consumption; see the module docstring.
        self.projector = nn.Linear(config.upstream_dim, config.project_dim)
        self.conditioner = CrossAttentionConditioner(
            config.project_dim,
            config.vocab_size,
            num_heads=config.conditioner_heads,
            dropout=config.conditioner_dropout,
            padding_idx=config.padding_idx,
        )
        self.head = RecurrentHead(config)

    @property
    def conditioning_parameters(self) -> list[nn.Parameter]:
        """The conditioner's parameters, for reporting the arms' size difference."""
        return list(self.conditioner.parameters())

    def forward(
        self,
        features: Tensor,
        feature_lens: Tensor,
        canonical_ids: Tensor | None = None,
        canonical_lens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Fused upstream features -> CTC logits.

        `canonical_ids`/`canonical_lens` are required when
        `config.use_canonical` is set and **ignored** when it is not. Ignoring
        rather than rejecting them is deliberate: the same evaluation loop feeds
        both arms, and making the prompt-free arm raise on receiving a canonical
        it does not use would mean two loops, which is one more place for them
        to diverge.
        """
        x = self.projector(features)
        if self.config.use_canonical:
            if canonical_ids is None or canonical_lens is None:
                raise ValueError(
                    "use_canonical=True but no canonical sequence was passed. This is the "
                    "text-dependent arm; it cannot run prompt-free. Build the model with "
                    "use_canonical=False for that."
                )
            x = self.conditioner(x, canonical_ids, canonical_lens)
        return self.head(x, feature_lens)


def ctc_loss(
    logits: Tensor,
    logit_lens: Tensor,
    targets: Tensor,
    target_lens: Tensor,
    *,
    blank: int = 0,
    zero_infinity: bool = True,
) -> Tensor:
    """CTC loss over `(batch, time, vocab)` logits.

    Identical in both arms -- that is the reason the conditioning was built as a
    feature-space residual rather than as a decoder. `blank` defaults to 0
    because `arabic_mdd.models.vocab` puts `<pad>` there and s3prl's expert
    passes `blank=tokenizer.pad_idx`.

    `zero_infinity` defaults to True to match `sws.yaml`. It matters more than
    it looks: an utterance whose target is longer than its frame count yields an
    infinite loss, and without this the whole batch's gradient becomes NaN.
    """
    log_probs = functional.log_softmax(logits, dim=-1).transpose(0, 1)
    return functional.ctc_loss(
        log_probs,
        targets,
        logit_lens.cpu(),
        target_lens.cpu(),
        blank=blank,
        zero_infinity=zero_infinity,
    )


def greedy_indices(logits: Tensor, logit_lens: Tensor) -> list[list[int]]:
    """Per-utterance argmax index sequences, truncated to each real length.

    Collapsing repeats and stripping blanks is `PhonemeVocab.decode`'s job; this
    only takes the argmax and trims the padding, so the two concerns stay
    testable apart.
    """
    best = logits.argmax(dim=-1)
    return [best[i, : int(length)].tolist() for i, length in enumerate(logit_lens)]


@dataclass
class Batch:
    """A collated batch. `canonical_*` are present whether or not an arm uses them."""

    features: Tensor
    feature_lens: Tensor
    targets: Tensor
    target_lens: Tensor
    canonical_ids: Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))
    canonical_lens: Tensor = field(default_factory=lambda: torch.zeros(0, dtype=torch.long))


def collate_batch(
    features: Sequence[Tensor],
    targets: Sequence[Sequence[int]],
    canonicals: Sequence[Sequence[int]] | None = None,
    *,
    padding_idx: int = 0,
) -> Batch:
    """Pad a list of per-utterance tensors into a batch.

    Targets are concatenated rather than padded into a rectangle, which is the
    1-D form `torch.nn.functional.ctc_loss` accepts without needing the padding
    value to be excluded by hand -- the padded form is where an off-by-one in
    `target_lens` turns into blanks being trained as real labels.
    """
    if len(features) != len(targets):
        raise ValueError(f"{len(features)} feature rows but {len(targets)} target rows")
    feature_lens = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    padded = pad_sequence(list(features), batch_first=True)
    target_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat_targets = torch.tensor([i for t in targets for i in t], dtype=torch.long)

    batch = Batch(
        features=padded,
        feature_lens=feature_lens,
        targets=flat_targets,
        target_lens=target_lens,
    )
    if canonicals is not None:
        if len(canonicals) != len(features):
            raise ValueError(f"{len(features)} feature rows but {len(canonicals)} canonical rows")
        batch.canonical_lens = torch.tensor([len(c) for c in canonicals], dtype=torch.long)
        batch.canonical_ids = pad_sequence(
            [torch.tensor(c, dtype=torch.long) for c in canonicals],
            batch_first=True,
            padding_value=padding_idx,
        )
    return batch
