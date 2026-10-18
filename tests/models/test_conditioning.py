"""The conditioning block's two guarantees: it starts as an exact no-op, and it
only ever reads the unpadded part of the canonical sequence."""

from __future__ import annotations

import pytest
import torch

from arabic_mdd.models.conditioning import CrossAttentionConditioner, sinusoidal_positions

DIM = 16
VOCAB = 20


@pytest.fixture
def conditioner() -> CrossAttentionConditioner:
    torch.manual_seed(0)
    block = CrossAttentionConditioner(DIM, VOCAB, num_heads=4)
    block.eval()  # no dropout, so every assertion below is exact
    return block


@pytest.fixture
def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1)
    frames = torch.randn(2, 5, DIM)
    canonical_ids = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 8, 9, 10]])
    canonical_lens = torch.tensor([3, 5])
    return frames, canonical_ids, canonical_lens


def test_zero_gate_makes_the_block_an_exact_identity(conditioner, inputs) -> None:
    # Bitwise, not approximately. This is what lets the two arms start from the
    # same function, so a divergence later is learned rather than inherited
    # from a different initialization.
    frames, canonical_ids, canonical_lens = inputs
    out = conditioner(frames, canonical_ids, canonical_lens)
    assert torch.equal(out, frames)


def test_block_stops_being_an_identity_once_the_gate_moves(conditioner, inputs) -> None:
    # The mirror of the test above: if the gate were somehow frozen at zero the
    # identity test would pass for the wrong reason and the arm would never
    # learn anything.
    frames, canonical_ids, canonical_lens = inputs
    with torch.no_grad():
        conditioner.gate.weight.fill_(0.01)
    out = conditioner(frames, canonical_ids, canonical_lens)
    assert not torch.equal(out, frames)


def test_gate_receives_gradient_on_the_first_step(conditioner, inputs) -> None:
    # The documented cost of the zero gate is that the attention weights get no
    # gradient at step 0. That is only acceptable because the *gate* does, so
    # the block unlocks immediately instead of staying stuck at zero forever.
    frames, canonical_ids, canonical_lens = inputs
    out = conditioner(frames, canonical_ids, canonical_lens)
    out.sum().backward()
    assert conditioner.gate.weight.grad is not None
    assert torch.any(conditioner.gate.weight.grad != 0)


def test_padded_canonical_positions_do_not_affect_the_output(conditioner, inputs) -> None:
    frames, canonical_ids, canonical_lens = inputs
    with torch.no_grad():
        conditioner.gate.weight.fill_(0.01)
        conditioner.gate.bias.fill_(0.02)
    before = conditioner(frames, canonical_ids, canonical_lens)

    # Row 0 has length 3, so positions 3 and 4 are padding. Scribble on them.
    scribbled = canonical_ids.clone()
    scribbled[0, 3:] = torch.tensor([11, 12])
    after = conditioner(frames, scribbled, canonical_lens)
    assert torch.equal(before, after)


def test_canonical_content_changes_the_output(conditioner, inputs) -> None:
    # The experiment's whole premise is that swapping C_gold for C_auto at
    # inference changes the prediction. If this failed, the 2x2 would measure
    # nothing.
    frames, canonical_ids, canonical_lens = inputs
    with torch.no_grad():
        conditioner.gate.weight.fill_(0.01)
    before = conditioner(frames, canonical_ids, canonical_lens)
    swapped = canonical_ids.clone()
    swapped[1, 2] = 15
    after = conditioner(frames, swapped, canonical_lens)
    assert not torch.equal(before, after)


def test_a_frames_output_does_not_depend_on_other_frames(conditioner, inputs) -> None:
    # Cross-attention is per-query, so padded frame positions cannot contaminate
    # a valid one. The module docstring claims this to justify having no
    # query-side mask; this is the claim under test.
    frames, canonical_ids, canonical_lens = inputs
    with torch.no_grad():
        conditioner.gate.weight.fill_(0.01)
    before = conditioner(frames, canonical_ids, canonical_lens)
    perturbed = frames.clone()
    perturbed[0, 4] = 99.0
    after = conditioner(perturbed, canonical_ids, canonical_lens)
    assert torch.equal(before[0, :4], after[0, :4])


def test_zero_length_canonical_is_rejected_not_silently_nan(conditioner, inputs) -> None:
    # An all-masked attention row returns NaN rather than raising, which would
    # surface as a dead loss many hours into a run.
    frames, canonical_ids, _ = inputs
    with pytest.raises(ValueError, match="zero"):
        conditioner(frames, canonical_ids, torch.tensor([0, 5]))


def test_length_longer_than_the_padded_width_is_rejected(conditioner, inputs) -> None:
    frames, canonical_ids, _ = inputs
    with pytest.raises(ValueError, match="exceeds"):
        conditioner(frames, canonical_ids, torch.tensor([3, 9]))


def test_batch_mismatch_is_rejected(conditioner, inputs) -> None:
    frames, canonical_ids, canonical_lens = inputs
    with pytest.raises(ValueError, match="batch mismatch"):
        conditioner(frames[:1], canonical_ids, canonical_lens)


def test_rejects_a_dim_not_divisible_by_the_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CrossAttentionConditioner(10, VOCAB, num_heads=4)


def test_sinusoidal_positions_are_deterministic_and_shaped() -> None:
    first = sinusoidal_positions(7, DIM)
    second = sinusoidal_positions(7, DIM)
    assert first.shape == (7, DIM)
    assert torch.equal(first, second)


def test_sinusoidal_positions_extend_rather_than_repeat() -> None:
    # Fixed rather than learned so an unusually long verse is still positioned
    # sensibly at evaluation time; a longer table must agree with a shorter one
    # on their shared prefix.
    short = sinusoidal_positions(4, DIM)
    long = sinusoidal_positions(64, DIM)
    assert torch.equal(short, long[:4])


def test_sinusoidal_positions_reject_an_odd_dim() -> None:
    with pytest.raises(ValueError, match="even"):
        sinusoidal_positions(4, 7)
