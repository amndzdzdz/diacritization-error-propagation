"""The arms must differ in exactly one property, and the model must train.

The first half of this file is the "exactly one property" claim from plan §3,
turned into assertions. The second half is the shape-and-overfit smoke test
week-07 task 7 asks for before anything touches the queue.
"""

from __future__ import annotations

import pytest
import torch

from arabic_mdd.models.mdd import (
    MddConfig,
    MddModel,
    collate_batch,
    ctc_loss,
    greedy_indices,
)

VOCAB = 12
UPSTREAM_DIM = 8
PROJECT_DIM = 8


def make_config(*, use_canonical: bool) -> MddConfig:
    """A small model with sws.yaml's *structure* and a testable width."""
    return MddConfig(
        vocab_size=VOCAB,
        upstream_dim=UPSTREAM_DIM,
        project_dim=PROJECT_DIM,
        rnn_dims=(4, 4),
        rnn_dropout=(0.0, 0.0),
        rnn_layer_norm=(False, False),
        rnn_proj=(True, True),
        conditioner_heads=2,
        use_canonical=use_canonical,
    )


def make_model(*, use_canonical: bool, seed: int = 7) -> MddModel:
    torch.manual_seed(seed)
    model = MddModel(make_config(use_canonical=use_canonical))
    model.eval()
    return model


@pytest.fixture
def batch():
    torch.manual_seed(3)
    features = [torch.randn(12, UPSTREAM_DIM), torch.randn(9, UPSTREAM_DIM)]
    targets = [[3, 4, 5, 1], [6, 7, 1]]
    canonicals = [[3, 4, 5], [6, 7, 8, 9]]
    return collate_batch(features, targets, canonicals)


# --- "exactly one property" ------------------------------------------------


def test_both_arms_have_identical_head_and_projector_weights() -> None:
    # The reason the conditioner is constructed even in the ablated arm. If it
    # were built conditionally it would consume less RNG, the recurrent head
    # would initialize differently, and the arms would differ in two properties
    # with nothing in any result to reveal it.
    free = make_model(use_canonical=False)
    conditioned = make_model(use_canonical=True)
    free_state = dict(free.named_parameters())
    for name, parameter in conditioned.named_parameters():
        if name.startswith("conditioner."):
            continue
        assert torch.equal(parameter, free_state[name]), name


def test_both_arms_have_the_same_parameter_names() -> None:
    free = set(dict(make_model(use_canonical=False).named_parameters()))
    conditioned = set(dict(make_model(use_canonical=True).named_parameters()))
    assert free == conditioned


def test_conditioned_arm_at_init_matches_the_prompt_free_arm_bit_for_bit(batch) -> None:
    # The headline property. At step 0 the zero-gated residual adds exactly
    # zero, so the text-dependent arm *is* the prompt-free arm as a function.
    free = make_model(use_canonical=False)
    conditioned = make_model(use_canonical=True)
    free_logits, free_lens = free(batch.features, batch.feature_lens)
    cond_logits, cond_lens = conditioned(
        batch.features, batch.feature_lens, batch.canonical_ids, batch.canonical_lens
    )
    assert torch.equal(free_logits, cond_logits)
    assert torch.equal(free_lens, cond_lens)


def test_prompt_free_arm_ignores_a_canonical_it_is_given(batch) -> None:
    # One evaluation loop feeds both arms. Making the prompt-free arm raise on
    # an unused canonical would mean two loops, which is one more place for them
    # to drift apart.
    free = make_model(use_canonical=False)
    without, _ = free(batch.features, batch.feature_lens)
    with_canonical, _ = free(
        batch.features, batch.feature_lens, batch.canonical_ids, batch.canonical_lens
    )
    assert torch.equal(without, with_canonical)


def test_conditioned_arm_refuses_to_run_without_a_canonical(batch) -> None:
    conditioned = make_model(use_canonical=True)
    with pytest.raises(ValueError, match="use_canonical=True"):
        conditioned(batch.features, batch.feature_lens)


def test_conditioner_parameters_get_no_gradient_in_the_prompt_free_arm(batch) -> None:
    free = make_model(use_canonical=False)
    free.train()
    logits, logit_lens = free(batch.features, batch.feature_lens)
    ctc_loss(logits, logit_lens, batch.targets, batch.target_lens).backward()
    assert all(p.grad is None for p in free.conditioning_parameters)
    assert free.projector.weight.grad is not None


# --- shapes, lengths, decoding ---------------------------------------------


def test_logit_shape_and_lengths_follow_the_inputs(batch) -> None:
    model = make_model(use_canonical=True)
    logits, logit_lens = model(
        batch.features, batch.feature_lens, batch.canonical_ids, batch.canonical_lens
    )
    assert logits.shape == (2, 12, VOCAB)
    # sample_rate is 1, so frames pass through the head without downsampling.
    assert torch.equal(logit_lens, batch.feature_lens)


def test_padded_frames_do_not_change_a_shorter_utterances_logits() -> None:
    # Packing is what guarantees this. If it regressed, the shorter utterance in
    # every batch would silently train against the longer one's padding.
    model = make_model(use_canonical=False)
    short = torch.randn(5, UPSTREAM_DIM)
    alone, alone_lens = model(short.unsqueeze(0), torch.tensor([5]))
    padded = torch.zeros(1, 11, UPSTREAM_DIM)
    padded[0, :5] = short
    batched, _ = model(padded, torch.tensor([5]))
    assert int(alone_lens[0]) == 5
    assert torch.allclose(alone[0, :5], batched[0, :5], atol=1e-6)


def test_greedy_indices_trims_to_each_real_length(batch) -> None:
    model = make_model(use_canonical=False)
    logits, logit_lens = model(batch.features, batch.feature_lens)
    decoded = greedy_indices(logits, logit_lens)
    assert [len(d) for d in decoded] == [12, 9]


# --- collation -------------------------------------------------------------


def test_collate_pads_features_and_flattens_targets(batch) -> None:
    assert batch.features.shape == (2, 12, UPSTREAM_DIM)
    assert torch.equal(batch.feature_lens, torch.tensor([12, 9]))
    # Concatenated, not a padded rectangle: the padded form is where an
    # off-by-one in target_lens trains blanks as real labels.
    assert batch.targets.tolist() == [3, 4, 5, 1, 6, 7, 1]
    assert torch.equal(batch.target_lens, torch.tensor([4, 3]))


def test_collate_pads_canonicals_and_records_their_lengths(batch) -> None:
    assert batch.canonical_ids.shape == (2, 4)
    assert torch.equal(batch.canonical_lens, torch.tensor([3, 4]))
    assert int(batch.canonical_ids[0, 3]) == 0  # padding_idx


def test_collate_without_canonicals_leaves_them_empty() -> None:
    plain = collate_batch([torch.randn(4, UPSTREAM_DIM)], [[3, 1]])
    assert plain.canonical_ids.numel() == 0
    assert plain.canonical_lens.numel() == 0


def test_collate_rejects_mismatched_row_counts() -> None:
    with pytest.raises(ValueError, match="target rows"):
        collate_batch([torch.randn(4, UPSTREAM_DIM)], [[3], [4]])
    with pytest.raises(ValueError, match="canonical rows"):
        collate_batch([torch.randn(4, UPSTREAM_DIM)], [[3]], [[3], [4]])


# --- config guards ---------------------------------------------------------


def test_config_rejects_mismatched_per_layer_lists() -> None:
    with pytest.raises(ValueError, match="per-layer lists"):
        MddConfig(vocab_size=VOCAB, rnn_dims=(4, 4), rnn_dropout=(0.0,))


def test_config_rejects_a_sample_rate_the_port_does_not_implement() -> None:
    # s3prl's head has time-downsampling branches that sws.yaml never reaches.
    # This port omits them, so it must refuse the configurations that need them
    # rather than silently ignore the request.
    with pytest.raises(ValueError, match="sample_rate"):
        MddConfig(vocab_size=VOCAB, sample_rate=2)


def test_defaults_match_the_organizers_config() -> None:
    # sws.yaml: project_dim 1024, dim [1024, 1024], dropout [0.2, 0.2],
    # layer_norm [False, False], proj [True, True], zero_infinity True.
    config = MddConfig(vocab_size=71)
    assert config.project_dim == 1024
    assert config.rnn_dims == (1024, 1024)
    assert config.rnn_dropout == (0.2, 0.2)
    assert config.rnn_layer_norm == (False, False)
    assert config.rnn_proj == (True, True)
    assert config.rnn_module == "LSTM"
    assert config.bidirectional is True
    assert config.zero_infinity is True
    # The prompt-free arm is the default, so a config built without thinking
    # about it cannot accidentally be the text-dependent one.
    assert config.use_canonical is False


# --- it actually trains ----------------------------------------------------


@pytest.mark.parametrize("use_canonical", [False, True])
def test_one_batch_overfits(batch, use_canonical: bool) -> None:
    """Both arms drive the CTC loss down on a single repeated batch.

    The cheap end-to-end check: a shape error, a lengths error, a blank-index
    error or a detached graph all show up here as a loss that does not move,
    and all of them otherwise show up hours into a queued job.
    """
    torch.manual_seed(11)
    model = MddModel(make_config(use_canonical=use_canonical))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        logits, logit_lens = model(
            batch.features,
            batch.feature_lens,
            batch.canonical_ids if use_canonical else None,
            batch.canonical_lens if use_canonical else None,
        )
        loss = ctc_loss(logits, logit_lens, batch.targets, batch.target_lens)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert all(torch.isfinite(torch.tensor(value)) for value in losses)
    assert losses[-1] < losses[0] * 0.5, losses[:: len(losses) // 4]


def test_conditioning_block_trains_away_from_the_identity(batch) -> None:
    # The zero gate must be a starting point, not a fixed point. If the gate
    # stayed at zero the text-dependent arm would be the prompt-free arm with
    # extra parameters, and the 2x2 would measure nothing.
    torch.manual_seed(11)
    model = MddModel(make_config(use_canonical=True))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        optimizer.zero_grad()
        logits, logit_lens = model(
            batch.features, batch.feature_lens, batch.canonical_ids, batch.canonical_lens
        )
        ctc_loss(logits, logit_lens, batch.targets, batch.target_lens).backward()
        optimizer.step()
    assert torch.any(model.conditioner.gate.weight != 0)


def test_the_two_arms_diverge_once_trained(batch) -> None:
    # Same seed, same data, same steps; the only difference is whether C is
    # read. They must not end up at the same place, or conditioning is a no-op.
    def train() -> torch.Tensor:
        torch.manual_seed(11)
        model = MddModel(make_config(use_canonical=True))
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(10):
            optimizer.zero_grad()
            logits, logit_lens = model(
                batch.features, batch.feature_lens, batch.canonical_ids, batch.canonical_lens
            )
            ctc_loss(logits, logit_lens, batch.targets, batch.target_lens).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            gold, _ = model(
                batch.features, batch.feature_lens, batch.canonical_ids, batch.canonical_lens
            )
            # The D10 swap: same trained model, a different canonical at
            # inference. This is the operation the whole 2x2 rests on.
            corrupted = batch.canonical_ids.clone()
            corrupted[0, 0] = 10
            swapped, _ = model(batch.features, batch.feature_lens, corrupted, batch.canonical_lens)
        return gold - swapped

    difference = train()
    assert torch.any(difference != 0)
