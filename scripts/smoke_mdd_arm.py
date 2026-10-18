#!/usr/bin/env python3
"""Smoke-test the MDD downstream at `sws.yaml`'s real dimensions.

Week-07 task 7 asks for an end-to-end smoke test "before it touches the queue
-- a 48h job that dies at hour 40 on a shape error costs two days". This is the
part of that which needs no GPU and no audio: the downstream from fused
features onward, at the real widths, on synthetic features of realistic length.

What it does **not** cover, stated so the coverage is not overclaimed: the
upstream forward pass, the s3prl featurizer's learned layer-weighted sum, the
dataloader, and the specaug block. Those are s3prl's and are exercised by
`run/smoke_train.slurm`. What this catches is everything that is ours --
shapes, lengths, the blank index, the packing, the CTC call, and whether the
conditioned arm's extra block costs more than it looks.

    uv run python scripts/smoke_mdd_arm.py
    uv run python scripts/smoke_mdd_arm.py --device cuda --steps 20

On a GPU the per-step time printed here is the downstream's share only. The
useful number is the *ratio* between the arms: it says how much of the
text-dependent arm's wall clock is the conditioning, which is what decides
whether the two arms can share one time limit.
"""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import torch

from arabic_mdd.models import MddConfig, MddModel, collate_batch, ctc_loss, load_vocab

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB = REPO_ROOT / "scripts" / "baseline_reproduction" / "vocab" / "sws_arabic.txt"

# 16000 Hz / 320 = 50 frames per second, so 300 frames is 6 s -- about the
# middle of Iqra_train's utterance lengths.
FRAMES_PER_SECOND = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--batch-size", type=int, default=16, help="sws.yaml's batch_size")
    parser.add_argument("--seconds", type=float, default=6.0, help="synthetic utterance length")
    parser.add_argument("--canonical-len", type=int, default=40, help="canonical phoneme count")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vocab = load_vocab(args.vocab)
    device = torch.device(args.device)
    frames = int(args.seconds * FRAMES_PER_SECOND)

    print(f"vocab {vocab.size} classes (blank={vocab.blank_idx}, eos={vocab.eos_idx})")
    print(
        f"batch {args.batch_size} x {frames} frames ({args.seconds} s), "
        f"canonical {args.canonical_len} phonemes, device {device}"
    )

    torch.manual_seed(args.seed)
    config_dims = MddConfig(vocab_size=vocab.size)
    features = [torch.randn(frames, config_dims.upstream_dim) for _ in range(args.batch_size)]
    # Content is irrelevant to shapes and timing; a single repeated phoneme
    # keeps the target length exact and the output readable.
    targets = [vocab.encode(["b"] * args.canonical_len) for _ in range(args.batch_size)]
    canonicals = [
        vocab.encode(["b"] * args.canonical_len, append_eos=False) for _ in range(args.batch_size)
    ]
    batch = collate_batch(features, targets, canonicals)

    timings: dict[bool, float] = {}
    for use_canonical in (False, True):
        torch.manual_seed(args.seed)
        model = MddModel(MddConfig(vocab_size=vocab.size, use_canonical=use_canonical))
        model.to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)

        total = sum(p.numel() for p in model.parameters())
        conditioner = sum(p.numel() for p in model.conditioning_parameters)
        arm = "text-dependent" if use_canonical else "prompt-free   "

        losses = []
        start = time.perf_counter()
        for _ in range(args.steps):
            optimizer.zero_grad()
            logits, logit_lens = model(
                batch.features.to(device),
                batch.feature_lens,
                batch.canonical_ids.to(device) if use_canonical else None,
                batch.canonical_lens if use_canonical else None,
            )
            loss = ctc_loss(
                logits,
                logit_lens,
                batch.targets.to(device),
                batch.target_lens,
                blank=vocab.blank_idx,
                zero_infinity=config_dims.zero_infinity,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_step = (time.perf_counter() - start) / args.steps
        timings[use_canonical] = per_step

        trainable = total - (0 if use_canonical else conditioner)
        print(
            f"{arm}  logits {tuple(logits.shape)}  "
            f"params {total / 1e6:.2f}M (trainable {trainable / 1e6:.2f}M)  "
            f"loss {losses[0]:.3f} -> {losses[-1]:.3f}  {per_step:.3f} s/step"
        )
        if not all(torch.isfinite(torch.tensor(value)) for value in losses):
            print("FAIL: a non-finite loss. Check the blank index and the lengths.")
            return 1

    overhead = timings[True] / timings[False] - 1.0
    print(f"conditioning overhead: {overhead:+.1%} of downstream step time")
    # Observed on this sandbox: +12.9% at 3 steps, -9.4% at 2. A *negative*
    # overhead is impossible -- the conditioned arm strictly does more work --
    # so a negative or wildly varying figure is measuring scheduler noise on a
    # shared CPU, not the block. Say so rather than let the number be quoted.
    if overhead < 0 or args.steps < 10:
        print(
            f"  ^ do not quote this: {args.steps} steps is too few to time, and a "
            "negative value is noise by construction. Re-run with --steps 20 or more, "
            "on the device whose wall clock you actually care about."
        )
    print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
    if device.type == "cuda":
        print(f"peak GPU: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
