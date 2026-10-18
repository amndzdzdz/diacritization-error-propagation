"""Construct and run `downstream/ctc_cond`'s expert without a GPU or a dataset.

`scripts/baseline_reproduction/ctc_cond/expert.py` lives in a ruff-excluded
directory and cannot be unit-tested the way `src/arabic_mdd/models/` is: it
imports from the s3prl package and is instantiated by s3prl's runner. The
failures it is exposed to are consequently the dull, expensive kind -- a config
key spelled `bidirectional` where s3prl writes `bidirection`, a tensor handed
to `tokenizer.decode` where a list of ints was wanted -- each of which costs a
queue slot and an hour to discover.

So: load the real `sws_cond.yaml`, build the real `DownstreamExpert` against a
real s3prl checkout, and run `forward` and `log_records` on random features, for
**both** arms. No audio, no dataloader, no upstream. Catches everything in the
glue; catches nothing about whether the model learns, which is what
`tests/models/` and `scripts/smoke_mdd_arm.py` are for.

    uv run python scripts/smoke_ctc_cond_expert.py

Run it on a login node before submitting `run/train_mdd_arm.slurm`.

Two things it does that look like side effects and are:

* It copies `scripts/baseline_reproduction/ctc_cond/*` into
  `<checkout>/s3prl/downstream/ctc_cond/`, which is exactly what
  `run/prepare_baseline.sh` step 7b does, into a directory that is untracked
  there by design.
* If `editdistance` or `torchaudio` is missing it installs a stub into
  `sys.modules`. Both are imported eagerly and neither is reached on this path;
  the stub makes the printed `wer` meaningless, and the script says so. The
  cluster venv has both for real.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import types
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_SRC = REPO_ROOT / "scripts" / "baseline_reproduction"


def install_stubs() -> list[str]:
    """Stub the import-time-only dependencies this script never exercises.

    Returns the names that needed stubbing, so the caller can say so. Both are
    imported eagerly at module level and neither is reached on this path:
    `editdistance` backs `ctc/metric.py`'s `wer`, and `torchaudio.load` is
    called only by `ctc_cond/data.py`'s collate function, which needs real
    audio and a dataloader. The cluster venv has both for real.
    """
    stubbed = []

    try:
        import editdistance  # noqa: F401
    except ImportError:
        module = types.ModuleType("editdistance")
        module.eval = lambda a, b: abs(len(a) - len(b))  # type: ignore[attr-defined]
        sys.modules["editdistance"] = module
        stubbed.append("editdistance (so `wer` below is nonsense)")

    try:
        import torchaudio  # noqa: F401
    except ImportError:
        module = types.ModuleType("torchaudio")

        def _load(_path):
            raise RuntimeError("torchaudio is stubbed; this script reads no audio")

        module.load = _load  # type: ignore[attr-defined]
        sys.modules["torchaudio"] = module
        stubbed.append("torchaudio (no audio is read)")

    return stubbed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--s3prl",
        type=Path,
        default=REPO_ROOT / "run" / "s3prl_checkout",
        help="an s3prl checkout; its parent of `s3prl/` goes on sys.path",
    )
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--upstream-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import yaml

    checkout = args.s3prl.resolve()
    if not (checkout / "s3prl" / "downstream" / "ctc" / "expert.py").is_file():
        print(
            f"ERROR: no s3prl checkout at {checkout}.\n"
            "Pass --s3prl, or run `APPLY_HOLDOUT=0 bash run/prepare_baseline.sh` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    target = checkout / "s3prl" / "downstream" / "ctc_cond"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "data.py", "expert.py"):
        shutil.copy(BASELINE_SRC / "ctc_cond" / name, target / name)
    print(f"materialized ctc_cond into {target}")

    sys.path.insert(0, str(checkout))
    stubbed = install_stubs()
    if stubbed:
        print("NOTE: stubbed " + ", ".join(stubbed))

    from s3prl.downstream.ctc_cond.expert import DownstreamExpert

    config = yaml.safe_load((BASELINE_SRC / "config" / "sws_cond.yaml").read_text())
    expert_config = config["downstream_expert"]
    # The vocab path in the config is relative to the checkout's `s3prl/`
    # directory, which is run_downstream.py's cwd. Point it at the tracked copy
    # instead so this works before prepare_baseline.sh has staged anything.
    expert_config["text"]["vocab_file"] = str(BASELINE_SRC / "vocab" / "sws_arabic.txt")

    vocab_tokens = len((BASELINE_SRC / "vocab" / "sws_arabic.txt").read_text().split())
    print(f"vocab: {vocab_tokens} phonemes + 3 reserved = {vocab_tokens + 3} classes")

    for use_canonical in (False, True):
        expert_config["model"]["use_canonical"] = use_canonical
        torch.manual_seed(args.seed)
        expert = DownstreamExpert(
            upstream_dim=args.upstream_dim,
            # 320 samples per frame, matching every wav2vec-family upstream
            # here and `RNNs.total_rate: 320`, so the derived sample_rate is 1.
            upstream_rate=320,
            downstream_expert=expert_config,
            expdir=str(Path(args.s3prl).parent / "smoke_expdir"),
            **{k: v for k, v in config.items() if k != "downstream_expert"},
        )
        Path(expert.expdir).mkdir(parents=True, exist_ok=True)

        torch.manual_seed(args.seed + 1)
        features = [
            torch.randn(args.frames - 7 * i, args.upstream_dim) for i in range(args.batch_size)
        ]
        # Targets and canonicals as `ctc_cond/data.py` yields them: index lists
        # over the real vocabulary, targets with an eos and canonicals without.
        labels = [[5, 9, 13, 4, 1] for _ in range(args.batch_size)]
        canonicals = [[5, 9, 13, 4], [5, 9, 11, 4, 7], [5, 9]]
        filenames = [f"utt_{i}" for i in range(args.batch_size)]

        records = defaultdict(list)
        # "dev" so log_records takes the branch that writes the three .ark
        # files, which is the part this script exists to exercise.
        loss = expert("dev", features, labels, canonicals, filenames, records)
        loss.backward()

        grads = {
            "conditioner": any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in expert.model.conditioning_parameters
            ),
            "head": any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in expert.model.head.parameters()
            ),
        }
        expert.log_records("dev", records, _NullLogger(), 0)

        arks = sorted(p.name for p in Path(expert.expdir).glob("dev-*.ark"))
        print(
            f"use_canonical={use_canonical}: loss={float(loss.detach()):.4f}  "
            f"grads={grads}  arks={arks}"
        )
        canon_ark = Path(expert.expdir) / "dev-canon.ark"
        print(f"  dev-canon.ark line 1: {canon_ark.read_text().splitlines()[0]!r}")

    print(
        "\nBoth arms' step-0 loss should agree to the digit: the conditioner's gate is\n"
        "zero-initialized, so at step 0 the two arms are the same function bitwise.\n"
        "The conditioner should show grads=False when ablated and True when not."
    )


class _NullLogger:
    """s3prl passes a tensorboard SummaryWriter; `log_records` only adds scalars."""

    def add_scalar(self, *_args, **_kwargs) -> None:
        return None


if __name__ == "__main__":
    main()
