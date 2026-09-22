"""Print the training Args/Config embedded in an S3PRL downstream checkpoint.

Project-authored (not vendored). S3PRL's runner stores the full argparse
Namespace and the resolved YAML config inside every downstream checkpoint it
saves, so the organizers' own `mhubert.ckpt` states outright what produced
the published F1 = 44.14% -- no guessing from the paper prose.

Two week-3 questions this answers directly:

  1. **Which upstream.** `run/train_baseline.slurm` passes `-u hubert_base`
     (English HuBERT Base, LibriSpeech) as an unresolved stand-in for the
     paper's frozen mHuBERT-147; that mismatch cost 1,282 extra false
     rejections and 0.036 F1 (insights/week-03.md). `Args.upstream` and
     `Args.upstream_ckpt` give the real identifier to pass instead.
  2. **How many steps.** The vendored `config/sws.yaml` says
     `runner.total_steps: 200000` while the benchmark paper's prose says
     12.5k updates. Our own run's logs came back empty, so neither was
     confirmed. `Config['runner']` and the saved `Step` settle it -- and the
     answer changes the cost of a retrain by ~16x.

Needs only `torch` and the stdlib -- no `s3prl`, no GPU -- so it runs in
either environment. The checkpoint may be a local path or a URL (downloaded
to a temp file).

    # cluster, S3PRL venv, inspecting a local checkpoint:
    source run/s3prl_venv/bin/activate
    python scripts/baseline_reproduction/inspect_s3prl_ckpt.py hubert_base_per/dev-best.ckpt

    # anywhere with this project's env, inspecting the organizers' checkpoint:
    uv run python scripts/baseline_reproduction/inspect_s3prl_ckpt.py

Defaults to the organizers' public checkpoint. Pass a path to inspect one of
our own runs instead, e.g. `hubert_base_per/dev-best.ckpt`.
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

import torch

OFFICIAL_CKPT = "https://huggingface.co/IqraEval/Iqra_mhubert_base/resolve/main/mhubert.ckpt"

# The keys worth pulling out of Config and printing first, because they are
# the ones the week-3 writeup actually needs. Everything else still gets
# dumped below, just unprioritised.
HIGHLIGHT_ARGS = ("upstream", "upstream_ckpt", "upstream_model_config", "downstream", "expdir")
HIGHLIGHT_CONFIG_SECTIONS = ("runner", "optimizer", "scheduler")


def _download_if_needed(path: str) -> str:
    """Fetch `path` to a temp file if it's a URL, else return it unchanged.

    Deliberately a local copy of `s3prl_inference.download_if_needed` rather
    than an import: that module pulls in `s3prl` and `torchaudio` at import
    time, which would pin this script to the S3PRL venv even though reading a
    checkpoint's metadata needs nothing but torch.
    """
    if not str(path).startswith(("http://", "https://")):
        return path
    suffix = Path(path).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        with urllib.request.urlopen(path) as response:  # noqa: S310 - fixed https URL
            while chunk := response.read(1 << 20):
                tmp.write(chunk)
        return tmp.name


def _describe(value: object, indent: int = 0) -> str:
    """Render a checkpoint value compactly, summarising tensors by shape.

    State dicts hold hundreds of tensors whose contents are noise here; only
    their presence and size matter for identifying what was trained.
    """
    pad = "  " * indent
    if torch.is_tensor(value):
        return f"Tensor{tuple(value.shape)} {value.dtype}"
    if isinstance(value, dict):
        if not value:
            return "{}"
        tensor_count = sum(1 for v in value.values() if torch.is_tensor(v))
        if tensor_count and tensor_count == len(value):
            total = sum(v.numel() for v in value.values())
            return f"<state dict: {len(value)} tensors, {total:,} params>"
        lines = [""]
        for key, sub in value.items():
            lines.append(f"{pad}  {key}: {_describe(sub, indent + 1)}")
        return "\n".join(lines)
    if isinstance(value, (list, tuple)) and len(value) > 8:
        return f"{type(value).__name__} of {len(value)} items: {value[:4]} ..."
    return repr(value)


def main(ckpt_path: str) -> None:
    resolved = _download_if_needed(ckpt_path)
    print(f"Loading {ckpt_path}")
    if resolved != ckpt_path:
        print(f"  downloaded to {resolved}")

    # Same loading incantation as the vendored s3prl_inference.S3PRLModel:
    # these checkpoints pickle an argparse.Namespace, which newer torch
    # refuses to unpickle under the default weights_only=True.
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([argparse.Namespace])
    state = torch.load(resolved, map_location="cpu", weights_only=False)

    print("\n=== top-level keys ===")
    for key in state:
        print(f"  {key}")

    args = state.get("Args")
    if args is not None:
        args_dict = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
        print("\n=== Args: the answer to 'which upstream' ===")
        for key in HIGHLIGHT_ARGS:
            if key in args_dict:
                print(f"  {key:24} = {args_dict[key]!r}")
        print("\n--- all other Args ---")
        for key in sorted(args_dict):
            if key not in HIGHLIGHT_ARGS:
                print(f"  {key:24} = {args_dict[key]!r}")

    print("\n=== training progress: the answer to '200k or 12.5k steps' ===")
    for key in ("Step", "Epoch"):
        if key in state:
            print(f"  {key:24} = {state[key]!r}")

    config = state.get("Config")
    if config is not None:
        print("\n=== Config ===")
        for section in HIGHLIGHT_CONFIG_SECTIONS:
            if section in config:
                print(f"  {section}: {_describe(config[section], 1)}")
        for section in sorted(config):
            if section not in HIGHLIGHT_CONFIG_SECTIONS:
                print(f"  {section}: {_describe(config[section], 1)}")

    print("\n=== remaining entries ===")
    for key, value in state.items():
        if key not in ("Args", "Config", "Step", "Epoch"):
            print(f"  {key}: {_describe(value, 1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ckpt",
        nargs="?",
        default=OFFICIAL_CKPT,
        help="Checkpoint path or URL (default: the organizers' mhubert.ckpt).",
    )
    main(parser.parse_args().ckpt)
