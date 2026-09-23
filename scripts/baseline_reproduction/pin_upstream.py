"""Pin the frozen upstream (`utter-project/mHuBERT-147`) to a commit sha.

**Why this is not a one-line flag.** `s3prl/run_downstream.py` does expose
`--upstream_revision`, documented as "The commit hash of the specified
HuggingFace Repository", and it is tempting to just add it. It would do
nothing. That argument is read in exactly one place --
`s3prl/downstream/runner.py:125`,
`snapshot_download(self.args.upstream, self.args.upstream_revision, ...)` --
which is the code path for `--upstream` *itself* naming an HF repo that
holds an s3prl checkpoint. Our configuration is `-u hf_hubert_custom -k
utter-project/mHuBERT-147`, which goes to
`s3prl/upstream/hf_hubert/expert.py`:

    class UpstreamExpert(torch.nn.Module):
        def __init__(self, ckpt, **kwds):
            self.extracter = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)
            self.model = HubertModel.from_pretrained(ckpt)

`**kwds` is accepted and never read, and neither `from_pretrained` call is
given a `revision`. So `--upstream_revision` on our command line would be
recorded in the checkpoint's `Args`, look like provenance, and pin nothing.
That is worse than not pinning, because it is undetectable later.

**What actually pins it.** `from_pretrained` accepts a local directory. So
we resolve the repo at a fixed sha ourselves, materialize it on disk in a
sha-named directory, and hand `-k` that path. The sha is then visible in
the checkpoint's own `Args.upstream_ckpt`, which makes every trained
checkpoint self-describing under
`scripts/baseline_reproduction/inspect_s3prl_ckpt.py`.

**The pin is safe to adopt retroactively.** Checked against the Hub on
2026-09-23: no weight or config file in the repo has been touched since
2024-06-12, and every commit after that is README/metadata churn (the
current HEAD is "Fix YAML language metadata issue for Norwegian"). So
pinning to HEAD does not change the weights our validated 0.4406 run
consumed -- it only removes the ability for a future Hub edit to change
them silently.

    python scripts/baseline_reproduction/pin_upstream.py --output-dir run/upstreams

Prints a human-readable report; the **last line of stdout is the resolved
directory path**, so callers can do:

    UPSTREAM_DIR="$(python .../pin_upstream.py --output-dir ... | tail -1)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The frozen upstream, pinned. Changing either of these two lines
# invalidates the validated recipe and must be recorded in insights/.
MHUBERT_REPO = "utter-project/mHuBERT-147"
MHUBERT_REVISION = "7ad3fc0bc5106c58c9c13526abccad527150d135"

# Weights/config last modified on the Hub (verified 2026-09-23). Everything
# committed after this date is README-only, which is why pinning to HEAD is
# equivalent to what the 0.4406 run actually trained on.
WEIGHTS_LAST_CHANGED = "2024-06-12"

# `HubertModel.from_pretrained` + `Wav2Vec2FeatureExtractor.from_pretrained`
# need only these. Deliberately NOT downloading `checkpoint_best.pt`
# (1,085 MB fairseq original), `mhubert147_faiss.index`, or `manifest/*.tsv`
# -- none are read by the transformers loader, and $HOME is NFS
# quota-limited on the cluster.
CONFIG_FILES = ["config.json", "preprocessor_config.json"]
SAFETENSORS_WEIGHTS = "model.safetensors"
TORCH_WEIGHTS = "pytorch_model.bin"


def _weights_filename() -> str:
    """Prefer safetensors, but only if this interpreter can actually read it.

    transformers has depended on `safetensors` since 4.31, so on any modern
    install this picks the 360 MB safetensors file. The fallback exists
    because the S3PRL venv is Python 3.8 with an unpinned `pip install
    transformers`, and an old resolution there would need the .bin instead.
    Choosing here rather than downloading both saves 360 MB.
    """
    try:
        import safetensors  # noqa: F401
    except ImportError:
        return TORCH_WEIGHTS
    return SAFETENSORS_WEIGHTS


def _report_drift(repo: str, pinned: str) -> None:
    """Say whether the Hub has moved past the pin. Informational only.

    A moved HEAD is not an error -- that is the entire point of pinning --
    but it should be visible, because it is the trigger to re-verify that
    the new commits are still README-only.
    """
    try:
        from huggingface_hub import HfApi

        head = HfApi().model_info(repo).sha
    except Exception as exc:  # network, auth, API change -- all non-fatal
        print(f"  (could not check Hub HEAD: {type(exc).__name__}: {exc})")
        return
    if head == pinned:
        print(f"  Hub HEAD == pin ({head[:12]}). No drift.")
    else:
        print(f"  Hub HEAD is {head[:12]}, pin is {pinned[:12]} -- HEAD has MOVED.")
        print("  This is not an error; the pin is doing its job. But before")
        print("  advancing the pin, confirm the new commits are README-only:")
        print(f"    https://huggingface.co/{repo}/commits/main")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=MHUBERT_REPO)
    parser.add_argument("--revision", default=MHUBERT_REVISION)
    parser.add_argument(
        "--output-dir",
        default="run/upstreams",
        help="Parent directory; a sha-named subdirectory is created inside it.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report drift and the target path, download nothing.",
    )
    args = parser.parse_args()

    short = args.revision[:12]
    dest = Path(args.output_dir).resolve() / f"{args.repo.split('/')[-1]}-{short}"

    print(f"repo     : {args.repo}")
    print(f"revision : {args.revision}")
    print(f"weights last changed on the Hub: {WEIGHTS_LAST_CHANGED}")
    print(f"target   : {dest}")
    _report_drift(args.repo, args.revision)

    weights = _weights_filename()
    patterns = [*CONFIG_FILES, weights]
    print(f"files    : {patterns}")

    if args.check_only:
        print(dest)
        return

    marker = dest / weights
    if marker.exists() and all((dest / f).exists() for f in CONFIG_FILES):
        print("  already materialized; skipping download (idempotent).")
    else:
        from huggingface_hub import snapshot_download

        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            local_dir=str(dest),
            allow_patterns=patterns,
        )

    # Fail loudly rather than let a half-download reach a GPU queue slot.
    missing = [f for f in patterns if not (dest / f).exists()]
    if missing:
        raise SystemExit(
            f"ERROR: snapshot incomplete, missing {missing} in {dest}.\n"
            "Delete the directory and re-run; do NOT fall back to the "
            "unpinned repo id, which would silently un-pin the upstream."
        )

    print("  OK: upstream materialized at the pinned revision.")
    # Last line, machine-readable: the path to hand to `-k`.
    print(dest)


if __name__ == "__main__":
    sys.exit(main())
