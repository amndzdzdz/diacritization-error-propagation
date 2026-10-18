"""Pin a frozen S3PRL upstream to a commit sha.

Two upstreams are registered: `mhubert147`, the organizers' own (week 3), and
`xlsr300m`, the modern prompt-free arm (week 7 task 3). Select with
`--upstream`; `--repo`/`--revision` still override for a one-off.

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
    python scripts/baseline_reproduction/pin_upstream.py --upstream xlsr300m \
        --output-dir run/upstreams

Prints a human-readable report; the **last line of stdout is the resolved
directory path**, so callers can do:

    UPSTREAM_DIR="$(python .../pin_upstream.py --output-dir ... | tail -1)"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# `*Model.from_pretrained` + `Wav2Vec2FeatureExtractor.from_pretrained`
# need only these. Deliberately NOT downloading e.g. mHuBERT's
# `checkpoint_best.pt` (1,085 MB fairseq original), `mhubert147_faiss.index`,
# or `manifest/*.tsv` -- none are read by the transformers loader, and $HOME is
# NFS quota-limited on the cluster.
CONFIG_FILES = ["config.json", "preprocessor_config.json"]
SAFETENSORS_WEIGHTS = "model.safetensors"
TORCH_WEIGHTS = "pytorch_model.bin"

# The frozen upstreams, pinned. Changing any `revision` here invalidates the
# validated recipe for that arm and must be recorded in insights/.
#
# `s3prl_upstream` is the `-u` name the pinned directory must be paired with,
# and it is recorded here rather than left to the caller because getting it
# wrong is a whole queue slot: `hf_hubert_custom` builds a `HubertModel` and
# *cannot* load an XLS-R checkpoint, and `hf_wav2vec2_custom` the reverse.
# Both live at the pinned S3PRL commit (`s3prl/upstream/hf_{hubert,wav2vec2}/
# hubconf.py`) and expose the same interface: `from_pretrained(ckpt)`,
# `output_hidden_states=True`, `get_downsample_rates() == 320`. So swapping
# the upstream really does change only the upstream.
#
# `has_safetensors` is per-repo and not a global preference, because
# `facebook/wav2vec2-xls-r-300m` has never published one -- its file list is
# exactly [config.json, preprocessor_config.json, pytorch_model.bin]. Asking
# for `model.safetensors` there downloads nothing and the completeness check
# at the bottom of this file fires; declaring it here means we never get that
# far.
UPSTREAMS = {
    "mhubert147": {
        "repo": "utter-project/mHuBERT-147",
        "revision": "7ad3fc0bc5106c58c9c13526abccad527150d135",
        "s3prl_upstream": "hf_hubert_custom",
        # Weights/config last modified on the Hub (verified 2026-09-23).
        # Everything committed after this date is README-only, which is why
        # pinning to HEAD is equivalent to what the 0.4406 run trained on.
        "weights_last_changed": "2024-06-12",
        "has_safetensors": True,
        "note": "the organizers' own upstream; week-3 reproduction",
    },
    "xlsr300m": {
        "repo": "facebook/wav2vec2-xls-r-300m",
        # Hub HEAD as of 2026-10-16, title "Add language tags (#1)".
        "revision": "1a640f32ac3e39899438a2931f9924c02f080a54",
        "s3prl_upstream": "hf_wav2vec2_custom",
        # Checked the full 25-commit history on 2026-10-16: the only
        # non-README commits are `initial commit` and `up` (2021-10-26, the
        # weights) and `Update config.json` (2021-11-03). HEAD is two years of
        # metadata churn on top of frozen weights, so pinning to it is safe.
        "weights_last_changed": "2021-11-03",
        "has_safetensors": False,
        # 24 layers / hidden_size 1024, against mHuBERT's 12 / 768 -- so 25
        # hidden states reach the featurizer rather than 13. Nothing to
        # configure: s3prl sizes the weighted-sum and the downstream input dim
        # from a dummy forward pass, not from the config file.
        "note": "modern prompt-free arm; 24 layers, 1024-dim, ~1.26 GB .bin",
    },
}


def _weights_filename(spec: dict) -> str:
    """Which weights file to fetch for this repo, under this interpreter.

    Two independent conditions. The repo has to *have* a safetensors file, and
    this interpreter has to be able to read one -- the S3PRL venv is Python 3.8
    with an unpinned `pip install transformers` (week-7 task 1), and an old
    resolution there predates the safetensors dependency. Choosing here rather
    than downloading both saves 360 MB on mHuBERT.
    """
    if not spec["has_safetensors"]:
        return TORCH_WEIGHTS
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
    parser.add_argument(
        "--upstream",
        default="mhubert147",
        choices=sorted(UPSTREAMS),
        help="which registered upstream to pin (default: the week-3 baseline's)",
    )
    parser.add_argument("--repo", help="override the registered repo id")
    parser.add_argument("--revision", help="override the registered sha")
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
    parser.add_argument(
        "--print-s3prl-upstream",
        action="store_true",
        help="print only the `-u` name this upstream must be paired with, and exit",
    )
    args = parser.parse_args()

    spec = UPSTREAMS[args.upstream]
    if args.print_s3prl_upstream:
        print(spec["s3prl_upstream"])
        return
    repo = args.repo or spec["repo"]
    revision = args.revision or spec["revision"]

    short = revision[:12]
    dest = Path(args.output_dir).resolve() / f"{repo.split('/')[-1]}-{short}"

    print(f"upstream : {args.upstream} -- {spec['note']}")
    print(f"repo     : {repo}")
    print(f"revision : {revision}")
    print(f"-u name  : {spec['s3prl_upstream']}")
    print(f"weights last changed on the Hub: {spec['weights_last_changed']}")
    print(f"target   : {dest}")
    if args.repo or args.revision:
        # An override is legitimate for a one-off probe and poisonous in a
        # training run, because the registry is what insights/ cites.
        print("  WARNING: repo/revision overridden on the command line.")
        print("  A checkpoint trained this way is NOT described by UPSTREAMS.")
    _report_drift(repo, revision)

    weights = _weights_filename(spec)
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
            repo_id=repo,
            revision=revision,
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
