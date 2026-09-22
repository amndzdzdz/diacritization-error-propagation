#!/bin/bash
#SBATCH --job-name=iqra_inspect_ckpt
#SBATCH --output=run/logs/output_logs/output_iqra_inspect_ckpt_%j.log
#SBATCH --error=run/logs/error_logs/error_iqra_inspect_ckpt_%j.log
#SBATCH -J iqra_inspect_ckpt
#SBATCH -p NvidiaAll
#SBATCH -t 00:20:00

# Print the Args/Config an S3PRL downstream checkpoint was trained with.
# CPU-only and single-stage: `torch` is a declared project dependency, so
# this runs in the uv env and needs neither the S3PRL Python-3.8 venv nor a
# GPU. The only slow part is pulling the 616 MB checkpoint.
#
#   sbatch run/inspect_ckpt.slurm                              # organizers' ckpt
#   CKPT=hubert_base_per/dev-best.ckpt sbatch run/inspect_ckpt.slurm   # one of ours
#
# A relative CKPT is resolved against the s3prl checkout, since that is where
# training writes its exp dirs; an absolute path or an http(s) URL is used
# as-is. Output lands in run/logs/output_logs/.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# Batch jobs don't source ~/.zshrc or ~/.bashrc, so make sure `uv` is found.
export PATH="$HOME/.local/bin:$PATH"

# The checkpoint is downloaded to a temp file. Keep that off $HOME, which is
# NFS-mounted and quota-limited -- 616 MB is enough to hit it.
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

S3PRL_DIR="${S3PRL_DIR:-$REPO_ROOT/run/s3prl_checkout}"
CKPT="${CKPT:-https://huggingface.co/IqraEval/Iqra_mhubert_base/resolve/main/mhubert.ckpt}"
case "$CKPT" in
    http://*|https://*|/*) ;;
    *) CKPT="$S3PRL_DIR/s3prl/$CKPT" ;;
esac

echo "checkpoint: $CKPT"
uv run python scripts/baseline_reproduction/inspect_s3prl_ckpt.py "$CKPT"
