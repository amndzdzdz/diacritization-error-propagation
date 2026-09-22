#!/bin/bash
#SBATCH --job-name=iqra_mhubert_baseline_eval
#SBATCH --output=run/logs/output_logs/output_iqra_mhubert_baseline_eval_%j.log
#SBATCH --error=run/logs/error_logs/error_iqra_mhubert_baseline_eval_%j.log
#SBATCH -J iqra_mhubert_baseline_eval
#SBATCH -p NvidiaAll
#SBATCH -t 04:00:00

# Evaluation-only counterpart to run/train_baseline.slurm -- runs
# `run_downstream.py -m evaluate` against an already-trained checkpoint,
# without repeating environment setup, data download, or training. Use this
# to re-score an existing checkpoint (e.g. after fixing something in
# scoring/config) without paying for another multi-hour training run.
#
# Assumes run/train_baseline.slurm (or `bash run/prepare_baseline.sh` +
# a manual train step) has already been run at least once, so the S3PRL
# venv/checkout and the checkpoint below all exist. This script does NOT
# call run/prepare_baseline.sh -- it fails fast with a clear error instead
# of silently re-downloading/re-installing anything.
#
# Override via environment variables if your run used non-default values:
#   EXP_DIR   -- experiment dir name passed to `-p` during training
#                (default: hubert_base_per, matching train_baseline.slurm)
#   CKPT      -- checkpoint path to evaluate, relative to $S3PRL_DIR/s3prl
#                (default: $EXP_DIR/dev-best.ckpt)
#   S3PRL_DIR, S3PRL_VENV -- same meaning as in train_baseline.slurm/
#                            prepare_baseline.sh
#
# Example: re-evaluate a differently-named experiment dir:
#   EXP_DIR=hubert_base_per_rerun sbatch run/evaluate_baseline.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Batch jobs don't source ~/.zshrc or ~/.bashrc, so make sure `uv` (installed
# to ~/.local/bin via the standalone installer) is reachable here too, in
# case anything downstream needs it.
export PATH="$HOME/.local/bin:$PATH"

S3PRL_VENV="${S3PRL_VENV:-$(pwd)/run/s3prl_venv}"
if [ ! -f "$S3PRL_VENV/bin/activate" ]; then
    echo "ERROR: no S3PRL venv found at $S3PRL_VENV." >&2
    echo "Run 'sbatch run/train_baseline.slurm' (or 'bash run/prepare_baseline.sh')" >&2
    echo "at least once first -- this script only evaluates, it does not set up" >&2
    echo "the environment." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$S3PRL_VENV/bin/activate"

S3PRL_DIR="${S3PRL_DIR:-$(pwd)/run/s3prl_checkout}"
cd "$S3PRL_DIR/s3prl"

EXP_DIR="${EXP_DIR:-hubert_base_per}"
CKPT="${CKPT:-${EXP_DIR}/dev-best.ckpt}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $S3PRL_DIR/s3prl/$CKPT" >&2
    echo "Set EXP_DIR/CKPT to point at an existing checkpoint, or run" >&2
    echo "'sbatch run/train_baseline.slurm' first to produce one." >&2
    exit 1
fi

python3 run_downstream.py -m evaluate -e "$CKPT"

# Predictions now live under run/s3prl_checkout/s3prl/${EXP_DIR}/.
# Feed them into this project's own gate check per "Step 6" of
# scripts/baseline_reproduction/README.md (score_predictions + compute_metrics).
