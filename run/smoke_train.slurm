#!/bin/bash
#SBATCH --job-name=iqra_smoke_train
#SBATCH --output=run/logs/output_logs/output_iqra_smoke_train_%j.log
#SBATCH --error=run/logs/error_logs/error_iqra_smoke_train_%j.log
#SBATCH -J iqra_smoke_train
#SBATCH -p NvidiaAll
#SBATCH -t 01:00:00

# A 20-step throwaway version of run/train_baseline.slurm, to prove the
# upstream loads and a train->save->evaluate cycle completes before
# committing 48 GPU-hours to the real thing.
#
# It exercises exactly the parts that are new and therefore untested:
# `transformers` being present in the S3PRL venv, `-u hf_hubert_custom -k
# utter-project/mHuBERT-147` resolving and downloading, the 13 mHuBERT hidden
# states wiring into the featurizer, a checkpoint being written, and
# `-m evaluate` reading it back. Everything else (data prep, config, vocab)
# is already known good from the hubert_base run.
#
#   sbatch run/smoke_train.slurm
#   STEPS=50 sbatch run/smoke_train.slurm      # longer, if you want loss to move
#
# If this passes, submit the real job: sbatch run/train_baseline.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

export PATH="$HOME/.local/bin:$PATH"

# mHuBERT-147 is a ~2.4 GB HF download. Keep it off $HOME, which is NFS and
# quota-limited. Override HF_HOME to persistent scratch if you have it, to
# avoid re-downloading on every submit.
export TMPDIR="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR"
export HF_HOME="${HF_HOME:-$TMPDIR/hf_cache}"
mkdir -p "$HF_HOME"

# Idempotent, and the reason it runs here: it is what installs `transformers`
# into the S3PRL venv, which the hf_hubert_custom upstream needs.
bash run/prepare_baseline.sh

S3PRL_VENV="${S3PRL_VENV:-$REPO_ROOT/run/s3prl_venv}"
# shellcheck source=/dev/null
source "$S3PRL_VENV/bin/activate"

S3PRL_DIR="${S3PRL_DIR:-$REPO_ROOT/run/s3prl_checkout}"
cd "$S3PRL_DIR/s3prl"

STEPS="${STEPS:-20}"
# S3PRL refuses to reuse a populated expdir, so key it to the job id -- that
# way repeated smoke runs don't collide and none of them can touch the real
# mhubert147_per / hubert_base_per experiments.
exp_dir="smoke_mhubert_${SLURM_JOB_ID:-$$}"

echo "== smoke train: $STEPS steps into $exp_dir =="
# `,,` is S3PRL's override separator, not a typo.
python3 run_downstream.py -m train -c downstream/ctc/cv_config/sws.yaml -p "${exp_dir}" \
    -u hf_hubert_custom -k utter-project/mHuBERT-147 -d ctc \
    -o "config.runner.total_steps=${STEPS},,config.runner.eval_step=${STEPS},,config.runner.save_step=${STEPS},,config.runner.log_step=1"

echo "== smoke evaluate: reading the checkpoint back =="
python3 run_downstream.py -m evaluate -e "${exp_dir}/dev-best.ckpt"

echo "== smoke run OK -- safe to submit run/train_baseline.slurm =="
echo "Artifacts left in $S3PRL_DIR/s3prl/$exp_dir (delete when done)."
