#!/bin/bash
#SBATCH --job-name=iqra_smoke_train
#SBATCH --output=run/logs/output_logs/output_iqra_smoke_train_%j.log
#SBATCH --error=run/logs/error_logs/error_iqra_smoke_train_%j.log
#SBATCH -J iqra_smoke_train
#SBATCH -p NvidiaAll
#SBATCH -t 01:00:00

# A short throwaway training run, to prove an upstream loads and a
# train->save->evaluate cycle completes before committing 48 GPU-hours to the
# real thing. Originally written for mHuBERT (week 3); parameterized over the
# upstream in week 7, because the XLS-R arm needs exactly the same check and
# the cost of a bad guess is a whole queue slot.
#
# It exercises the parts that are new and therefore untested: `transformers`
# resolving in the S3PRL venv, `-u <name> -k <pinned dir>` loading, the
# upstream's hidden states wiring into the featurizer (13 for mHuBERT, 25 for
# XLS-R -- s3prl sizes the weighted sum from a dummy forward, so this is the
# check that it really does), a checkpoint being written, and `-m evaluate`
# reading it back.
#
#   sbatch run/smoke_train.slurm                         # mHuBERT, 20 steps
#   STEPS=50 sbatch run/smoke_train.slurm                # longer, if you want
#                                                        # the loss to move
#   UPSTREAM_PIN=xlsr300m STEPS=200 sbatch run/smoke_train.slurm
#                                                        # week-07 task 3: the
#                                                        # step-rate measurement
#
# Week-07 task 3 wants 200 steps on XLS-R and three numbers out of it:
# **steps/s**, **peak GPU memory**, and the projected wall clock for the full
# step count. steps/s is in the log (`log_step=1`); read peak memory with
# `sacct -j <jobid> --format=JobID,MaxRSS,Elapsed` plus whatever GPU accounting
# this cluster exposes, or `nvidia-smi --query-gpu=memory.used` sampled from a
# second shell on the node.
#
# If this passes, submit the real job.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

export PATH="$HOME/.local/bin:$PATH"

# The upstream weights are a 1.3-2.4 GB HF download. Keep them off $HOME, which
# is NFS and quota-limited. Override HF_HOME to persistent scratch if you have
# it, to avoid re-downloading on every submit.
export TMPDIR="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR"
export HF_HOME="${HF_HOME:-$TMPDIR/hf_cache}"
mkdir -p "$HF_HOME"

# Which registered upstream to smoke. The name indexes pin_upstream.py's
# UPSTREAMS, which owns the `-u` name / `-k` directory pairing -- so this
# script never names `hf_hubert_custom` or `hf_wav2vec2_custom` itself, and
# cannot pair them wrongly.
export UPSTREAM_PIN="${UPSTREAM_PIN:-mhubert147}"

# Inherited, with a default, because a smoke run measures whether the wiring
# works and how fast a step is -- neither of which depends on which rows are in
# the manifest. The default is 0 so that a bare `sbatch run/smoke_train.slurm`
# smokes the reproduction path. Pass APPLY_HOLDOUT=1 when you are smoking a run
# that will itself be held out, so the filter's own failure modes (a missing
# training_holdout.json, an empty output manifest) surface in a 20-step job
# rather than at the start of a 48-hour one.
export APPLY_HOLDOUT="${APPLY_HOLDOUT:-0}"

# Idempotent, and the reason it runs here: it is what installs `transformers`
# into the S3PRL venv, which both hf_* upstreams need at module load.
bash run/prepare_baseline.sh

S3PRL_VENV="${S3PRL_VENV:-$REPO_ROOT/run/s3prl_venv}"
# shellcheck source=/dev/null
source "$S3PRL_VENV/bin/activate"

# Same sha-pinned upstream the real training run uses -- the smoke test is
# worthless if it validates a different set of weights. Read from the per-pin
# env file prepare step 8 writes, so an XLS-R prepare followed by an mHuBERT
# smoke cannot pick up the wrong directory.
UPSTREAM_ENV="$REPO_ROOT/run/.upstream_$UPSTREAM_PIN.env"
if [ ! -f "$UPSTREAM_ENV" ]; then
    echo "ERROR: $UPSTREAM_ENV not found." >&2
    echo "Expected run/prepare_baseline.sh step 8 to write it. Run:" >&2
    echo "  APPLY_HOLDOUT=$APPLY_HOLDOUT UPSTREAM_PIN=$UPSTREAM_PIN bash run/prepare_baseline.sh" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$UPSTREAM_ENV"
if [ -z "${UPSTREAM_DIR:-}" ] || [ ! -d "$UPSTREAM_DIR" ]; then
    echo "ERROR: no pinned upstream at '${UPSTREAM_DIR:-<unset>}'." >&2
    echo "Do NOT work around this by passing a bare HF repo id to -k: that" >&2
    echo "silently un-pins the upstream (see pin_upstream.py)." >&2
    exit 1
fi
echo "pinned upstream: $S3PRL_UPSTREAM -k $UPSTREAM_DIR"

S3PRL_DIR="${S3PRL_DIR:-$REPO_ROOT/run/s3prl_checkout}"
cd "$S3PRL_DIR/s3prl"

STEPS="${STEPS:-20}"
# S3PRL refuses to reuse a populated expdir, so key it to the upstream and the
# job id -- that way repeated smoke runs don't collide and none of them can
# touch the real mhubert147_per / hubert_base_per experiments.
exp_dir="smoke_${UPSTREAM_PIN}_${SLURM_JOB_ID:-$$}"

echo "== smoke train: $STEPS steps into $exp_dir =="
# `,,` is S3PRL's override separator, not a typo.
python3 run_downstream.py -m train -c downstream/ctc/cv_config/sws.yaml -p "${exp_dir}" \
    -u "$S3PRL_UPSTREAM" -k "$UPSTREAM_DIR" -d ctc \
    -o "config.runner.total_steps=${STEPS},,config.runner.eval_step=${STEPS},,config.runner.save_step=${STEPS},,config.runner.log_step=1"

echo "== smoke evaluate: reading the checkpoint back =="
python3 run_downstream.py -m evaluate -e "${exp_dir}/dev-best.ckpt"

echo "== smoke run OK ($UPSTREAM_PIN) =="
echo "Artifacts left in $S3PRL_DIR/s3prl/$exp_dir (delete when done)."
