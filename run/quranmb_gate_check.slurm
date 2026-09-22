#!/bin/bash
#SBATCH --job-name=iqra_mhubert_quranmb_gate
#SBATCH --output=run/logs/output_logs/output_iqra_mhubert_quranmb_gate_%j.log
#SBATCH --error=run/logs/error_logs/error_iqra_mhubert_quranmb_gate_%j.log
#SBATCH -J iqra_mhubert_quranmb_gate
#SBATCH -p NvidiaAll
#SBATCH -t 02:00:00

# Week-3 gate check: runs a trained checkpoint over QuranMB.v2 audio and
# scores it with this project's own hierarchical MDD metric, printing F1
# against the gate (0.4414 +/- 0.02). Two stages, two different Python
# environments, chained in one job:
#
#   Stage A (S3PRL Python-3.8 venv, needs the GPU):
#     scripts/baseline_reproduction/quranmb_gate_inference.py runs the
#     checkpoint over QuranMB.v2 audio -> {id: predicted_string} JSON.
#   Stage B (this project's own uv/Python-3.12 env, CPU-only, needs
#     network access to pull `safikhan/quran_mbv2_formatted`'s ground
#     truth):
#     scripts/quranmb_gate_score.py scores that JSON with
#     src/arabic_mdd/metrics/hierarchical.py and prints the gate verdict.
#
# By default this scores the organizers' OWN trained checkpoint, pulled
# straight from HuggingFace -- no training run required, and no dependence
# on getting the S3PRL upstream identifier right. That is the decisive
# gate check: if it reproduces F1 = 0.4414 through this pipeline, the
# metric and data path are validated and any shortfall from our own
# training run is isolated to that run.
#
# To score a locally trained checkpoint instead (run/train_baseline.slurm),
# override EXP_DIR or CKPT the same way as run/evaluate_baseline.slurm:
#   EXP_DIR=hubert_base_per sbatch run/quranmb_gate_check.slurm
#   CKPT=hubert_base_per/dev-best.ckpt sbatch run/quranmb_gate_check.slurm
# OUTPUT_JSON then defaults to run/quranmb_predictions.json instead, so the
# two runs never overwrite each other's predictions.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# Batch jobs don't source ~/.zshrc or ~/.bashrc, so make sure `uv` is
# reachable here (needed for Stage B).
export PATH="$HOME/.local/bin:$PATH"

# This cluster's per-job TMPDIR ([/tmp/user/$UID]) isn't creatable
# (permission denied), so Slurm falls back to the node's local /tmp --
# which is fine, that's node-local disk with real room (unlike $HOME,
# which is NFS-mounted and quota-limited -- do NOT point TMPDIR/HF_HOME
# there, that was tried and just redirects into the same full quota).
# Force both explicitly at a job-scoped subdir of the node's local /tmp,
# so this doesn't depend on Slurm's fallback behavior and doesn't collide
# with other users'/jobs' files.
export TMPDIR="/tmp/${USER}_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export HF_HOME="$TMPDIR/hf_cache"
mkdir -p "$HF_HOME"

S3PRL_VENV="${S3PRL_VENV:-$REPO_ROOT/run/s3prl_venv}"
if [ ! -f "$S3PRL_VENV/bin/activate" ]; then
    echo "ERROR: no S3PRL venv found at $S3PRL_VENV." >&2
    echo "Run 'sbatch run/train_baseline.slurm' at least once first." >&2
    exit 1
fi

S3PRL_DIR="${S3PRL_DIR:-$REPO_ROOT/run/s3prl_checkout}"
BASELINE_SRC="$REPO_ROOT/scripts/baseline_reproduction"
DICT_PATH="${DICT_PATH:-downstream/ctc/cv_vocab/sws_arabic.txt}"

# The organizers' trained baseline checkpoint -- public and ungated.
# S3PRLModel downloads an http(s) CKPT itself via
# s3prl_inference.download_if_needed, so this needs no pre-fetch.
OFFICIAL_CKPT="https://huggingface.co/IqraEval/Iqra_mhubert_base/resolve/main/mhubert.ckpt"
if [ -n "${EXP_DIR:-}" ]; then
    CKPT="${CKPT:-${EXP_DIR}/dev-best.ckpt}"
else
    CKPT="${CKPT:-$OFFICIAL_CKPT}"
fi

# Default the output path off the checkpoint, so scoring the official
# checkpoint can never silently overwrite the predictions from our own
# training run (the file insights/week-03.md's F1 = 0.4058 entry rests on).
if [ -z "${OUTPUT_JSON:-}" ]; then
    case "$CKPT" in
        http://*|https://*) OUTPUT_JSON="run/quranmb_predictions_official_ckpt.json" ;;
        *)                  OUTPUT_JSON="run/quranmb_predictions.json" ;;
    esac
fi
# Stage A runs with cwd = the s3prl checkout, Stage B with cwd = the repo
# root, and quranmb_gate_inference.py writes to `Path(output_json)` -- i.e.
# relative to *its* cwd. Resolve against the repo root here so a relative
# OUTPUT_JSON doesn't get written inside the s3prl checkout and then go
# missing when Stage B looks for it.
case "$OUTPUT_JSON" in
    /*) ;;
    *) OUTPUT_JSON="$REPO_ROOT/$OUTPUT_JSON" ;;
esac
mkdir -p "$(dirname "$OUTPUT_JSON")"

echo "checkpoint:  $CKPT"
echo "dict:        $DICT_PATH"
echo "predictions: $OUTPUT_JSON"

echo "== Stage A: inference over QuranMB.v2 (S3PRL venv) =="
# shellcheck source=/dev/null
source "$S3PRL_VENV/bin/activate"
cd "$S3PRL_DIR/s3prl"

# CKPT may be an http(s) URL (the default), so only check local paths for
# existence. Note local paths are relative to the s3prl checkout, which is
# the cwd here.
case "$CKPT" in
    http://*|https://*)
        echo "Using remote checkpoint: $CKPT"
        ;;
    *)
        if [ ! -f "$CKPT" ]; then
            echo "ERROR: checkpoint not found: $S3PRL_DIR/s3prl/$CKPT" >&2
            echo "Set EXP_DIR/CKPT to point at an existing checkpoint, or run" >&2
            echo "'sbatch run/train_baseline.slurm' first to produce one." >&2
            exit 1
        fi
        ;;
esac

python3 "$BASELINE_SRC/quranmb_gate_inference.py" \
    --ckpt "$CKPT" \
    --dict_path "$DICT_PATH" \
    --output_json "$OUTPUT_JSON"

echo "== Stage B: scoring against the gate (this project's uv env) =="
# Leave the S3PRL venv before invoking uv -- `uv run` resolves its own
# project env, but an active foreign VIRTUAL_ENV can still confuse it.
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
cd "$REPO_ROOT"

uv run python scripts/quranmb_gate_score.py "$OUTPUT_JSON"
