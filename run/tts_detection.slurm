#!/bin/bash
#SBATCH --job-name=iqra_tts_detect
#SBATCH --output=run/logs/output_logs/output_iqra_tts_detect_%j.log
#SBATCH --error=run/logs/error_logs/error_iqra_tts_detect_%j.log
#SBATCH -J iqra_tts_detect
#SBATCH -p NvidiaAll
#SBATCH -t 06:00:00

# The MSA arm's detection metric on REAL positives (docs/msa-arm.md §3.3).
#
# Everything measured on the MSA arm so far has had to fake `A` by setting
# A := C, which zeroes TR and FA and leaves precision/recall/F1 undefined
# as quantities of interest. `Iqra_TTS`'s AUGMENTED rows fix that: they
# carry `phoneme_ref` (C) and `phoneme_mis` (A, the sequence actually
# synthesized), which differ in ~99% of rows. That completes the (C, A, P)
# triple and makes the hierarchical metric computable for real.
#
# This is the only MSA-domain data in the project with genuine detection
# positives. It yields an MSA-side F1 directly comparable to the Qur'anic
# arm's 0.4406, and a fallback headline if the real-speech base rate turns
# out too low for option 1.
#
#   Stage A (S3PRL Python-3.8 venv, GPU): same inference script as the
#     control, with --label augmented
#       -> run/tts_augmented_predictions.json
#   Stage B (uv env, CPU): scripts/tts_detection_metric.py
#
# CAVEAT to carry into any write-up: injected mispronunciations, synthetic
# audio, and Iqra_TTS is inside the 131 h training set. Upper bound and
# sanity check, not the paper's MSA result -- and it says nothing about the
# real-speech base rate, which only step 4 settles.
#
# Overrides:
#   EXP_DIR=some_other_run sbatch run/tts_detection.slurm
#   CKPT=https://... sbatch run/tts_detection.slurm
#   LIMIT=200 sbatch run/tts_detection.slurm    # smoke test
#   LIMIT=0   sbatch run/tts_detection.slurm    # all augmented rows (slow)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# Batch jobs don't source ~/.zshrc or ~/.bashrc, so make sure `uv` is
# reachable here (needed for Stage B).
export PATH="$HOME/.local/bin:$PATH"

# $HOME is NFS-mounted and quota-limited; TMPDIR and HF_HOME go to
# node-local /tmp. Iqra_TTS is ~6 GB, so this matters here.
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
SPLIT="${SPLIT:-train}"
LABEL="${LABEL:-augmented}"

EXP_DIR="${EXP_DIR:-mhubert147_per}"
CKPT="${CKPT:-${EXP_DIR}/dev-best.ckpt}"

# Stage A runs with cwd = the s3prl checkout and Stage B with cwd = the
# repo root, so resolve a relative output path against the repo root.
OUTPUT_JSON="${OUTPUT_JSON:-run/tts_augmented_predictions.json}"
case "$OUTPUT_JSON" in
    /*) ;;
    *) OUTPUT_JSON="$REPO_ROOT/$OUTPUT_JSON" ;;
esac
mkdir -p "$(dirname "$OUTPUT_JSON")"

echo "checkpoint:  $CKPT"
echo "dict:        $DICT_PATH"
echo "split:       $SPLIT   label: $LABEL"
echo "records:     $OUTPUT_JSON"

echo "== Stage A: inference over Iqra_TTS [label=${LABEL}] (S3PRL venv) =="
# shellcheck source=/dev/null
source "$S3PRL_VENV/bin/activate"
cd "$S3PRL_DIR/s3prl"

case "$CKPT" in
    http://*|https://*)
        echo "Using remote checkpoint: $CKPT"
        ;;
    *)
        if [ ! -f "$CKPT" ]; then
            echo "ERROR: checkpoint not found: $S3PRL_DIR/s3prl/$CKPT" >&2
            echo "Set EXP_DIR/CKPT to an existing checkpoint, or train one first." >&2
            exit 1
        fi
        ;;
esac

python3 "$BASELINE_SRC/tts_control_inference.py" \
    --ckpt "$CKPT" \
    --dict_path "$DICT_PATH" \
    --split "$SPLIT" \
    --label "$LABEL" \
    --output_json "$OUTPUT_JSON" \
    ${LIMIT:+--limit "$LIMIT"}

echo "== Stage B: hierarchical metric on the real (C, A, P) triple (uv env) =="
# Leave the S3PRL venv before invoking uv -- an active foreign VIRTUAL_ENV
# can confuse `uv run`'s own env resolution.
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
cd "$REPO_ROOT"

uv run python scripts/tts_detection_metric.py "$OUTPUT_JSON"

echo "== Done. Upper bound + sanity check; step 4 still decides the base rate. =="
