#!/bin/bash
#SBATCH --job-name=iqra_tts_control
#SBATCH --output=run/logs/output_logs/output_iqra_tts_control_%j.log
#SBATCH --error=run/logs/error_logs/error_iqra_tts_control_%j.log
#SBATCH -J iqra_tts_control
#SBATCH -p NvidiaAll
#SBATCH -t 06:00:00

# Zero-base-rate control for the MSA arm (docs/msa-arm.md §3.3).
#
# run/cv_dev_baserate.slurm returned a CV-Ar dev disagreement rate of
# 0.0708 against QuranMB's 0.1241. That reads as "no headroom for a
# mispronunciation base rate", but the comparison is confounded:
# dev-best.ckpt was model-selected on that very split (§5.2), so a lower
# rate there is expected regardless of whether speakers deviate.
#
# This job removes the confound by measuring the model's error rate where
# the base rate is ZERO BY CONSTRUCTION: `IqraEval/Iqra_TTS`'s `original`
# rows are synthesized straight from `phoneme_ref`, so A == C exactly.
#
#   Stage A (S3PRL Python-3.8 venv, needs the GPU):
#     scripts/baseline_reproduction/tts_control_inference.py
#       -> run/tts_control_predictions.json  {row_index: {prediction, phoneme_ref, speaker}}
#   Stage B (this project's uv/Python-3.12 env, CPU-only):
#     scripts/tts_control_baserate.py scores it and prints the verdict
#
# Stage A stores the reference alongside the prediction so that Stage B
# never re-fetches this ~6 GB dataset.
#
# Read the outcome ONE-SIDEDLY. This control is in-domain (Iqra_TTS is part
# of the 131 h training set) and synthetic (acoustically easy), and both
# push its error rate down. So only "control FR >= 0.0708" is decisive;
# a lower number is an upper bound on the base rate, not a measurement.
# Step 4's listening pilot stays on the critical path either way.
#
# Premise verified before writing this, over 800 sampled rows and after
# stripping `<sil>`: label=original has phoneme_mis == phoneme_ref in
# 500/500, label=augmented in only 3/300. The clean half really is clean.
#
# ~15.6k of the 46,851 rows are `original`, which would overrun any sane
# wall clock, so Stage A scores a SEEDED RANDOM subsample (default 3,000 --
# standard error ~0.002 on the FR rate, far finer than the ~0.04 gap being
# tested). Random, not head-N: the dataset is ordered by speaker, so the
# first N rows would measure one or two voices instead of the model.
#
# Same overrides as the other jobs here:
#   EXP_DIR=some_other_run sbatch run/tts_control.slurm
#   CKPT=https://... sbatch run/tts_control.slurm
#   LIMIT=200 sbatch run/tts_control.slurm     # smoke test
#   LIMIT=0   sbatch run/tts_control.slurm     # all ~15.6k rows (slow)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# Batch jobs don't source ~/.zshrc or ~/.bashrc, so make sure `uv` is
# reachable here (needed for Stage B).
export PATH="$HOME/.local/bin:$PATH"

# $HOME is NFS-mounted and quota-limited; TMPDIR and HF_HOME go to
# node-local /tmp. See run/quranmb_gate_check.slurm for the full rationale.
# Iqra_TTS is ~6 GB, so this matters more here than in the other jobs.
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
LABEL="${LABEL:-original}"

EXP_DIR="${EXP_DIR:-mhubert147_per}"
CKPT="${CKPT:-${EXP_DIR}/dev-best.ckpt}"

# Stage A runs with cwd = the s3prl checkout and Stage B with cwd = the
# repo root, so resolve a relative output path against the repo root --
# otherwise Stage A writes it inside the checkout and Stage B can't find it.
OUTPUT_JSON="${OUTPUT_JSON:-run/tts_control_predictions.json}"
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

echo "== Stage B: score the control (this project's uv env) =="
# Leave the S3PRL venv before invoking uv -- an active foreign VIRTUAL_ENV
# can confuse `uv run`'s own env resolution.
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
cd "$REPO_ROOT"

uv run python scripts/tts_control_baserate.py "$OUTPUT_JSON"

echo "== Done. Read the verdict one-sidedly; step 4 still decides. =="
