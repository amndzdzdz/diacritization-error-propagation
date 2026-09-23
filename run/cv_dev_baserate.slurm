#!/bin/bash
#SBATCH --job-name=iqra_cv_dev_baserate
#SBATCH --output=run/logs/output_logs/output_iqra_cv_dev_baserate_%j.log
#SBATCH --error=run/logs/error_logs/error_iqra_cv_dev_baserate_%j.log
#SBATCH -J iqra_cv_dev_baserate
#SBATCH -p NvidiaAll
#SBATCH -t 03:00:00

# MSA-arm base-rate probe (docs/msa-arm.md §3.3) -- step 3 of the MSA-arm
# work order. Runs a trained checkpoint over `IqraEval/Iqra_train`'s dev
# audio (2,588 Common Voice Arabic utterances) and compares its output
# against `phoneme_ref`, to find out whether read MSA speech contains
# enough real mispronunciations for the MSA arm's detection metric to mean
# anything.
#
# Two stages, two Python environments, same structure as
# run/quranmb_gate_check.slurm:
#
#   Stage A (S3PRL Python-3.8 venv, needs the GPU):
#     scripts/baseline_reproduction/cv_dev_inference.py
#       -> run/cv_dev_predictions.json  {id: predicted_phoneme_string}
#   Stage B (this project's uv/Python-3.12 env, CPU-only, needs network):
#     scripts/cv_dev_baserate.py scores it with A := C, contrasts the
#     resulting FR rate against the QuranMB figure, and writes the
#     stratified 50-utterance manifest for the step-4 listening pilot
#       -> run/msa_pilot_sample.json  (+ optional wavs)
#
# Defaults to OUR OWN retrained checkpoint (mhubert147_per), not the
# organizers' -- this probe is about the data, and using the checkpoint we
# control keeps the comparison against our own QuranMB numbers exact.
# Override the usual way:
#   EXP_DIR=some_other_run sbatch run/cv_dev_baserate.slurm
#   CKPT=https://... sbatch run/cv_dev_baserate.slurm
#
# Stage A also dumps every utterance's wav to run/cv_dev_audio (~77 MB for
# dev), and Stage B copies the 50 sampled ones into run/pilot_audio, so the
# listening pilot can start straight off this job's output with no second
# GPU run. Override with AUDIO_DIR / PILOT_AUDIO.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# Batch jobs don't source ~/.zshrc or ~/.bashrc, so make sure `uv` is
# reachable here (needed for Stage B).
export PATH="$HOME/.local/bin:$PATH"

# Same cache redirection as the other jobs in this directory: $HOME is
# NFS-mounted and quota-limited, so TMPDIR and HF_HOME go to node-local
# /tmp. See run/quranmb_gate_check.slurm for the full explanation.
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
SPLIT="${SPLIT:-dev}"

EXP_DIR="${EXP_DIR:-mhubert147_per}"
CKPT="${CKPT:-${EXP_DIR}/dev-best.ckpt}"

# Stage A runs with cwd = the s3prl checkout and Stage B with cwd = the
# repo root, so resolve a relative output path against the repo root --
# otherwise Stage A writes it inside the checkout and Stage B can't find
# it. (Same bug class as run/quranmb_gate_check.slurm.)
OUTPUT_JSON="${OUTPUT_JSON:-run/cv_dev_predictions.json}"
case "$OUTPUT_JSON" in
    /*) ;;
    *) OUTPUT_JSON="$REPO_ROOT/$OUTPUT_JSON" ;;
esac
mkdir -p "$(dirname "$OUTPUT_JSON")"

# Stage A decodes every utterance anyway, so it keeps a wav copy here --
# Stage B's uv env cannot decode HF Audio features without `torchcodec`,
# which isn't installed. The step-4 listening pilot needs these.
AUDIO_DIR="${AUDIO_DIR:-run/cv_dev_audio}"
case "$AUDIO_DIR" in
    /*) ;;
    *) AUDIO_DIR="$REPO_ROOT/$AUDIO_DIR" ;;
esac
mkdir -p "$AUDIO_DIR"

echo "checkpoint:  $CKPT"
echo "dict:        $DICT_PATH"
echo "split:       $SPLIT"
echo "predictions: $OUTPUT_JSON"
echo "audio dump:  $AUDIO_DIR"

echo "== Stage A: inference over Iqra_train/${SPLIT} (S3PRL venv) =="
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

python3 "$BASELINE_SRC/cv_dev_inference.py" \
    --ckpt "$CKPT" \
    --dict_path "$DICT_PATH" \
    --split "$SPLIT" \
    --output_json "$OUTPUT_JSON" \
    --save_audio_dir "$AUDIO_DIR" \
    ${LIMIT:+--limit "$LIMIT"}

echo "== Stage B: base-rate probe + pilot sample (this project's uv env) =="
# Leave the S3PRL venv before invoking uv -- an active foreign VIRTUAL_ENV
# can confuse `uv run`'s own env resolution.
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
cd "$REPO_ROOT"

uv run python scripts/cv_dev_baserate.py "$OUTPUT_JSON" \
    --split "$SPLIT" \
    --audio-source "$AUDIO_DIR" \
    --dump-audio "${PILOT_AUDIO:-run/pilot_audio}"

echo "== Done. Next: docs/msa-arm.md §3.3 step 4 -- listen to run/msa_pilot_sample.json =="
