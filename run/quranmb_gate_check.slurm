#!/bin/bash
#SBATCH --job-name=iqra_mhubert_quranmb_gate
#SBATCH --output=run/logs/output_logs/output_iqra_mhubert_quranmb_gate_%j.log
#SBATCH --error=run/logs/error_logs/error_iqra_mhubert_quranmb_gate_%j.log
#SBATCH -J iqra_mhubert_quranmb_gate
#SBATCH -p NvidiaAll
#SBATCH --gres=gpu:1
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
# Assumes a trained checkpoint already exists (run/train_baseline.slurm).
# Override EXP_DIR/CKPT the same way as run/evaluate_baseline.slurm, e.g.:
#   EXP_DIR=hubert_base_per_rerun sbatch run/quranmb_gate_check.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

# Batch jobs don't source ~/.zshrc or ~/.bashrc, so make sure `uv` is
# reachable here (needed for Stage B).
export PATH="$HOME/.local/bin:$PATH"

S3PRL_VENV="${S3PRL_VENV:-$REPO_ROOT/run/s3prl_venv}"
if [ ! -f "$S3PRL_VENV/bin/activate" ]; then
    echo "ERROR: no S3PRL venv found at $S3PRL_VENV." >&2
    echo "Run 'sbatch run/train_baseline.slurm' at least once first." >&2
    exit 1
fi

S3PRL_DIR="${S3PRL_DIR:-$REPO_ROOT/run/s3prl_checkout}"
BASELINE_SRC="$REPO_ROOT/scripts/baseline_reproduction"
EXP_DIR="${EXP_DIR:-hubert_base_per}"
CKPT="${CKPT:-${EXP_DIR}/dev-best.ckpt}"
DICT_PATH="${DICT_PATH:-downstream/ctc/cv_vocab/sws_arabic.txt}"
OUTPUT_JSON="${OUTPUT_JSON:-$REPO_ROOT/run/quranmb_predictions.json}"

echo "== Stage A: inference over QuranMB.v2 (S3PRL venv) =="
# shellcheck source=/dev/null
source "$S3PRL_VENV/bin/activate"
cd "$S3PRL_DIR/s3prl"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $S3PRL_DIR/s3prl/$CKPT" >&2
    echo "Set EXP_DIR/CKPT to point at an existing checkpoint, or run" >&2
    echo "'sbatch run/train_baseline.slurm' first to produce one." >&2
    exit 1
fi

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
