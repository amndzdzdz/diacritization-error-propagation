#!/bin/bash
# Idempotent setup for the S3PRL mHuBERT+BiLSTM+CTC baseline reproduction.
# Safe to re-run: every step checks whether its output already exists and
# skips if so, so a resubmitted/retried SLURM job doesn't redo finished work
# (e.g. a multi-hour dataset download).
#
# Called from run/train_baseline.slurm -- not meant to be the thing you
# submit directly, though you can run it standalone to pre-stage everything
# from a login node before submitting the training job.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$REPO_ROOT/run"
BASELINE_SRC="$REPO_ROOT/scripts/baseline_reproduction"
S3PRL_DIR="${S3PRL_DIR:-$RUN_DIR/s3prl_checkout}"
S3PRL_VENV="${S3PRL_VENV:-$RUN_DIR/s3prl_venv}"
DATA_DIR="${BASELINE_DATA_DIR:-$RUN_DIR/data}"

echo "== [1/6] Python 3.8 environment for S3PRL (stdlib venv, no conda/uv) =="
# S3PRL needs Python 3.8, incompatible with this project's own Python-3.12
# env -- kept as a fully separate venv here, never touching this project's
# pyproject.toml/uv.lock. No conda and no uv install permissions needed:
# this uses whatever python3.8 interpreter is already on the cluster, via
# the stdlib `venv` module.
#
# Interpreter resolution order (see also the comment block at the top of
# run/train_baseline.slurm):
#   1. $PYTHON38_BIN, if you set it (path to a python3.8 executable)
#   2. `python3.8` already on PATH
#   3. `python3` on PATH, if it reports version 3.8.x
#   4. a handful of common install locations
PYTHON38_BIN="${PYTHON38_BIN:-}"
if [ -z "$PYTHON38_BIN" ] && command -v python3.8 >/dev/null 2>&1; then
    PYTHON38_BIN="$(command -v python3.8)"
fi
if [ -z "$PYTHON38_BIN" ] && command -v python3 >/dev/null 2>&1 \
    && python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 8) else 1)' 2>/dev/null; then
    PYTHON38_BIN="$(command -v python3)"
fi
if [ -z "$PYTHON38_BIN" ]; then
    for candidate in /usr/bin/python3.8 /usr/local/bin/python3.8 /opt/python3.8/bin/python3.8; do
        if [ -x "$candidate" ]; then
            PYTHON38_BIN="$candidate"
            break
        fi
    done
fi
if [ -z "$PYTHON38_BIN" ]; then
    echo "ERROR: could not find a Python 3.8 interpreter." >&2
    echo "S3PRL requires Python 3.8. If your cluster provides it via environment" >&2
    echo "modules, load it yourself and pass its path, e.g.:" >&2
    echo "  module load python/3.8 && PYTHON38_BIN=\$(which python3.8) sbatch run/train_baseline.slurm" >&2
    exit 1
fi
echo "Using Python 3.8 interpreter: $PYTHON38_BIN ($("$PYTHON38_BIN" --version))"

if [ ! -f "$S3PRL_VENV/bin/activate" ]; then
    "$PYTHON38_BIN" -m venv "$S3PRL_VENV"
fi
# shellcheck source=/dev/null
source "$S3PRL_VENV/bin/activate"
python -m pip install --quiet --upgrade pip

echo "== [2/6] S3PRL toolkit install =="
if [ ! -d "$S3PRL_DIR" ]; then
    git clone https://github.com/s3prl/s3prl.git "$S3PRL_DIR"
fi
python -c "import s3prl" 2>/dev/null || pip install -e "${S3PRL_DIR}[all]"
python -c "import datasets" 2>/dev/null || pip install datasets
python -c "import joblib, tqdm, pandas, numpy" 2>/dev/null || pip install joblib tqdm pandas numpy

echo "== [3/6] Download Iqra_train + Iqra_TTS =="
mkdir -p "$DATA_DIR"
if [ ! -d "$DATA_DIR/CV-Ar/train/wav" ] || [ -z "$(ls -A "$DATA_DIR/CV-Ar/train/wav" 2>/dev/null)" ]; then
    python "$BASELINE_SRC/download_hugg_data.py" --path "IqraEval/Iqra_train" --split "train" --output_dir "$DATA_DIR/CV-Ar"
fi
if [ ! -d "$DATA_DIR/CV-Ar/dev/wav" ] || [ -z "$(ls -A "$DATA_DIR/CV-Ar/dev/wav" 2>/dev/null)" ]; then
    python "$BASELINE_SRC/download_hugg_data.py" --path "IqraEval/Iqra_train" --split "dev" --output_dir "$DATA_DIR/CV-Ar"
fi
if [ ! -d "$DATA_DIR/TTS/train/wav" ] || [ -z "$(ls -A "$DATA_DIR/TTS/train/wav" 2>/dev/null)" ]; then
    python "$BASELINE_SRC/download_hugg_data_tts.py" --path "IqraEval/Iqra_TTS" --split "train" --output_dir "$DATA_DIR/TTS" --dev_name "Amer"
fi

echo "== [4/6] Merge TTS into the combined 131h train split =="
# The 131h training set (per docs/weeks/week-02.md) is Iqra_train (79h) +
# Iqra_TTS (52h) combined. The bucketing step below treats whatever is under
# CV-Ar/train as one split, so the TTS train audio/transcripts are merged in
# here (hard-linked/copied, not moved, so re-running is idempotent and the
# original TTS download stays intact).
if [ -d "$DATA_DIR/TTS/train/wav" ]; then
    find "$DATA_DIR/TTS/train/wav" -name '*.wav' -exec cp -n {} "$DATA_DIR/CV-Ar/train/wav/" \;
    find "$DATA_DIR/TTS/train/transcripts" -name '*.txt' -exec cp -n {} "$DATA_DIR/CV-Ar/train/transcripts/" \;
fi

echo "== [5/6] Bucketing metadata + TSV conversion =="
mkdir -p "$DATA_DIR/len_for_bucket"
if [ ! -f "$DATA_DIR/len_for_bucket/train.csv" ] || [ ! -f "$DATA_DIR/len_for_bucket/dev.csv" ]; then
    python "$BASELINE_SRC/generate_len_for_bucket_sdaia.py" \
        -i "$DATA_DIR/CV-Ar" -o "$DATA_DIR" --splits train dev
fi
if [ ! -f "$DATA_DIR/CV-Ar/train/train.tsv" ]; then
    python "$BASELINE_SRC/csv_to_tsv_with_transcripts.py" \
        --csv_path "$DATA_DIR/len_for_bucket/train.csv" \
        --transcript_root "$DATA_DIR/CV-Ar/" \
        --output_path "$DATA_DIR/CV-Ar/train/train.tsv"
fi
if [ ! -f "$DATA_DIR/CV-Ar/dev/dev.tsv" ]; then
    python "$BASELINE_SRC/csv_to_tsv_with_transcripts.py" \
        --csv_path "$DATA_DIR/len_for_bucket/dev.csv" \
        --transcript_root "$DATA_DIR/CV-Ar/" \
        --output_path "$DATA_DIR/CV-Ar/dev/dev.tsv"
fi

echo "== [6/6] Downstream task config + vocab =="
mkdir -p "$S3PRL_DIR/s3prl/downstream/ctc/cv_vocab" "$S3PRL_DIR/s3prl/downstream/ctc/cv_config"
cp "$BASELINE_SRC/vocab/sws_arabic.txt" "$S3PRL_DIR/s3prl/downstream/ctc/cv_vocab/sws_arabic.txt"
# Materialize the vendored config with real paths substituted -- the
# vendored copy keeps the organizers' placeholder paths so it stays a
# verbatim reference; this generated copy is what actually gets used.
sed \
    -e "s#'./sws_data/'#'$DATA_DIR/CV-Ar/'#" \
    -e "s#path_to_tsv_file/train/train.tsv#$DATA_DIR/CV-Ar/train/train.tsv#" \
    -e "s#path_to_tsv_file/dev/dev.tsv#$DATA_DIR/CV-Ar/dev/dev.tsv#" \
    "$BASELINE_SRC/config/sws.yaml" > "$S3PRL_DIR/s3prl/downstream/ctc/cv_config/sws.yaml"

echo "Baseline reproduction data/environment ready. S3PRL_DIR=$S3PRL_DIR"
