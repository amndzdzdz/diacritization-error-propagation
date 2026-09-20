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
DATA_DIR="${BASELINE_DATA_DIR:-$RUN_DIR/data}"

echo "== [1/6] S3PRL conda environment =="
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -qE '^\s*s3prl\s'; then
    conda create -y -n s3prl python=3.8
fi
conda activate s3prl

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
