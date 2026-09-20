#!/bin/bash
#SBATCH --job-name=iqra_mhubert_baseline
#SBATCH --output=run/logs/output_logs/output_iqra_mhubert_baseline_%j.log
#SBATCH --error=run/logs/error_logs/error_iqra_mhubert_baseline_%j.log
#SBATCH -J iqra_mhubert_baseline
#SBATCH -p NvidiaAll
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00

# Reproduces the mHuBERT + BiLSTM + CTC organizer baseline end to end:
# environment setup, data download/prep, training, evaluation. Submit with
# `sbatch run/train_baseline.slurm` from the repo root -- nothing else needs
# to be done first.
#
# Only two things you may need to change for your cluster:
#   -p / --gres above (partition and GPU request naming vary by cluster --
#   "NvidiaAll"/"gpu:1" are placeholders)
#   The conda base path assumed below (~/miniconda3 or ~/anaconda3).
#
# Everything else (conda env creation, S3PRL install, dataset download,
# data prep, config wiring) is handled by run/prepare_baseline.sh, which is
# idempotent -- if the job dies partway through and you resubmit, already
# finished steps (e.g. a completed download) are skipped rather than redone.
# This does assume the compute node has internet access (to clone S3PRL and
# pull the HF datasets); if your cluster's compute nodes are offline, run
# `bash run/prepare_baseline.sh` once from a login node first, then submit
# this job -- it will skip everything already done.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

bash run/prepare_baseline.sh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate s3prl

S3PRL_DIR="${S3PRL_DIR:-$(pwd)/run/s3prl_checkout}"
cd "$S3PRL_DIR/s3prl"

exp_dir=hubert_base_per
python3 run_downstream.py -m train -c downstream/ctc/cv_config/sws.yaml -p "${exp_dir}" -u hubert_base -d ctc
python3 run_downstream.py -m evaluate -e "${exp_dir}/dev-best.ckpt"

# Predictions/checkpoint now live under run/s3prl_checkout/s3prl/${exp_dir}/.
# Feed them into this project's own gate check per "Step 6" of
# scripts/baseline_reproduction/README.md (score_predictions + compute_metrics).
