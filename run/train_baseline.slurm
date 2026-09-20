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
# Things you may need to change for your cluster:
#   -p / --gres above (partition and GPU request naming vary by cluster --
#   "NvidiaAll"/"gpu:1" are placeholders)
#   Python 3.8: S3PRL runs in a plain stdlib `venv`, never this project's own
#   uv/Python-3.12 env. run/prepare_baseline.sh looks for `python3.8` on
#   PATH, then a `python3` that reports 3.8.x, then a few common install
#   paths, then falls back to `uv python install 3.8` (a standalone
#   interpreter uv manages itself -- doesn't touch this project's
#   pyproject.toml/uv.lock) if `uv` is available. If your cluster only
#   exposes Python 3.8 via an environment module, load it yourself before
#   submitting and pass its path, e.g.:
#     module load python/3.8 && PYTHON38_BIN=$(which python3.8) sbatch run/train_baseline.slurm
#
# Everything else (venv creation, S3PRL install, dataset download, data
# prep, config wiring) is handled by run/prepare_baseline.sh, which is
# idempotent -- if the job dies partway through and you resubmit, already
# finished steps (e.g. a completed download) are skipped rather than redone.
# This does assume the compute node has internet access (to clone S3PRL and
# pull the HF datasets); if your cluster's compute nodes are offline, run
# `bash run/prepare_baseline.sh` once from a login node first, then submit
# this job -- it will skip everything already done.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Batch jobs don't source ~/.zshrc or ~/.bashrc, so make sure `uv` (installed
# to ~/.local/bin via the standalone installer, not through a package
# manager) is reachable here too -- needed both as this project's own
# dependency manager and as the Python 3.8 fallback in prepare_baseline.sh.
export PATH="$HOME/.local/bin:$PATH"

bash run/prepare_baseline.sh

S3PRL_VENV="${S3PRL_VENV:-$(pwd)/run/s3prl_venv}"
# shellcheck source=/dev/null
source "$S3PRL_VENV/bin/activate"

S3PRL_DIR="${S3PRL_DIR:-$(pwd)/run/s3prl_checkout}"
cd "$S3PRL_DIR/s3prl"

exp_dir=hubert_base_per
python3 run_downstream.py -m train -c downstream/ctc/cv_config/sws.yaml -p "${exp_dir}" -u hubert_base -d ctc
python3 run_downstream.py -m evaluate -e "${exp_dir}/dev-best.ckpt"

# Predictions/checkpoint now live under run/s3prl_checkout/s3prl/${exp_dir}/.
# Feed them into this project's own gate check per "Step 6" of
# scripts/baseline_reproduction/README.md (score_predictions + compute_metrics).
