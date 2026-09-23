#!/bin/bash
#SBATCH --job-name=iqra_mhubert_baseline
#SBATCH --output=run/logs/output_logs/output_iqra_mhubert_baseline_%j.log
#SBATCH --error=run/logs/error_logs/error_iqra_mhubert_baseline_%j.log
#SBATCH -J iqra_mhubert_baseline
#SBATCH -p NvidiaAll
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

# The hf_hubert_custom upstream pulls mHuBERT-147 from the Hub (~2.4 GB) on
# top of the Iqra_train/Iqra_TTS dataset downloads. The default cache is
# under $HOME, which is NFS-mounted and quota-limited on this cluster, so
# point it at node-local scratch instead. Not trapped for cleanup, unlike
# run/quranmb_gate_check.slurm: this job is long enough and resubmitted often
# enough that re-downloading everything each time is the worse trade. Set
# HF_HOME yourself if you have persistent scratch that survives the node.
export TMPDIR="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR"
export HF_HOME="${HF_HOME:-$TMPDIR/hf_cache}"
mkdir -p "$HF_HOME"

bash run/prepare_baseline.sh

S3PRL_VENV="${S3PRL_VENV:-$(pwd)/run/s3prl_venv}"
# shellcheck source=/dev/null
source "$S3PRL_VENV/bin/activate"

# The sha-pinned upstream directory, materialized by prepare_baseline.sh
# step 7. Read from the file it writes, since prepare runs in a child shell.
UPSTREAM_DIR="${UPSTREAM_DIR:-$(cat "$(pwd)/run/.upstream_dir" 2>/dev/null || true)}"
if [ -z "$UPSTREAM_DIR" ] || [ ! -d "$UPSTREAM_DIR" ]; then
    echo "ERROR: no pinned upstream at '${UPSTREAM_DIR:-<unset>}'." >&2
    echo "Expected run/prepare_baseline.sh step 7 to create it." >&2
    echo "Do NOT work around this by passing -k utter-project/mHuBERT-147:" >&2
    echo "that silently un-pins the upstream (see pin_upstream.py)." >&2
    exit 1
fi
echo "pinned upstream: $UPSTREAM_DIR"

S3PRL_DIR="${S3PRL_DIR:-$(pwd)/run/s3prl_checkout}"
cd "$S3PRL_DIR/s3prl"

# Upstream read straight out of the organizers' own published checkpoint
# (scripts/baseline_reproduction/inspect_s3prl_ckpt.py): Args.upstream =
# 'hf_hubert_custom', Args.upstream_ckpt = 'utter-project/mHuBERT-147',
# upstream_trainable = False. This replaces the earlier `-u hubert_base`
# stand-in (English HuBERT Base, LibriSpeech), which cost 0.036 F1 and 1,282
# excess false rejections -- see insights/week-03.md.
#
# `-k` is a sha-pinned LOCAL DIRECTORY rather than the bare repo id. Same
# weights (nothing but README has changed in that repo since 2024-06-12),
# but now fixed. `--upstream_revision` would NOT achieve this -- it does not
# reach the hf_hubert_custom code path at all; see pin_upstream.py. The sha
# is in the directory name, so it lands in the trained checkpoint's own
# Args.upstream_ckpt and inspect_s3prl_ckpt.py can read it back.
exp_dir=mhubert147_per
python3 run_downstream.py -m train -c downstream/ctc/cv_config/sws.yaml -p "${exp_dir}" \
    -u hf_hubert_custom -k "$UPSTREAM_DIR" -d ctc
python3 run_downstream.py -m evaluate -e "${exp_dir}/dev-best.ckpt"

# Predictions/checkpoint now live under run/s3prl_checkout/s3prl/${exp_dir}/.
# Feed them into this project's own gate check per "Step 6" of
# scripts/baseline_reproduction/README.md (score_predictions + compute_metrics).
