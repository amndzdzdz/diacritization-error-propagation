#!/bin/bash
#SBATCH --job-name=der_measurement
#SBATCH --output=run/logs/output_logs/output_der_measurement_%j.log
#SBATCH --error=run/logs/error_logs/error_der_measurement_%j.log
#SBATCH -J der_measurement
#SBATCH -p NvidiaAll
#SBATCH -t 04:00:00

# Week-05 exit criterion 2, both clauses, in one job.
#
#   Stage A  scripts/calibrate_der.py        -> run/wikinews_der.json
#            Is our DER the literature's DER? Leg A scores CATT's own
#            benchmark files (reproduces all 16 Table-5 figures exactly,
#            already verified on CPU); Leg B runs our four tools on
#            WikiNewsMax against the pre-registered +-0.5 band.
#   Stage B  scripts/measure_quranic_der.py  -> run/quranic_der.json
#            The first DER this project has measured against genuinely gold
#            text: all 6,236 Tanzil verses plus the 98 QuranMB prompts.
#
# WHY THIS IS A CLUSTER JOB AND NOT A LOCAL ONE
#
# `catt-ed` decodes autoregressively, one position at a time. Measured on
# CPU it runs at ~0.32 s/verse batched, which is ~35 min for the Qur'anic
# corpus and ~45 min for WikiNews -- and a local attempt to run both at once
# was OOM-killed at 14 GB.
#
# The GPU is picked up without any code change: catt_tashkeel builds its
# sessions as `ort.InferenceSession(path)` with no `providers=` argument, so
# onnxruntime uses its default priority order and takes
# CUDAExecutionProvider when a device is visible. That is also why the
# assertion below exists -- a silent CPU fallback would still produce
# correct numbers, just hours later, and we want to know which happened.
#
# Both stages are CPU-cheap otherwise and write only JSON. Neither touches
# the GPU through torch.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=== host $(hostname), job ${SLURM_JOB_ID:-local} ==="
nvidia-smi || echo "no nvidia-smi -- stage A/B will fall back to CPU and be slow"

uv sync --all-groups

echo
echo "=== execution providers actually available ==="
# Not a gate: a CPU fallback is correct, only slow. Logged so that a
# four-hour run and a twenty-minute run can be told apart after the fact.
uv run python -c "
import onnxruntime as ort
print('onnxruntime', ort.__version__)
print('available  ', ort.get_available_providers())
import onnxruntime.capi._pybind_state as C
print('device     ', C.get_available_providers()[0])
"

# The two corpora are gitignored (CC-BY-SA-4.0 share-alike for WikiNews, a
# copyright block for Tanzil), so fail early and loudly rather than halfway
# through stage B.
for f in run/WikiNewsMax.tsv.local run/quran_simple.xml.local; do
  [ -s "$f" ] || { echo "MISSING $f -- see the scripts' docstrings for how to fetch it"; exit 1; }
done

echo
echo "=== Stage A: calibration against CATT Table 5 ==="
# --fetch-benchmark needs network; drop it if the compute nodes are offline
# and copy run/catt_benchmark/ over from the login node instead.
uv run python -u scripts/calibrate_der.py --fetch-benchmark --write

echo
echo "=== Stage B: DER on gold Qur'anic text ==="
uv run python -u scripts/measure_quranic_der.py --write

echo
echo "=== done; artifacts ==="
ls -la run/wikinews_der.json run/quranic_der.json
