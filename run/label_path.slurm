#!/bin/bash
#SBATCH --job-name=label_path
#SBATCH --output=run/logs/output_logs/output_label_path_%j.log
#SBATCH --error=run/logs/error_logs/error_label_path_%j.log
#SBATCH -J label_path
#SBATCH -p NvidiaAll
#SBATCH -t 01:00:00

# Week-06 exit criterion, clause 1: the label-path column on the Qur'anic arm.
#
#   scripts/measure_label_path.py  -> run/quranic_label_path.json
#
# `A` (what the reciter said) and `P` (the official checkpoint's predictions)
# are held fixed; only `C` is swapped. Four diacritizers produce four `C_auto`
# conditions against one `C_gold`, and the output is dF1 per tool with a
# cluster bootstrap over the 98 prompts.
#
# WHY THE WALL IS ONE HOUR AND NOT TWELVE
#
# This is a *small* job and the time limit says so deliberately, so that a
# hang is reported in an hour instead of occupying a GPU node overnight. The
# measurement is 98 prompts of 4,442 characters total. Week 5 measured
# catt-eo at 0.015 s/verse and catt-ed at 1.205 s/verse over 6,236 verses, so
# the slowest tool here is ~2 minutes, not hours. The 10,000-replicate
# bootstrap is arithmetic on six integers per prompt and costs seconds.
#
# The expensive part of week 5 was the 6,236-verse corpus. That is not
# re-measured here: run/quranic_der.json already has it, and this script
# recomputes DER only on the 98 prompts, so the dF1-vs-DER relation can be
# read from one artifact.
#
# WHY IT IS A CLUSTER JOB AT ALL
#
# Honestly: it does not need to be. It is submitted here for two reasons.
# First, shakkala runs out-of-process under its own Python 3.10 environment
# and needs TensorFlow 2.9.3 fetched on first use, which is the one step that
# can be slow and is better off not competing with an interactive session.
# Second, catt-ed decodes autoregressively and picks up CUDA through
# onnxruntime's default provider order with no code change, so a GPU node
# makes the one slow tool faster for free.
#
# Note for the ONNX-provider item still open in week 6's housekeeping: job
# 224461 printed SELECTED ['CPUExecutionProvider'] on a node with a visible
# RTX 2060. If that happens again here it costs ~2 minutes rather than the
# 3h14m it cost in stage A of the DER job, so this job is a *cheap* place to
# observe the bug, not a victim of it.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "=== host $(hostname), job ${SLURM_JOB_ID:-local} ==="
nvidia-smi || echo "no nvidia-smi -- catt-ed falls back to CPU; costs ~2 min, not hours"

uv sync --all-groups

echo
echo "=== which execution provider a session actually SELECTS ==="
# Same probe as run/der_measurement.slurm: builds a real session on a real
# model and asks it, rather than printing ORT's static priority order.
uv run python - <<'PY'
from pathlib import Path

import catt_tashkeel
import onnxruntime as ort

print("onnxruntime", ort.__version__)
print("available  ", ort.get_available_providers())

models = sorted(Path(catt_tashkeel.__file__).parent.glob("onnx_models/**/*.onnx"))
if models:
    print("SELECTED   ", ort.InferenceSession(str(models[0])).get_providers())
else:
    print("SELECTED    (models not downloaded yet; this prints on the next run)")
PY

# Fail early rather than halfway through. Both inputs are committed, so a
# miss here means the checkout is incomplete, not that something needs
# fetching.
for f in run/quranmb_verse_map.json run/quranmb_predictions_official_ckpt.json; do
  [ -s "$f" ] || { echo "MISSING $f -- cannot measure the label path without it"; exit 1; }
done

echo
echo "=== pre-flight: can the ground truth be reached? ==="
# The canonical and annotation columns come from the public safikhan join on
# the HF hub, which is a network call. Compute nodes are sometimes offline,
# and finding that out *after* four diacritizers have run would waste the
# whole job. This also warms the HF cache so the real run does not re-fetch.
uv run python -u - <<'PY'
from arabic_mdd.data.quranmb import load_ground_truth

gt = load_ground_truth()
prompts = {g.prompt_index for g in gt.values()}
print(f"ground truth: {len(gt)} clips, {len(prompts)} prompts")
assert len(gt) == 1642, f"expected 1,642 clips, got {len(gt)}"
assert len(prompts) == 98, f"expected 98 prompts, got {len(prompts)}"
PY

echo
echo "=== smoke: one fast tool, 8 prompts, so a broken harness fails in seconds ==="
uv run python -u scripts/measure_label_path.py --limit 8 --tools catt-eo --bootstrap 200

echo
echo "=== the measurement: all four tools, 98 prompts, 10,000 bootstrap replicates ==="
date
uv run python -u scripts/measure_label_path.py --write
date

echo
echo "=== done; artifact ==="
ls -la run/quranic_label_path.json

# What to read first in the output above:
#
#   1. leg B's anchor F1. It must be 0.4415. If it is not, the harness has
#      drifted and no delta in the artifact is trustworthy -- stop there.
#   2. CLAUSE 1: PASS / NOT MET. The criteria were fixed in
#      docs/weeks/week-06.md before this ran: the CI must exclude zero for at
#      least mishkal, and |dF1| must be monotone in CE DER.
#   3. The dominant moving cell per tool. Prediction 1 says dF1 is negative
#      and FA-driven; a positive dF1, or a negative one moving some other
#      cell, falsifies it. Smoke runs on 8 prompts already showed mishkal
#      going *positive* and catt-eo negative, so expect to have to explain a
#      sign, and write it up in insights/week-06.md either way.
#   4. A "NO" on monotonicity is the most interesting possible failure: three
#      unrelated instruments already agree on that tool ordering, so an
#      inversion is first evidence for RQ3's miscalibration claim.
