#!/bin/bash
#SBATCH --job-name=xlsr_promptfree
#SBATCH --output=run/logs/output_logs/output_xlsr_promptfree_%j.log
#SBATCH --error=run/logs/error_logs/error_xlsr_promptfree_%j.log
#SBATCH -J xlsr_promptfree
#SBATCH -p NvidiaAll
# `NvidiaAll` imposes no maximum walltime (confirmed 18 Oct 2026), and 0 means
# "no limit" to SLURM. Stated explicitly rather than omitted: if an admin ever
# sets a partition DefaultTime, a missing -t silently reintroduces a cap, and
# the symptom would be a run killed somewhere past step 100k.
#SBATCH -t 0

# Week-07 task 6: the modern prompt-free arm. XLS-R-300m (frozen) + s3prl's
# learned layer-weighted sum + BiLSTM + CTC, trained on the contamination-held-
# out Iqra_train manifest.
#
#   sbatch run/train_xlsr_promptfree.slurm
#   BATCH_SIZE=8 ACCUM=2 sbatch run/train_xlsr_promptfree.slurm   # OOM fallback
#   STEPS=50000 sbatch run/train_xlsr_promptfree.slurm            # shorter, if ever wanted
#
# The step-rate smoke run (week-07 task 3,
# `UPSTREAM_PIN=xlsr300m STEPS=200 sbatch run/smoke_train.slurm`) is **optional
# now that walltime is unlimited** -- it existed to size a limit that no longer
# exists. What it also measured, and what is therefore still unverified here, is
# whether `batch_size: 16` fits in this GPU's memory. That failure is loud and
# arrives in the first minutes, so it costs a resubmission rather than a run;
# `BATCH_SIZE`/`ACCUM` above are the fallback, and they must be changed together
# or the recipe changes rather than only its memory footprint.
#
# ---------------------------------------------------------------------------
# Two things this job deliberately does NOT change, and one it does.
#
# **It does not change the config.** It runs the organizers' own
# `scripts/baseline_reproduction/config/sws.yaml`, verbatim, the same file the
# mHuBERT baseline runs. That is not laziness, it is the comparison: the only
# difference between this run and the week-3 baseline is the upstream, so a
# difference in the result is attributable to the upstream. Checked at the
# pinned s3prl commit that no config edit is even needed for a 24-layer,
# 1024-dim upstream in place of a 12-layer, 768-dim one:
#   * `upstream/interfaces.py` Featurizer runs a dummy forward on 1 s of noise,
#     sets `layer_num = len(hidden_states)` and `output_dim = feature.size(-1)`
#     from it, and allocates `weights = nn.Parameter(torch.zeros(layer_num))`.
#     So 25 hidden states and 1024 channels are both discovered, not declared.
#   * `downstream/ctc/expert.py` then does
#     `nn.Linear(upstream_dim, modelrc["project_dim"])`, and `project_dim` is
#     already 1024 -- so the BiLSTM sees a 1024-dim input either way and the
#     head is literally the same size in both runs.
#
# **It does not use a TCN**, which `docs/weeks/week-07.md` task 6 named while
# describing the whu-iasp recipe. s3prl's ctc downstream offers exactly two
# heads, `RNNs` (the BiLSTM here) and `Wav2Letter` (an undilated conv stack,
# not a TCN), and `expert.py` reaches them by `eval(model_select)` over its own
# import namespace. Adding a TCN means either editing a tracked file in the
# sha-pinned checkout -- which breaks the pin's provenance story and trips
# prepare_baseline.sh's dirty-tree warning on every run -- or materializing a
# whole forked `downstream/ctc_tcn/` directory, which re-vendors data.py,
# expert.py and metric.py to own 60 lines of model.
#
# Not worth it, and the comparison is arguably better without it: holding the
# head fixed makes the upstream the single variable, which is what "the effect
# is not an artifact of one model" needs. The escalation trigger, if it is ever
# needed: **if this run does not beat the mHuBERT baseline's F1**, the head
# becomes a live hypothesis and the forked downstream is the next thing to try.
#
# **It does change the training data**, and that is the point of APPLY_HOLDOUT
# below.
# ---------------------------------------------------------------------------

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

export PATH="$HOME/.local/bin:$PATH"

# XLS-R-300m is a ~1.26 GB HF download on top of the dataset. Keep it off
# $HOME, which is NFS and quota-limited.
export TMPDIR="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR"
export HF_HOME="${HF_HOME:-$TMPDIR/hf_cache}"
mkdir -p "$HF_HOME"

export UPSTREAM_PIN=xlsr300m

# Hardcoded to 1, and it is the whole reason this file is separate from
# train_baseline.slurm. This is one of our own reported systems, so it must not
# train on -- or select against -- rows that overlap either evaluation pool.
# Nothing this week evaluates on MSA, which is exactly the trap: the model
# trained here is the model week 8 evaluates on MSA, and re-deriving it later
# costs 4-5 days of queue. 379 train rows (0.53%) and 505 dev rows (19.51%)
# leave; see docs/weeks/week-07.md task 4 and run/training_holdout.json.
export APPLY_HOLDOUT=1

bash run/prepare_baseline.sh

S3PRL_VENV="${S3PRL_VENV:-$REPO_ROOT/run/s3prl_venv}"
# shellcheck source=/dev/null
source "$S3PRL_VENV/bin/activate"

UPSTREAM_ENV="$REPO_ROOT/run/.upstream_$UPSTREAM_PIN.env"
# shellcheck source=/dev/null
[ -f "$UPSTREAM_ENV" ] && source "$UPSTREAM_ENV"
if [ -z "${UPSTREAM_DIR:-}" ] || [ ! -d "$UPSTREAM_DIR" ]; then
    echo "ERROR: no pinned upstream at '${UPSTREAM_DIR:-<unset>}'." >&2
    echo "Expected run/prepare_baseline.sh step 8 to write $UPSTREAM_ENV." >&2
    echo "Do NOT work around this by passing -k facebook/wav2vec2-xls-r-300m:" >&2
    echo "that silently un-pins the upstream (see pin_upstream.py)." >&2
    exit 1
fi
echo "pinned upstream: $S3PRL_UPSTREAM -k $UPSTREAM_DIR"

# Evidence, in this job's own log, that the holdout really was applied. The
# report is also committed to run/holdout_applied_train.json; printing the
# counts here means a log alone answers "was this run contaminated?".
for split in train dev; do
    report="$REPO_ROOT/run/holdout_applied_$split.json"
    if [ ! -f "$report" ]; then
        echo "ERROR: $report missing -- APPLY_HOLDOUT=1 but no filter report." >&2
        echo "prepare_baseline.sh step 6 should have written it. Do not proceed:" >&2
        echo "an unfiltered manifest here silently contaminates week 8's numbers." >&2
        exit 1
    fi
    python3 - "$report" <<'PY'
import json
import sys

r = json.load(open(sys.argv[1]))
print(
    "holdout %s: %d -> %d rows (%d by id, %d by phoneme ref)"
    % (
        r["split"],
        r["rows_in"],
        r["rows_out"],
        r["dropped_by_id"],
        r["dropped_by_phoneme_ref"],
    )
)
PY
done

S3PRL_DIR="${S3PRL_DIR:-$REPO_ROOT/run/s3prl_checkout}"
cd "$S3PRL_DIR/s3prl"

exp_dir="${EXP_DIR:-xlsr300m_promptfree_heldout}"

# Overrides are opt-in and empty by default, so an unmodified submit runs the
# organizers' config with nothing changed -- and `Config` in the saved
# checkpoint then matches the baseline's field for field except the upstream.
OVERRIDES=""
if [ -n "${STEPS:-}" ]; then
    OVERRIDES="config.runner.total_steps=${STEPS}"
fi
if [ -n "${BATCH_SIZE:-}" ]; then
    # Halving batch_size while doubling ACCUM keeps the *effective* batch at
    # the baseline's 16, which is what has to be held constant for the
    # comparison. Changing batch_size alone would change the recipe, not just
    # its memory footprint.
    [ -n "$OVERRIDES" ] && OVERRIDES="${OVERRIDES},,"
    OVERRIDES="${OVERRIDES}config.downstream_expert.corpus.batch_size=${BATCH_SIZE}"
    if [ -n "${ACCUM:-}" ]; then
        OVERRIDES="${OVERRIDES},,config.runner.gradient_accumulate_steps=${ACCUM}"
    else
        echo "WARNING: BATCH_SIZE set without ACCUM. The effective batch now" >&2
        echo "differs from the mHuBERT baseline's 16, so a difference in the" >&2
        echo "result is no longer attributable to the upstream alone." >&2
    fi
fi

echo "== train XLS-R-300m prompt-free into $exp_dir =="
# No apostrophe in the fallback: inside ${VAR:-...} bash still treats a single
# quote as a quote opener, even within double quotes, and it swallows the rest
# of the file.
echo "overrides: ${OVERRIDES:-<none - the organizers config, verbatim>}"
if [ -n "$OVERRIDES" ]; then
    python3 run_downstream.py -m train -c downstream/ctc/cv_config/sws.yaml -p "${exp_dir}" \
        -u "$S3PRL_UPSTREAM" -k "$UPSTREAM_DIR" -d ctc -o "$OVERRIDES"
else
    python3 run_downstream.py -m train -c downstream/ctc/cv_config/sws.yaml -p "${exp_dir}" \
        -u "$S3PRL_UPSTREAM" -k "$UPSTREAM_DIR" -d ctc
fi

echo "== evaluate dev-best =="
python3 run_downstream.py -m evaluate -e "${exp_dir}/dev-best.ckpt"

# Provenance from the checkpoint rather than from this script (msa-arm.md
# §5.3). Week 3 is the proof it matters: the organizers' README specified an
# upstream their own published checkpoint contradicts. The sha is inside
# Args.upstream_ckpt because -k is a sha-named directory.
echo "== recorded provenance =="
python3 "$REPO_ROOT/scripts/baseline_reproduction/inspect_s3prl_ckpt.py" \
    "${exp_dir}/dev-best.ckpt" | tee "$REPO_ROOT/run/xlsr_promptfree_provenance.txt"

echo "== done. Artifacts in $S3PRL_DIR/s3prl/$exp_dir =="
echo "Next: score the predictions per scripts/baseline_reproduction/README.md step 6."
