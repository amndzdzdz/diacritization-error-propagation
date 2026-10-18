#!/bin/bash
#SBATCH --job-name=mdd_arm
#SBATCH --output=run/logs/output_logs/output_mdd_arm_%j.log
#SBATCH --error=run/logs/error_logs/error_mdd_arm_%j.log
#SBATCH -J mdd_arm
#SBATCH -p NvidiaAll
# `NvidiaAll` imposes no maximum walltime (confirmed 18 Oct 2026), and 0 means
# "no limit" to SLURM. Stated explicitly rather than omitted: if an admin ever
# sets a partition DefaultTime, a missing -t silently reintroduces a cap, and
# the symptom would be a run killed somewhere past step 100k. Both arms must
# carry the same value -- a limit that truncates one arm and not the other is a
# difference between them that no result would reveal as such.
#SBATCH -t 0

# Week-07 task 7: both arms of the 2x2, from one module and one config.
#
#   USE_CANONICAL=0 sbatch run/train_mdd_arm.slurm    # the prompt-free arm
#   USE_CANONICAL=1 sbatch run/train_mdd_arm.slurm    # the text-dependent arm
#
# Those two submissions differ in **one** value, which is passed to
# `MddConfig.use_canonical` and decides whether the cross-attention block in
# `src/arabic_mdd/models/conditioning.py` runs. Everything else -- the upstream,
# the manifest, the optimizer, the step count, the seed, specaug, bucketing,
# the checkpoint-selection metric -- is the same file read twice. That is what
# plan 3's "two of your own systems differing in exactly one property" requires,
# and it is the reason this is one slurm script rather than two.
#
# ---------------------------------------------------------------------------
# Three things to know before submitting.
#
# **This is a THIRD training run, not a reuse of task 6's.** Task 6 trains
# s3prl's own `downstream/ctc` with XLS-R; this trains `downstream/ctc_cond`
# with our `MddModel`. The 2x2's prompt-free row has to be *this* model
# ablated, because a row filled by a differently-implemented system would make
# the row difference a difference of two things. Task 6's run keeps its own
# job: it is the modern-system anchor for the architecture-independence claim,
# and it is the control on this port -- if our USE_CANONICAL=0 arm comes out
# materially below task 6's s3prl run at the same config and upstream, the port
# is wrong and the conditioned number means nothing. Compare those two first.
#
# **At step 0 the two arms are the same function, bitwise.** The conditioner's
# output projection is zero-initialized, so `frames + gate(context)` is exactly
# `frames` until the first optimizer step. Expect the two runs' step-0 loss to
# agree to the digit; if they do not, the arms differ in something other than
# the conditioning and the comparison is void. `src/arabic_mdd/models/
# conditioning.py` explains why it is built that way.
#
# **Single GPU, deliberately.** The ablated arm constructs the conditioner (for
# RNG parity -- see mdd.py) and routes nothing through it, so its parameters
# receive no gradient. Under DDP that is the classic "parameters that did not
# receive grad" abort. Plain `python3 run_downstream.py` does not initialize
# distributed, so there is nothing to work around here; just do not add
# torchrun.
# ---------------------------------------------------------------------------

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

export PATH="$HOME/.local/bin:$PATH"

# No default. Both values are legitimate and the failure is silent in the
# direction that matters: a conditioned run submitted with the switch off
# trains happily and fills the text-dependent row of the 2x2 with a
# prompt-free result, which is a wrong paper rather than a failed job.
if [ -z "${USE_CANONICAL:-}" ]; then
    echo "ERROR: USE_CANONICAL is unset, and there is no safe default." >&2
    echo "" >&2
    echo "  USE_CANONICAL=0  the prompt-free arm. Reads the three-column" >&2
    echo "                   manifest and ignores column three." >&2
    echo "  USE_CANONICAL=1  the text-dependent arm. Conditions on column" >&2
    echo "                   three via cross-attention." >&2
    echo "" >&2
    echo "e.g. USE_CANONICAL=1 sbatch run/train_mdd_arm.slurm" >&2
    exit 1
fi
case "$USE_CANONICAL" in
    0) COND_FLAG=False; ARM=promptfree ;;
    1) COND_FLAG=True;  ARM=conditioned ;;
    *) echo "ERROR: USE_CANONICAL must be 0 or 1, got '$USE_CANONICAL'." >&2; exit 1 ;;
esac

export TMPDIR="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR"
export HF_HOME="${HF_HOME:-$TMPDIR/hf_cache}"
mkdir -p "$HF_HOME"

# Same upstream as task 6, so that run is a valid control on this port.
export UPSTREAM_PIN=xlsr300m

# Hardcoded, both of them. These are our own reported systems, so they must not
# train on -- or select against -- rows that overlap either evaluation pool.
export APPLY_HOLDOUT=1
# Builds the three-column manifest and materializes downstream/ctc_cond. Set
# for BOTH arms: the prompt-free arm reads the same manifest and ignores column
# three, which is one fewer difference between the two runs than pointing them
# at different files would be.
export BUILD_CONDITIONED=1

bash run/prepare_baseline.sh

S3PRL_VENV="${S3PRL_VENV:-$REPO_ROOT/run/s3prl_venv}"
# shellcheck source=/dev/null
source "$S3PRL_VENV/bin/activate"

# `downstream/ctc_cond/expert.py` imports MddConfig/MddModel from this repo, so
# that the arms the cluster trains are the arms tests/models/ tests. This is
# the line that makes that import resolve; prepare_baseline.sh step 7b has
# already verified it works on this interpreter.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

UPSTREAM_ENV="$REPO_ROOT/run/.upstream_$UPSTREAM_PIN.env"
# shellcheck source=/dev/null
[ -f "$UPSTREAM_ENV" ] && source "$UPSTREAM_ENV"
if [ -z "${UPSTREAM_DIR:-}" ] || [ ! -d "$UPSTREAM_DIR" ]; then
    echo "ERROR: no pinned upstream at '${UPSTREAM_DIR:-<unset>}'." >&2
    echo "Expected run/prepare_baseline.sh step 8 to write $UPSTREAM_ENV." >&2
    exit 1
fi
echo "pinned upstream: $S3PRL_UPSTREAM -k $UPSTREAM_DIR"

# Evidence in this job's own log that both data-side guards fired. A log alone
# should answer "was this run contaminated?" and "did it have canonicals?".
for split in train dev; do
    for report in "holdout_applied_$split" "canonical_manifest_$split"; do
        path="$REPO_ROOT/run/$report.json"
        if [ ! -f "$path" ]; then
            echo "ERROR: $path missing -- prepare_baseline.sh did not write it." >&2
            exit 1
        fi
    done
done
python3 - "$REPO_ROOT/run" <<'PY'
import json
import os
import sys

run_dir = sys.argv[1]
for split in ("train", "dev"):
    holdout = json.load(open(os.path.join(run_dir, "holdout_applied_%s.json" % split)))
    canon = json.load(open(os.path.join(run_dir, "canonical_manifest_%s.json" % split)))
    print(
        "holdout %s: %d -> %d rows (%d by id, %d by phoneme ref)"
        % (
            split,
            holdout["rows_in"],
            holdout["rows_out"],
            holdout["dropped_by_id"],
            holdout["dropped_by_phoneme_ref"],
        )
    )
    print(
        "canonical %s: %d rows (%d from sentence, %d from file, %d unknown tokens)"
        % (
            split,
            canon["rows"],
            canon["canonical_from_sentence"],
            canon["canonical_from_file"],
            canon["unknown_canonical_tokens"],
        )
    )
    # The number that decides whether the conditioned arm can be trusted at
    # all. `canonical_from_file` counts the Iqra_TTS rows -- the only rows
    # where the CTC target differs from the canonical, and therefore the only
    # training signal that punishes copying the canonical instead of reading
    # the audio (src/arabic_mdd/models/collapse.py). At zero on train, the
    # text-dependent arm has nothing stopping it from collapsing into an echo,
    # and a spectacular-looking |dF1| in the 2x2 would be a pure artifact.
    if split == "train" and canon["canonical_from_file"] == 0:
        print("", file=sys.stderr)
        print(
            "ERROR: no train row took its canonical from a file, so every row has "
            "canonical == target.",
            file=sys.stderr,
        )
        print(
            "That is the whole 52 h of anti-collapse signal missing. Either Iqra_TTS "
            "was never merged (prepare_baseline.sh step 4) or its canonicals were "
            "never dumped (step 3b). Do not train on this.",
            file=sys.stderr,
        )
        sys.exit(1)
PY

S3PRL_DIR="${S3PRL_DIR:-$REPO_ROOT/run/s3prl_checkout}"
cd "$S3PRL_DIR/s3prl"

exp_dir="${EXP_DIR:-xlsr300m_mdd_${ARM}}"

# The arm switch, and the only override either submission needs. s3prl's
# `-o` runs the value through `eval` (utility/helper.py:84), so True/False
# arrive as Python booleans rather than as the strings "True"/"False" --
# which matters, because a non-empty string is truthy and `use_canonical`
# would then be on in both arms.
OVERRIDES="config.downstream_expert.model.use_canonical=${COND_FLAG}"
if [ -n "${STEPS:-}" ]; then
    OVERRIDES="${OVERRIDES},,config.runner.total_steps=${STEPS}"
fi
if [ -n "${BATCH_SIZE:-}" ]; then
    # Halve batch_size and double ACCUM together, so the effective batch stays
    # at the baseline's 16. The conditioner's keys are short (a canonical is
    # tens of tokens, not thousands of frames), so if this arm OOMs where task
    # 6 did not, suspect the attention's frames x canonical score matrix.
    OVERRIDES="${OVERRIDES},,config.downstream_expert.corpus.batch_size=${BATCH_SIZE}"
    if [ -n "${ACCUM:-}" ]; then
        OVERRIDES="${OVERRIDES},,config.runner.gradient_accumulate_steps=${ACCUM}"
    else
        echo "WARNING: BATCH_SIZE set without ACCUM. The effective batch now" >&2
        echo "differs from the other arm's, so the one-property claim is false." >&2
    fi
fi

echo "== train the $ARM arm (use_canonical=$COND_FLAG) into $exp_dir =="
echo "overrides: $OVERRIDES"
python3 run_downstream.py -m train \
    -c downstream/ctc_cond/cv_config/sws_cond.yaml -p "${exp_dir}" \
    -u "$S3PRL_UPSTREAM" -k "$UPSTREAM_DIR" -d ctc_cond -o "$OVERRIDES"

echo "== evaluate dev-best =="
python3 run_downstream.py -m evaluate -e "${exp_dir}/dev-best.ckpt"

echo "== collapse diagnostic (indicative only on this split) =="
# Reads the three .ark files the expert wrote during evaluation. Run for BOTH
# arms: the prompt-free arm's copy rate -- against a canonical it never saw --
# is the null this number has to be read against.
#
# Expect `canonical_preference: nan` here and do not treat it as a failure. The
# dev split is held-out Iqra_train, where the CTC target IS phoneme_ref, so no
# row has canonical != target and there is nothing for the model to resolve
# either way. The decisive measurement is the same three sequences against
# QuranMB.v2, whose human annotation disagrees with the canonical on ~12.4% of
# phonemes -- that is week 8's evaluation, and it needs no new dataloader.
# Non-fatal, because a 40-hour training run must not be thrown away by a
# reporting script.
if command -v uv >/dev/null 2>&1; then
    COLLAPSE_PY=(uv run --project "$REPO_ROOT" python)
else
    COLLAPSE_PY=(env "PYTHONPATH=$REPO_ROOT/src" python3)
fi
"${COLLAPSE_PY[@]}" "$REPO_ROOT/scripts/report_collapse.py" \
    --expdir "${exp_dir}" --split dev \
    --output "$REPO_ROOT/run/collapse_${ARM}.json" \
    || echo "WARNING: collapse diagnostic failed; the .ark files are still in $exp_dir"

echo "== recorded provenance =="
python3 "$REPO_ROOT/scripts/baseline_reproduction/inspect_s3prl_ckpt.py" \
    "${exp_dir}/dev-best.ckpt" | tee "$REPO_ROOT/run/mdd_${ARM}_provenance.txt"

echo "== done. Artifacts in $S3PRL_DIR/s3prl/$exp_dir =="
echo "Next: compare the $ARM arm's F1 against the other arm's, and both against"
echo "task 6's s3prl run. Then docs/weeks/week-07.md task 7."
