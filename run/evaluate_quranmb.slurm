#!/bin/bash
#SBATCH --job-name=quranmb_eval
#SBATCH --output=run/logs/output_logs/output_quranmb_eval_%j.log
#SBATCH --error=run/logs/error_logs/error_quranmb_eval_%j.log
#SBATCH -J quranmb_eval
#SBATCH -p NvidiaAll
# Same reasoning as run/train_mdd_arm.slurm: NvidiaAll imposes no maximum and a
# missing -t would silently inherit any DefaultTime an admin later sets. This
# job is minutes, not days, so the cap would not bite -- stated anyway, because
# a truncated evaluation pass writes a *short* .ark rather than failing, and the
# 1,642-line assertion below is the only thing that would notice.
#SBATCH -t 0

# Week-08 tasks 3, 4 and 6: one arm, one canonical, over QuranMB.v2.
#
#   EXP_DIR=xlsr300m_mdd_conditioned CANONICAL_VARIANT=gold \
#       sbatch run/evaluate_quranmb.slurm
#
# One submission is **one cell's worth of inference**. Plan section 1's 2x2 has
# rows that cost a GPU pass (which canonical the model receives) and columns
# that are free (which canonical the prediction is scored against, offline over
# a .ark file). So the full table is:
#
#   xlsr300m_mdd_conditioned   x {gold, catt-eo, catt-ed, shakkala, mishkal}
#   xlsr300m_mdd_promptfree    x {gold, mishkal}   <- see the identity check
#   xlsr300m_promptfree_heldout x {gold}
#
# and `scripts/measure_label_path.py` then sweeps the scoring column of each
# pass without touching a GPU again.
#
# ---------------------------------------------------------------------------
# Three things about this job that are not obvious from the command line.
#
# **The test manifest is set with `-o`, and a generated YAML would NOT work.**
# At the pinned commit, `run_downstream.py -m evaluate -e <ckpt>` takes
# `config = ckpt['Config']` and never reads `-c` (run_downstream.py:132; `-c` is
# on the `else` branch for new experiments). A `sws_cond_quranmb.yaml` passed
# with `-c` would be *silently ignored* and every pass would re-score the 2,069
# dev clips the checkpoint's own `corpus.test` points at -- which is precisely
# the week-7 mix-up this job exists to avoid, arriving again and looking like a
# successful run. `-o` is applied *after* the checkpoint's config
# (utility/helper.py `override`) and its values go through `eval()`, so a list
# literal arrives as a list. Hence the override, plus the hard line-count
# assertion: 1,642 is QuranMB.v2 and 2,069 is the dev set, and the difference
# must be caught by a number rather than by a reader.
#
# **`args.expdir` is overridden too, and that is not cosmetic.** `expdir` is not
# in upstream's `cannot_overwrite_args`, so without the override it comes from
# the checkpoint and every pass would write `test-hyp.ark` into the *training*
# expdir -- five variants overwriting each other, and week 7's dev-split .ark
# files gone with them.
#
# **The .ark files are renamed `test-* -> quranmb-*` on copy-out.** s3prl calls
# the evaluated split "test" no matter what is in it, and in week 7 that name
# meant the dev set. Carrying "test" forward into run/quranmb_eval/ would
# preserve exactly the ambiguity that cost week 7 a re-run.
# ---------------------------------------------------------------------------

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="$(pwd)"

export PATH="$HOME/.local/bin:$PATH"

VARIANTS="gold catt-eo catt-ed shakkala mishkal"

# No defaults for either. A default EXP_DIR would quietly evaluate whichever arm
# was alphabetically convenient and fill a row of the 2x2 with another arm's
# numbers; a default CANONICAL_VARIANT would make the conditioned arm's five
# passes indistinguishable in the log.
if [ -z "${EXP_DIR:-}" ] || [ -z "${CANONICAL_VARIANT:-}" ]; then
    echo "ERROR: EXP_DIR and CANONICAL_VARIANT are both required." >&2
    echo "" >&2
    echo "  EXP_DIR             an s3prl expdir holding dev-best.ckpt, relative" >&2
    echo "                      to \$S3PRL_DIR/s3prl. Week 7 produced three:" >&2
    echo "                        xlsr300m_mdd_conditioned    (use_canonical=True)" >&2
    echo "                        xlsr300m_mdd_promptfree     (use_canonical=False)" >&2
    echo "                        xlsr300m_promptfree_heldout (s3prl's own -d ctc)" >&2
    echo "  CANONICAL_VARIANT   one of: $VARIANTS" >&2
    echo "" >&2
    echo "e.g. EXP_DIR=xlsr300m_mdd_conditioned CANONICAL_VARIANT=mishkal \\" >&2
    echo "         sbatch run/evaluate_quranmb.slurm" >&2
    exit 1
fi
case " $VARIANTS " in
    *" $CANONICAL_VARIANT "*) ;;
    *) echo "ERROR: CANONICAL_VARIANT='$CANONICAL_VARIANT' is not one of: $VARIANTS" >&2
       echo "These are the five manifests scripts/build_quranmb_manifests.py writes." >&2
       exit 1 ;;
esac

export TMPDIR="${TMPDIR:-/tmp/${USER}_${SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR"
export HF_HOME="${HF_HOME:-$TMPDIR/hf_cache}"
mkdir -p "$HF_HOME"

S3PRL_DIR="${S3PRL_DIR:-$REPO_ROOT/run/s3prl_checkout}"
S3PRL_VENV="${S3PRL_VENV:-$REPO_ROOT/run/s3prl_venv}"
DATA_DIR="${BASELINE_DATA_DIR:-$REPO_ROOT/run/data}"
MANIFEST="$DATA_DIR/CV-Ar/quranmb/quranmb.${CANONICAL_VARIANT}.tsv"
EVAL_EXPDIR="${EXP_DIR}_quranmb_${CANONICAL_VARIANT}"
OUT_DIR="$REPO_ROOT/run/quranmb_eval/${EXP_DIR}/${CANONICAL_VARIANT}"

# deliberately NOT `bash run/prepare_baseline.sh`: that script rebuilds the
# train/dev manifests and would re-run the holdout filter. Nothing here needs
# it, and a rebuild between the five conditioned passes could change the
# training-side artifacts mid-table. The one thing this job needs from the data
# directory is the QuranMB manifest, which is built once on a login node.
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found." >&2
    echo "" >&2
    echo "Build the five manifests once, on a LOGIN NODE (they need the four" >&2
    echo "diacritizers and a HuggingFace download, not a GPU):" >&2
    echo "  uv run python scripts/build_quranmb_manifests.py \\" >&2
    echo "      --data-root $DATA_DIR/CV-Ar" >&2
    echo "" >&2
    echo "That run also performs week-08 task 2's vocabulary check, which must" >&2
    echo "pass before any GPU time: an out-of-vocabulary canonical token becomes" >&2
    echo "<unk> silently and degrades the conditioning signal in a way whose only" >&2
    echo "symptom is a disappointing |dF1|." >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$S3PRL_VENV/bin/activate"
# `downstream/ctc_cond/expert.py` imports MddConfig/MddModel from this repo.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$S3PRL_DIR/s3prl"
CKPT="${EXP_DIR}/dev-best.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: $S3PRL_DIR/s3prl/$CKPT not found." >&2
    echo "Expdirs live inside the pinned s3prl checkout, so a stray \`git clean\`" >&2
    echo "there costs a 200k-step run. Check run/mdd_*_provenance.txt." >&2
    exit 1
fi

echo "== preflight: what the checkpoint thinks it is, and where the wavs are =="
# Everything here is read from the checkpoint rather than assumed, because every
# one of these values silently decides what the pass measures. `use_canonical`
# in particular selects which of the two identity checks at the end applies.
PREFLIGHT="$TMPDIR/preflight_${SLURM_JOB_ID:-$$}.env"
python3 - "$CKPT" "$MANIFEST" "$PREFLIGHT" <<'PY'
import csv
import os
import sys

import torch

ckpt_path, manifest, out = sys.argv[1], sys.argv[2], sys.argv[3]
ckpt = torch.load(ckpt_path, map_location="cpu")
args, config = ckpt["Args"], ckpt["Config"]
downstream = getattr(args, "downstream", None)
expert = config["downstream_expert"]
corpus_path = expert["corpus"]["path"]
# Absent for s3prl's own `-d ctc`, whose model has no such switch. `None` is
# therefore meaningful and distinct from False.
use_canonical = expert.get("model", {}).get("use_canonical")

with open(manifest, newline="") as handle:
    rows = list(csv.reader(handle, delimiter="\t"))
header, body = rows[0], rows[1:]
if [c.strip() for c in header] != ["path", "sentence", "canonical"]:
    sys.exit(f"{manifest}: header is {header!r}, expected ['path','sentence','canonical']")

# The join that no other check performs. The manifest's paths are relative to
# the *checkpoint's* corpus.path, and that value was baked in at training time
# -- so a BASELINE_DATA_DIR that differs between the training job and this one
# produces a manifest full of paths torchaudio cannot open. The failure would
# surface mid-pass as a FileNotFoundError after the upstream had loaded, which
# is a confusing place to learn it.
first = os.path.join(corpus_path, body[0][0])
if not os.path.isfile(first):
    sys.exit(
        f"{first} does not exist.\n"
        f"The checkpoint records corpus.path = {corpus_path!r} and the manifest's "
        f"first row is {body[0][0]!r}. Either the wav dump went to a different "
        "--data-root than the training run used, or it has not been run."
    )

print(f"  downstream      : {downstream}")
print(f"  use_canonical   : {use_canonical}")
print(f"  corpus.path     : {corpus_path}")
print(f"  manifest rows   : {len(body)}")
print(f"  first wav       : {first}")
with open(out, "w") as handle:
    handle.write(f"CKPT_DOWNSTREAM={downstream}\n")
    handle.write(f"CKPT_USE_CANONICAL={use_canonical}\n")
    handle.write(f"MANIFEST_ROWS={len(body)}\n")
PY
# shellcheck source=/dev/null
source "$PREFLIGHT"

EXPECTED_CLIPS=1642
if [ "$MANIFEST_ROWS" != "$EXPECTED_CLIPS" ]; then
    echo "ERROR: $MANIFEST has $MANIFEST_ROWS rows, expected $EXPECTED_CLIPS." >&2
    echo "QuranMB.v2 as safikhan/quran_mbv2_formatted is 1,642 clips. A different" >&2
    echo "count means the manifest build dropped rows, and the 2x2's denominator" >&2
    echo "would differ between variants." >&2
    exit 1
fi

echo "== evaluate $EXP_DIR on QuranMB.v2 with C = $CANONICAL_VARIANT =="
# Two overrides, both load-bearing; see the header comment. Absolute manifest
# path, because `-m evaluate` runs from $S3PRL_DIR/s3prl and a relative one
# would resolve against the checkout.
OVERRIDES="config.downstream_expert.corpus.test=['${MANIFEST}']"
OVERRIDES="${OVERRIDES},,args.expdir=${EVAL_EXPDIR}"
echo "overrides: $OVERRIDES"
python3 run_downstream.py -m evaluate -e "$CKPT" -o "$OVERRIDES"

echo "== check what was actually scored =="
HYP="$EVAL_EXPDIR/test-hyp.ark"
if [ ! -f "$HYP" ]; then
    echo "ERROR: $HYP was not written." >&2
    echo "The expert writes .ark files only when the split name contains 'test'" >&2
    echo "or 'dev' (expert.py log_records). -m evaluate uses -t/--evaluate_split," >&2
    echo "which defaults to 'test', so this should not happen -- if -t was passed," >&2
    echo "that is why." >&2
    exit 1
fi
LINES=$(wc -l < "$HYP")
if [ "$LINES" != "$EXPECTED_CLIPS" ]; then
    echo "ERROR: $HYP holds $LINES utterances, expected $EXPECTED_CLIPS." >&2
    echo "" >&2
    echo "2069 means the corpus.test override did not take and this pass re-scored" >&2
    echo "the dev split -- the exact week-7 failure. Anything less means the pass" >&2
    echo "was truncated. Do not copy this out." >&2
    exit 1
fi
echo "  $LINES utterances, as expected"

echo "== copy out to run/quranmb_eval =="
mkdir -p "$OUT_DIR"
for name in hyp ref canon; do
    src="$EVAL_EXPDIR/test-${name}.ark"
    # canon.ark exists only for downstream/ctc_cond. s3prl's own `-d ctc` writes
    # hyp and ref alone, which is why the collapse diagnostic cannot run on the
    # xlsr300m_promptfree_heldout arm: without the canonical there is no copy to
    # detect. Absence here is information, not an error.
    [ -f "$src" ] && cp "$src" "$OUT_DIR/quranmb-${name}.ark"
done
cp "$MANIFEST" "$OUT_DIR/manifest.tsv"
ls -la "$OUT_DIR"

if command -v uv >/dev/null 2>&1; then
    PY=(uv run --project "$REPO_ROOT" python)
else
    PY=(env "PYTHONPATH=$REPO_ROOT/src" python3)
fi

echo "== .ark -> predictions JSON =="
PREDICTIONS="$REPO_ROOT/run/quranmb_predictions_${EXP_DIR}_${CANONICAL_VARIANT}.json"
"${PY[@]}" "$REPO_ROOT/scripts/ark_to_predictions.py" \
    --ark "$OUT_DIR/quranmb-hyp.ark" --output "$PREDICTIONS"

if [ -f "$OUT_DIR/quranmb-canon.ark" ]; then
    echo "== task 6: the collapse diagnostic, on the canonical this pass received =="
    # D12's null is the ablated arm's number on the SAME manifest, so --null is
    # passed only when that report already exists. Order the submissions
    # ablated-first and the conditioned arm's report carries the verdict; submit
    # them the other way round and the conditioned report is still correct, just
    # without the comparison -- rerun this one line afterwards.
    COLLAPSE="$REPO_ROOT/run/quranmb_collapse_${EXP_DIR}_${CANONICAL_VARIANT}.json"
    NULL_REPORT="$REPO_ROOT/run/quranmb_collapse_${NULL_EXP_DIR:-xlsr300m_mdd_promptfree}_${CANONICAL_VARIANT}.json"
    NULL_ARGS=()
    if [ "$CKPT_USE_CANONICAL" = "True" ] && [ -f "$NULL_REPORT" ]; then
        NULL_ARGS=(--null "$NULL_REPORT")
        echo "  applying D12 against $NULL_REPORT"
    elif [ "$CKPT_USE_CANONICAL" = "True" ]; then
        echo "  no ablated-arm report at $NULL_REPORT yet, so no D12 verdict."
        echo "  D12 is a DIFFERENCE from the arm that cannot consume a canonical;"
        echo "  week 7 measured canonical_preference = 1.0000 on that arm, so the"
        echo "  conditioned number alone is not evidence of anything."
    fi
    "${PY[@]}" "$REPO_ROOT/scripts/report_collapse.py" \
        --expdir "$OUT_DIR" --split quranmb \
        --output "$COLLAPSE" "${NULL_ARGS[@]}"
else
    echo "== no quranmb-canon.ark: this is s3prl's own -d ctc, so no collapse number =="
fi

echo "== task 4: the routing check, against this arm's other variants =="
# A free end-to-end proof that the canonical goes where we think it does, and
# the only check in the week that can catch the canonical being wired to the
# wrong arm. Compares against every sibling variant already on disk, so it
# becomes meaningful from the second submission for an arm onwards.
#
#   use_canonical=False : hypotheses MUST be byte-identical across variants,
#                         while the canonicals differ. Identical canonicals
#                         would mean the override never took; differing
#                         hypotheses would mean an arm that is supposed to
#                         ignore column three is reading it, and every number
#                         this week would be void.
#   use_canonical=True  : the canonicals must differ, and the hypotheses are
#                         then *expected* to differ. Identical hypotheses are
#                         not a crash but they are the whole input-path row
#                         coming out at zero, which is a result that has to be
#                         noticed here rather than inferred from a flat 2x2.
ROUTING_FAILED=0
for sibling in "$REPO_ROOT/run/quranmb_eval/${EXP_DIR}"/*; do
    other="$(basename "$sibling")"
    [ "$other" = "$CANONICAL_VARIANT" ] && continue
    [ -f "$sibling/quranmb-hyp.ark" ] || continue
    if [ -f "$sibling/quranmb-canon.ark" ] && [ -f "$OUT_DIR/quranmb-canon.ark" ]; then
        if cmp -s "$sibling/quranmb-canon.ark" "$OUT_DIR/quranmb-canon.ark"; then
            echo "ERROR: the canonicals fed to variant '$other' and '$CANONICAL_VARIANT'" >&2
            echo "are byte-identical. Two different manifests cannot produce the same" >&2
            echo "canonical column, so the corpus.test override did not take on one of" >&2
            echo "the two passes." >&2
            ROUTING_FAILED=1
            continue
        fi
        echo "  canonicals differ vs '$other': the manifest override took"
    fi
    if cmp -s "$sibling/quranmb-hyp.ark" "$OUT_DIR/quranmb-hyp.ark"; then
        if [ "$CKPT_USE_CANONICAL" = "True" ]; then
            echo "  WARNING: hypotheses are byte-identical to variant '$other' on a" >&2
            echo "  use_canonical=True arm. The canonical reached the dataloader (above)" >&2
            echo "  but changed nothing, so the input-path row of the 2x2 is exactly" >&2
            echo "  zero. Report it as measured; do not assume it is a bug." >&2
        else
            echo "  hypotheses byte-identical to '$other': the arm ignores C, as it must"
        fi
    else
        if [ "$CKPT_USE_CANONICAL" = "True" ]; then
            echo "  hypotheses differ vs '$other': conditioning has an inference-time effect"
        else
            echo "ERROR: hypotheses differ from variant '$other' on an arm with" >&2
            echo "use_canonical=$CKPT_USE_CANONICAL, which never consumes the canonical." >&2
            echo "A prompt-free arm's output cannot depend on column three. Either the" >&2
            echo "ablation is not actually ablating or the two passes differ in something" >&2
            echo "other than the manifest -- and in both cases every number produced this" >&2
            echo "week is void. This is week-08 task 4's assertion." >&2
            ROUTING_FAILED=1
        fi
    fi
done
if [ "$ROUTING_FAILED" != 0 ]; then
    exit 1
fi

echo "== done =="
echo "arks + manifest : $OUT_DIR"
echo "predictions     : $PREDICTIONS"
echo ""
echo "Next, with no GPU: sweep the scoring column of this pass."
echo "  uv run python scripts/measure_label_path.py \\"
echo "      --predictions $PREDICTIONS \\"
echo "      --output $REPO_ROOT/run/quranmb_2x2_${EXP_DIR}_${CANONICAL_VARIANT}.json --write"
echo "and again with --clean-prompts-only for the 82-prompt denominator, which is"
echo "the footing for any comparison against week 6's official-checkpoint numbers."
