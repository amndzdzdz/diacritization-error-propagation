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

# S3PRL supplies the featurizer, the downstream model AND the training loop, so
# an unpinned checkout can change the result of a run without changing a line
# of this repository. Pinned 16 Oct 2026.
#
# Provenance: this is the HEAD of the cluster checkout that produced the week-3
# gate -- F1 = 0.4415 with the organizers' checkpoint, within 0.0001 of the
# published 0.4414 on every component rate -- read off `citrin` on 16 Oct 2026.
# It is upstream's own v0.4.18 release commit, dated 2025-06-13. The tracked
# tree was clean at that point (`git status --porcelain -uno` empty), so this
# SHA reproduces the code exactly; the only untracked content was three
# training expdirs, which are outputs and not inputs.
#
# Do not "update" this to upstream main. The point of the pin is that this
# commit is the one the reproduced baseline was measured against; a newer one
# would make the gate a claim about software we no longer run.
S3PRL_COMMIT="${S3PRL_COMMIT:-ec8064b5889f81ca460fbe2c094ce576a6f120b7}"
DATA_DIR="${BASELINE_DATA_DIR:-$RUN_DIR/data}"

# --- Python dependency pins (week-07 task 1) --------------------------------
# Pinning S3PRL's *commit* only pins half the training loop. `transformers`
# supplies the upstream encoder, and `torch`/`torchaudio` supply the kernels,
# the RNG and the CTC implementation -- so an unpinned resolution can change
# the result of a run without changing a line of this repository, exactly the
# argument the S3PRL pin above is based on.
#
# Provenance: `pip freeze` from `run/s3prl_venv` on the cluster, read off
# `danburit` on 17 Oct 2026 -- i.e. these are the versions actually present in
# the venv, adopted as the pin, not versions chosen in the abstract. Caveat
# worth keeping: the S3PRL commit pin above was read off a host named `citrin`
# on 16 Oct 2026. If those are two different machines rather than two login
# nodes sharing $HOME, then this freeze describes `danburit`'s venv and the
# week-3 gate was produced on another one; the venv lives in the repo tree, so
# shared storage makes them the same, but that has not been verified.
#
# These are *verified*, not installed (see step 2) -- and a mismatch on the
# first three is fatal. Changing any of them invalidates the validated recipe
# and must be recorded in insights/.
PIN_TORCH="${PIN_TORCH:-2.4.1}"
PIN_TORCHAUDIO="${PIN_TORCHAUDIO:-2.4.1}"
PIN_TRANSFORMERS="${PIN_TRANSFORMERS:-4.46.3}"
# Advisory rather than fatal: these two do not determine the computation, but a
# large jump in either is worth seeing in the log.
PIN_NUMPY="${PIN_NUMPY:-1.24.4}"
PIN_SAFETENSORS="${PIN_SAFETENSORS:-0.5.3}"
export PIN_TORCH PIN_TORCHAUDIO PIN_TRANSFORMERS PIN_NUMPY PIN_SAFETENSORS

# --- Contamination holdout: deliberately WITHOUT a default ------------------
# `run/training_holdout.json` lists the Iqra_train rows that share a
# consonantal skeleton with one of the two evaluation pools -- the 500
# hand-annotated MSA rows (docs/msa-arm.md 5.2.1) and the QuranMB.v2 prompts
# (docs/weeks/week-07.md task 5). Dropping them is right for our own models and
# WRONG for the week-3 reproduction, which has to keep matching the organizers'
# full training set or it stops certifying anything.
#
# So there is no safe default and no default is offered. Either answer is
# defensible; silently picking one is not, because both failures are invisible
# in the logs -- a contaminated "clean" model trains perfectly happily, and a
# held-out reproduction misses the published number by an amount you would then
# spend days debugging. Every run states which experiment it is.
if [ -z "${APPLY_HOLDOUT:-}" ]; then
    echo "ERROR: APPLY_HOLDOUT is unset, and there is no safe default." >&2
    echo "" >&2
    echo "  APPLY_HOLDOUT=0  the week-3 baseline reproduction. Trains on the" >&2
    echo "                   organizers' full set, contamination included, because" >&2
    echo "                   reproducing their number is the entire point." >&2
    echo "  APPLY_HOLDOUT=1  any of our own models. Drops run/training_holdout.json" >&2
    echo "                   from BOTH the train manifest and the dev manifest --" >&2
    echo "                   dev too, because the MSA evaluation pool *is* 500 dev" >&2
    echo "                   rows, so dev-best selection would otherwise be" >&2
    echo "                   performed on the evaluation set." >&2
    echo "" >&2
    echo "e.g. APPLY_HOLDOUT=1 sbatch run/train_xlsr_promptfree.slurm" >&2
    exit 1
fi
case "$APPLY_HOLDOUT" in
    0|1) ;;
    *) echo "ERROR: APPLY_HOLDOUT must be 0 or 1, got '$APPLY_HOLDOUT'." >&2; exit 1 ;;
esac

# --- The text-dependent arm's extra staging (week-07 task 7) ----------------
# Unlike APPLY_HOLDOUT this *does* get a default, and the default is off,
# because the two mistakes are not symmetric. Forgetting APPLY_HOLDOUT yields a
# run that looks correct and is contaminated; forgetting BUILD_CONDITIONED
# yields a run that cannot start -- `downstream/ctc_cond/` and the three-column
# manifest will simply not be there. A failure that is impossible to miss needs
# no prompt.
#
# With it set, step 7 additionally:
#   * dumps Iqra_TTS's `phoneme_ref` to disk, if it is not there already;
#   * builds `*.cond.tsv`, the three-column manifest, from whichever manifest
#     APPLY_HOLDOUT selected -- so the holdout and the conditioning cannot
#     disagree about which rows exist;
#   * materializes `scripts/baseline_reproduction/ctc_cond/` into the pinned
#     checkout as `downstream/ctc_cond/`, and `sws_cond.yaml` beside it; and
#   * imports `arabic_mdd.models` in the 3.8 venv, which is the only thing that
#     enforces that package's Python 3.8 ceiling.
BUILD_CONDITIONED="${BUILD_CONDITIONED:-0}"
case "$BUILD_CONDITIONED" in
    0|1) ;;
    *) echo "ERROR: BUILD_CONDITIONED must be 0 or 1, got '$BUILD_CONDITIONED'." >&2; exit 1 ;;
esac

echo "== [1/8] Python 3.8 environment for S3PRL (stdlib venv) =="
# S3PRL needs Python 3.8, incompatible with this project's own Python-3.12
# env -- kept as a fully separate venv here, never touching this project's
# pyproject.toml/uv.lock.
#
# Interpreter resolution order (see also the comment block at the top of
# run/train_baseline.slurm):
#   1. $PYTHON38_BIN, if you set it (path to a python3.8 executable)
#   2. `python3.8` already on PATH
#   3. `python3` on PATH, if it reports version 3.8.x
#   4. a handful of common install locations
#   5. `uv python install 3.8` -- a standalone interpreter uv downloads and
#      manages itself, used only to get a 3.8 binary; the resulting venv
#      below is still a plain stdlib venv, not a uv-managed project env
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
    UV_BIN="${UV_BIN:-}"
    if [ -z "$UV_BIN" ] && command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    elif [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
        UV_BIN="$HOME/.local/bin/uv"
    fi
    if [ -n "$UV_BIN" ]; then
        "$UV_BIN" python install 3.8
        PYTHON38_BIN="$("$UV_BIN" python find 3.8)"
    fi
fi
if [ -z "$PYTHON38_BIN" ]; then
    echo "ERROR: could not find a Python 3.8 interpreter." >&2
    echo "S3PRL requires Python 3.8. If your cluster provides it via environment" >&2
    echo "modules, load it yourself and pass its path, e.g.:" >&2
    echo "  module load python/3.8 && PYTHON38_BIN=\$(which python3.8) sbatch run/train_baseline.slurm" >&2
    echo "Otherwise install uv (curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2
    echo "so this script can fetch Python 3.8 itself via 'uv python install 3.8'." >&2
    exit 1
fi
echo "Using Python 3.8 interpreter: $PYTHON38_BIN ($("$PYTHON38_BIN" --version))"

if [ ! -f "$S3PRL_VENV/bin/activate" ]; then
    "$PYTHON38_BIN" -m venv "$S3PRL_VENV"
fi
# shellcheck source=/dev/null
source "$S3PRL_VENV/bin/activate"
python -m pip install --quiet --upgrade pip

echo "== [2/8] S3PRL toolkit install =="
if [ ! -d "$S3PRL_DIR" ]; then
    git clone https://github.com/s3prl/s3prl.git "$S3PRL_DIR"
    # Detached on purpose: there is no branch to be on, and a detached HEAD
    # makes an accidental `git pull` in this tree fail instead of silently
    # moving the toolkit under a half-finished experiment.
    git -C "$S3PRL_DIR" checkout --detach "$S3PRL_COMMIT"
fi

# Enforced on every run, not only on the clone, because the exposure this pin
# closes is a checkout that already exists and has moved -- which is the state
# a `git pull`, a re-clone, or a second machine produces. A mismatch is fatal:
# training on a different S3PRL than the gate was measured on is precisely the
# contamination the pin exists to prevent, and it is silent if allowed through.
S3PRL_HEAD="$(git -C "$S3PRL_DIR" rev-parse HEAD)"
if [ "$S3PRL_HEAD" != "$S3PRL_COMMIT" ]; then
    echo "ERROR: S3PRL checkout is not at the pinned commit." >&2
    echo "  expected: $S3PRL_COMMIT" >&2
    echo "  found:    $S3PRL_HEAD" >&2
    echo "  checkout: $S3PRL_DIR" >&2
    echo "" >&2
    echo "S3PRL supplies the featurizer, the downstream model and the training" >&2
    echo "loop, so results from this tree are not comparable with the week-3" >&2
    echo "gate (F1 = 0.4415) until it is back on the pin:" >&2
    echo "  git -C $S3PRL_DIR checkout --detach $S3PRL_COMMIT" >&2
    echo "" >&2
    echo "If the move was deliberate, change S3PRL_COMMIT in this script and" >&2
    echo "re-run the gate -- do not just silence this check." >&2
    exit 1
fi
# Tracked modifications are a warning rather than an error: `pip install -e`
# means any hand-edit here is already live in every run, so the useful thing is
# to say so loudly at the start of a job rather than to refuse to start one.
S3PRL_DIRTY="$(git -C "$S3PRL_DIR" status --porcelain -uno)"
if [ -n "$S3PRL_DIRTY" ]; then
    echo "WARNING: S3PRL checkout has uncommitted changes to tracked files." >&2
    echo "The commit pin does NOT describe this tree. Changes:" >&2
    echo "$S3PRL_DIRTY" >&2
fi
echo "S3PRL pinned at $S3PRL_COMMIT (upstream v0.4.18)"
python -c "import s3prl" 2>/dev/null || pip install -e "${S3PRL_DIR}[all]"
python -c "import datasets" 2>/dev/null || pip install datasets
python -c "import joblib, tqdm, pandas, numpy" 2>/dev/null || pip install joblib tqdm pandas numpy
# The `hf_*_custom` upstreams (s3prl/upstream/hf_{hubert,wav2vec2}/expert.py)
# import HubertModel/Wav2Vec2Model/Wav2Vec2FeatureExtractor at module load, so
# `transformers` must be present or training dies the moment the upstream is
# constructed. s3prl's [all] extra normally pulls it in; make that explicit
# rather than assumed, since discovering it is missing costs a queue slot.
python -c "import transformers" 2>/dev/null \
    || pip install "transformers==$PIN_TRANSFORMERS"

# --- Verify the pins (week-07 task 1) --------------------------------------
# `import X || pip install X` above is an "install if absent" idiom, which
# cannot pin anything: a venv that already holds the wrong version is never
# corrected and never complained about. The pins are therefore *verified*
# rather than installed, and a mismatch on the three that determine the
# computation is fatal.
#
# Why fatal rather than auto-corrected: reinstalling torch under a job that is
# about to use a GPU is how you get a half-written environment and a confusing
# CUDA error two hours in. The script says exactly what to run and stops.
python - <<'PY'
import importlib
import os
import sys

# (module, env var holding the pin, fatal?)
PINS = [
    ("torch", "PIN_TORCH", True),
    ("torchaudio", "PIN_TORCHAUDIO", True),
    ("transformers", "PIN_TRANSFORMERS", True),
    ("numpy", "PIN_NUMPY", False),
    ("safetensors", "PIN_SAFETENSORS", False),
]

bad = []
for name, env, fatal in PINS:
    want = os.environ.get(env, "")
    if not want:
        continue
    try:
        got = importlib.import_module(name).__version__
    except Exception as exc:
        print("  %-13s MISSING (%s)" % (name, type(exc).__name__))
        bad.append((name, want, "missing", fatal))
        continue
    if got == want:
        print("  %-13s %s" % (name, got))
    else:
        print("  %-13s %s  != pin %s" % (name, got, want))
        bad.append((name, want, got, fatal))

if not bad:
    sys.exit(0)

# The per-package report above goes to stdout and the complaint below to
# stderr. Flush first, or an interactive run prints the remedy before the
# problem it is a remedy for.
sys.stdout.flush()
fatal = [b for b in bad if b[3]]
for name, want, got, is_fatal in bad:
    label = "ERROR" if is_fatal else "WARNING"
    print("%s: %s is %s, pinned at %s." % (label, name, got, want), file=sys.stderr)
if fatal:
    print("", file=sys.stderr)
    print("These versions determine the numbers this repository reports, so a", file=sys.stderr)
    print("silent drift makes every trained checkpoint unreproducible. Fix with:", file=sys.stderr)
    print(
        "  pip install " + " ".join("%s==%s" % (n, w) for n, w, _, f in fatal if f),
        file=sys.stderr,
    )
    print("", file=sys.stderr)
    print("If the move was deliberate, change the PIN_* values in", file=sys.stderr)
    print("run/prepare_baseline.sh and record it in insights/ -- do not just", file=sys.stderr)
    print("silence this check.", file=sys.stderr)
    sys.exit(1)
PY

echo "== [3/8] Download Iqra_train + Iqra_TTS =="
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

if [ "$BUILD_CONDITIONED" = "1" ]; then
    echo "== [3b/8] Iqra_TTS canonicals (phoneme_ref) for the conditioned arm =="
    # Before the merge in step 4, so the canonicals travel with the audio and
    # transcripts in one copy rather than needing a second pass later.
    #
    # Iqra_train needs nothing here: download_hugg_data.py writes phoneme_ref as
    # the transcript, so for those rows the canonical and the CTC target are the
    # same string and build_canonical_manifest.py reads it from the manifest.
    # Only Iqra_TTS, whose transcript is phoneme_mis, needs a separate file.
    if [ ! -d "$DATA_DIR/TTS/train/canonicals" ] \
        || [ -z "$(ls -A "$DATA_DIR/TTS/train/canonicals" 2>/dev/null)" ]; then
        python "$BASELINE_SRC/dump_tts_canonicals.py" \
            --path "IqraEval/Iqra_TTS" --split "train" \
            --output_dir "$DATA_DIR/TTS" --dev_name "Amer"
    else
        echo "canonicals already present under $DATA_DIR/TTS/train/canonicals -- skipping"
    fi
fi

echo "== [4/8] Merge TTS into the combined 131h train split =="
# The 131h training set (per docs/weeks/week-02.md) is Iqra_train (79h) +
# Iqra_TTS (52h) combined. The bucketing step below treats whatever is under
# CV-Ar/train as one split, so the TTS train audio/transcripts are merged in
# here (hard-linked/copied, not moved, so re-running is idempotent and the
# original TTS download stays intact).
if [ -d "$DATA_DIR/TTS/train/wav" ]; then
    find "$DATA_DIR/TTS/train/wav" -name '*.wav' -exec cp -n {} "$DATA_DIR/CV-Ar/train/wav/" \;
    find "$DATA_DIR/TTS/train/transcripts" -name '*.txt' -exec cp -n {} "$DATA_DIR/CV-Ar/train/transcripts/" \;
    # The canonicals ride along in the same merge, so they cannot end up in a
    # different directory from the audio they describe. Absent unless
    # dump_tts_canonicals.py has run (step 7, BUILD_CONDITIONED=1); harmless
    # when present, since nothing but the conditioned arm reads them.
    if [ -d "$DATA_DIR/TTS/train/canonicals" ]; then
        mkdir -p "$DATA_DIR/CV-Ar/train/canonicals"
        find "$DATA_DIR/TTS/train/canonicals" -name '*.txt' \
            -exec cp -n {} "$DATA_DIR/CV-Ar/train/canonicals/" \;
    fi
fi

echo "== [5/8] Bucketing metadata + TSV conversion =="
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

echo "== [6/8] Contamination holdout (APPLY_HOLDOUT=$APPLY_HOLDOUT) =="
# Which TSVs step 7 wires into the config. Filtering writes siblings rather
# than editing in place, so the unfiltered manifests stay on disk and a
# reproduction run and an experiment run can share one 131h data directory.
TRAIN_TSV="$DATA_DIR/CV-Ar/train/train.tsv"
DEV_TSV="$DATA_DIR/CV-Ar/dev/dev.tsv"
if [ "$APPLY_HOLDOUT" = "1" ]; then
    HOLDOUT_JSON="${HOLDOUT_JSON:-$RUN_DIR/training_holdout.json}"
    if [ ! -f "$HOLDOUT_JSON" ]; then
        echo "ERROR: APPLY_HOLDOUT=1 but $HOLDOUT_JSON does not exist." >&2
        echo "Build it first (needs the Iqra_train Arrow cache, no GPU):" >&2
        echo "  uv run python scripts/build_training_holdout.py --write" >&2
        exit 1
    fi
    # The filter is this project's code, not S3PRL's, so it runs on this
    # project's Python 3.12 -- not the 3.8 venv that is currently active. It is
    # a one-shot TSV rewrite outside the training loop, so there is no cost to
    # shelling out. `uv run` is preferred; the PYTHONPATH fallback exists
    # because the filter imports nothing but the standard library, and a
    # cluster node without a synced project env should not block a run.
    if command -v uv >/dev/null 2>&1; then
        HOLDOUT_PY=(uv run --project "$REPO_ROOT" python)
    else
        echo "WARNING: uv not found; running the filter with PYTHONPATH=src." >&2
        HOLDOUT_PY=(env "PYTHONPATH=$REPO_ROOT/src" python3)
    fi
    # Both splits. Dev is not an afterthought: the MSA evaluation pool *is* 500
    # Iqra_train dev rows, so leaving them in dev means `dev-best` selection is
    # performed on the evaluation set -- the exact objection docs/msa-arm.md
    # 5.2 raises against the organizers' published checkpoint. It costs 19.5%
    # of dev (505 of 2,588 rows, 2,083 left for selection), against 0.53% of
    # train. The dev cost is the real one and is reported, not absorbed.
    for split in train dev; do
        # The report goes to $RUN_DIR, not next to the TSV: $DATA_DIR holds 131h
        # of audio and is not tracked, and the drop counts are the evidence that
        # this run was actually held out. They belong where they can be committed.
        "${HOLDOUT_PY[@]}" "$REPO_ROOT/scripts/apply_holdout.py" \
            --manifest "$DATA_DIR/CV-Ar/$split/$split.tsv" \
            --split "$split" \
            --holdout "$HOLDOUT_JSON" \
            --report "$RUN_DIR/holdout_applied_$split.json"
    done
    TRAIN_TSV="$DATA_DIR/CV-Ar/train/train.heldout.tsv"
    DEV_TSV="$DATA_DIR/CV-Ar/dev/dev.heldout.tsv"
else
    echo "holdout NOT applied -- this run trains on the organizers' full set,"
    echo "including the rows that overlap both evaluation pools. Correct for the"
    echo "week-3 reproduction (docs/weeks/week-03.md); wrong for anything we report"
    echo "as our own system (docs/weeks/week-07.md task 4)."
fi
echo "train manifest: $TRAIN_TSV"
echo "dev manifest:   $DEV_TSV"

echo "== [7/8] Downstream task config + vocab =="
mkdir -p "$S3PRL_DIR/s3prl/downstream/ctc/cv_vocab" "$S3PRL_DIR/s3prl/downstream/ctc/cv_config"
cp "$BASELINE_SRC/vocab/sws_arabic.txt" "$S3PRL_DIR/s3prl/downstream/ctc/cv_vocab/sws_arabic.txt"
# Materialize the vendored config with real paths substituted -- the
# vendored copy keeps the organizers' placeholder paths so it stays a
# verbatim reference; this generated copy is what actually gets used.
sed \
    -e "s#'./sws_data/'#'$DATA_DIR/CV-Ar/'#" \
    -e "s#path_to_tsv_file/train/train.tsv#$TRAIN_TSV#" \
    -e "s#path_to_tsv_file/dev/dev.tsv#$DEV_TSV#" \
    "$BASELINE_SRC/config/sws.yaml" > "$S3PRL_DIR/s3prl/downstream/ctc/cv_config/sws.yaml"

if [ "$BUILD_CONDITIONED" = "1" ]; then
    echo "== [7b/8] Three-column manifest + downstream/ctc_cond =="

    # Same Python split as the holdout filter: this is our code, so it runs on
    # the project's 3.12, not the 3.8 venv that is currently active.
    if command -v uv >/dev/null 2>&1; then
        COND_PY=(uv run --project "$REPO_ROOT" python)
    else
        echo "WARNING: uv not found; building the manifest with PYTHONPATH=src." >&2
        COND_PY=(env "PYTHONPATH=$REPO_ROOT/src" python3)
    fi

    # Built from $TRAIN_TSV/$DEV_TSV -- i.e. from whichever manifest step 6
    # selected -- so the holdout and the conditioning cannot disagree about
    # which rows exist. Exits nonzero if any Iqra_TTS row has no canonical,
    # rather than substituting its `sentence`; see the script's docstring.
    COND_TRAIN_TSV="${TRAIN_TSV%.tsv}.cond.tsv"
    COND_DEV_TSV="${DEV_TSV%.tsv}.cond.tsv"
    "${COND_PY[@]}" "$REPO_ROOT/scripts/build_canonical_manifest.py" \
        --manifest "$TRAIN_TSV" --data-root "$DATA_DIR/CV-Ar" \
        --output "$COND_TRAIN_TSV" --report "$RUN_DIR/canonical_manifest_train.json" \
        --vocab "$BASELINE_SRC/vocab/sws_arabic.txt"
    "${COND_PY[@]}" "$REPO_ROOT/scripts/build_canonical_manifest.py" \
        --manifest "$DEV_TSV" --data-root "$DATA_DIR/CV-Ar" \
        --output "$COND_DEV_TSV" --report "$RUN_DIR/canonical_manifest_dev.json" \
        --vocab "$BASELINE_SRC/vocab/sws_arabic.txt"
    echo "conditioned train manifest: $COND_TRAIN_TSV"
    echo "conditioned dev manifest:   $COND_DEV_TSV"

    # Materialized rather than committed into the checkout, for the same reason
    # the vocab and config above are: the sha pin's provenance story depends on
    # the tracked tree staying clean, and step 2's dirty-tree check uses
    # `git status --porcelain -uno`, which ignores untracked files. So an
    # untracked `downstream/ctc_cond/` is invisible to it by design.
    mkdir -p "$S3PRL_DIR/s3prl/downstream/ctc_cond/cv_config"
    cp "$BASELINE_SRC/ctc_cond/__init__.py" \
       "$BASELINE_SRC/ctc_cond/data.py" \
       "$BASELINE_SRC/ctc_cond/expert.py" \
       "$S3PRL_DIR/s3prl/downstream/ctc_cond/"
    sed \
        -e "s#'./sws_data/'#'$DATA_DIR/CV-Ar/'#" \
        -e "s#path_to_tsv_file/train/train.tsv#$COND_TRAIN_TSV#" \
        -e "s#path_to_tsv_file/dev/dev.tsv#$COND_DEV_TSV#" \
        "$BASELINE_SRC/config/sws_cond.yaml" \
        > "$S3PRL_DIR/s3prl/downstream/ctc_cond/cv_config/sws_cond.yaml"

    # The only thing enforcing src/arabic_mdd/models/'s Python 3.8 ceiling.
    # `ctc_cond/expert.py` imports MddModel from there so that the arms the
    # cluster trains are the arms tests/models/ tests; a 3.10-only construct
    # slipping into that package would otherwise surface as a SyntaxError
    # several minutes into a queued job.
    echo "-- checking arabic_mdd.models imports under $(python --version 2>&1) --"
    PYTHONPATH="$REPO_ROOT/src" python - <<'PY'
import sys

try:
    from arabic_mdd.models.mdd import MddConfig, MddModel
except SyntaxError as exc:
    print("", file=sys.stderr)
    print("ERROR: src/arabic_mdd/models/ is not importable on this Python.", file=sys.stderr)
    print("  %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
    print("", file=sys.stderr)
    print("That package is held to Python 3.8 syntax on purpose: S3PRL needs 3.8,", file=sys.stderr)
    print("and downstream/ctc_cond/expert.py imports MddModel from it so the two", file=sys.stderr)
    print("arms are one tested implementation rather than two. The usual culprit", file=sys.stderr)
    print("is zip(..., strict=True). See that package's mdd.py docstring and the", file=sys.stderr)
    print("per-file B905 ignore in pyproject.toml.", file=sys.stderr)
    sys.exit(1)

model = MddModel(MddConfig(vocab_size=71, use_canonical=True))
total = sum(p.numel() for p in model.parameters())
conditioning = sum(p.numel() for p in model.conditioning_parameters)
print("  ok: MddModel builds, %d params (%d conditioning)" % (total, conditioning))
PY
fi

echo "== [8/8] Pin the frozen upstream to a commit sha =="
# s3prl's `--upstream_revision` does NOT apply to `-u hf_hubert_custom`
# (it is read only on the `--upstream`-is-an-HF-repo path, runner.py:125),
# so the upstream is pinned by materializing it at a fixed sha and passing
# the resulting directory to `-k`. See the module docstring of
# pin_upstream.py for the full argument.
#
# `UPSTREAM_PIN` names a registered upstream in pin_upstream.py's UPSTREAMS,
# not a repo id, because the `-k` directory and the `-u` name have to agree:
# `hf_hubert_custom` builds a HubertModel and cannot load XLS-R,
# `hf_wav2vec2_custom` builds a Wav2Vec2Model and cannot load mHuBERT. The
# registry owns that pairing so no caller can get it half right.
UPSTREAM_PARENT="${UPSTREAM_PARENT:-$RUN_DIR/upstreams}"
UPSTREAM_PIN="${UPSTREAM_PIN:-mhubert147}"
UPSTREAM_DIR="$(python "$BASELINE_SRC/pin_upstream.py" \
    --upstream "$UPSTREAM_PIN" --output-dir "$UPSTREAM_PARENT" | tail -1)"
S3PRL_UPSTREAM="$(python "$BASELINE_SRC/pin_upstream.py" \
    --upstream "$UPSTREAM_PIN" --print-s3prl-upstream)"
export UPSTREAM_DIR S3PRL_UPSTREAM
echo "UPSTREAM_PIN=$UPSTREAM_PIN"
echo "UPSTREAM_DIR=$UPSTREAM_DIR"
echo "S3PRL_UPSTREAM=$S3PRL_UPSTREAM"
# Written to disk as well, because the training scripts run in a sibling shell
# and should not have to redo the resolution (which touches the network). One
# file per pin, keyed by name rather than "whatever ran last" -- otherwise an
# XLS-R prepare followed by a baseline submit hands mHuBERT's `-u` an XLS-R
# directory, and that is a wasted queue slot with a confusing traceback.
cat > "$RUN_DIR/.upstream_$UPSTREAM_PIN.env" <<EOF
# Generated by run/prepare_baseline.sh step 8. Do not edit; re-run instead.
UPSTREAM_PIN=$UPSTREAM_PIN
UPSTREAM_DIR=$UPSTREAM_DIR
S3PRL_UPSTREAM=$S3PRL_UPSTREAM
EOF
# Legacy single-file form, still read by anything written before week 7.
echo "$UPSTREAM_DIR" > "$RUN_DIR/.upstream_dir"

echo "Baseline reproduction data/environment ready. S3PRL_DIR=$S3PRL_DIR"
