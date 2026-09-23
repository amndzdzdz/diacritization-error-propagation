# Week 2 — Metric, data pipeline, baseline reproduction materials

Plan: [docs/weeks/week-02.md](../docs/weeks/week-02.md). Update this file as
each task below produces a result — do not wait until the end of the week.

## Logistics log

Track non-experimental tasks here as they close. One line each, dated.

- [x] Hierarchical TA/TR/FA/FR/CD/ED metric implemented,
      `src/arabic_mdd/metrics/hierarchical.py`, 20 Sep 2026 — see experiment
      below for validation.
- [x] Data loaders implemented, `src/arabic_mdd/data/{phonemes,quranmb,iqra_train}.py`,
      20 Sep 2026, with fixture-based tests in `tests/data/`.
- [x] `QuranMB.v2` label-gating question (open since week 1) resolved, 20 Sep
      2026: `IqraEval/QuranMB.v2` ships only `ID`/`audio` — canonical
      reference + human annotation live in a separate gated dataset,
      `IqraEval/IqraEval_Test_GT`. User confirmed access to it directly.
      Rather than joining the two gated/public datasets ourselves,
      `src/arabic_mdd/data/quranmb.py` reads from `safikhan/quran_mbv2_formatted`
      (public HF dataset the user found, which already performs that join:
      audio + `reference_phoneme_string` + `annotation_phoneme_string` +
      a recovered Arabic reference text). Explicit licence-terms
      confirmation for `QuranMB.v2`/`Iqra_Extra_IS26` itself is still open,
      carried to week 3.
- [x] S3PRL baseline reproduction materials vendored,
      `scripts/baseline_reproduction/` (config, vocab, data-prep/training
      helper scripts, README with exact ordered cluster commands), 20 Sep
      2026 — copied verbatim from `github.com/Iqra-Eval/interspeech_IqraEval`,
      not reimplemented, per the week 2 decision (ambiguous-gate-result
      risk of a from-scratch reimplementation). Paper-vs-config discrepancy
      (12.5k updates prose vs. `total_steps: 200000` in the shipped config)
      flagged explicitly in that README; config treated as source of truth.
- [x] Schema-drift bug found and fixed in the vendored download scripts,
      20 Sep 2026: `download_hugg_data.py`/`download_hugg_data_tts.py`, as
      fetched from upstream, read column names (`ID`/`phoneme`/`phoneme_aug`)
      and default `--path` values from different, older datasets than
      `IqraEval/Iqra_train`/`IqraEval/Iqra_TTS` — the ones the organizers'
      own README instructs running them against. Verified the real schemas
      via `datasets-server.huggingface.co` (`id`/`phoneme_ref` for
      Iqra_train, `phoneme_mis` for Iqra_TTS) and patched the vendored
      copies to match; documented as a deliberate, commented deviation from
      "verbatim" in the README, since it only affects column-name mapping
      during data materialization, not the training algorithm/config.
      Caught by reading the vendored scripts closely before automating
      around them, not by a failed run — worth remembering that "vendored
      from the official repo" isn't the same as "verified compatible with
      the dataset as currently published."
- [x] Full pipeline automated end-to-end into a single SLURM entry point,
      `run/train_baseline.slurm` (calls the idempotent
      `run/prepare_baseline.sh`: conda/S3PRL setup, dataset download,
      combined-131h-set merge, bucketing, config wiring), 20 Sep 2026, per
      user request that submitting the job require no manual pre-steps.
      Added a non-interactive `--splits` flag to
      `generate_len_for_bucket_sdaia.py` since its original interactive
      `input()` prompt would hang a batch job. Not yet run — that's the
      user's cluster-side job, week 3.
- [x] First real submission (`sbatch run/train_baseline.slurm`) failed
      immediately, 20 Sep 2026: `conda: command not found`. Cause: SLURM
      batch jobs run a non-interactive shell, so `~/.bashrc`'s conda-init
      block (guarded by `[ -z "$PS1" ] && return` or similar) never runs,
      and `conda info --base` — which the script used to locate conda's
      shell hook — fails before conda is even findable. Fixed by adding
      `run/find_conda.sh`, sourced by both `prepare_baseline.sh` and
      `train_baseline.slurm`, which checks `CONDA_BASE_OVERRIDE`, an
      optional `CONDA_MODULE` to `module load`, `conda` already on PATH,
      then common install locations, before giving up with an actionable
      error. (A separate `Unable to create TMPDIR` warning in the same
      failed run is a harmless SLURM message, not related to this bug.)

## Experiment: Hierarchical MDD metric port validation

### Hypothesis

The hierarchical TA/TR/FA/FR/CD/ED metric, re-derived from the official
repo's `mdd_eval/{metric.py,align_data.py,ins_del_cor_sub_analysis.py}`
(three pairwise Needleman-Wunsch alignments — canonical-vs-annotation,
annotation-vs-prediction, canonical-vs-prediction — merged position-by-
position), can be implemented independently in this project's own code and
made to match the official algorithm's classification logic exactly,
verifiable against small hand-constructed cases before ever being trusted
on real baseline output.

### Methodology

Implemented `src/arabic_mdd/metrics/hierarchical.py` from the formulas in
the official repo (read via GitHub, not copied — cited in the module
docstring for traceability). Wrote `tests/metrics/test_hierarchical.py`
with canonical/human-annotation/prediction triples covering every
classification branch, each with expected TA/TR/FA/FR/CD/ED counts computed
by hand via full manual Needleman-Wunsch traces before running the code, so
the test would catch a wrong port rather than just confirm "it runs."

### Configuration

- 12 test cases: exact match, false rejection, correctly-diagnosed
  substitution, wrongly-diagnosed substitution, missed substitution
  (false acceptance), correctly-detected deletion, missed deletion,
  correctly-diagnosed insertion, a multi-error utterance, aggregation
  across utterances, and the `evaluate()` convenience wrapper.
- `uv run pytest tests/metrics/`, `uv run ruff check .`, `uv run ruff format .`

### Results

All 12 tests passed against hand-computed expected values. Two arithmetic
errors were caught in my own hand-derivations while writing the *data
loader* tests later in the week (`tests/data/test_quranmb.py`) — I initially
under-counted `TA` for fully-correct multi-phoneme utterances (each
correctly-pronounced-and-predicted phoneme contributes its own `TA`, not one
`TA` per utterance) — fixed before those tests were committed. The
hierarchical metric module's own tests were correct on the first pass.

### Observations

Re-deriving from formulas rather than transcribing the official code
surfaced a real subtlety not obvious from the formulas alone: the two
independently-computed alignments that share one underlying sequence (e.g.
canonical-vs-annotation and canonical-vs-prediction both containing the
canonical sequence) need a two-pointer resync step to merge correctly when
insertions/deletions cause the two alignments' indices to drift apart. This
is implicit in the official repo's file-based, offset-tracking implementation
but had to be made explicit and tested here.

### Conclusion

**Metric port validated — trusted for week 3 gate scoring.** No blockers.
Next: feed real S3PRL predictions through `score_predictions` +
`compute_metrics` once the user's cluster job (week 3) produces them.
