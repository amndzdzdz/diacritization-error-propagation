# Week 2 implementation plan

Dates: 28 Sep – 4 Oct 2026. Part of the week 1–3 block in
[mdd-paper-project-plan-v2.md](../mdd-paper-project-plan-v2.md) §6, whose gate
(reproduce the mHuBERT baseline at F1 ≈ 0.4414 ± 0.02 on `QuranMB.v2`) falls
due at the end of week 3. Week 1–3 share one plan-level task list ("cluster
setup; pull `Iqra_train`, `Iqra_TTS`, `QuranMB.v2`; implement the official
hierarchical metric; reproduce the mHuBERT baseline") and one gate — this
file and `week-03.md` split that block into what happens in-session
(code, testable without a GPU) versus what only the user can run
(the actual training job, on their cluster). See
[insights/week-02.md](../../insights/week-02.md) for the running log.

## Scope

Week 2 produces everything needed *before* the cluster training job can
run: the hierarchical MDD metric (so results can be scored once produced),
the data loaders (so the training/eval scripts have something to point
at), and the vendored S3PRL reproduction materials (so the user has exact,
ready-to-run commands). No model is trained this week; week 3 covers the
actual run and the gate check.

## Baseline reproduction strategy (decided this week)

The published organizer baseline (mHuBERT frozen + BiLSTM + CTC, F1 =
44.14% on `QuranMB.v2`, from the 2025 Iqra'Eval shared task per week 1's
citation fix) was trained with
[S3PRL](https://github.com/s3prl/s3prl) via the shared task's own
companion repo, `github.com/Iqra-Eval/interspeech_IqraEval`. S3PRL requires
Python 3.8, and ships its own checkpoint format — incompatible with this
project's Python-3.12/uv conventions.

Decision (confirmed with the user): reproduce the baseline using S3PRL
directly, in its own isolated Python 3.8 environment (a plain stdlib
`venv`, not conda or uv — the user's cluster has neither installed nor
installable), rather than reimplementing the mHuBERT+BiLSTM+CTC pipeline
in this project's own `transformers`/uv stack. A reimplementation that
failed to hit the gate
would be ambiguous — no way to tell whether it's a real finding or an
implementation bug — whereas running the organizers' own code removes
that risk. S3PRL is therefore **not** added to `pyproject.toml`; it lives
entirely under `scripts/baseline_reproduction/`, documented as an
external, unpinned-by-this-project tool dependency, per the `scripts/`
convention in `CLAUDE.md`.

Execution model (also confirmed): this sandbox has no GPU
(`nvidia-smi`: not found; `torch.cuda.is_available()`: False) and no
cluster access. I prepare all code/config/scripts here; the user runs the
actual multi-hour training job on their cluster and reports the resulting
predictions back for scoring (week 3).

Data access: `Iqra_train` and `Iqra_TTS` (the 131h combined training set:
79h Common Voice Arabic + 52h TTS) are confirmed already accessible to the
user, alongside the `QuranMB.v2`/`Iqra_Extra_IS26` access confirmed in
week 1.

## Tasks

1. **Hierarchical TA/TR/FA/FR/CD/ED metric.** Implement
   `src/arabic_mdd/metrics/hierarchical.py`, porting the algorithm found in
   the official repo's `mdd_eval/ins_del_cor_sub_analysis.py` (three
   pairwise Needleman-Wunsch alignments — canonical-vs-prediction,
   canonical-vs-human-annotation, human-annotation-vs-prediction — merged
   position-by-position into TA/TR/FA/FR and, within TR, Correct/Error
   Diagnosis). Expose a typed
   `evaluate(canonical, human_annotation, prediction) -> MDDResult` API
   with precision/recall/F1, FAR, FRR, DER, and Detection_Accuracy fields.
   This is the metric the week-3 gate is scored with, so it must be
   validated against hand-built cases before being trusted on real output
   (task 2).
2. **Metric tests.** `tests/metrics/test_hierarchical.py`: hand-constructed
   canonical/annotation/prediction triples exercising every classification
   branch (exact match, correctly-diagnosed substitution, wrongly-diagnosed
   substitution, insertion, deletion, a multi-error sequence), asserting
   exact TA/TR/FA/FR/CD/ED counts and derived metrics against manually
   computed expected values.
3. **Data loaders.** `src/arabic_mdd/data/`: thin loaders for `QuranMB.v2`
   test set (gold-by-convention canonical + human annotation, via
   `safikhan/quran_mbv2_formatted` — see task 6) and `IqraEval/Iqra_train`
   + `IqraEval/Iqra_TTS` (combined train set), normalizing into a common
   structure the metric module and later model code can consume, plus a
   `score_predictions` helper that runs a system's predictions through
   `hierarchical.evaluate_utterance`/`aggregate` against the QuranMB.v2
   ground truth — this is what week 3 uses to check the gate. Not a full
   training pipeline — that's the user's cluster-side S3PRL job this
   round, not code this project owns yet.
4. **Data loader tests.** `tests/data/`: fixture-based (small synthetic
   data, not the real corpus) tests of loader correctness — column
   presence, phoneme normalization, expected shapes — kept fast and
   offline.
5. **Vendored baseline reproduction materials.**
   `scripts/baseline_reproduction/`: the official repo's `config/sws.yaml`
   training config, S3PRL environment setup commands, and train/eval
   invocation, plus a `README.md` giving the user the exact ordered shell
   commands to run on their cluster (Python 3.8 venv, S3PRL install,
   dataset pull/prep, checkpoint download, train, evaluate, then feed
   output into
   `src/arabic_mdd/metrics/hierarchical.py`). Note explicitly in this
   README the discrepancy between the paper's prose (12.5k updates) and
   the shipped config (`total_steps: 200000`, `eval_step: 5000`) — the
   config is treated as source of truth since it's what actually produced
   the published number.
6. **Licence/access follow-up.** Partially resolved this week: week 1 left
   open whether `QuranMB.v2` test labels are public or leaderboard-held.
   Checked the actual HF schemas — `IqraEval/QuranMB.v2` ships only
   `ID`/`audio`; the canonical reference + human annotation
   (`Reference_phn`/`Annotation_phn`) live in a separate, gated dataset,
   `IqraEval/IqraEval_Test_GT` (distinct from `Iqra_Extra_IS26`, which is
   what week 1 confirmed access to). So: **held separately, access-gated,
   not bundled with the audio.** The user confirmed access to
   `IqraEval_Test_GT` directly. Rather than joining the two gated/public
   datasets ourselves, `src/arabic_mdd/data/quranmb.py` uses
   `safikhan/quran_mbv2_formatted`, a public HF dataset that already
   performs that join (audio + `reference_phoneme_string` +
   `annotation_phoneme_string`, plus a recovered Arabic reference text) —
   this is what the loaders in task 3 actually read from. Explicit
   licence-terms confirmation for `QuranMB.v2`/`Iqra_Extra_IS26` itself is
   still open; carry forward to week 3.

## Out of scope for week 2

- Running the actual training job (week 3, user-executed on the cluster).
- Scoring the gate (week 3, once predictions exist).
- Any diacritizer-pipeline or MDD-model code beyond the metric and data
  loaders above (weeks 4+).

## Handoff to week 3

By the end of week 2, the user should have everything needed in
`scripts/baseline_reproduction/` to start the cluster job without further
back-and-forth. Week 3 covers running it, scoring the result, and the
go/no-go gate decision.
