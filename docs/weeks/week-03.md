# Week 3 implementation plan

Dates: 5–11 Oct 2026. Closes out the week 1–3 block in
[mdd-paper-project-plan-v2.md](../mdd-paper-project-plan-v2.md) §6. This is
**the real go/no-go gate** for the whole project: "if a fully specified
published baseline cannot be reproduced, nothing downstream is
trustworthy" (plan §6). See [week-02.md](week-02.md) for the preceding
setup (metric implementation, data loaders, vendored S3PRL materials) that
this week depends on, and
[insights/week-03.md](../../insights/week-03.md) for the running log.

## Scope

Week 3 is the baseline run and the gate check, not new pipeline code.
Everything code-side needed to run and score the baseline should already
exist from week 2 (`src/arabic_mdd/metrics/hierarchical.py`,
`src/arabic_mdd/data/`, `scripts/baseline_reproduction/`).

## Gate

**F1 ≈ 0.4414 ± 0.02 on `QuranMB.v2`**, reproducing the 2025 Iqra'Eval
shared task's organizer baseline (mHuBERT frozen + BiLSTM + CTC; citation
corrected in week 1 to `2025.arabicnlp-sharedtasks.61`, not the IQRA 2026
leaderboard paper which merely carries this number forward unchanged).

- **Pass:** proceed to week 4–6 (diacritizer pipeline) as planned.
- **Fail:** stop and debug per the plan — do not proceed to build
  downstream experiments on an unverified foundation. Debugging order:
  (1) check the metric implementation against the official repo's output
  on the same predictions, to rule out a scoring bug; (2) check data
  prep/config against `scripts/baseline_reproduction/README.md` for
  drift from the official recipe; (3) only then treat it as a genuine
  reproduction gap and escalate to the user/co-author.

## Tasks

1. **User runs the cluster job.** Per
   `scripts/baseline_reproduction/README.md` (produced in week 2): Python
   3.8 venv + S3PRL install, dataset pull/prep, checkpoint download
   (`mhubert.ckpt`), training (`run_downstream.py -m train ...`),
   evaluation (`run_downstream.py -m evaluate ...`). This step happens
   outside this sandbox — no GPU or cluster access here. I prepared the
   materials in week 2; this is user-executed.
2. **Score the gate.** Once the user reports back predictions (or their
   location), run them through
   `src/arabic_mdd/metrics/hierarchical.py`'s `evaluate()` against
   `QuranMB.v2`'s canonical references to compute F1, and any of
   precision/recall/FAR/FRR/DER/Detection_Accuracy worth recording
   alongside it.
3. **Log the experiment.** `insights/week-03.md`, using the standard
   Hypothesis → Methodology → Configuration → Results → Observations →
   Conclusion structure, stating the pass/fail verdict against the gate
   explicitly. If the shipped config's `total_steps`/`eval_step` diverged
   from the paper's stated update count (flagged in week 2), note whether
   that discrepancy is implicated in the result.
4. **Close out week 1–3 logistics.** Anything still open from weeks 1–2
   (licence-terms confirmation for `QuranMB.v2`/`Iqra_Extra_IS26`, actual
   wall-clock cost of one XLS-R-300m fine-tune run for later weeks'
   budget planning, the work-split decision deferred since week 1) should
   be resolved or explicitly carried forward with a reason by the end of
   this week, since week 7 is when the two arms (model/training vs.
   diacritization/annotation) converge and need an assigned owner.

## Out of scope for week 3

- Any diacritizer-pipeline or MDD-model code (week 4+).
- Building the XLS-R-300m "strong prompt-free" or text-dependent systems
  (later weeks per plan §3's systems table).
- Re-running or extending the novelty check (closed in week 1).

## Contingency if the gate fails

Per the plan's explicit instruction: stop, do not proceed to week 4-6
work, and debug. Document the failure honestly in `insights/week-03.md`
rather than adjusting the gate's target after the fact. If debugging
points to a genuine non-reproduction (not a bug in this project's scoring
or data prep), escalate to the user for a decision on how to proceed
before any further code is written.
