# Week 1 implementation plan

Dates: 21–27 Sep 2026. Part of the week 1–3 block in
[mdd-paper-project-plan-v2.md](../mdd-paper-project-plan-v2.md) §6, whose gate
(reproduce the mHuBERT baseline at F1 ≈ 0.4414 ± 0.02) falls due at the end of
week 3, not week 1. This plan covers only what §7 of that document assigns to
week 1 itself.

Log outcomes in [insights/week-01.md](../../insights/week-01.md) as they land.

## Scope

Week 1 is setup and access, not modeling. Nothing here should require a
trained model or a downloaded corpus to have finished — those start once
access is confirmed (item 3) and continue into weeks 2–3.

## Tasks (from plan §7)

1. **Differentiation text.** Read Kadambi et al. 2024 and Mathad et al. 2021
   (already done per the user), then write the 2–3 paragraphs distinguishing
   window-shifting (alignment) corruption from label-shifting (diacritic)
   corruption. Draft in `docs/differentiation-kadambi-mathad.md`; this text is
   destined for the introduction and related work.
2. **Pre-committed effect-size position.** One paragraph, signed off by both
   students, committing in advance to report a small effect honestly rather
   than reframe under deadline pressure. Draft in
   `docs/effect-size-position.md`.
3. **Dataset access.** Create the HF account, request access to the IqraEval
   org datasets, confirm licence terms on `Iqra_Extra_IS26` and
   `QuranMB.v2`, and establish whether v2 test labels are public or
   leaderboard-held. Blocking — has latency outside our control, start first.
4. **CATT token-probability check.** Confirm CATT
   ([github.com/abjadai/catt](https://github.com/abjadai/catt)) exposes
   token-level confidence through its public interface. RQ4 depends on this
   entirely; if it does not expose probabilities, patch the inference code or
   pick a substitute diacritizer now. This is the one concrete technical
   experiment in week 1 — log it in `insights/week-01.md` using the
   Hypothesis → Methodology → Configuration → Results → Observations →
   Conclusion structure.
5. **Cluster confirmation.** Confirm A100 cluster allocation and queue
   limits; estimate wall-clock cost of one XLS-R-300m fine-tuning run, since
   the 21-week timeline assumes several are affordable.
6. **QuranMB v1/v2 overlap.** Resolve the leaderboard-row-matches-a-2025-
   baseline oddity noted in the deep dive. Does not change the experimental
   design either way, but the framing needs to be accurate.
7. **Work split.** One student owns the model/training pipeline, the other
   owns the diacritization pipeline and annotation. Record the split in
   `insights/week-01.md`; both converge at week 7.
8. **Residual novelty-check items.** Arabic-language queries (تشكيل + تقييم
   النطق, كشف أخطاء النطق + التشكيل الآلي, plus Arabic thesis repositories) and
   the manual "cited by" traversal on CATT, Halabi & Wald, and Kadambi 2024.
   Record results directly in [novelty-check.md](../novelty-check.md), per its
   own instructions.

## Out of scope for week 1

- Implementing the hierarchical TA/TR/FA/FR metric (week 1–3 block, but not
  gated until week 3).
- Reproducing the mHuBERT baseline (week 3 gate).
- Any diacritizer pipeline code beyond the CATT probability check (week 4–6).
- Model training code (week 7+).

## Repository foundation (this session)

The project skeleton, tooling, and CI/pre-commit setup were established
ahead of item 3–4 so that dataset and modeling code have somewhere to land
immediately once access is confirmed. See [CLAUDE.md](../../CLAUDE.md) for
the resulting structure and conventions.
