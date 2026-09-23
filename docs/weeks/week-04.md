# Week 4 implementation plan

Dates: 12–18 Oct 2026. Opens the week 4–6 block in
[mdd-paper-project-plan-v2.md](../mdd-paper-project-plan-v2.md) §6:
"diacritizer pipeline, all tools; RQ1 measurements on both domains; confirm
CATT exposes token probabilities; begin annotation; run the full label-path
column on the Qur'anic arm." See [week-03.md](week-03.md) for the closed
baseline gate and [insights/week-03.md](../../insights/week-03.md) for what
that block validated — and, importantly, for the section "Carried into week
4+: what this gate does **not** cover", which sets this week's priorities.

## Where week 3 left things

The week 1–3 gate passed three independent ways, agreeing within 0.001 F1 of
the published 0.4414: the organizers' checkpoint scored through our pipeline
(0.4415), and our own from-scratch 200k-step retrain (0.4406). The
evaluation pipeline *and* the training recipe are validated as a unit.

What that does **not** validate is the text→phoneme path this week builds.
The gate consumed phoneme strings that shipped with the dataset; from week 4
onward we generate them, and nothing published exists to check them against.

## Scope

Week 4 is the **text side** of the pipeline: diacritizer wrappers and a
phonetizer, behind tested interfaces, plus the inventory check that decides
whether the MSA arm is even measurable with the current vocab. No RQ
measurements, no annotation, no MDD retraining.

Provisional split of the block, to keep this week bounded:

| Week | Work |
|---|---|
| **4** | Diacritizer wrappers + phonetizer + inventory diff |
| 5 | RQ1 measurements on both domains; begin annotation |
| 6 | Label-path column on the Qur'anic arm; RQ5 interaction power estimate (**block gate**) |

## Week 4 exit criterion

**The phonetizer round-trips QuranMB.v2.** The dataset ships both
`reference_arabic_string` (gold-diacritized Arabic) and
`reference_phoneme_string` (the canonical phoneme sequence the baseline was
scored against). So the phonetizer has an exact, free, 1,642-utterance
reference:

```
phonetize(reference_arabic_string) == reference_phoneme_string
```

Target ≥99% exact sequence match, with every mismatch inspected and
explained rather than tolerated. This is the cheapest possible validation of
the single riskiest new component, and it exists only on the Qur'anic arm —
which is precisely why it must be spent here rather than discovered missing
on the MSA arm.

If the round-trip cannot be made to work, the label-path decomposition that
is the paper's spine cannot be computed correctly, and that is a week-4
finding rather than a week-6 surprise.

## Tasks

1. **Diacritizer wrappers** under `src/arabic_mdd/diacritizers/`, one
   module per tool behind a common interface: CATT (EO and ED), Shakkala,
   and Mishkal or Farasa. Plan §3 wants the degradation plotted as a *curve*
   against diacritizer quality, so the weak points matter as much as CATT
   does. Tests per `CLAUDE.md`; mock or cache tool output so the suite stays
   offline and fast.
2. **Vendor the CATT token-probability extraction** into that same package,
   with its own tests. Week 1 established feasibility through undocumented
   internals and explicitly deferred the production interface to this block
   (`insights/week-01-03.md`). RQ4 depends on it entirely. Confirm the same
   technique works on the **ED** checkpoint before committing to a variant —
   ED is the more accurate one per CATT's own numbers.
3. **Phonetizer**, `src/arabic_mdd/data/phonemes.py` or a sibling: diacritized
   Arabic → phoneme sequence in the 68-token `sws_arabic.txt` inventory.
   Validate by the round-trip above.
4. **Phoneme-inventory diff — the risk (a) mitigation.** Run the phonetizer
   over Common Voice Arabic transcripts diacritized by each tool, and diff
   the resulting phoneme inventory against the 68-token vocab. Any phoneme
   the CTC head cannot emit becomes a *guaranteed* false rejection for a
   mechanical reason, which is indistinguishable in the metric from
   "diacritization corrupts MDD" — the paper's own claim. Decide explicitly:
   extend the vocab (breaking cross-arm comparability), or map/drop (losing
   coverage). Record the choice and its justification; it goes in the paper's
   methods, not its rebuttal.
5. **NAACL 2024 audio-informed diacritic restoration, one hour.** Plan §Data:
   if audio-informed restoration beats text-only CATT on Common Voice, it is
   the better correct-the-machine baseline *and* the anchoring bias points
   away from the system under test rather than toward it. This decides the
   annotation starting point and must happen before week 5's annotation
   begins.
6. **Chase the 2026 diacritizer survey** ("Evaluating Arabic Diacritization
   Models: Self-hosted to Commercial"), currently an OpenAlex record with no
   venue or text retrieved. Plan §Open questions wants this resolved before
   the diacritizer set is finalised — which is this week.
7. **Pin `utter-project/mHuBERT-147` to a commit sha.** Both the organizers'
   run and ours took whatever the Hub served that day
   (`upstream_revision: None`). Cheap now; it keeps the validated recipe
   validated across every week 4+ training run.

## Carried forward

- Licence-terms confirmation for `QuranMB.v2`/`Iqra_Extra_IS26` — open since
  week 2. Common Voice is CC0 and the annotated slice is a releasable
  artifact in its own right (plan §Data), so this now gates a deliverable,
  not just diligence.
- Wall-clock cost of one XLS-R-300m fine-tune and the week-7 work-split
  decision — open since week 1. Week 7 is when the two arms converge and
  need an assigned owner, so this cannot slip past week 6.
- `scripts/verify_quranmb_labels.py` — optional. The official checkpoint
  reproducing 0.4415 on our labels settled the provenance question in
  aggregate; the residual value is per-row, plus identifying the one
  utterance by which the `safikhan` join (1,642) differs from the published
  count (1,643).

## Out of scope for week 4

- RQ1 measurements (week 5) and the label-path column (week 6).
- Any annotation work — task 5 decides its starting point, but the protocol
  must be pre-registered in writing before a single utterance is annotated.
- Training any MDD model. The validated `mhubert147_per` recipe stands; the
  XLS-R prompt-free and text-dependent arms are weeks 7–9.
- Revisiting the paper's framing. Pre-committed in plan §5 and explicitly not
  to be reopened under deadline pressure.
