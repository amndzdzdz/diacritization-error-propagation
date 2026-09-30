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

Everything this week feeds the MSA arm, whose constraints and failure modes
are collected once in [docs/msa-arm.md](../msa-arm.md) — read §4 (inventory
mismatch) and §8 (build order) before starting tasks 3–4.

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
the single riskiest new component.

**Second reference, added 2026-09-23 — and it covers MSA orthography.**
This criterion originally noted that the round-trip "exists only on the
Qur'anic arm". That turned out to be wrong. `IqraEval/Iqra_train` ships a
`tashkeel_sentence` column (the organizers' in-house vowelizer's output,
fully diacritized in every row) alongside `phoneme_ref`, so

```
phonetize(tashkeel_sentence) == phoneme_ref
```

is a second round-trip over **73,979 utterances** — 45× larger, and
crucially it exercises the digits, Latin script, punctuation and loanwords
that Qur'anic text never contains. See `insights/week-04.md` and
[docs/msa-arm.md](../msa-arm.md) §4.1.

Hold both to ≥99%, and report them separately: a phonetizer that passes on
Qur'anic text and fails on MSA is precisely the failure this week exists to
catch, and averaging the two would hide it. Expect the MSA side to need the
normalization contract (§4.1) settled first — the vowelizer strips
punctuation, drops Latin script, and **inserts** U+0670 (dagger alef) of its
own accord in 3.8% of rows while stripping every one it is given, and the
phonetizer has to match all three to close the loop.

(An earlier draft of this line said the vowelizer "rewrites U+0670 as a
literal alef". That was inferred from eight dev rows and is wrong at scale:
0/120 given dagger alefs survive, and 2,738 new ones appear. The practical
consequence is the opposite of what was planned for — the phonetizer must
handle U+0670 in its *input*, since `tashkeel_sentence` contains it.)

If the round-trip cannot be made to work, the label-path decomposition that
is the paper's spine cannot be computed correctly, and that is a week-4
finding rather than a week-6 surprise.

## Tasks

1. ~~**Diacritizer wrappers** under `src/arabic_mdd/diacritizers/`, one
   module per tool behind a common interface: CATT (EO and ED), Shakkala,
   and Mishkal or Farasa.~~ **DONE** 2026-09-28 — `base.Diacritizer` plus
   `catt.py` (EO and ED), `shakkala.py`, `mishkal.py` and a `DIACRITIZERS`
   registry; all four verified against the real tools, tests offline via
   injected backends. Farasa not wrapped (Mishkal is pure Python and is the
   tool RQ3's rank prediction names).

   Three things worth carrying forward. **Shakkala cannot run in this
   environment** — it pins `tensorflow==2.9.3` (cp310 max) against Python
   3.12 — so it runs out-of-process under its own `uv`-managed 3.10
   interpreter, and the version pin is enforced at runtime rather than
   assumed. **Mishkal emits U+0001** in place of sentence-final punctuation
   *and the following space*, merging two words into one; the wrapper
   restores the space. **Wrappers deliberately do not normalize their
   input** — the §4.1 normalization contract stays a single pinned step, so
   it remains visible and identical across tools. Full write-up:
   `insights/week-04.md`.
2. ~~**Vendor the CATT token-probability extraction** into that same package,
   with its own tests. Confirm the same technique works on the **ED**
   checkpoint before committing to a variant.~~ **DONE** 2026-09-28 —
   `CattDiacritizer.diacritize_batch_with_confidence` returns a
   `DiacriticConfidence(letter, diacritic, confidence, forced)` per input
   position, for both variants; `tests/diacritizers/test_catt_confidence.py`
   asserts the reconstruction identity against the package's own decode with
   fake ONNX sessions, and `scripts/check_catt_probabilities.py` (rewritten,
   superseding the week-1 EO-only probe) re-confirms it on the real weights.

   Three things worth carrying forward. **ED needed its own extraction
   path** — it decodes autoregressively and keeps only `preds[:, -1, :]`, so
   the loop had to be re-walked; confirming it was not box-ticking. **RQ4
   must ignore space positions**: `_apply_space_mask` overrides the model
   there, so the confidence describes a discarded prediction — hence the
   `forced` flag, and `catt-tashkeel` is now pinned `==1.0.2` with a runtime
   check, because the whole path rides on private internals. **The variant
   choice is deferred to week 5's RQ1 DER measurement**, not taken here: on
   the one long sentence tested ED was visibly worse than EO, against CATT's
   own claim, and three sentences cannot settle it. Full write-up:
   `insights/week-04.md`.
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
   methods, not its rebuttal. Full argument and the options table:
   [docs/msa-arm.md](../msa-arm.md) §4.

   Same task, second half: MSA transcripts contain Latin script, digits,
   punctuation, abbreviations and loanwords that Qur'anic text does not, so
   the task-3 round-trip says nothing about them. Inspect the Common Voice
   transcripts and fix the normalization rules now — annotators hit them on
   day one.
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
7. ~~**Pin `utter-project/mHuBERT-147` to a commit sha.**~~ **DONE** —
   pinned to `7ad3fc0bc5106c58c9c13526abccad527150d135` by
   `scripts/baseline_reproduction/pin_upstream.py`, wired into
   `run/prepare_baseline.sh` step 7 and consumed by `train_baseline.slurm`
   and `smoke_train.slurm` as `-k <sha-named local dir>`.

   Two things worth carrying forward. First, the obvious fix does not work:
   s3prl's `--upstream_revision` never reaches the `hf_hubert_custom`
   upstream, so adding it would have written *false* provenance into every
   checkpoint's `Args`. Second, the pin is retroactively safe — no weight or
   config file in that repo has changed since 2024-06-12, so the validated
   0.4406 run consumed exactly these bytes. Full argument:
   [docs/msa-arm.md](../msa-arm.md) §5.1.

8. ~~**50-utterance listening pilot** (MSA-arm step 4).~~ **DONE** — pulled
   forward from week 5 because step 3b retired the proxy strategy and left
   it the only arbiter of the MSA base rate. Protocol pre-registered
   (`docs/annotation-protocol-pilot.md`), 50/50 annotated blind
   (`run/msa_pilot_annotated.json`), scored by `scripts/pilot_baserate.py`.

   **Outcome: the base rate is unresolved and the pre-registered decision
   cannot be taken.** 2.3% lenient vs 4.7% counting case endings — the
   definition alone straddles the 3% band boundary, and every interval spans
   it. The design lost ~40% of its power to unusable rows (7 of 25 uniform).

   Two results it was not designed to produce, both larger than its
   headline: the **vowelizer is wrong on 43.6%** of usable utterances
   (~10:1 against speaker error), and the corpus is **10% Qur'anic / 20%
   pre-vowelized** — contamination in the arm defined by *not* being
   Qur'anic. Items 1–4 under "Carried forward" are the consequences. Full
   write-up `insights/week-04.md`; standing reference
   [docs/msa-arm.md](../msa-arm.md) §3.3a.

## Carried forward

### Created by the 50-utterance pilot (24 Sep 2026)

Ordered by what blocks the 300–500 annotation block. See msa-arm.md §3.3a.

1. **Filter the MSA pool before annotating anything.** The pilot found 10%
   Qur'anic verses and 20% pre-vowelized source text in the corpus that is
   supposed to be the *non*-Qur'anic arm. Qur'anic rows attack the arm's
   premise (§1); pre-vowelized rows never exercise the diacritization path,
   and where the text is fully vowelized the pipeline passes the input's own
   diacritics straight through, so there is nothing to evaluate. Both are
   detectable **automatically** — a diacritic regex over the text, no
   listening — so this is cheap. Run it over the full dev split, not the
   50-row sample, before sampling the annotation block; the pilot's own
   intervals are wide ([3.3%, 21.8%] and [10.0%, 33.7%]) and the sample was
   never meant to estimate them. Expect the usable pool to shrink by more
   than the headline 22% unusable rate, since the categories overlap only
   partly.

2. **Second annotator on the same 50, blind.** Already required by plan
   line 134 (~50 double-annotated for IAA *and* anchoring). Now urgent
   rather than scheduled: the pilot is single-annotator with no agreement
   measure, and it is currently the only evidence for both the base rate and
   the 43.6% vowelizer error rate. `scripts/pilot_annotate.py` supports it
   as-is via `--output`; report on the clinical model (plan line 138), not
   bare kappa.

3. **Settle whether dropped mid-sentence case endings count as `A ≠ C`.**
   This single definition moves the base rate 2.3% → 4.7%, across the 3%
   threshold the pre-committed bands hang on. It is a modelling decision,
   not a measurement — more annotation cannot resolve it. It interacts
   directly with §3.4.1 option 4 and should be decided in the same place and
   reported both ways.

4. **Budget for a 44% correction rate.** Correct-the-machine annotation
   (plan line 134) assumed the machine is mostly right. On MSA it is wrong
   on 43.6% of utterances, so nearly half of the 300–500 block needs real
   editing. This is a schedule fact, and it also sharpens the anchoring risk
   that the double-annotation in item 2 exists to measure.

### Older

- **Pin the S3PRL checkout itself.** `run/prepare_baseline.sh:81` clones
  `s3prl/s3prl.git` at whatever `main` serves that day. S3PRL supplies the
  featurizer, the downstream BiLSTM+CTC model *and* the training loop, so
  this is a strictly larger reproducibility exposure than the upstream
  weights were. Surfaced while doing task 7; not fixed in the same change
  because the commit has to be chosen against a venv known to build, which
  means a cluster round-trip.

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
