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

> **REWRITTEN 2026-10-01, after the criterion was run.** The original
> criterion below named QuranMB.v2 as the primary reference. That reference
> turned out to be **circular and unscoreable**, so the criterion now rests
> on the MSA arm alone. The original wording is kept struck-through rather
> than deleted, because the reason it failed is a week-4 finding in its own
> right. Result: **PASS, 99.85% on `Iqra_train` dev.** Full write-up:
> `insights/week-04.md` § "Phonetizer round-trip".

### The criterion, as it now stands

**The phonetizer round-trips `Iqra_train`.** `tashkeel_sentence` is the
organizers' vowelizer's output and is the string `phoneme_ref` was derived
from, so

```
phonetize(tashkeel_sentence) == phoneme_ref
```

Target ≥99% exact sequence match, every mismatch inspected and explained
rather than tolerated, and zero phonemes emitted outside the 68-token
`sws_arabic.txt` inventory.

**Met: 2584/2588 = 99.85% on the dev split, TER 0.000048, 0 out-of-inventory
tokens.** All four failures are one orthographic case (a shadda written on
the alef of `إِلاَّ`), where the reference itself is malformed; neither side
is correct and there is nothing to fix. The train split (71,391 rows) has
**not** been run — the 73,979-utterance figure below is the reference that
*exists*, not the number validated.

### ~~The criterion as originally written~~ — void

> ~~**The phonetizer round-trips QuranMB.v2.** The dataset ships both
> `reference_arabic_string` (gold-diacritized Arabic) and
> `reference_phoneme_string` (the canonical phoneme sequence the baseline was
> scored against). So the phonetizer has an exact, free, 1,642-utterance
> reference:~~ `phonetize(reference_arabic_string) == reference_phoneme_string`

`reference_arabic_string` is **not** gold-diacritized Arabic shipped with
the benchmark. The third-party uploader did not have the text; he
reconstructed it by phonemising the Qur'an and matching the result against
`reference_phoneme_string`, and shipped the residual as `match_distance`
(with `match_type` ∈ {exact, fuzzy}). The text is therefore a function of
the phonemes, and our independently computed edit distances reproduce
`match_distance` row for row — mean 2.47, max 10 on fuzzy rows, 0 on exact
ones.

Consequence: the round-trip score on that arm is **definitionally** the
fraction of rows with `match_distance == 0`. We measured 904/1642 = 55.05%,
and "100% on the exact subset" means only "rows already selected for
round-tripping do round trip." The metric returns the uploader's recovery
residual no matter what the phonetizer does. It can neither pass nor fail
anything.

Two further things it hides: the 1,642 rows collapse to **96 distinct
sentences**, so any per-row percentage over that corpus is a weighted
average over repeats; and the dagger-alef normalization rule was originally
tuned on this circular metric (55.05% → 49.39%), a justification now void
and re-grounded non-circularly at 177/177 on MSA dev rows containing U+0670.

This does **not** affect the Qur'anic arm's *phoneme* columns.
`reference_phoneme_string` and `annotation_phoneme_string` are validated —
week 3 reproduced the published F1 to four decimal places with the
organizers' own checkpoint. Only the Arabic **text** column is circular.
The consequence lands on the label path, not on detection scoring, and the
fix is clip → verse recovery against an authoritative Qur'an text
(`docs/msa-arm.md` §3.3b D1, where it is now a precondition rather than a
chore).

### How the MSA reference was found — kept, because it became the criterion

**Added 2026-09-23.** This criterion originally noted that the round-trip
"exists only on the Qur'anic arm". That turned out to be wrong, and the
correction is what saved the criterion when the Qur'anic arm collapsed.
`IqraEval/Iqra_train` ships a `tashkeel_sentence` column (the organizers'
in-house vowelizer's output, fully diacritized in every row) alongside
`phoneme_ref`, giving a reference over **73,979 utterances** — 45× larger
than QuranMB, and crucially it exercises the digits, Latin script,
punctuation and loanwords that Qur'anic text never contains. See
`insights/week-04.md` and [docs/msa-arm.md](../msa-arm.md) §4.1.

~~Hold both to ≥99%, and report them separately: a phonetizer that passes on
Qur'anic text and fails on MSA is precisely the failure this week exists to
catch, and averaging the two would hide it.~~ Only one is holdable, so the
separate-reporting instruction is moot — but its *motivation* survives and
now points the other way. Validation rests entirely on an MSA corpus, so
the untested direction is Qur'anic orthography. Measured incidentally on
MSA dev: dagger alef U+0670 177/177, alef madda 152/153, tanwin 1155/1158,
shadda 1687/1691 — but **alef wasla U+0671 occurs in 0 rows** and its
handling is asserted, never measured. That gap cannot be closed from the
MSA side and the Qur'anic reference is circular, so it stays open.

Expect the MSA side to need the
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
3. ~~**Phonetizer**, `src/arabic_mdd/data/phonemes.py` or a sibling: diacritized
   Arabic → phoneme sequence in the 68-token `sws_arabic.txt` inventory.
   Validate by the round-trip above.~~ **DONE** 2026-10-01 —
   `src/arabic_mdd/data/phonetizer.py`, wrapping the Halabi phonetiser
   vendored verbatim at `_halabi_phonetiser.py` (commit `e75e06b`,
   CC BY-NC 4.0, asserted by test). 30 offline tests. Validated at
   **99.85%** on `Iqra_train` dev; the exit criterion was rewritten above
   because the QuranMB reference it originally named is circular.

   Four things worth carrying forward. **Every normalization rule is
   justified by measurement, and two plausible ones were refuted** — mapping
   the dagger alef to a full alef, and deleting the alef that carries
   tanwin-fath (costs ~500 `aa` tokens, since that alef is a real long vowel
   in `phoneme_ref`). **The convention switch is mechanical**, so annotators
   never need to learn the pausal rules; `normalize` strips tanwin and the
   utterance-final mark, which makes 6.52% of diacritic positions harmless
   by construction (`docs/msa-arm.md` §3.3b D3 for the per-class table).
   **Zero out-of-inventory tokens on the reference path** — risk (a) does
   not arise here, but task 4's diacritizer path is a different input
   distribution and is still unchecked. **Faulty vowelizer diacritics do not
   contaminate the result**: both sides of the round-trip receive the same
   string, so a wrong diacritic cannot create or hide disagreement between
   two functions of it. Full write-up: `insights/week-04.md`.
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

   **DONE** (1 Oct 2026). `scripts/check_phoneme_inventory.py`,
   `src/arabic_mdd/data/normalize.py` (14 tests), artifact
   `run/phoneme_inventory_diff.json`; decision in `docs/msa-arm.md` §4 and the
   contract in §4.1.1.

   **The diff was unfalsifiable as specified, and that is the finding.** The
   vendored phonetiser passes an unknown character straight through its
   Buckwalter stage, where it matches no phoneme rule, so **nothing is emitted
   for it**. It therefore cannot produce an out-of-inventory token — the
   failure mode is silent *deletion*, not overflow. Running only the specified
   diff would have returned a confident all-clear for a structural reason.
   Two further instruments were added: a character-level diff over each tool's
   output, and phoneme-count conservation against `phoneme_ref`.

   All three are clean, for all four tools over all 2,588 dev utterances: 0
   out-of-inventory tokens, 0 characters with no phonetiser rule, and length
   conserved to ≤1% (`catt-eo` 0.9998, `catt-ed` 0.9955, `shakkala` 0.9984,
   `mishkal` 0.9900). So **none** of extend/map/drop is needed; risk (a) is
   retired rather than mitigated. 66/68 phoneme types is not a shortfall —
   the dev reference itself uses 66/68, missing the same `<<` and `gg`.

   Carry forward, three things:
   - **The per-utterance conservation tail is a result, not a check**:
     utterances >10% short run 0.08% / 1.04% / 1.39% / **5.14%**, an
     order-of-magnitude spread that ranks the tools before any annotation
     exists. `mishkal`'s tail is it declining to diacritize — a
     diacritization error, which is the paper's subject. Check RQ1's ordering
     against this.
   - **Alef wasla U+0671 is the live version of risk (a)**, now on the
     *Qur'anic* arm. It is absent from the 44-character Buckwalter map, so it
     is dropped along with the phonemes it carries (`ٱلْحَمْدُ` loses `< a`), and
     it appears in 0 MSA dev rows, so week 4's round-trip could not see it.
     Moved into `docs/msa-arm.md` §9 as a live falsifier, gated on the clip →
     verse recovery probe.
   - **The digit rule is vacuous**: 0 rows out of 71,391 contain a digit,
     ASCII or Arabic-Indic. No transliteration policy was needed and the
     annotation guidelines do not need an entry.
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

   **[RESOLVED 2026-10-01 — `docs/msa-arm.md` §3.3b D3.]** A second
   annotator is available but is not a graduate-level Arabic specialist.
   Rather than accept or reject on credential, both annotators sit a
   qualification test against known-gold diacritization
   ([docs/annotator-qualification.md](../annotator-qualification.md)) and
   the measured per-bucket result decides what each may annotate
   unsupervised. Note the reason this is the *stronger* instrument, not a
   consolation prize: under correct-the-machine, two annotators who both
   tend to accept the machine's suggestion agree almost perfectly and are
   both wrong in the same places, so agreement is structurally blind to the
   anchoring failure it was meant to catch. Accuracy against gold is not.

3. **Settle whether dropped mid-sentence case endings count as `A ≠ C`.**
   ~~This single definition moves the base rate 2.3% → 4.7%, across the 3%
   threshold the pre-committed bands hang on.~~ **MOOT as a blocker,
   2026-10-01 — §3.3b D1 dropped `A` from the arm**, so there is no `A ≠ C`
   judgment left to define. The underlying finding survives and is the
   reason for the decision: the MSA "mispronunciation" rate is dominated by
   case-ending convention rather than by mispronunciation, which makes the
   quantity definitional rather than measurable. Retained as a limitations
   paragraph and as support for §3.4.1 option 4.

4. **Budget for a 44% correction rate.** Correct-the-machine annotation
   (plan line 134) assumed the machine is mostly right. On MSA it is wrong
   on 43.6% of utterances, so nearly half of the block needs real editing.
   This is a schedule fact, and it also sharpens the anchoring risk that
   item 2's design exists to measure.

   **Revised 2026-10-01.** Two changes pull in opposite directions and
   roughly cancel, so the budget stands. Against it: §3.3b D1 removed the
   listening pass, so the task is text-only and much faster per utterance,
   and there are now two annotators. For it: the 43.6% is the *per-utterance*
   rate, so the correction load is real, and D4's pre-annotation filter
   shrinks the usable pool rather than the per-item cost. Plan line 134's
   300–500 is now **conservative**, but set the target in D4 from the
   qualification test's recorded timings rather than by guessing.

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
