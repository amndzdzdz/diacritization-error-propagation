# The MSA arm — standing reference

What the MSA arm is, what has to be built before it produces a number, and
the failure modes that would make that number wrong in a way nobody
downstream could detect.

This is a **standing reference**, not a week plan. Week plans
(`docs/weeks/week-NN.md`) cite it and schedule the work; this file is where
the reasoning and the constraints live so that they are decided once. It is
written against [mdd-paper-project-plan-v2.md](mdd-paper-project-plan-v2.md)
§1, §3 and §6, and against the week 1–3 gate outcome recorded in
[insights/week-03.md](../insights/week-03.md).

Status at time of writing (23 Sep 2026): the Qur'anic arm is validated end
to end. **Nothing in the MSA arm exists yet.** Weeks 4–9 build it.

**Update, same day — steps 1–3 of §8 are done, plus a follow-up control.**
Full write-ups in [insights/week-04.md](../insights/week-04.md); the
affected sections below are marked **[MEASURED]** and carry the numbers
rather than the original assumption.

| Step | Outcome |
|---|---|
| 1. `Iqra_train` text inspection (dev + all 71,391 train rows) | A 73,979-utterance MSA round-trip reference exists (§4.1); risk (a) downgraded — 2.3 M tokens, **zero** vocab overflow, 68/68 types used (§4); vowelizer content loss quantified at **0.8%** (§4.1); normalization contract added (§4.1) |
| 2. Cross-arm word-final convention | **The two arms use opposite conventions** — 34.1% vs 80.1% case endings realized. Confirmed outright (§3.4) |
| 3. Base-rate screen on CV-Ar dev | FR 0.0708 vs QuranMB 0.1241 — **inconclusive**, confounded by the checkpoint being in-domain (§3.3, §5.2) |
| 3b. Zero-base-rate control on `Iqra_TTS` | FR **0.0066**. Bound is 91% of the disagreement, i.e. vacuous. **The proxy strategy is retired**; the listening pilot is the only arbiter (§3.3) |
| 3c. Case-ending impact on QuranMB | The pausal/prescriptive mismatch is **in the published benchmark**: `C` 72.0% vowel-final vs `P` 32.7%, on both checkpoints; worth 5–10% of its FR. **§3.4.1 decided: option 4** (§3.4.1) |
| 4. 50-utterance listening pilot | Base rate **2.3%** lenient / **4.7%** counting case (CI spans the 3% boundary, **unresolved**). The dominant finding is elsewhere: the **vowelizer is wrong on 43.6% of usable utterances**, ~10:1 against speaker error. Corpus contamination found: 10% Qur'anic, 20% pre-vowelized, 22% unusable (§3.3a) |
| 5. Phonetizer round-trip | **PASS at 99.85%** on `Iqra_train` dev (2584/2588, TER 0.000048, **0** out-of-inventory tokens) — the week-4 exit criterion, met on the MSA arm. The Qur'anic arm's is **unscoreable**: its Arabic text was recovered *from* its own phoneme string, so the metric returns the uploader's residual. Consequence for §3.3b D1 |
| 6. Scope decision | **D1: the arm drops `A`** — label-path only, text-only annotation. **D3: two annotators, qualified against gold** rather than by credential. **D4: 500 utterances, seeded draw from a 2,092/2,588-row pool** (80.8%), three exclusion predicates and five flag-only ones. D2/D5/D6 open (§3.3b) |
| 6b. Sampling frame | The 8.4% "vowelizer dropped a word" rate was **punctuation**, not loss — the real rate is **3 rows (0.1%)**. The split is **not** duplicate-free: 2,588 rows → 2,542 skeletons, and 33 of 44 duplicate groups are the same verse bare *and* pre-vowelized, which made filter order worth 16 rows (§3.3b D4) |

Two things changed the plan rather than just the numbers. The
pausal-convention defect (§3.4) had to be resolved before any MSA number
was generated — **now decided, §3.4.1 option 4**, which keeps MSA `C_gold`
pausal and turns the cross-arm mismatch into a reported result. And §3.3
option 1 is no longer the *preferred* route but the *only* one — with the
consolation that `Iqra_TTS`'s augmented rows supply a complete `(C, A, P)`
triple, making option 3 available on existing data.

Two corrections worth flagging. An early dev-split reading of the U+0670
rule was wrong and was caught by re-checking at scale (§4.1). And after the
pilot returned a low mispronunciation rate, the first reading of that result
was that the MSA arm needed reframing — which was wrong twice over: the arm
was never a detection study (plan line 130, "No — you create them"), and
plan §Pre-committed position on effect size forbids exactly that kind of
pivot. A low base rate is *favourable* for the deployment-harm contribution.
See §3.3a.

---

## 1. Why the arm exists

The paper's opening frame, from plan §Data:

> Qur'anic text is canonically diacritized and fixed, so `C_gold` is free
> there. The field's flagship benchmark is therefore dominated by the one
> Arabic domain in which this failure mode *cannot occur*, while deployment
> is MSA, where it always does.

That is the natural experiment. The Qur'anic arm is the controlled
condition — strip the diacritics, re-diacritize automatically, compare
against a gold that was there all along. The MSA arm is the realistic
condition — the text genuinely has no diacritics, so `C_gold` has to be
manufactured by a human, which is exactly what a deployed system cannot do.

The two arms are **not two samples of one population** and the paper must
say so early. They differ in domain, in speaker population, in error base
rate, and in how `C_gold` came to exist. Every one of those differences is a
candidate confound for the Qur'anic-vs-MSA comparison, and §5 below lists
the ones that must be pinned.

### The asymmetry, concretely

| | Qur'anic arm | MSA arm |
|---|---|---|
| Corpus | QuranMB.v2, 1,642 utts (joined) | Common Voice Arabic, 300–500 annotated |
| Undiacritized text | Produced by stripping | **Native condition** |
| `C_gold` (canonical phonemes) | Ships with the dataset | **Must be annotated** |
| `A` (verbatim production) | Ships with the dataset | **Must be annotated** — `phoneme_aug` is *not* it, §3.3 |
| `C_auto` | Diacritizer + phonetizer | Diacritizer + phonetizer |
| Acoustic model | Validated, F1 = 0.4406 | Same recipe, new eval set |
| Published number to check against | **0.4414** | **None** |
| Phoneme inventory | 68-token `sws_arabic.txt`, native | 68-token vocab, **no overflow observed**, §4 |
| Licence | Open since week 2 | CC0 (verify) |

The two bolded "must be annotated" rows are the schedule's main risk, per
the plan, and the "None" row is why §5 exists.

---

## 2. What already exists and transfers

The week 1–3 gate passed three independent ways, agreeing within 0.001 F1
of the published 0.4414. What that bought the MSA arm:

- **A validated training recipe.** `-u hf_hubert_custom -k
  utter-project/mHuBERT-147`, frozen upstream, SUPERB weighted layer-sum,
  2×1024 BiLSTM + CTC, `runner.total_steps: 200000`, 68-phoneme
  `sws_arabic.txt`. Our from-scratch retrain scored 0.4406; the organizers'
  own checkpoint scored 0.4415 through the same pipeline. Both are recorded
  in `insights/week-03.md`.
- **A validated scorer.** `src/arabic_mdd/metrics/hierarchical.py`,
  cross-checked against the organizers' `mdd_eval/`. Takes
  `(C, A, P)` as `Sequence[str]` and returns TA/FR/FA/TR + CD/ED. It is
  domain-agnostic — it does not care which arm produced the sequences.
- **Data loaders.** `src/arabic_mdd/data/iqra_train.py` already reads
  `IqraEval/Iqra_train` (Common Voice Arabic, 79 h) and
  `IqraEval/Iqra_TTS`; `phonemes.py` fixes the tokenization convention
  (whitespace split) shared by loaders and metric.
- **Provenance tooling.** `scripts/baseline_reproduction/inspect_s3prl_ckpt.py`
  reads `Args`/`Config` back out of any trained checkpoint. §5 makes this
  mandatory rather than optional for every MSA-arm run.

### The convenient accident: the MSA arm is *in-domain* for the baseline

The organizers train on Iqra_train (Common Voice Arabic) + Iqra_TTS and
evaluate on QuranMB.v2. So the validated baseline is already an MSA-trained
model tested out of domain. Evaluating it on Common Voice Arabic is the
*easier* direction, not the harder one, and needs no new training run.

This removes the obvious worry — "can the acoustic model even handle MSA" —
but introduces a subtler one, in §5.2: the CV-Ar dev split was used for
checkpoint selection (`dev-best.ckpt`), so it is not cleanly held out.

---

## 3. The four objects the arm needs, and where each comes from

Restating plan §1's diagram with the MSA arm's provenance attached:

```
undiacritized CV-Ar transcript → [diacritizer] → vowelized → [phonetizer] → C_auto
                                     ↑ CATT EO/ED, Shakkala, Mishkal/Farasa
human diacritization of the same transcript    → [phonetizer] → C_gold   ← ANNOTATE
audio → [MDD model]                                            → P
human listening to the audio                                   → A       ← ANNOTATE
```

### 3.1 `C_auto` — mechanical, week 4

Diacritizer wrappers plus the phonetizer, both scheduled in
[docs/weeks/week-04.md](weeks/week-04.md). Nothing MSA-specific except the
inventory question in §4.

### 3.2 `C_gold` — the diacritization annotation

Plan §Data, unchanged: 300–500 utterances, **correct-the-machine** rather
than from scratch (run a diacritizer, present its output, have the
annotator correct it). Faster and less error-prone, but it anchors toward
the tool under evaluation, so ~50 utterances must be **double-annotated
blind** — both annotators independently, without seeing any machine output
— to measure inter-annotator agreement *and* the anchoring effect. Report
both. Agreement on the clinical model (Pearson + ICC(2,1), the Harf-Speech
template), not bare Cohen's kappa.

**Revised 2026-10-01 by §3.3b.** Three changes, none of which touch the
correct-the-machine choice itself:

- `C_gold` is now the arm's **sole** annotation output (D1 dropped `A`), so
  the task is text-only — annotators never open the audio. All of the arm's
  quality risk is concentrated here, which is what motivates the next point.
- Suitability is established by **measurement against gold**, not by
  credential: [annotator-qualification.md](annotator-qualification.md),
  scored separately on word-internal marks versus case endings, because a
  combined score is dominated by the easy 78.6% of positions.
- Agreement remains reported but is **demoted from the primary instrument**.
  Under correct-the-machine, two annotators who both tend to accept the
  machine's suggestion agree almost perfectly and are both wrong in the same
  places — agreement is structurally blind to the exact anchoring failure it
  was introduced to catch. Accuracy against gold is not. The clinical
  reporting model (Pearson + ICC(2,1)) is unchanged for the agreement
  number itself.
- The anchoring measurement becomes a **crossover**: annotator 1 sees the
  machine's suggestion while annotator 2 works blank-page on the same
  utterances, then they swap on a second block. This yields anchoring and
  agreement from one pass, and it **cannot be reconstructed after the fact**
  — it is the remaining week-5 blocker here. D4 has since fixed the block it
  runs on: the first 50 utterances of the drawn 500, in draw order
  (`run/msa_annotation_block.json`, `crossover: true`).

Week 4 task 5 decides what to anchor on: if the NAACL 2024 audio-informed
restoration method beats text-only CATT on Common Voice, it is the better
starting point *and* the anchoring bias then points away from the system
under test rather than toward it.

**Pre-register the protocol in writing before a single utterance is
annotated.** This is not a formality — the anchoring measurement is only
interpretable if the procedure was fixed in advance.

### 3.3 `A` — the verbatim-production annotation, and the base-rate problem **[MEASURED, partly]**

This is the part the plan does not spell out and it is the largest open
design question in the arm.

The hierarchical metric needs all three of `(C, A, P)`. A position counts
as a detection *positive* only where `A ≠ C` — i.e. where the speaker
actually mispronounced something. On QuranMB.v2 those positives are
plentiful: the recordings are learners reciting, and
`annotation_phoneme_string` captures what they really produced.

**Common Voice Arabic is read speech by fluent readers.** The expected
mispronunciation rate is low, and it may be very low. If `A ≈ C_gold`
everywhere, then TR and FA go to ~0 by construction, precision collapses
toward 0, and F1 stops being a meaningful quantity. The metric does not
break loudly in that case — it returns a number, and the number is
uninformative.

There is direct evidence the organizers hit this themselves: `Iqra_train`
(real CV-Ar speech) ships only `phoneme_ref`, with **no** annotation
column, whereas `Iqra_TTS` ships `phoneme_mis` — synthesized speech with
*injected* mispronunciations. The TTS half of the 131 h training set exists
precisely because real read MSA speech does not supply enough error
examples.

**[MEASURED]** `Iqra_train` does ship a second phoneme column,
`phoneme_aug`, which might have been an annotation of actual production.
It is not: it is byte-identical to `phoneme_ref` in **2,588 of 2,588** dev
rows and **71,376 of 71,376** train rows — the entire corpus, without a
single exception. Whatever "aug" was meant to denote, it carries no
mispronunciation signal. The MSA arm therefore has no verbatim-production
annotation of any kind, and `A` must be created from nothing. The worry
stands undiminished.

**[MEASURED — the screen ran, and it did not settle the question.]**
`run/cv_dev_baserate.slurm` scored all 2,588 dev utterances with `A := C`:

| | |
|---|---|
| CV-Ar dev disagreement (FR) rate | **0.0708** |
| QuranMB FR rate | 0.1241 |
| excess | **−0.0533** |
| utterances with zero disagreement | 752/2,588 (29.1%) |
| per-utterance disagreement p50 / p90 | 0.050 / 0.194 |

Taken at face value this says "no headroom". **Do not take it at face
value.** `dev-best.ckpt` was model-*selected* on this very split (§5.2), so
its error rate here is expected to fall well below the QuranMB figure
whether or not speakers deviate at all. A negative excess is equally
consistent with "no mispronunciations" and with "mispronunciations masked
by an in-domain model". QuranMB is the wrong yardstick for an in-domain
split — different domain, different speaker population, out-of-domain
checkpoint. The screen is **inconclusive in the negative direction**.

The fix is a control where the base rate is zero *by construction*:
`Iqra_TTS`'s `original` rows are synthesized from `phoneme_ref`, so
`A == C` exactly, and their FR rate is the model's pure error rate on MSA
phonetics (`run/tts_control.slurm`). Read one-sidedly: `Iqra_TTS` is both
in-domain and acoustically easy, so only `control FR ≥ 0.0708` is decisive.
Below that, the gap is an upper bound on the base rate, not an estimate.

Premise verified over 800 sampled rows — after stripping `<sil>`,
`phoneme_mis == phoneme_ref` in 500/500 `original` rows vs 3/300
`augmented`. ~1/3 of the 46,851 rows are `original`. Note `<sil>` is **not**
in the 68-token vocab (`<` alone is, the glottal stop) and appears only in
`phoneme_ref`; it must be stripped on both sides or the clean rows look
fully mispronounced and the FR rate inflates.

**[MEASURED] The control ran, and it does not constrain.** 3,000 rows:
FR rate **0.0066** — 10.7× below CV-Ar dev, 18.8× below QuranMB. The
nominal headroom (0.0642) is 91% of the dev disagreement, so the bound
excludes almost nothing. At 0.66 errors per 100 phonemes the model is close
to memorizing `Iqra_TTS` (~7 synthetic voices, clean, in the 131 h training
set), so the 10.7× gap to real speech is explained by acoustics alone
without invoking mispronunciation.

**Conclusion: the proxy strategy is retired.** Both available reference
points are confounded in the same direction — QuranMB by domain, TTS by
ease and in-domain training — and there is no third. §3.3 option 1 (the
listening pilot) is not the preferred route, it is the only one.

**What the control did surface** is that `Iqra_TTS`'s *augmented* rows
carry both `phoneme_ref` (`C`) and `phoneme_mis` (`A`), differing in ~99%
of rows — a complete `(C, A, P)` triple, and the only MSA-domain data in
the project with genuine detection positives. `run/tts_detection.slurm`
scores it: this is option 3 below, on data that already exists.

Net effect: option 1 below is now the **critical path**, not the
preferred-first option. Nothing cheaper can settle this.

Options, none yet chosen:

1. **Annotate `A` verbatim and measure the base rate first.** Common Voice
   Arabic does contain misreadings, hesitations, dialectal substitutions and
   non-native speakers. The rate is unknown. A 50-utterance pilot before
   committing 300–500 is cheap and decides everything downstream. **This
   should happen first** — and after the inconclusive screen it is the only
   unconfounded measurement available. The manifest and audio are prepared
   (`run/msa_pilot_sample.json`, `run/pilot_audio/`).
2. **Restrict the MSA arm to label-path measurement.** The corruption-
   attributable false-rejection metric (plan §4.2) counts FRs, which need
   positions where the speaker was *correct* and the system flagged anyway —
   abundant even at a zero error rate. This keeps the deployment-facing
   number alive without needing positives. It does **not** give the
   2×2's headline F1 bias figure.
3. **Inject controlled mispronunciations**, mirroring the organizers' own
   TTS strategy, and report the MSA arm on a mixed real+synthetic set.
   Cleanest statistically, weakest for the "realistic condition" framing
   that justifies the arm in the first place. If used, the synthetic
   fraction must be stated in the abstract-adjacent text, not a footnote.
4. **Source a non-Common-Voice MSA read-aloud corpus with learner speech.**
   Out of scope on this timeline; recorded here so it is a deliberate
   omission rather than an oversight.

The pilot in option 1 gates the choice. Schedule it in week 5 alongside the
first annotation block.

### 3.3a The pilot ran — what it decided and what it did not **[MEASURED, 24 Sep 2026]**

50 utterances, blinded, single annotator, one pass. Protocol pre-registered
in `docs/annotation-protocol-pilot.md`; full write-up in
`insights/week-04.md`. Raw: `run/msa_pilot_annotated.json`.

**The base rate is unresolved, and more annotation will not resolve it.**
Uniform stratum, 18 usable utterances / 86 words:

| definition | rate | 95% CI |
|---|---|---|
| count everything | 19.8% | [12.0%, 29.8%] |
| drop `refbad` only | 5.8% | [1.9%, 13.0%] |
| drop `refbad` + `dialect` (**case counts**) | 4.7% | [1.3%, 11.5%] |
| drop `case` too (pre-registered headline) | 2.3% | [0.3%, 8.1%] |

The `case` decision alone moves the answer across the 3% threshold the
pre-committed bands hang on. That is not a sampling problem — it is a
definitional one, and it is the substantive result: **the MSA
"mispronunciation" rate is dominated by case-ending convention rather than
by mispronunciation.** It also promotes §3.4.1 from a Qur'anic-arm finding
to a cross-arm one. Every interval above spans 3%; n=86 cannot separate
them, and n=500 would not settle a definition.

The power loss was not anticipated: the design assumed 25 uniform rows
(~142 words) and 7 were unusable, delivering 86. ~40% of the pilot's power
went to corpus quality.

**The dominant finding is on the label path, not the speaker.** Two words
in the whole uniform stratum are genuine mispronunciations. Against that:

| | k/n | rate | 95% CI |
|---|---|---|---|
| utterances where the **vowelizer** is wrong | 17/39 | **43.6%** | [27.8%, 60.4%] |

~10:1 in words (23 `refbad` vs 2 `sub`). This is far above the ~12% DER the
diacritizer literature reports, because per-utterance exposure compounds
per-word error. Three specimens are in `insights/week-04.md`; `72328` is the
cleanest — ى written as ي, so the vowelizer produced *layl-ī* ("my night")
for *Laylā*, and `phoneme_ref` carries `l a y l ii`, manufacturing a false
rejection against a correct speaker.

**Corpus contamination, all 50 rows:**

| | k/50 | 95% CI |
|---|---|---|
| Qur'anic verse in the *non*-Qur'anic arm | 5 | [3.3%, 21.8%] |
| pre-vowelized source text | 10 | [10.0%, 33.7%] |
| unusable, any reason | 11 | [11.5%, 36.0%] |

The Qur'anic rows attack the arm's premise directly (§1: the two arms are
supposed to differ in exactly this respect). The pre-vowelized rows never
exercise the diacritization path — for the 4 fully-vowelized ones the
pipeline passes the input's own diacritics through unchanged, so there is
nothing to evaluate. Both must be filtered before the 300–500 block, and
filtering shrinks the usable pool further. §7's licence question is
unaffected.

**Declared deviation.** Only 5 of the 11 exclusions fall within the
protocol's pre-registered wording for `unusable` (audio/text defects). Five
more are Qur'anic — a *domain* criterion that §1 already implies but that
was applied during annotation rather than pre-registered — and one
(`69711`) is a fully-vowelized hadith. The pre-vowelized exclusion **raises**
the vowelizer error rate (43.6% excluded vs 39.5% included-as-correct), so
it must always be reported with that sensitivity: it is the defensible
choice, but it moves the paper's key number in the paper's favour. The
conclusion survives either reading. Full table and base-rate sensitivity:
`docs/annotation-protocol-pilot.md` §9a. For the 300–500 block this becomes
a sampling-frame filter applied *before* annotation, which removes the
deviation rather than repeating it.

#### What this does NOT change

The low base rate is **favourable**, not damaging, and the first reading of
it here was wrong on both counts:

- **The arm was never a detection study.** Plan line 130 records the MSA
  arm's references as "No — you create them". The annotation output is
  `C_gold`, the correct *diacritization* — plan contribution 5, "a
  gold-diacritized Common Voice Arabic evaluation subset, released". A 43.6%
  tool error rate is the argument *for* that artifact's value, not against
  it.
- **Option 2 above is strengthened.** Corruption-attributable false
  rejections need positions where the speaker was correct and the system
  flagged anyway. At a near-zero error rate essentially every flag is one —
  the same logic as the `Iqra_TTS` zero-base-rate control (step 3b).
- **Plan §"Pre-committed position on effect size"** forbids converting a
  small effect into a different paper. That pre-commitment was made in
  week 0 for this exact situation and it holds.

#### Limitation that must be fixed

Single annotator, one pass, **no inter-annotator agreement**. Plan line 134
already requires ~50 double-annotated blind utterances; the pilot makes that
urgent rather than optional, and the 50 are already prepared. Every number
in this section is provisional on it. **Resolved by §3.3b D3 below** — a
second annotator is available, qualified by measurement rather than
credential.

### 3.3b Scope decision: the arm drops `A` **[DECIDED 1 Oct 2026]**

§3.3's four options were left open pending the pilot. The pilot ran (§3.3a)
and settled two of the six decisions the arm's scope depends on; D4 closed on
1 Oct 2026 once the sampling frame was measured. The other three are listed at
the end and remain open.

#### D1 — DECIDED: option 2, label-path only. The arm does not annotate `A`.

The MSA arm measures reference corruption and releases `C_gold`. It does
**not** attempt mispronunciation detection. No listening pass; annotation is
text-only.

**Rationale.** Three independent reasons, in descending order of force.

1. **The base rate is unresolvable by annotation, not just unresolved.** The
   pilot's four definitions span 2.3%–19.8% and the intervals all cross the
   3% threshold the pre-committed bands hang on (§3.3a). The `case` decision
   alone moves the answer across it. That is a definitional choice, and
   n=500 settles definitions no better than n=86.
2. **The positives would be too few to carry a result.** At the pilot's
   ~4.8 words/utterance, 500 utterances is ~2,400 words — **~55**
   mispronounced words at the pre-registered 2.3% rate, ~475 at 19.8%. A
   precision/recall decomposition resting somewhere in a 9× band whose
   position is a modelling choice is not a reportable number.
3. **Listening is the expensive half.** Dropping it turns the task into
   text editing, which is what makes D4's volume and D3's two-annotator
   design affordable at all.

**What this costs.** The MSA instance of plan contribution 3 — the
bias direction/magnitude figure for *real* MSA speech. That is a real loss
and should be named as such in the limitations, not smoothed over.

**What it preserves.** Contribution 4 (corruption-attributable false
rejections) needs positions where the speaker was correct and the system
flagged anyway, which are abundant at a near-zero base rate — the same
logic as the `Iqra_TTS` zero-base-rate control. Contribution 5 (`C_gold`
released) is untouched and is now the arm's sole deliverable. RQ1 on the
deployment corpus is untouched and is strengthened by D5.

**Where contribution 3 relocates — with a correction.** The argument for D1
is partly that the bias figure can be measured on the Qur'anic arm instead,
which plan line 265 says needs no annotation. That holds, but only after
separating two columns that were previously conflated:

| object | status |
|---|---|
| `reference_phoneme_string` (`C`) | available via the third-party join, **validated** — week 3 reproduced the published F1 to 4 dp with the organizers' checkpoint |
| `annotation_phoneme_string` (`A`) | same |
| `reference_arabic_string` | **circular** — reconstructed by phonemising the Qur'an and matching against `C`; 55% of rows round-trip (see `insights/week-04.md`) |

So detection scoring on the Qur'anic arm is sound and contribution 3 does
relocate there. But the **label path** on that arm needs diacritized Arabic
text to strip and re-diacritize, and the only text column we have is the
circular one. The fix is not more annotation: the Qur'an is public and
authoritatively vowelized, so the dependency is **clip → verse recovery**
(consonant-skeleton matching against an authoritative text), already parked
for week 5/6. **D1 is therefore conditional on that recovery succeeding** —
if it does not, both arms lose the label path and the paper has a much
larger problem than the MSA arm's scope.

**Hedge, cheap, recommended.** Annotate `A` on the ~50 reliability
utterances only (D3's blind block). The base-rate estimate keeps
accumulating at negligible cost and the main block stays text-only. This
does not reopen D1; it preserves the option of a limitations-section
sentence with n=136 instead of n=86.

#### D3 — DECIDED: two annotators, qualified against gold, not by credential.

A second annotator is available but is not a graduate-level Arabic
specialist. Rather than treat that as disqualifying, both annotators sit a
**qualification test** on text whose correct diacritization is already
known, and the measured result decides what each is allowed to annotate.
Protocol: [docs/annotator-qualification.md](annotator-qualification.md).

This supersedes the test–retest workaround that was the fallback while no
second annotator existed. Test–retest stays in the design as a secondary
check, not as the primary one.

**Why a gold test is the stronger instrument, not a substitute for a weaker
one.** Inter-annotator agreement measures consistency, not correctness, and
under correct-the-machine it is actively vulnerable: two annotators who
both tend to accept the machine's suggestion agree almost perfectly and are
both wrong in the same places. That is the anchoring failure mode plan
line 134 exists to catch, and agreement cannot see it. Accuracy against
gold can. Report both; the gold number is the one that licenses the
release.

**Qualification does not neutralize the credential gap — it localizes it.**
The task has an easy majority and a hard minority, and the hard minority is
where a non-specialist native speaker is expected to be weak: word-final
case endings are grammatically determined (*i'rab*), which is a schooling
skill rather than a speaking skill. The literature puts the error there
too — CATT's WikiNews DER roughly doubles when case endings are counted
(5.43% vs 3.11%).

**[MEASURED 1 Oct 2026] The pausal convention relieves less of this than
assumed.** §3.4.1 fixes `C_gold` as pausal, and the phonetizer applies that
mechanically (`phonetizer.py` `normalize`), so errors in the stripped
positions are harmless by construction. Over all 50,188 diacritic marks in
`Iqra_train` dev's `tashkeel_sentence`:

| position class | marks | share |
|---|---|---|
| word-internal — survives | 36,410 | 72.55% |
| case ending (word-final, non-final word) — survives | 7,474 | 14.89% |
| shadda — survives | 3,030 | 6.04% |
| tanwin — **stripped** | 1,754 | 3.49% |
| final word's final mark — **stripped** | 1,520 | 3.03% |

**Only 6.52% of positions are stripped; 93.48% are scored, and case endings
are 15.93% of those.** An earlier guess in conversation that the pausal
choice removes "a good chunk" of the case-ending difficulty was wrong —
tanwin is a much smaller share of marks than it is of *utterances* (it
appears somewhere in 44.7% of them). The case-ending problem is not
convention-relieved and has to be handled by the qualification test's
flag-and-adjudicate fallback instead.

Consequence: the test scores the three surviving classes **separately**. A
single combined score would be dominated by the 72.55% easy class and would
pass a candidate who is poor at exactly the positions the pilot showed
dominate the arm's findings.

#### D4 — DECIDED: 500 utterances, seeded draw from a 2,092-row pool, thin filter.

Annotate **500** utterances from `Iqra_train` **dev**, drawn with a fixed seed
from an explicitly filtered pool. Filter predicates and the draw live in
`src/arabic_mdd/data/msa_pool.py` (tested); the runner and report are
`scripts/draw_annotation_block.py`; the result is committed as
`run/msa_annotation_block.json`. The block is a *function of* (seed, pool), so
it can be regenerated and audited rather than described.

**The filter is thin by design: three predicates exclude, five only flag.**

| predicate | rows (of 2,588) | action |
|---|---|---|
| `pre_vowelized` — `sentence` diacritization rate ≥ 0.50 | 489 (18.9%) | **exclude** |
| `duplicate` — same consonantal skeleton as a kept row | 46 (1.8%) | **exclude** |
| `no_arabic` / `no_tashkeel` | 0 / 0 | **exclude** |
| `partially_vowelized` — rate in [0.05, 0.50) | 252 (12.0% of pool) | keep + flag |
| `short` — fewer than 3 Arabic words | 195 (9.3%) | keep + flag |
| `word_count_mismatch` — vowelizer dropped/merged a word | 3 (0.1%) | keep + flag |
| `latin` / `digits` | 3 / 0 | keep + flag |

**Pool: 2,092/2,588 = 80.8%**, 61,683 phoneme positions, 11,018 words (29.5
and 5.3 per utterance). The 0.50/0.05 thresholds are §4.1's, reused rather
than reinvented, so the pool is comparable with the corpus-level figures.

**Why exclusion and flagging are not symmetric.** A `word_count_mismatch`
means the vowelizer dropped or merged a word — a genuine tool error, which is
the arm's *subject*. Excluding those rows would delete real errors from D5's
denominator and bias the headline **down**: the same class of mistake as
§3.3a's declared deviation, pointing the other way. Flagged rows therefore
appear as a results breakdown, never as a silent deletion. Only rows that
cannot answer the question at all are excluded (nothing to diacritize, or
already diacritized so the measurement would be of someone else's vowelizer).

**[MEASURED 1 Oct 2026] Two figures quoted in conversation were wrong and are
corrected here.**

1. **Word-count mismatch is 3 rows (0.1%), not 218 (8.4%).** All 218 raw
   whitespace-token mismatches contain a non-Arabic token: the 8.4% was
   entirely the vowelizer's punctuation removal, which §4.1 had already
   documented (99.5% skeleton agreement once depunctuated). Counting only
   tokens that carry an Arabic letter leaves 3, and all 3 are §4.1's known
   modes — two catastrophic truncations (6→2 and 9→2 words, leaving a
   Qur'anic pause mark) and one word *split*, `تعذریننی` → `تَعَذَّرَ نَنْ`,
   on a token containing Farsi yeh (U+06CC) from §4.1's non-MSA tail. The
   keep-and-flag decision stands, but the reason is now "it costs 3 rows",
   not "it would delete 8.4% of the data". Regression-tested, because
   re-deriving 218 and acting on it is the trap.
2. **The split is not duplicate-free.** The earlier claim — 2,588 rows,
   2,588 distinct sentences, "unlike QuranMB's 1,642→96" — was true of the
   *raw* strings only. On consonantal skeletons there are 2,542 distinct
   sentences: 44 groups, 46 redundant rows. Small, but the groups are
   structured, not random — 33 of 44 are the **same verse once bare and once
   pre-vowelized** (e.g. `إياك نعبد وإياك نستعين` / `إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ`).
   That made filter *order* load-bearing: deduplicating by lowest id
   discarded the usable member of 16 of those 33 groups. Eligibility now
   outranks id order, which recovered exactly those 16 rows (2,076 → 2,092).

**Volume: 500 committed, ~1,592 declared as an extension.** Plan line 134's
300–500 was costed for a *listening* task; D1 made it text editing and D3
added a second annotator, so 500 is the conservative end. The block is 14,754
phoneme positions and 2,638 words — ~105 flagged utterances (21.0%), of which
59 partially vowelized and 47 short. Whether the remaining ~1,592 eligible
rows are annotated is decided **after** the qualification test gives real
per-utterance timings, and is declared now so a later extension is not
mistaken for a post-hoc enlargement. The seeded draw makes the extension a
superset of the block rather than a different sample.

**Annotate dev, not fresh Common Voice.** `tashkeel_sentence` exists only in
`Iqra_train`, and D5's whole measurement is `tashkeel_sentence` vs `C_gold`,
so fresh Common Voice text would have no `C_auto` to compare against without
first re-running the organizers' vowelizer, which we do not have. The cost is
that dev was selected by the organizers and is not a random slice of MSA; it
is mostly cancelled by the measurement being a delta (§3.3b D2), and is listed
in §9 as a limitation rather than argued away.

**The crossover sub-block is the first 50 in draw order**, taken from inside
the 500 rather than drawn separately, so D3's reliability estimate describes
the block that was actually annotated.

**[1 Oct 2026] The `latin` flag and §4.1.1's `UNFAIR` are not in conflict,
and the seam is the flag.** §4.1.1 decides that a Latin-script row cannot be
scored at phoneme level — the vowelizer deleted the Latin words, and the
phonetizer invents phonemes for them — yet D4 keeps such rows in the pool, and
one landed in the block (rank 380, id 64376: `أثبت استوديو Eddie عكس ذلك، حيث
أصر Eddie على ...`, whose `tashkeel_sentence` is missing both occurrences of
`Eddie`). That is deliberate: the pool is the *sampling frame*, `UNFAIR` is
the *evaluation denominator*, and the flag is what joins them. Dropping the row
from the frame would also drop the annotation that demonstrates the vowelizer's
100% Latin-deletion rate — a real `C_auto` error, which is D5's subject. So it
is annotated, flagged, excluded from the phoneme-level number, and reported as
its own line. 1 row of 500; no reseed.

**What the filter deliberately does *not* do.** There is no Qur'anic/classical
predicate, and the artifact records `quranic_filter_applied: false` so a later
block cannot be mistaken for this one. Qur'anic pause marks (U+06D6–U+06ED)
catch only 6 rows (2 still eligible) and the duplicate groups show plainly
Qur'anic verses carrying none, so the predicate genuinely needs an
authoritative full Qur'an text — the same artifact D1 is conditional on. It is
blocked on the same work, not forgotten.

#### Still open

- **D2** — whether `A := C_gold` is stated as an assumption with the
  2.3%/19.8% band as a sensitivity row, or whether the residual
  interaction term is bounded explicitly. The measurement is a *delta*
  (FR under `C_auto` − FR under `C_gold`, same `P`), so real
  mispronunciations appear in both terms and cancel to first order; this is
  about how much rigour to spend on the residual.
- **D5** — whether the 43.6% vowelizer error rate is promoted from a
  single-annotator listening judgment to an objective `tashkeel_sentence`
  vs `C_gold` DER/WER over the whole annotated set. Recommended yes; the
  pilot's figure is retained as the pilot's own, with its 39.5% sensitivity.
- **D6** — whether `Iqra_TTS`'s augmented rows are reported as a
  clearly-labelled synthetic detection sub-arm to partially recover what D1
  gives up (§3.3 option 3; `run/tts_detection.slurm` exists, zero
  annotation cost). If used, the synthetic fraction goes in
  abstract-adjacent text, not a footnote.

#### What would reverse D1

- Clip → verse recovery fails on the Qur'anic arm, so contribution 3 has
  nowhere to relocate to. Then `A` on the MSA arm becomes the only route to
  a detection number and its weakness has to be accepted.
- A source of MSA learner speech with a materially higher base rate appears
  (§3.3 option 4). Out of scope on this timeline, recorded as deliberate.

### 3.4 Pausal form — the constraint that decides what `C_gold` even means **[MEASURED]**

MSA speakers routinely drop case endings (*i'rab*) in pausal form. So there
are two defensible golds for the same sentence:

- **Prescriptive**: full i'rab, what a grammarian says the sentence is.
- **Descriptive**: what a competent reader would actually produce, pausal
  forms included.

The choice is load-bearing for RQ3. If annotators mark full i'rab while
speakers read in pausal form, every word-final case ending becomes an
`A ≠ C` position — a mass of "mispronunciations" that are nothing of the
kind, which would swamp the base rate and simultaneously manufacture the
exact case-ending effect the paper predicts. That is a self-fulfilling
result and a reviewer will find it.

**Decision, to be pre-registered:** `C_gold` is prescriptive (full i'rab),
`A` is verbatim, and the gap between them is *reported as a measured
quantity* — it is the empirical pausal-form rate — rather than silently
counted as error. The selective-scoring mitigation (plan §4.3) and the
"exclude case endings by rule" argument in RQ3 both operate on exactly this
gap, so measuring it is a contribution, not an inconvenience.

Both arms must use the same convention. QuranMB's `Reference_phn` follows
Qur'anic recitation convention; confirm what it does with word-final
endings before assuming the two are comparable.

**[MEASURED] They are not comparable.** At the utterance-final position,
where the final word carries a case mark:

| Arm | n | realized as a vowel | dropped (pausal) |
|---|---|---|---|
| MSA — `Iqra_train` reference | 2,451 | **34.1%** | **65.9%** |
| Qur'anic — QuranMB reference | 1,425 | **80.1%** | 19.9% |
| Qur'anic — QuranMB annotation | 1,425 | 79.1% | 20.9% |

Close to an exact inversion. The organizers' MSA pipeline applies pausal
treatment; the Qur'anic data follows recitation convention. Mid-utterance
the arms agree (all-word mark distributions nearly coincide), so the
divergence is specifically at the pause — which is what a pausal rule
predicts, and argues this is a real convention difference rather than an
artifact of the counting heuristic.

Three consequences:

1. **The pre-registered decision above stands, and is now load-bearing
   rather than precautionary.** Cross-arm case-ending comparison is
   confounded until the convention is chosen and applied uniformly.
2. **The empirical pausal rate on the Qur'anic arm is ~1 point** (80.1%
   reference vs 79.1% annotation). Reciters realize case endings almost as
   often as the reference says they should, so `C_gold`/`A` disagreement at
   this position is not a meaningful error source there.
3. **The MSA reference largely does not contain utterance-final case
   endings at all.** That hands the "case endings inflate DER but cause
   little downstream harm" hypothesis a concrete mechanism instead of an
   assumption — a result for RQ3 rather than merely a hazard.

Caveats: this is alignment-free and therefore reaches only the final word
of each utterance (the per-word version needs the week-4 phonetizer), and
`safikhan`'s `reference_arabic_string` is a *recovered* text, so the
Qur'anic contrast is not fully independent of its own phoneme string.

#### 3.4.1 The convention choice is not free — it collides with the model

Consequence 1 above says "choose a convention and apply it uniformly", as
if that were a clerical act. It is not, and the collision was only visible
once §4.1's findings landed.

`phoneme_ref` is the **CTC training target** of `mhubert147_per`. That
target is pausal: 65.9% of utterances end on a consonant. So the model has
learned to emit nothing at the utterance-final case-ending position. The
two options are therefore not symmetric:

| Choice | Cost |
|---|---|
| **Keep MSA pausal** (match the model) | Cross-arm case-ending comparison stays confounded — the Qur'anic arm is 80.1% prescriptive. RQ3's cross-arm contrast is not interpretable, and any "MSA has fewer case-ending errors" result is an artifact of the reference, not a finding. |
| **Switch MSA to prescriptive** (match the Qur'anic arm) | `C_gold` now marks a case ending the model was trained never to produce. Every utterance-final position becomes a deletion against the reference — a **systematic, mechanical false rejection**, roughly one per utterance (~3% FR inflation on a ~32-phoneme mean). Indistinguishable in the metric from "diacritization corrupts MDD", which is the paper's own claim. |

The second row is risk (a)'s mechanism again (§4), arriving through the
convention door rather than the inventory door, and it is the more
dangerous option precisely because it is the one that *looks* principled.

This does **not** contaminate the phonetizer round-trip (§4.1): that
validates the mark→phoneme mapping, which is convention-independent, since
`tashkeel_sentence` carries whatever marks it carries and the phonetizer
only has to read them faithfully. The round-trip survives either choice.

**[MEASURED] The mismatch is already present in the published benchmark,
and it is large.** `scripts/case_ending_impact.py`, over all 1,642
QuranMB.v2 utterances, for both checkpoints:

| utterance-final phoneme | `C` reference | `A` annotation | `P` ours | `P` organizers' |
|---|---|---|---|---|
| short vowel | 55.4% | 54.9% | 11.4% | 10.7% |
| long vowel | 16.7% | 16.4% | 21.3% | 22.8% |
| consonant | 28.0% | 28.7% | 67.3% | 66.5% |
| **ends on a vowel** | **72.0%** | **71.3%** | **32.7%** | **33.5%** |

The `C − P` gap is **+39.3 points** (ours) and **+38.6** (the organizers'
own checkpoint). Three things follow.

1. **It is not our training run.** Both checkpoints show it to within 0.7
   points, so it is a property of the shared recipe — specifically of the
   pausal `Iqra_train` CTC target — not of our reproduction.
2. **The reference is not a prescriptive fiction.** `A`, the human
   annotation of what the reciters actually produced, tracks `C` to within
   0.7 points (71.3% vs 72.0%). The reciters *do* realize the case endings;
   the model simply does not predict them.
3. **It is pausal treatment, not generic CTC end-truncation.** Truncation
   would depress *every* vowel-final class. Instead the short-vowel-final
   share collapses (55.4% → 11.4%) while the long-vowel-final share
   *rises* (16.7% → 21.3%). Dropping short final vowels while keeping long
   ones is exactly waqf: the short vowel is the i'rab, the long vowel is
   part of the stem.

**[MEASURED, weak] The cost inside the metric is real but secondary.**
Ablating the final token from `C`, `A` and `P` on the 909/1642 utterances
where `C` ends in a short vowel moves F1 **+0.0168** (0.4406 → 0.4574),
precision +0.0126, recall +0.0243, and removes **322 of 6,180 false
rejections (5.2%)**; on the organizers' checkpoint, 339 of 6,207 (5.5%).
The ablation also shifts the alignment, so read it as an order of
magnitude. A class-count bound agrees: ~39% of 1,642 ≈ 640 utterances
differ in final-phoneme class ≈ 10% of all FR, so the true share is
plausibly **5–10%**.

This corrects an over-reading made *before* the measurement, which put the
share at ~27%. The mismatch is dramatic **as a distributional fact** and
modest **as a metric contribution** — two different claims, and only the
first was ever supported.

Options, none free, to be decided before any MSA number:

1. Keep pausal on both arms — i.e. *re-derive the Qur'anic reference*
   pausally rather than the MSA one prescriptively. Preserves the
   model-reference match on both arms; costs a re-derivation of QuranMB's
   canonical and breaks comparability with the published 0.4414.
2. Switch MSA to prescriptive and **retrain** on prescriptive targets.
   Clean, and the recipe is validated, but it is a training run plus a new
   baseline number that no longer matches the organizers'.
3. Keep both as they are and **restrict RQ3's cross-arm claim** to
   non-final positions, reporting the utterance-final position separately
   as a measured convention difference rather than a comparable error rate.
   Cheapest, and arguably the most honest; costs the strongest form of the
   RQ3 result.
4. Option 3 **plus report the mismatch as a result**. The table above is a
   finding about the field's flagship benchmark, not only about us: the
   Iqra'Eval baseline is trained pausally and scored prescriptively, and
   5–10% of its published false rejections are a convention artifact rather
   than model error. Same cost as option 3 — the measurement already
   exists — and it converts the confound that forces the restriction into
   the justification for it.

#### DECIDED (23 Sep 2026): option 4

Keep both arms' conventions as they are; restrict RQ3's cross-arm claim to
non-final positions; report the utterance-final position separately as a
measured convention difference; **and publish that difference as a finding**.

Rationale. Options 1 and 2 both spend the **0.4414 reproduction anchor** —
the single hardest-won asset the project has, and the thing that licenses a
reader to trust every other number in the paper — in exchange for a
*secondary* research question. That trade is bad at any price. Option 4
costs one table, which `scripts/case_ending_impact.py` already produces.

Option 4 is preferred over option 3 because it changes the rhetorical role
of the constraint. Under option 3 the restriction is a limitations-section
apology ("we could not compare utterance-final positions across arms").
Under option 4 it is the consequence of a stated result, which is the same
sentence read forward instead of backward.

**The claim, as it should appear in the paper:**

> The field's flagship Arabic MDD baseline is trained pausally and scored
> prescriptively. The mismatch is visible in the organizers' own released
> checkpoint — 72% of references end on a vowel, 33% of its predictions do —
> and accounts for 5–10% of its published false rejections. Cross-corpus MDD
> comparison in Arabic is silently confounded by case-ending convention.

Every number in that passage is measured and reproducible from public
artifacts plus `scripts/case_ending_impact.py`.

**Two constraints on how it is written up**, both load-bearing:

- **Do not exceed 5–10%.** The 39-point distributional gap is not the metric
  effect; conflating them is precisely the error made before the ablation was
  run, and it is the sentence a reviewer would attack first.
- **Do not imply the benchmark is invalidated.** The honest framing is "a
  known, quantified, correctable bias", not "the published result is wrong".
  5–10% of FR is ~0.017 F1 — material, not decisive.

Consequences for the rest of the arm: no re-derivation of QuranMB's
canonical, no retrain on prescriptive targets, and `C_gold` for the MSA arm
stays **pausal** to match the model. §3.4's consequence 1 ("choose a
convention and apply it uniformly") is therefore satisfied *within* each arm
and explicitly not across them.

---

## 4. Risk (a): phoneme-inventory mismatch can manufacture the headline **[CLOSED 1 Oct 2026 — and it was the wrong risk]**

> **Verdict.** The diacritizer path emits **zero** out-of-inventory phonemes
> for all four tools over all 2,588 dev utterances, so the extend/map/drop
> choice never arises. The real mechanism is the *opposite* of the one this
> section was written against — the phonetiser cannot overflow, it silently
> **deletes** — and that channel is now instrumented and also clean (zero
> unhandled characters, length conserved to ≤1%). The decision block is
> below; the normalization contract it depends on is §4.1.1.

**[MEASURED, full corpus] The organizers' own MSA data stays inside the
vocab.** Every one of the **2,336,971** `phoneme_ref` tokens across the
`Iqra_train` train split (71,391 rows) is in the 68-token `sws_arabic.txt`,
and all **68 of 68** types are used — the two the dev split never reached
(`<<`, `gg`) do occur at scale. Zero overflow over the whole corpus. They
phonetized MSA text into this inventory exactly, with nothing left over.

**[MEASURED] One real out-of-vocab token does exist, in the sibling
dataset.** `Iqra_TTS`'s `phoneme_ref` carries `<sil>`, which is **not** one
of the 68 tokens (`<` alone is — the glottal stop). `Iqra_train` has none.
It is a worked example of exactly the mechanism this section warns about:
the CTC head cannot emit `<sil>`, so scoring against an unstripped
reference charges a guaranteed false rejection at every silence. Left in,
it would have inflated the very number the §3.3 control was built to
measure. Strip it on both sides. The general lesson for the week-4
inventory diff is that OOV tokens arrive as *markup* — silence, boundary
and unknown symbols — at least as often as as genuine phonemes.

That is evidence, not proof. It constrains *their* phonetizer, not ours,
and says nothing about what CATT/Shakkala/Mishkal/Farasa will emit on text
they never processed. The check below still runs — but it is now expected
to confirm rather than to rescue, and the risk of the inventory
manufacturing the headline result is much reduced. The original argument
follows, since it is what the week-4 task is written against.

---


Carried verbatim in intent from `insights/week-03.md`, restated because it
is the single failure mode most likely to produce a *publishable-looking
but false* result.

`sws_arabic.txt` is a **68-phoneme inventory derived from Qur'anic
recitation**. The MSA arm runs undiacritized Common Voice transcripts
through a diacritizer and then a phonetizer. Any phoneme that path emits
which the CTC head cannot produce is a **guaranteed false rejection for a
purely mechanical reason** — the model is structurally incapable of
outputting the target. In the metric, that is indistinguishable from
"automatic diacritization corrupts MDD", which is the paper's claim.

The check is cheap and must run before any MSA number is generated
(week 4, task 4): run the phonetizer over CV-Ar transcripts diacritized by
each tool and diff the resulting inventory against the 68 tokens.

If they differ, the choice is explicit and goes in the paper's **methods**,
not its rebuttal:

| Option | Cost |
|---|---|
| Extend the vocab | Breaks cross-arm comparability; the Qur'anic baseline is no longer the same model |
| Map onto the 68 | Loses phonetic distinctions; must be stated per-phoneme |
| Drop affected positions | Loses coverage; biases which utterances survive |

Record the decision with its justification and the affected phoneme counts.

#### DECIDED (1 Oct 2026): none of the three. The choice does not arise — and the risk was pointing the wrong way.

**[MEASURED, all 2,588 dev utterances, all four tools]** The diacritizer path
(`sentence → normalize → strip → diacritizer → phonetizer`) emits **zero
out-of-inventory phonemes**, for every tool:

| Tool | Phoneme types | OOV tokens | Chars with no phonetiser rule | Predicted/reference length | Utterances >10% short |
|---|---|---|---|---|---|
| `catt-eo` | 66/68 | **none** | none | 83,830 / 83,849 = **0.9998** | 2 (0.08%) |
| `catt-ed` | 67/68 | **none** | none | 83,474 / 83,849 = **0.9955** | 27 (1.04%) |
| `shakkala` | 66/68 | **none** | none | 83,716 / 83,849 = **0.9984** | 36 (1.39%) |
| `mishkal` | 66/68 | **none** | none | 83,012 / 83,849 = **0.9900** | 133 (5.14%) |

Artifact: `run/phoneme_inventory_diff.json`; script
`scripts/check_phoneme_inventory.py`. 66/68 is not a shortfall — the dev
`phoneme_ref` **reference** also uses 66/68, missing the same `<<` and `gg`
that §4's full-corpus count found only at train scale. Three tools match the
reference's type coverage exactly; `catt-ed` reaches one more (necessarily
`<<` or `gg`, both inside the inventory).

So **no vocab is extended, nothing is mapped, no positions are dropped**, and
the risk of the inventory manufacturing the headline is closed, not merely
downgraded. Risk (a) is **retired as stated**.

**But the inventory diff on its own would have returned a false all-clear,
because it cannot fail.** The vendored Halabi phonetiser's
`arabicToBuckwalter` passes an unknown character through unchanged (`else:
res += letter`); it then matches no phoneme rule, and **nothing is emitted
for it**. Read from the source and confirmed by probe. The consequence is
that risk (a) was written backwards: an unhandled character does not
announce itself as an out-of-vocabulary token, it **silently shortens the
canonical sequence**, and at evaluation the speaker's audio for it becomes an
insertion against the reference — the same mechanical false rejection risk
(a) predicted, arriving through a channel no inventory diff can see. A
phoneme-level diff is therefore necessary and worthless alone.

**Hence three diffs, not one**, which is what the table above reports:

1. **phoneme-level** — tokens outside the 68. Structurally cannot fire;
   kept as the regression guard it is.
2. **character-level** — characters in each tool's *output* that the
   phonetiser has no rule for (`normalize.unsupported_characters`). This is
   the diff that could have fired. **It did not: zero, for all four tools.**
   No diacritizer invents a character the phonetizer then eats.
3. **conservation** — phoneme count against the reference. A silent drop
   looks like a short sequence, not a bad token. All four tools land within
   1% in aggregate; the per-utterance tail is the real signal, and it
   separates the tools by an order of magnitude (0.08% → 5.14%).

**The conservation column is a result, not a check.** It is the first
tool-ranking evidence the arm has produced, it is independent of any
annotation, and it is ordered as expected on architecture: the two neural
seq2seq tools and Shakkala conserve length to ≤0.2%, while `mishkal`'s rule
base is 1% short in aggregate and >10% short on 5.14% of utterances. That
tail is not an inventory problem — it is `mishkal` declining to diacritize,
which is a *diacritization* error and squarely the paper's subject. Pinned
here so RQ1's per-tool ordering can be checked against a number that
predates it. The 0.9998 for `catt-eo` is also the strongest evidence yet that
the pipeline is wired correctly end to end: an independently diacritized path
reproducing the organizers' own reference length to 2 parts in 10,000 is not
something a broken normalization step would permit.

**What is *not* closed.** The audio was never consulted. Conservation
compares our phonemes to the organizers' reference phonemes, both derived
from text, so a tool can conserve length perfectly while being wrong about
every vowel. Phoneme *accuracy* per tool is RQ1 and needs §3.3b D5's
`C_gold`.

### 4.1 The normalization contract **[MEASURED — new item]**

The week-4 phonetizer exit criterion is a round-trip on QuranMB's own
Arabic strings, which validates the phonetizer **on Qur'anic orthography
only**. The inspection of what MSA text actually adds turned up more than
expected.

**A second, much larger round-trip reference exists.** `Iqra_train` ships
`tashkeel_sentence` — the organizers' in-house vowelizer's output, fully
diacritized in every row (mean rate 0.783, zero rows below 0.20) —
alongside `phoneme_ref`. So

```
phonetize(tashkeel_sentence) == phoneme_ref
```

is a **73,979-utterance round-trip that covers MSA orthography**, against
QuranMB's 1,642. ~~This should be folded into the week-4 exit criterion~~ —
**done, and it became the criterion outright** (2026-10-01): QuranMB's
round-trip turned out to be circular and unscoreable, so this reference is
the only one there is. **PASS at 99.85%** on the dev split (2584/2588, TER
0.000048, 0 out-of-inventory tokens); the 71,391-row train split has not
been run, so 73,979 is the reference that *exists*, not the number
validated. The direction now untested is the reverse of the one this section
was written to flag — Qur'anic orthography, where alef wasla U+0671 occurs
in 0 MSA rows. On the
train split `tashkeel_sentence` is non-empty in 71,391/71,391 rows (mean
diacritization rate 0.785, p05 0.660) and `phoneme_ref` is empty in only
**15**, so the usable reference is 71,376 rows from train plus the dev
split.

**`sentence` is not the bare input the arm assumed.** It is heterogeneous,
and more so at scale than the dev split suggested: on train, 54.4% is
effectively undiacritized (rate < 0.05) but **32.3% is heavily diacritized**
(≥ 0.50) — against 18.9% on dev. Fully a third of the corpus arrives
pre-vowelized. The MSA arm needs an explicit, pinned stripping step; "the
transcripts are undiacritized" is not true as stated, and is least true
exactly where the text is Qur'anic or classical.

> **Partly closed by §3.3b D4** (2026-10-01). For the *annotation* block the
> resolution is exclusion rather than stripping: rows at ≥ 0.50 are dropped
> from the sampling frame (489/2,588 on dev), because stripping and
> re-diacritizing them would measure whoever vowelized them, not our
> diacritizers. These thresholds are now executable in
> `arabic_mdd.data.msa_pool`. The pinned stripping step is still needed for
> the *RQ1 deployment corpus*, which keeps every row.

**The vowelizer's own normalization is punctuation removal.** Consonantal
skeletons of `sentence` and `tashkeel_sentence` agree in only 49.7% of
train rows — but **99.5%** once punctuation is stripped, and 50.0% of rows
contain punctuation. Any diacritizer we run must see the same depunctuated
text or the comparison is not like-for-like.

**The vowelizer loses content — and now we know the rate.** The residual
0.5% is genuine loss, not punctuation:

| Behaviour | Rate on train | Example |
|---|---|---|
| Rows losing >30% of their words | **552/71,391 (0.8%)** | "إنَّ النَّفْسَ لَأَمَّارَةٌ بِالسُّوءِ" → `ۚ` (everything but a pause mark) |
| Latin script silently dropped | 21/71,391 rows contain it; **dropped in 21/21 (100%)** | "ظهرت الأغنية في الألبوم المصغر \"Memorial Address\"" → no Latin |
| Ordinary vowelizer errors | — | "عَمَلٍ" for "عَمِلَ"; "تعذریننی" → "تَعَذَّرَ نَنْ" |

`phoneme_ref` is the CTC **training target**, so a dropped word is audio
the model was trained to emit nothing for, and at evaluation it is a
guaranteed insertion against the reference — the same
mechanical-false-rejection mechanism as risk (a), already present in the
organizers' own data and therefore in the validated baseline. **The verdict
is "real but not material": 0.8%**, with a median word-count ratio of
exactly 1.000. Too small to explain any headline effect, large enough that
these rows must be excluded from the round-trip's denominator rather than
counted as phonetizer failures. Latin script is the opposite case — a 100%
drop rate, but on 0.03% of rows.

**U+0670 moves in both directions, and my dev-split reading of it was
wrong.** The vowelizer strips every superscript alef it is given
(**0/120 survive**) and then inserts its own on **2,738 rows (3.8%)** where
the source had none, following its own orthographic convention
("الرحمن" → "الرَّحْمَٰن", "موسى" → "مُوسَىٰ", "على" → "عَلَىٰ"). So this is
not a rewrite-to-alef rule; it is a normalization-then-restoration. The
consequence for us is the reverse of what §4.1 first said: U+0670 is a
**live character in the phonetizer's input**, present in 3.8% of rows, not
a rarity to be stripped. Handle it in `tashkeel_sentence`, not in
`sentence`.

#### 4.1.1 The contract, decided and executable **[DECIDED 1 Oct 2026]**

~~**Decide before annotation**, since annotators hit all of this on day
one: the rule for Latin script and digits (transliterate / spell out / drop
the utterance), the punctuation rule, and the U+0670 rule.~~ **Decided.** The
contract is `arabic_mdd.data.normalize` (14 tests), run identically on the
reference path and on every diacritizer path — which is what makes `C_auto`
and `C_gold` comparable at all. It deliberately does **not** strip
diacritics; that is the separate `strip` step and applies to one path only.
`base.Diacritizer` still refuses to normalize its own input, so this stays a
single visible stage rather than four tool-specific ones.

Rates are over the whole `Iqra_train` train split (71,391 rows):

| Input class | Rule | Rows | Row usable? |
|---|---|---|---|
| Punctuation | → **space**, not deleted | 35,687 (49.99%) | yes |
| Non-phonemic marks (tatweel, U+06D6/U+06DA Qur'anic annotation) | dropped | 222 (0.31%) | yes |
| Persian-script look-alikes (farsi yeh, keheh, heh doachashmee) | mapped to the Arabic letter | 108 (0.15%) | yes |
| Presentation forms (`ﻻ` → `لا`) | NFKC-folded | 4 (0.01%) | yes |
| Latin script | characters dropped, row flagged | 21 (0.03%) | **no** |
| Letters with no MSA phoneme (`چ`, `ڨ`) | dropped, row flagged | 4 (0.01%) | **no** |
| Digits | spell out — **vacuous rule** | **0** | — |
| Genuinely unexpected characters | dropped, row flagged | 22 (0.03%) — but 21 are the Latin rows above; the 22nd is one U+262D HAMMER AND SICKLE | yes |
| U+0670 dagger alef | **kept** here; handled in `phonetizer.normalize` | 2,738 (3.8%) | yes |

**Total exclusion cost: 25 rows (0.035%).** `UNFAIR = {latin, digits,
no_msa_equivalent}` — the classes where the audio contains speech the
reference has no phonemes for, so cleaning the text does not fix the row,
only dropping it does. Everything else is cleaned and kept. After
normalization, residual characters the phonetiser has no rule for: **none**,
over all 71,391 rows.

Four of these need their reasoning on the record, because the obvious rule
is wrong in each case:

1. **Punctuation maps to a space, not to deletion.** Deleting it merges the
   neighbouring words, and the phonetizer applies vowel-length and glide
   rules across the join: `فِي الشَّمْسِ` is 6 tokens spaced and 8 merged
   (`f i y aa ...` — a spurious glide and a lengthened vowel). At 50.0% of
   rows this is the single highest-traffic rule in the contract.
2. **Digits: the rule is vacuous and that is worth recording.** Zero rows
   contain a digit, ASCII or Arabic-Indic, in all 71,391. The flag exists so
   that a future corpus cannot slip them in silently, but no transliteration
   policy had to be invented, and the annotation guidelines do not need an
   entry for it.
3. **Latin excludes the row rather than merely cleaning it.** Latin letters
   are *valid Buckwalter symbols*, so `Memorial` is transliterated as though
   it were Arabic and yields `m r i a l` — five invented phonemes. The
   vowelizer's own answer was to delete it (21/21 rows, §4.1 above), so
   `phoneme_ref` is short by exactly those words. Neither side can be right
   here: cleaning gives a reference with a hole, keeping gives one with
   fiction.
4. **Qur'anic annotation marks are dropped on evidence, not by assumption.**
   82 of them sit in `tashkeel_sentence`, which round-tripped at 99.85% with
   the phonetizer discarding them, so dropping them is the validated
   behaviour. Classified by Unicode category (`Mn`) rather than by a
   hardcoded range, so the whole annotation block is covered. They are
   reported as `non_phonemic_mark`, not as unexpected characters — an
   earlier version flagged 136 dev rows as "unsupported" on this account.

**Two false alarms the contract had to be built around.** `PRESENTATION_FORM`
was first detected as "NFKC changed something", which reported 6% of rows as
containing presentation forms when the whole corpus holds four such
characters: NFKC *also* canonically reorders `<shadda><harakah>`, on ~24.5%
of rows. It is now an explicit codepoint-range test. And the `SUPPORTED` set
is **derived** from the vendored phonetiser's own Buckwalter map (44
characters: 36 letters + 8 combining marks) rather than hand-listed, so it
cannot drift away from what the phonetizer really handles if the vendored
module is re-pinned.

**Ordering constraint, now written into both modules.** Unicode
normalization must run *here*, before phonetization, and never after it.
Canonical order for a shadda'd, voweled consonant is `<harakah><shadda>`
(combining classes 30 then 33); Halabi's rules want the non-canonical
`<shadda><harakah>`, which `phonetizer.normalize` produces. An NFC pass
afterwards would reorder it back and silently degeminate every geminate in
the corpus. The comment in `phonetizer.py` previously had this backwards and
called Halabi's order "canonical"; corrected, and pinned by a test that
phonetizes both spellings of `إِنَّ` and asserts they agree.

**[MEASURED] Alef wasla closes week 4's "asserted, never measured" gap, in
the direction nobody was watching.** U+0671 is absent from the 44-character
Buckwalter map, so it is dropped — and with it the phonemes it carries:
`ٱلْحَمْدُ` loses the leading `< a` that `الْحَمْدُ` produces. It occurs in **0**
MSA dev rows, which is why the MSA round-trip could not see it, but it is
ordinary Qur'anic orthography and therefore sits on the *Qur'anic* arm's
label path — the one D1 is conditional on. Carried to §9 as a live gap, not
closed.

**What the annotator is told** (feeds D3's guidelines): the normalizer runs
*before* the text reaches them, so they never see punctuation, tatweel,
Qur'anic annotation marks or Persian look-alikes. They do see Latin tokens,
because dropping those silently would present them a mutilated sentence. The
instruction is to leave Latin tokens untouched rather than transliterate
them — the row is excluded from phoneme-level scoring anyway, so inventing a
transliteration would be unpaid work on a row that cannot be scored.

---

## 5. Risk (b): no reference point makes a weak model unfalsifiable

The `-u hubert_base` mistake in week 3 cost 1,282 excess false rejections
and was caught only because 0.4414 existed to contradict 0.4058. **The MSA
arm has no such number.** A silently wrong or weak upstream there reads as
"MSA is harder" — a confound sitting directly underneath the
Qur'anic-vs-MSA comparison that is the paper's argument.

### 5.1 Pin everything, across both arms

Anything that differs between arms and is not deliberate is a confound.
Pin and state:

| Held fixed across arms | Where it is set |
|---|---|
| Upstream identifier | `utter-project/mHuBERT-147` |
| Upstream **revision** | **[PINNED]** `7ad3fc0bc5106c58c9c13526abccad527150d135`, via `scripts/baseline_reproduction/pin_upstream.py` |
| Frozen vs fine-tuned | `upstream_trainable: False` |
| Featurizer | SUPERB weighted layer-sum over 13 hidden states |
| Vocab | 68-token `sws_arabic.txt` (+3 specials → 71 output units) |
| Training budget | `runner.total_steps: 200000` |
| Training data | Iqra_train + Iqra_TTS, 131 h |

Any deliberate difference — for instance extending the vocab under §4 — is
stated in the methods and its effect on cross-arm comparison argued
explicitly.

**[MEASURED, 2026-09-23] How the upstream is pinned, and the trap in the
obvious way of doing it.** `s3prl/run_downstream.py` exposes
`--upstream_revision`, documented as "The commit hash of the specified
HuggingFace Repository". Adding it to our command line would pin **nothing**
while *looking* like provenance — the worst of both, because it would be
recorded in the checkpoint's `Args` and read as settled. Two facts:

- `--upstream_revision` is consumed at exactly one site,
  `s3prl/downstream/runner.py:125`,
  `snapshot_download(self.args.upstream, self.args.upstream_revision, ...)`.
  That is the path where **`--upstream` itself** names an HF repo holding an
  s3prl checkpoint. Ours is `-u hf_hubert_custom`, a built-in upstream.
- Our path reaches `s3prl/upstream/hf_hubert/expert.py`, whose
  `UpstreamExpert.__init__(self, ckpt, **kwds)` calls
  `Wav2Vec2FeatureExtractor.from_pretrained(ckpt)` and
  `HubertModel.from_pretrained(ckpt)`. `**kwds` is accepted and never read;
  neither call receives a `revision`.

So the pin is done the only way that works: resolve the repo at the sha,
materialize it into `run/upstreams/mHuBERT-147-<sha12>/`, and pass that
**directory** to `-k`. `from_pretrained` accepts a local path, the sha is in
the directory name, and it therefore lands in each trained checkpoint's own
`Args.upstream_ckpt` where `inspect_s3prl_ckpt.py` can read it back.

**The pin is retroactively safe.** No weight or config file in the repo has
changed since **2024-06-12** (`model.safetensors` 2024-06-12,
`pytorch_model.bin` / `config.json` / `preprocessor_config.json`
2024-03-14); every commit since is README/metadata churn, HEAD included
("Fix YAML language metadata issue for Norwegian"). Pinning to HEAD is
therefore byte-identical to what the validated 0.4406 run consumed — it
removes the possibility of silent drift rather than introducing a change.

**Still unpinned, and larger.** `run/prepare_baseline.sh:81` clones
`https://github.com/s3prl/s3prl.git` at whatever `main` is that day. S3PRL
is the featurizer, the downstream model and the training loop, so this is a
wider exposure than the upstream weights ever were. Not fixed here because
it needs a commit chosen against the venv that is known to work; logged in
`docs/weeks/week-04.md` → *Carried forward*.

### 5.2 The dev-split contamination

The validated checkpoint is `dev-best.ckpt`: selected on the CV-Ar **dev**
split. If the MSA arm's evaluation set is drawn from that same dev split,
the acoustic model has had model-selection exposure to it and the MSA
number is optimistic by an unknown amount. That biases the arm in the
direction that *weakens* the paper's claim (MSA looks better than it is),
so it is not fatal — but it is not defensible to leave undeclared either.

Cleanest fix: draw the MSA evaluation utterances from Common Voice Arabic
rows that `Iqra_train` never touched at all. Common Voice Arabic is far
larger than the 79 h Iqra_train slice, and the MSA arm does not need the
organizers' `phoneme_ref` — `C` is generated by our own pipeline either
way. This also side-steps any question about how their in-house vowelizer
built those references.

**Deduplicate by sentence text, not by utterance id.** Common Voice prompts
are read by many different speakers, so the same sentence recurs across
splits. An utterance-id-disjoint set can still be text-overlapping with
training, which for a text-conditioned system is leakage of exactly the
wrong kind.

### 5.3 Record provenance from the checkpoint, not the submission script

For every MSA-arm training or inference run, capture `Args`/`Config` with
`inspect_s3prl_ckpt.py` and store it alongside the results. Week 3 is the
proof this matters: the organizers' own README specified an upstream that
their own published checkpoint contradicts.

---

## 6. Risk (c): domain shift versus reference corruption

The 2×2 separates label-path from input-path corruption *within* an arm. It
does not separate either from the Qur'anic→MSA domain difference. The
comparison "F1 drops more on the MSA arm" has at least four candidate
causes:

1. Reference corruption — the claim.
2. Domain shift in the acoustics (recitation vs read prose, recording
   conditions, speaker population).
3. Inventory mismatch (§4).
4. Base-rate difference in mispronunciations (§3.3).

The mitigations, in order of how much they buy:

- **Run the full 2×2 inside each arm separately.** The label-corruption
  cell (`model given C_gold`, scored against `C_auto`) is a *within-arm*
  contrast, so domain shift cancels out of it. This is why the
  decomposition is the spine and the cross-arm magnitude is supporting
  evidence — a point worth making in the paper, not just in the plan.
- **Report the Qur'anic arm's `C_auto` condition too.** Stripping QuranMB's
  diacritics and re-diacritizing gives a corruption measurement in a domain
  where the acoustics are held fixed. Cross-arm comparison then contrasts
  two *effects*, not two absolute numbers.
- **State the remaining confound.** Don't argue it away.

---

## 7. Licence and release

The annotated slice is a **releasable artifact in its own right** — a
gold-diacritized evaluation subset of Common Voice Arabic — and is
contribution 5 in plan §5. Common Voice is CC0, which should be permissive,
but verify before promising a release in the paper. This now gates a
deliverable, not just diligence.

The separate `QuranMB.v2` / `Iqra_Extra_IS26` licence question (open since
week 2) does not block the MSA arm but blocks any combined release.

---

## 8. Build order

Mapped onto plan §6's timeline. Items marked ▲ are prerequisites that will
silently corrupt everything after them if skipped.

| When | Work | Ref |
|---|---|---|
| ✅ | ~~CV-Ar transcript inspection: non-Arabic tokens, digits, normalization rules~~ | §4.1, step 1 |
| ✅ | ~~Cross-arm word-final convention check~~ | §3.4, step 2 |
| Wk 4 | Diacritizer wrappers; phonetizer; round-trip on QuranMB **and on `tashkeel_sentence`** | [week-04](weeks/week-04.md) t1–3, §4.1 |
| ✅ | ~~▲ Pin the normalization contract: punctuation, Latin script, digits, U+0670~~ — **decided**, executable in `arabic_mdd.data.normalize`; exclusion cost 25/71,391 rows | §4.1.1 |
| ✅ | ~~Quantify vowelizer content loss on the train split~~ — 0.8% | §4.1 |
| ✅ | ~~▲ Phoneme-inventory diff against the 68-token vocab; record the decision~~ — **zero OOV for all four tools; extend/map/drop never arises.** The diff that mattered was the character-level one (also clean) | §4 |
| ✅ | ~~▲ Pin mHuBERT-147 to a commit sha~~ — `7ad3fc0bc510`, via a local snapshot; `--upstream_revision` is a no-op here | §5.1 |
| Wk 4 | Audio-informed vs text-only diacritization on CV-Ar — decides the annotation anchor | §3.2 |
| ✅ | ~~Base-rate probe on CV-Ar dev (step 3)~~ — 0.0708, inconclusive | §3.3 |
| ✅ | ~~Zero-base-rate control on `Iqra_TTS` original rows~~ — 0.0066, non-binding | §3.3 |
| **now** | ▲▲ 50-utterance listening pilot from `run/msa_pilot_sample.json` (step 4) — **the only remaining way to settle the base rate.** Originally scheduled Wk 5; pulled forward to *now* because both cheap proxies came back non-binding, and because its outcome decides what the MSA arm can claim (§9). Not a week-4 dependency, but the longest pole in the project — start it in parallel with week 4's text work. | §3.3 |
| **now** | MSA detection metric on real positives — `run/tts_detection.slurm` (option 3, existing data) | §3.3 |
| ✅ | ~~▲ Resolve the cross-arm case-ending convention~~ — **option 4 decided**; MSA `C_gold` stays pausal | §3.4.1 |
| Wk 5 | ▲ Select the MSA evaluation pool: CV-Ar rows disjoint from Iqra_train, deduped **by consonantal skeleton, not raw string** — raw-string dedup reports `Iqra_train` dev as duplicate-free and misses 46 rows (§3.3b D4) | §5.2 |
| ✅ | ~~▲ Fix the annotation sampling frame and draw the block~~ — **D4 decided**: 500 from a 2,092-row pool, seed 20261001, in `run/msa_annotation_block.json` | §3.3b D4 |
| Wk 5 | ▲ Pre-register the annotation protocol in writing — pausal-form convention included | §3.2, §3.4 |
| ~~Wk 5~~ | ~~▲ 50-utterance pilot: measure the real mispronunciation base rate~~ — **duplicate of the "now" row above**, kept struck rather than deleted so the schedule change is visible: this moved from Wk 5 to now. | §3.3 |
| ✅ | ~~Decide the §3.3 option in light of the pilot~~ — **D1 decided: option 2, label-path only, `A` dropped** | §3.3b |
| Wk 5 | RQ1 measurements on both domains | §3.3 |
| **now** | ▲ Qualification test for both annotators — blocks the annotation block, and the WikiNews gold set's availability/licence is unconfirmed | [annotator-qualification.md](annotator-qualification.md) |
| **now** | ▲▲ **Clip → verse recovery feasibility probe** on the Qur'anic arm. Promoted from chore to **precondition** by D1: the label path needs trustworthy diacritized text and the only text column we have is circular. Cheap to probe — 1,642 rows collapse to 96 distinct sentences. If it fails, *both* arms lose the label path | §3.3b D1 |
| Wk 5–8 | Annotation proper: the drawn **500** (D4 fixed the block; plan line 134's 300–500 was costed for a listening task and is conservative now that it is text-only with two annotators), crossover on the first 50 in draw order. ~1,592 further eligible rows declared as an extension, decided on qualification-test timings | §3.2, §3.3b D4 |
| Wk 6 | Label-path column on the Qur'anic arm (block gate) — **now gated on the clip → verse recovery above**; RQ5 interaction power estimate | plan §6 |
| Wk 7–9 | MSA-arm label path; text-dependent arm; XLS-R prompt-free | plan §6 |
| Wk 10–12 | Full 2×2; RQ3 weights; RQ4 abstention; feedback-validity number | plan §6 |

---

## 9. Things that would falsify the arm, stated in advance

So that a bad outcome is a finding rather than a scramble.

- **The mispronunciation base rate on CV-Ar is near zero.** Then the arm
  measures the label path only (§3.3 option 2), the headline F1-bias number
  comes from the Qur'anic arm's `C_auto` condition, and the MSA arm's
  contribution becomes the corruption-attributable FR rate. The paper
  survives; the framing shifts. **[PARTLY TESTED]** Two cheap screens both
  came back non-binding (§3.3): the CV-Ar dev probe is confounded by an
  in-domain checkpoint, and the `Iqra_TTS` control bounds nothing because
  the model is near-memorizing that set. Both available reference points
  are confounded in the *same* direction and there is no third, so this
  condition can now only be evaluated by the step-4 listening pilot.
  Budget for it accordingly — it is on the critical path, not optional.
  If it does fire, `Iqra_TTS`'s augmented rows keep option 3 open on data
  that already exists.
- ~~**The phoneme inventories are incompatible and no mapping is
  defensible.** Then the two arms cannot share a model and the cross-arm
  comparison is cut. RQ2 survives within-arm; the natural-experiment framing
  weakens to a discussion point.~~ **[DID NOT FIRE — 1 Oct 2026]** All four
  diacritizers stay inside the 68 tokens on all 2,588 dev utterances, and
  emit no character the phonetiser silently eats (§4). The arms can share a
  model.
- **[LIVE, replaces the above] A character the phonetiser cannot transliterate
  reaches the *Qur'anic* arm's label path.** The failure mode is deletion,
  not overflow, so it is invisible to an inventory diff (§4). One instance is
  already known: **alef wasla U+0671** is absent from the vendored Buckwalter
  map, so `ٱلْحَمْدُ` silently loses the `< a` that `الْحَمْدُ` produces. It occurs
  in **0** MSA dev rows — which is why the 99.85% MSA round-trip says nothing
  about it — and is ordinary Qur'anic orthography. If the Qur'anic text
  recovered by the clip → verse probe carries it, every affected position is
  a mechanical false rejection on the arm that supplies the headline number.
  The instrument exists (`normalize.unsupported_characters`); run it on that
  text the day it exists, before any label-path number.
- **Inter-annotator agreement is poor on diacritization.** Then `C_gold`
  itself is noisy and the "gold" label is not earned. Report the agreement
  figure regardless and treat `C_gold` as a high-quality reference rather
  than a gold standard, with the noise floor stated.
- **Anchoring is large.** Then correct-the-machine has to be abandoned for
  the affected subset, which costs schedule. Measured at ~50 utterances
  precisely so this is discovered in week 5, not week 9.
