# Week 6 implementation plan

Dates: 26 Oct – 1 Nov 2026 nominal. Closes the week 4–6 block in
[mdd-paper-project-plan-v2.md](../mdd-paper-project-plan-v2.md) §6, of which
this week owes the one remaining item — "**run the full label-path column on
the Qur'anic arm**" — and the block gate: "**label-path effect measurable, and
interaction power estimated**".

This is the first week whose deliverable is a *paper* number rather than
machinery. Plan §6 is explicit about why it matters more than its size
suggests: "If the decomposition does not work there, it will not work
anywhere, and you find out in November with fifteen weeks left."

See [week-05.md](week-05.md) for the nine closed tasks and
[insights/week-05.md](../../insights/week-05.md) for the twelve experiments.
[docs/msa-arm.md](../msa-arm.md) §3.3b (D1–D8), §4.1.1 (the normalization
contract) and §9 (falsifiers) are the sections this week acts on.

**Calendar note.** Week 5 closed on **14 Oct**, twelve days early, and week 4
closed nine days early before it. The slack is real and it is now ~3 weeks
against plan §6's "three weeks of genuine slack ... something will break."
Two things argue against spending it on pulling week 7 forward: annotation is
still the longest pole and is still blocked on a tool nobody has written
(task 4), and week 7's training work is blocked on an unpinned S3PRL checkout
(carried forward). Spend the slack on those two, in that order.

## Where week 5 left things

RQ1 is answered on both domains and the metric is calibrated exactly — all
sixteen of CATT's published Table-5 figures reproduced to the table's own
printing precision. Six things from that work set this week's shape, and the
first two change the design rather than informing it.

1. **There are two DER conventions and they are not interchangeable.** Ours
   (`metrics/der.py`) and the published one (`metrics/der_published.py`)
   differ by a *measured* +1.04 (`catt-eo`) to +4.66 (`mishkal`) on CE DER.
   Every rate this week reports must name one. This is now a project-wide
   rule, not a week-5 footnote.

2. **`catt-eo`'s MSA number was an agreement rate, and it understated its
   error by ~5×** (1.346 against `tashkeel_sentence` vs 6.736 against gold
   Qur'anic text), while the other three tools put the same comparison at
   1.05–1.22×. The asymmetry points at `tashkeel_sentence` being EO-derived.
   Consequence for this week: **the Qur'anic arm's `C_auto` must not be built
   with a tool correlated with the arm's own reference**, and on the Qur'anic
   arm it is not — the reference is the Qur'an. The MSA arm in weeks 7–9 has
   no such protection, which is D5's problem and is why asking the organizers
   which vowelizer they used is still worth the email (task 8).

3. **`phonetize(recovered_verse)` agrees with the benchmark's own
   `reference_phoneme_string` on only 54 of 98 prompts** (ceiling 97). This is
   the week's central design problem and task 1 exists for it.

   **[14 Oct 2026 — measured, and it is not the dagger alef.]** An earlier
   version of this item asserted the 44 inexact prompts were "dominated by
   one disagreement," U+0670. That was an inference from the DER work, not a
   measurement, and it is wrong: only **21 of 98** prompts contain a dagger
   alef at all, and only **10 of the 44** mismatches do. The other **34**
   contain none. The real picture, from 64 diff sites over the 44 prompts:
   **59 are utterance-internal** (so not a pausal/case-ending convention
   gap), they are long/short vowel and gemination disagreements, and the
   single most frequent site occurs **4** times — scattered, not systematic.
   The signatures that do recur (`uu` → `u w aa`, a dropped `n`) look like
   **tajwīd**: madd lengthening and nūn assimilation, which a general-purpose
   MSA phonetiser does not implement and which a recitation reference does.

   Consequence for task 1: the gap is a convention difference between a
   general phonetiser and a recitation-aware reference, **not a bug one rule
   fixes**. That makes leg B's separately-reported number mandatory rather
   than tidy, and it removes the option of closing the gap first and using
   candidate (i).

4. **The verse map is done and audio-verified.** 98/98 located,
   `unsupported_characters` **{}** over all 4,442 recovered characters, the
   alef-wasla falsifier measured as not firing on the `simple` edition
   (U+0671: 0), edition sha-pinned. So `C_gold` text exists for the Qur'anic
   arm, which is what unblocks this week.

5. **The baseline is reproduced and its predictions are on disk.** F1 =
   **0.4415** with the organizers' checkpoint (gate PASS, ±0.0001 of the
   published 0.4414 on every component rate), 0.4406 with our own retrain.
   `run/quranmb_predictions_official_ckpt.json` is `P`; QuranMB.v2's
   `annotation_phoneme_string` is `A`. Both are fixed inputs this week — only
   `C` moves.

6. **The evaluated set is final at four tools** (D8, 14 Oct). The frontier
   fifth condition is declined as a *condition* and scheduled as a single RQ1
   row on MSA gold once annotation closes — so **this week's label-path column
   has four `C_auto` cells and that number will not change under it.** The
   disqualifying criterion was RQ4: a frontier API exposes no per-position
   confidence, so it could not appear in the abstention curve. D8 carries
   three declared re-opening conditions, none of which can fire this week.

What did **not** advance: ~~annotation has not started, because no tool writes
the pre-registered logging schema;~~ annotator 2 has not sat the qualification
test; and S3PRL is still unpinned.

**[14 Oct 2026]** The tool now exists (task 4), so the blocker on the first
annotated utterance is no longer code — it is annotator 2's qualification
test, which is external latency and is task 8's first item for that reason.

**[16 Oct 2026]** S3PRL is pinned (task 5), at
`ec8064b5889f81ca460fbe2c094ce576a6f120b7` = upstream v0.4.18, which is still
upstream `main` — so the drift this had been carried for since week 4 never
occurred. `transformers` remains unpinned and is the same exposure.

## Scope

The whole week is one measurement plus the gate's second clause, and the
measurement is cheap in compute and expensive in definitions. Nothing here
needs a GPU: the Qur'anic `C_auto` is 98 prompts of 4,442 characters total,
and week 5 measured `catt-eo` at 0.015 s/verse and `catt-ed` at 1.205 s/verse
over 6,236 verses — so all four tools over 98 prompts is seconds, not hours.

| what | blocks | blocked on |
|---|---|---|
| the `C_gold` definition (task 1) | the entire label-path column | ~~nothing — a decision, not an experiment~~ **done 14 Oct, D9** |
| the label-path column (task 2) | the block gate, clause 1 | ~~task 1~~ **done 14 Oct, clause 1 PASSED** |
| the power estimate (task 3) | the block gate, clause 2; D2 | ~~task 2's per-utterance variance~~ **done 14 Oct, clause 2 DONE, D2 decided** |
| the annotation logging tool (task 4) | the first annotated utterance → weeks 7–9 | nothing |

The prompt-free baseline never consumes `C`, so for it the 2×2's rows
collapse and **only the scoring column exists** (plan §1). What week 6
delivers is therefore the top-right cell measured against the top-left:
**label-path corruption alone, with the model unimpaired.** The input-path
row needs the text-dependent arm and is weeks 7–9. Say this plainly in the
write-up — a reader who sees "2×2" in week 6 and finds one column will
reasonably ask what happened to the other.

Not this week: any model training, the text-dependent arm, the MSA label
path, RQ3's weights, RQ4's curves.

## Week 6 exit criterion — the block gate

> **The label-path effect is measurable on the Qur'anic arm, and the RQ5
> interaction is powered.**

Two clauses. Both pass/fail by criteria fixed **here**, before the
measurement runs.

### Clause 1 — label-path effect measurable **[PASSED 14 Oct 2026]**

> **[14 Oct 2026]** All four CIs exclude zero (the criterion required only the
> worst tool), point estimates monotone in CE DER, leg B reproduces 0.4415
> exactly, and the alignment-artifact check is clean — 0 prompts show damage at
> zero corrupted sites, damage/corrupted-site 0.108–0.218. The effect's **sign
> is positive**, which falsifies prediction 1; see that prediction's note and
> plan §1's annotation. The failure branch below did not fire and is retained
> as the record of what was pre-committed.


For each diacritizer, with `A` and `P` held fixed and only `C` swapped:
report ΔF1 = F1(scored against `C_auto`) − F1(scored against `C_gold`), with
a **95% CI from a cluster bootstrap resampling the 98 prompts** (not the
1,642 clips — see task 3 on why). "Measurable" means:

> **the CI excludes zero for at least the worst diacritizer** (`mishkal`, 18.499
> published-convention CE DER on this corpus), **and the per-tool point
> estimates are monotone in DER.**

Two-sided, because the sign is not known in advance and plan §1 says so
explicitly: "Nobody knows the sign." Pre-registering a direction here would be
pre-registering the thing the paper is supposed to find out. Task 2 states a
*mechanistic prediction* of the sign separately, which is a different and
weaker commitment.

**The failure branch, declared now.** If all four CIs include zero, that is a
null on the paper's spine, and plan §5's pre-committed position binds: the
paper reports a small effect and the contribution is the protocol, the metric
and the bias correction. It does **not** become a different paper. What a null
*does* trigger, mechanically:

- it must be distinguished from a power failure, which is exactly what clause
  2's MDE is for — a null with an MDE of 0.001 is a finding, a null with an
  MDE of 0.05 is an uninformative experiment;
- it reweights the paper toward RQ3/RQ4 within the existing framing, and the
  week-12 gate's cut order (§5) is unchanged;
- it is reported with the same prominence as a positive result would be.

### Clause 2 — interaction power estimated **[DONE 14 Oct 2026]**

> **[14 Oct 2026]** `scripts/power_rq5.py` → `run/rq5_power.json`.
> **Recommendation: KEEP RQ5.** Worst-corner MDE **0.0907 F1** (mishkal's
> per-prompt variance, `k = 2`, `ρ = 0`, prompt-clustered) against a **0.1635**
> threshold — a factor of 1.80, which is a pass and not comfort, and is
> recorded as such. At the more plausible corner (`k = 1.5`, `ρ = 0.5`) the MDE
> is 0.0591, a factor of 2.77.
>
> The threshold was **not evaluable when this clause was written**: our notes
> had only the endpoint (F1 → 40.52%), not the baseline it drops from. Pinned
> from arXiv:2604.22133 §6.4.3 table 4(c) — CROTTC-LLaMA **56.87%** vs the same
> system with canonical injection **40.52%**, L2-ARCTIC, 900 utterances. A
> within-system ablation, so **16.35 points**. The abstract's 71.77% CROTTC-IF
> figure is the wrong comparison: it is a different system and prices the
> *repair*, not the damage.
>
> **Design effect: 4.0–4.2×**, and it is corroborated analytically — for a
> purely `C`-driven effect the ICC is 1 and the classical design effect is
> `√(1 + (16.76 − 1))` = 4.09×. Quoting the naive MDE (0.018–0.022) would have
> overstated the power by a factor of four.
>
> **One finding the clause did not ask for and the write-up must carry.** RQ5 is
> powered against the *threshold*, not against every tool's own label-path
> effect. `MDE/|ΔF1|` is 0.32 (mishkal) and 0.52 (shakkala) but **1.33
> (catt-ed) and 1.96 (catt-eo)** — for the two good diacritizers the MDE exceeds
> the entire main effect, so an interaction there would have to be larger than
> the effect it modifies. The consequence is a scoping decision, not a cut: RQ5
> is answered on the high-corruption tools, and the low-DER tools are reported
> as lacking the resolution to say. **D2 decided** in `docs/msa-arm.md` §3.3b.

A number, not a verdict: the **minimum detectable difference-in-differences**
for RQ5 at 80% power, α = 0.05, two-sided, at n = 1,642 Qur'anic clips (98
prompt clusters) and at n = 500 MSA utterances, reported both naive and
cluster-corrected so the design-effect inflation is visible.

The decision rule is pre-registered, from plan §6: **if the MDE exceeds the
input-path main effect CROTTC-IF reports, RQ5 is cut in November** rather than
discovered to be underpowered in December. RQ5 is the designated cuttable RQ
(§5's cut order puts it first), so cutting it is a planned outcome and not a
failure.

One honesty constraint on this clause. The text-dependent arm does not exist
yet, so its variance cannot be measured — only assumed. The estimate must
therefore be reported as a **function of that assumption**, with a stated
sensitivity (variance equal to the prompt-free arm's, 1.5×, and 2×), not as a
single number that hides it. An MDE quoted without the assumption it rests on
is worse than no MDE, because it will be cited later as if it were measured.

## Tasks

Ordered by what blocks what.

1. **▲▲ Define `C_gold` on the Qur'anic arm — the decision the whole week
   rests on.** This is a definitions task with a measurement attached, and
   getting it wrong silently confounds the paper's primary number.

   Three candidates exist and they are not equivalent:

   | candidate | what it is | agreement with the benchmark |
   |---|---|---|
   | (i) `reference_phoneme_string` | the benchmark's own canonical; what the week-3 gate scored against | by definition |
   | (ii) `phonetize(recovered_verse)` | our pipeline on the recovered Qur'anic text | **54/98 prompts**, ceiling 97 |
   | (iii) (ii) with a declared U+0670 rule | as (ii), dagger alef realized as a long vowel | **48/98** — *worse*; see below |

   **The trap, stated so it is not walked into.** The tempting choice is (i),
   because it makes the `C_gold` cell reproduce the published F1 = 0.4415
   exactly. But `C_auto` must be built by *our* pipeline — strip,
   re-diacritize, phonetize — so pairing (i) with that `C_auto` makes ΔF1 the
   sum of two differences: the diacritizer's error, which is what we want, and
   our phonetizer's disagreement with the uploader's, which is 44/98 prompts
   of confound. The delta would be attributed entirely to diacritization in
   the write-up and ~45% of the prompts would carry an unmeasured second
   cause.

   **So the design is two legs, mirroring what worked for the week-5
   calibration**, and the legs answer different questions:

   - **Leg A — the delta.** `C_gold := phonetize(recovered_verse)` and
     `C_auto := phonetize(diacritize(strip_diacritics(recovered_verse)))`,
     same phonetizer, same text, same convention on both sides. The *only*
     difference between the cells is the diacritizer, so the delta is
     attributable. **This is the measurement.**
   - **Leg B — the anchor.** Score with `C := reference_phoneme_string` and
     confirm F1 = 0.4415 reproduces. This ties leg A to the published
     condition and makes **the phonetizer's own contribution a third,
     separately reported number** rather than an invisible one. Report
     F1 under (i) and under (ii) side by side; the difference is the price of
     owning the pipeline, and it belongs in the limitations, not in ΔF1.

   **The U+0670 rule must be declared, not defaulted** — the same discipline
   week 5 applied to it in DER, where it was worth 0.85 absolute. The facts:
   `normalize()` drops it on MSA evidence (177/177, and there are 0 MSA dev
   rows with U+0671 so the MSA evidence says nothing about Qur'anic
   orthography); the Tanzil `simple` edition writes it 3,218 times (0.973% of
   330,709 letter positions). Whichever way it goes it applies to **both**
   cells of leg A, so it cannot move ΔF1 much — it moves the *agreement with
   the benchmark*, which is leg B's number.

   **[14 Oct 2026 — measured, so this is now a recording job, not an open
   question.]** Three rules, scored as `phoneme_exact` against the
   benchmark's own canonical over all 98 prompts:

   | rule | exact | of the 21 dagger prompts |
   |---|---|---|
   | delete U+0670 (`normalize()` today) | **54/98** | 11/21 |
   | U+0670 → alef (the "long vowel" rule) | **48/98** | 5/21 |
   | context-dependent: keep after U+0649, alef after a bare fatha | **54/98** | 11/21 |

   The plain long-vowel rule **loses 6 prompts**, and the context-dependent
   one is an exact wash. The reason is in the context audit: of the 24 dagger
   alefs in the prompts, **14 follow alef maqsura** (U+0649, as in `عَلَىٰ`),
   where the carrier letter already supplies the length and adding an alef
   doubles it; only **10 follow a bare fatha** (as in `ذَٰلِكَ`), which is the
   case the long-vowel intuition was built on. So the intuition was right
   about 10 positions and wrong about 14.

   **Decision this implies: keep the deletion rule**, now resting on Qur'anic
   evidence (54 vs 48 on gold prompt text) and not only on the MSA 177/177.
   The rule lives at **`src/arabic_mdd/data/phonetizer.py:179`**, downstream of
   normalization, so `data/normalize.py` is not touched and the byte-identical
   regeneration guard below has nothing to catch — which is the cheapest
   possible outcome and the reason this was worth measuring before writing any
   code. Still record it as a numbered decision in `docs/msa-arm.md`; the
   measurement is what makes it a decision rather than a default.

   **If a later rule change reaches the text normalization the MSA artifacts
   were drawn through, they must regenerate byte-identically.** That is
   `data/normalize.py` (and `data/msa_pool.py`'s `skeleton`, which is the key
   the block was sampled on) — *not* `phonetizer.py`, whose output no committed
   MSA artifact contains. `run/msa_annotation_block.json` and
   `run/qualification/blank.txt` are the regression check — the same guard
   that protected D4's block through the week-5 refactors, and the reason the
   `tanzil`/`wikinews` loader promotions were safe. A domain-parameterized
   rule is acceptable; a globally changed one that silently moves the
   committed annotation block is not.

   Deliverable: the decision recorded in `docs/msa-arm.md` with a date, plus
   a measured `phoneme_exact` count under the chosen rule so the 54/98 figure
   is superseded by a number rather than an argument.

2. **▲▲ Run the label-path column — `scripts/measure_label_path.py`.** The
   paper's primary number for this week.

   - **Inputs fixed, one factor moved.** `A` = `annotation_phoneme_string`,
     `P` = `run/quranmb_predictions_official_ckpt.json`, both per clip (1,642);
     `C` per prompt (98), joined on `prompt_index`. Use the official
     checkpoint rather than our retrain: it is the one that reproduces the
     published condition to ±0.0001, so the ceiling cell is the field's own
     number and not ours.
   - **Conditions:** `C_gold` plus one `C_auto` per diacritizer in the final
     set, which D8 fixed at **four**. Four tools × 1,642 clips through
     `metrics/hierarchical.py`.
   - **Pool counts, never average ratios.** `compute_metrics` is applied once
     to the summed `MDDCounts`, matching the reference implementation —
     week 3's insights already flagged that per-utterance averaging biases
     toward short utterances. This also constrains the bootstrap in task 3:
     resample clusters, re-pool, recompute, rather than resampling
     per-utterance F1 values.
   - **Report the full TA/TR/FA/FR decomposition per condition, not just F1.**
     The mechanism lives in which cell moves, and a single F1 delta cannot
     show it.
   - **The gate's attribution check.** Of the FR and FA *delta*, what fraction
     falls at positions where `C_auto` and `C_gold` actually disagree? Without
     this, a nonzero ΔF1 has no established mechanism and could be an
     alignment artifact of three-way Needleman-Wunsch rather than label
     corruption. This is deliberately a **diagnostic for the gate, not plan
     §4.2's corruption-attributable FR rate**, which is a weeks-10–12
     deliverable with its own framing; keep it to one number and resist
     building the instrument early.
   - Writes `run/quranic_label_path.json` with per-tool counts, the `C_gold`
     rule from task 1, both DER conventions' rates for each tool (so the
     ΔF1-vs-DER relation can be read without a second artifact), tool
     provenance, and leg B's anchor F1.

   **Predictions, pre-registered, each with its falsifier.** Written before
   the run so they can be wrong — two of week 5's three failed, which is the
   point of writing them.

   1. **ΔF1 is negative** — `C_auto` scoring *deflates* F1 — and the
      mechanism is FA inflation. Where `C_auto` is wrong and the learner was
      right, the position looks like a mispronunciation the system failed to
      flag, so `sub_nosub` rises, recall = TR/(TR+FA) falls. *Falsifier:* a
      positive ΔF1, or a negative one whose dominant moving cell is not FA.

      Note what this does **not** predict. Plan §1 raises the opposite
      mechanism — label corruption inflating F1 because the system's own
      errors coincide with reference errors and cancel — and that requires a
      model that consumed `C`. ~~A prompt-free model cannot have correlated
      errors with `C_auto`. **So the sign may legitimately flip in weeks
      10–12 for the text-dependent arm, and that flip would be a result, not a
      contradiction.**~~ Writing this down now stops it being read as an
      inconsistency later.

      **[14 Oct 2026 — FALSIFIED, and so was the reasoning that fenced it off.
      Measured: `run/quranic_label_path.json`, `insights/week-06.md`.]** ΔF1 is
      **positive on all four tools** (+0.0265 `catt-eo` → +0.2818 `mishkal`,
      every CI excluding zero) and the dominant moving cell is **TR**, not FA.
      So prediction 1 fails on sign *and* on mechanism.

      The struck sentences are the more important error, because they are an
      argument rather than a guess. The claim was that inflation requires
      errors correlated with `C_auto`, which a prompt-free model cannot have.
      **Inflation needs no correlation at all.** TR counts positions where
      `C ≠ A` and the system's output matches `A`; corrupting `C` enlarges that
      set, so a position where the diacritizer errs and the model was *right*
      converts TA → TR — credit for detecting an error that only the reference
      committed. That is available to any detector that fires often, prompt-free
      or not, and the prompt-free arm had no protection against it.

      Two consequences. The sign is **not** reserved for weeks 10–12, so a
      positive ΔF1 on the text-dependent arm would no longer be the surprise
      this note was written to pre-absorb; the live question there becomes
      whether conditioning on `C_auto` adds the *cancellation* mechanism on top
      of this base-rate one, and therefore whether the inflation grows. And
      plan §1's "nobody knows the sign" is now answered for the label path and
      annotated there accordingly.
   2. **|ΔF1| is monotone in CE DER**: `catt-eo` < `catt-ed` < `shakkala` <
      `mishkal`. Three unrelated instruments already agree on that ordering
      (DER on three corpora, week-4 phoneme conservation, NAACL Table 1), so a
      fourth disagreeing is evidence about the metric and not about the tools.
      *Falsifier:* any inversion, which would be the first real evidence for
      RQ3's claim that DER is miscalibrated — i.e. the most interesting
      possible failure here.

      **[14 Oct 2026 — HOLDS.]** No inversion; a fourth instrument agrees.
      But the relation is strongly **non-linear**: CE DER rises 4.6× from
      `catt-eo` to `mishkal` (3.98 → 18.20) while ΔF1 rises 10.6×, so DER
      under-predicts label-path damage at the bad end. A weaker, more
      interesting version of RQ3's claim than an inversion would have been,
      and it survives into weeks 10–12.
   3. **The case-ending share of label damage is high on this arm — and that
      is a caveat, not a finding.** Qur'anic recitation *pronounces* case
      endings; tajwīd prescribes them. Plan §1's argument that case endings
      are "close to irrelevant to pronunciation teaching" is an **MSA, pausal
      argument**, and the Qur'anic arm is the worst available place to make
      it. *Falsifier:* a low CE share. Either way the write-up must carry the
      asymmetry explicitly, because an uncaveated case-ending result from this
      arm would overstate RQ3's headline and a reviewer who recites will catch
      it immediately.

      **[14 Oct 2026 — CONFIRMED at text level; phoneme level deliberately not
      measured.]** CE/noCE DER ratio is 1.46–1.68 across the four tools, so
      ~40% of each tool's text-level error sits on case endings — on the arm
      that pronounces them. The caveat stands and must be carried. The
      phoneme-level share of label *damage* needs word boundaries inside the
      phoneme sequence, and per-word phonetization diverges from
      whole-utterance phonetization (§4.1.1: one text comes out 6 tokens
      spaced and 8 merged), so it would need a word-aligned phonetizer. Not
      built; the artifact records `case_ending_share.phoneme_level: null` with
      the reason, so the gap is visible at the point of use rather than
      implied to have been measured.

3. **▲▲ Estimate the RQ5 interaction power — `scripts/power_rq5.py`.** The
   gate's second clause. **[DONE 14 Oct 2026 — `run/rq5_power.json`; KEEP RQ5;
   D2 decided. See clause 2 above for the numbers.]**

   - **The estimand is a difference-in-differences**: (ΔF1 for the
     text-dependent arm) − (ΔF1 for the prompt-free arm). CROTTC-IF has
     settled the main effect (conditioning hurts even with a correct `C`,
     F1 → 40.52%), so the remaining question is an interaction, and plan §2
     notes interactions need more power.
   - **Cluster on prompts, and show what that costs.** 1,642 clips carry only
     98 distinct prompts and 18 speakers, and `C` is constant within a
     prompt — so for a `C`-driven effect the effective n is far nearer 98 than
     1,642. Report the naive MDE and the cluster-bootstrap MDE together; the
     ratio is the design effect, and quoting only the naive figure would
     overstate the power by whatever that ratio turns out to be. Kadambi's
     mixed-effects specification (phoneme position and type as predictors,
     speaker and utterance as grouping factors) is the template plan §5 tells
     us to borrow, and this is where it earns its keep.
   - **Variance of the arm that does not exist yet is an assumption, reported
     as one.** Sensitivity at 1×, 1.5× and 2× the prompt-free arm's observed
     per-prompt variance.
   - Deliverable: `run/rq5_power.json` and a stated recommendation — keep RQ5,
     or cut it in November — against the pre-registered rule in the exit
     criterion.

   **This also closes D2.** Week 5 task 7 deferred D2 (whether `A := C_gold`
   is an assumption with a sensitivity row, or the residual interaction term
   is bounded) explicitly to "the week-6 power estimate rather than in the
   abstract". The estimate now exists, so decide it: if the MDE is comfortably
   below the effects of interest, bound the term; if it is not, the
   sensitivity row is the honest and cheaper option. Record either way.

4. **▲▲ Write the annotation logging tool. [DONE 14 Oct 2026]** Not a week-6 deliverable in plan
   §6, and the highest-value thing in this file after the gate, because it is
   the only thing standing between a pre-registered protocol and the project's
   longest pole. Week 5 task 4 closed with: "Still blocking the first
   annotated utterance: **no tool writes the schema.** `pilot_annotate.py` is
   a listening interface and cannot be reused."

   The schema is already fixed by the protocol and must not be redesigned
   here: `suggested` and `final` stored separately per word, plus
   `final_before` for adjudication — which closed pilot §6a's unrecoverable
   override-rate regret. The tool's job is to write exactly that, record
   elapsed time per utterance (D4's extension decision depends on real
   timings), and honour the crossover's role assignment by draw rank.

   Three things it must not do: reorder the block (the draw is committed),
   show `catt-eo`'s suggestion on the blank-page ranks, or allow a submission
   that alters the consonantal skeleton without flagging it.

   **Built, in three pieces, and one correction to the paragraph above.**

   - `src/arabic_mdd/data/annotation.py` — §4's role assignment, §6's 16-field
     record, the four ingest checks, R4 site detection, and the file-level
     report. In `src` rather than in the script because §6 requires the checks
     "automated at ingest" and ingest recurs in weeks 7–9, long after the tool
     stops running. `tests/data/test_annotation.py`, 37 tests, quotes the
     protocol wherever the two could drift: §4's role table row by row, §6's
     field list in order, §4's ≥45 rule, and **§3's "31 positions" as a
     regression assertion** — the implemented R4 rule finds exactly 31 sites
     in 29 rows of the committed block, which is what established that the
     rule coded is the rule the protocol counted. A skeleton-only reading of
     R4 — "any non-final word followed by one beginning `ال`" — finds **386
     candidates in 262 of the 500 rows**, nearly all of them legitimate R1
     case endings. That version would have warned on more than half the block
     and been tuned out by week 2.
   - `scripts/prepare_annotation_drafts.py` → `run/msa_annotation_drafts.json`,
     500 `catt-eo` drafts in 3.8s, committed rather than generated per session.
     §6 logs the `catt-tashkeel` version that produced `suggested`, which can
     only be a fact about the artifact and not about whichever machine the
     annotator sat at; and §8's Δ is a within-utterance contrast, so a
     crossover row's draft must be byte-identical whenever it is shown.
   - `scripts/annotate_msa.py` — the CLI. Text entry goes through `$EDITOR` so
     that Arabic is genuinely editable *and* `started_at`/`finished_at`
     bracket real work; `--validate` re-runs the whole gate on a finished
     file; `--status` reports progress, §4's extension rule, and the running
     median that D4 is waiting on. The three prohibitions are enforced
     structurally rather than by discipline: assignment order comes from the
     committed draw with no skip or jump, `build_record` forces `suggested` to
     `None` on the blank arm and the renderer is never handed the drafts
     mapping on those rows, and a skeleton-altering submission cannot be saved
     — it loops back into the editor, with `unusable` as the only exit.

   **The correction.** This task says "`suggested` and `final` stored
   separately **per word**". Protocol §6 stores them as **whole-utterance
   strings** with `flagged_words` as word indices, and the protocol is the
   pre-registered artifact, so it wins — the phrase above is a paraphrase
   written before §6 existed. Nothing is lost: per-word override rate, which
   is what pilot §6a regretted not having, is recoverable from the two strings
   by word index. Recorded here rather than silently diverged from.

   **Deliberately not built: the adjudication pass.** §6's `adjudicated` is
   written as `null` by this tool. Adjudication needs both annotators'
   finished files side by side, happens once, and §5's whole point is that
   `final_before` keeps annotator accuracy separable from adjudicated
   accuracy — which a UI threaded through the annotation loop makes easier to
   get wrong, not easier to do. It is a separate script, written when there
   are two files to read.

   **Two findings the build produced**, both recorded in the artifact, and
   both of the kind that would otherwise have surfaced as annotator confusion
   at row 1:

   - **`catt-eo` strips punctuation from all 299 rows that have any.** §6's
     skeleton check does not fire on it, because `skeleton()` ignores
     punctuation — so this was invisible to every check that existed. But §7
     says punctuation is left untouched, which is unfollowable without
     retyping on 299 of 500 rows, and the anchored arm would then have
     produced `final` strings differing from the blank arm's on something that
     is not diacritization, feeding straight into §8's within-utterance Δ. The
     generator splices punctuation back exactly — the draft is the source's
     letters plus inserted marks, so the merge is well-defined — repairing
     **299 rows, with 1 left**.
   - **The fix for that was worse than the bug, and §6's checks could not see
     either.** The first version of the merge restored any source character the
     draft had skipped. But **71 of the 500 rows already carry marks** (§7's 59
     flagged `partially_vowelized`, plus 12 under the flag's 0.05 threshold),
     and where the model marks a letter differently from the source, the
     "skipped" character is a disagreement rather than a deletion — carrying it
     across stacked two short vowels on one letter (`طَاغُونُ` → `طَاغُونَُ`) on **8
     rows**. §7 already settles it: the source's "existing marks have no
     authority", so the draft's mark stands and the annotator corrects it.
     The reason this is worth a paragraph is that **all four of §6's ingest
     checks compare skeletons or word counts, so the specified gate is blind to
     diacritics** — the one thing the arm exists to produce. Both bugs lived in
     that blind spot and both were found by printing a row. The generator now
     refuses to write the artifact if any letter carries two mutually exclusive
     marks; fatal, not advisory, because nothing downstream would reject it.
   - **Rank 380 (id 64376) is handed over failing §6's own ingest check.**
     `catt-eo` deletes `Eddie`, both occurrences, exactly as the organizers'
     vowelizer does — §7 already puts that deletion rate at 100% and calls it
     "a real `C_auto` error and D5's subject". §7 also says leave Latin
     untouched, so on this one row the annotator must put it back. It is also
     the one row the punctuation splice cannot align, for the same reason.
     Carried in `skeleton_differs_from_frame` and warned at that row, rather
     than left to fire mysteriously.

5. **Pin the S3PRL checkout. [DONE 16 Oct 2026]** Carried since week 4, and
   week 5 said "do it this week" without it being done. The exposure is
   specific: `run/prepare_baseline.sh:81` clones `s3prl/s3prl.git` at whatever
   `main` serves that day, and S3PRL supplies the featurizer, the downstream
   model *and* the training loop. **Week 7 starts training new arms**, and
   after that an unpinned training loop contaminates results rather than just
   the record. Needs a commit chosen against a venv known to build, i.e. a
   cluster round-trip: `git -C run/s3prl_checkout rev-parse HEAD` on the
   cluster is the first step and takes seconds.

   **Pinned at `ec8064b5889f81ca460fbe2c094ce576a6f120b7`** — upstream's own
   v0.4.18 release commit, dated 2025-06-13 — read off the `citrin` checkout
   that produced the week-3 gate. `status --porcelain -uno` was empty, so the
   SHA describes the tree exactly; the only untracked content was three
   training expdirs, which are outputs, not inputs.

   **The realized risk was zero, and saying so is part of doing the task.**
   Upstream `main` is *still* at that commit as of 16 Oct 2026, so it has not
   moved in the sixteen months spanning the clone — the paragraph above
   describes a real mechanism that had not fired and, as of today, a fresh
   clone would land on the pin anyway. Two things follow. The urgency framing
   was wrong but the policy was right: the pin's value is now entirely
   *prospective*, and it is wrong to record this as a near miss. And the
   valuable half is therefore **not** the SHA but the **enforcement**, which
   is what was actually missing: the script now verifies `HEAD` against the
   pin on **every** run and exits 1 on mismatch, rather than only choosing a
   commit at clone time. The states that check catches — a `git pull` in that
   tree, a re-clone after upstream eventually moves, a second machine — are
   the ones where the damage is silent, and none of them are the clone.

   A tracked-file modification warns instead of failing: `pip install -e`
   means any hand-edit there is already live in every run, so the useful
   action is to announce it at job start, not to refuse to start the job.
   Verified against a scratch repo on all four paths — matching HEAD,
   mismatched HEAD, untracked expdirs present (must stay silent, since that is
   the real cluster state), and a dirty tracked file.

   **Still unpinned, and it is the same exposure:** `transformers`, installed
   by name at what is now line 92. `hf_hubert_custom`'s expert imports it at
   module load and it supplies the featurizer's model code, so a floating
   version can move a result exactly as S3PRL can — and unlike S3PRL it is a
   package that *does* release often. One `pip freeze` on the cluster closes
   it; it was requested in the same round-trip and not returned.

6. **Establish the XLS-R-300m fine-tune wall-clock cost and the week-7 work
   split.** Open since week 1 and, per week 5's carried-forward note, this
   **cannot slip past week 6** — week 7 is when the two arms converge and need
   an assigned owner. The timeline assumes several fine-tunes are affordable;
   that assumption has never been measured.

   **[16 Oct 2026 — one datum in, and it is not the one plan §7.5 asked for.]**
   `run/train_baseline.slurm` took **24h**: 200,000 steps, batch 16, frozen
   mHuBERT-147 (`upstream_trainable = False`) plus the 2×1024 BiLSTM + CTC
   downstream, on `NvidiaAll`. That is ≈ **2.3 steps/s**, and it means the
   `-t 48:00:00` budget on that script was running with 2× headroom.

   It is a **lower bound, not the number**. Plan line 117 defines the arm as
   *"XLS-R-300m + layer fusion + TCN + CTC"*, which differs from what was
   measured in three ways at once — a 300m upstream instead of a base-size
   one, layer fusion, and a TCN instead of the BiLSTM. The 24h transfers to
   none of those directly.

   **Estimate, labelled as an estimate.** XLS-R-300m is 24 layers × 1024 wide
   against mHuBERT-147's 12 × 768, so the upstream forward is ≈ **3.6×** the
   FLOPs. The upstream is frozen, so that is forward-only; and the BiLSTM
   downstream is already roughly comparable in cost to the mHuBERT forward, so
   the ratio lands nearer **2–2.5× → 48–60h** than 3.6×. That is **at or over
   the current 48h budget**, which is the actionable part: the queue is
   unrestricted (`Timelimit: Infinite`, week 1), so this is a `-t` line to
   raise, not a constraint to design around — but a job that dies at hour 48
   of 55 wastes two days.

   **Do not take the estimate; measure it for the price of minutes.**
   `run/smoke_train.slurm` already exists for exactly this and the discipline
   is week 3's — `STEPS=200` with the XLS-R upstream gives steps/s directly
   off the log, extrapolates to 200k, and costs one short queue slot instead
   of a day. It also resolves a question that must be answered before any long
   job is submitted: **`hf_hubert_custom` cannot load XLS-R.** That expert
   builds a `HubertModel`; XLS-R is a Wav2Vec2 model. The pinned S3PRL commit
   does ship the exact parallel — `hf_wav2vec2_custom(ckpt)` in
   `s3prl/upstream/hf_wav2vec2/hubconf.py` — so the route is
   `-u hf_wav2vec2_custom -k <pinned dir>`, and `pin_upstream.py` already
   produces such directories. Both `smoke_train.slurm` and
   `train_baseline.slurm` hardcode `-u hf_hubert_custom`, so measuring this
   needs the upstream name and `-k` path parameterized first; that is a small
   change and it belongs to whoever owns training.

   **The run count is the real budget question, and it is undecided.** The
   2×2's columns are free (rescoring costs nothing). The rows are "Model
   **given** `C_gold`" vs "Model **given** `C_auto`". If *given* means at
   inference, the text-dependent arm is **one** training run fed two different
   canonicals. If the `C_auto` row must be *trained* on automatic canonicals —
   the deployment-realistic reading, since training on clean canonicals and
   testing on noisy ones is itself a train/test mismatch confound — it is one
   run per diacritizer, so **five**, because D8 fixed the set at four tools.
   At ~55h/run that is the difference between ≈ 4.6 days of GPU and ≈ 13.8
   days, inside two 21-day windows. Six runs fit, with no slack for a failed
   one, and each failure costs ~2.3 days. **An undecided sentence in plan §1
   is worth roughly nine GPU-days**, so decide it and record it.

   **One lever worth naming and not building.** Because the upstream is
   frozen, those 200k steps recompute identical upstream features for the same
   audio thousands of times, and every text-dependent variant shares them —
   the `C`-conditioning lives entirely in the downstream. Caching features
   once would collapse the 1-vs-5 question to nearly free and bring each run
   well under budget. Two reasons it is flagged rather than scheduled: layer
   fusion needs all 25 layer outputs retained, so the storage is substantial,
   and it means patching a dependency that was *just* pinned in task 5. Revisit
   only if the smoke run says a single XLS-R run exceeds ~72h.

   **What remains genuinely open after this**: the XLS-R steps/s measurement
   above, and the week-7 owner split (plan §7.7 — one owns model and training,
   the other diacritization and annotation, converging week 7), which is a
   decision rather than a measurement and is the half with the hard deadline.

7. ~~**Select the MSA evaluation pool for §5.2.** Carried from week 5 task 6,
   unstarted. CV-Ar rows disjoint from `Iqra_train`, deduped **by consonantal
   skeleton, not raw string** — D4 is the cautionary case, where raw-string
   dedup reported `Iqra_train` dev duplicate-free and missed 46 rows, 33 of
   whose 44 groups were the same verse once bare and once pre-vowelized.
   `msa_pool.skeleton` already does this; reuse it. Cite Sadeed's precedent
   while doing it (34.6% of the Fadel test set is fully present in the Abbad
   training set) — the discipline is standard practice the field frequently
   skips, and saying so costs one sentence.~~ — **DONE, and the task as
   written was the wrong task.** Decision recorded as §5.2.1.

   **"CV-Ar rows disjoint from `Iqra_train`" was foreclosed two weeks ago by
   our own decision.** D4 decided *annotate dev, not fresh Common Voice*,
   because `tashkeel_sentence` exists only in `Iqra_train` and D5's whole
   measurement is `tashkeel_sentence` vs `C_gold`. Fresh CV text has no
   `C_auto`. The block is drawn, the drafts are committed, the protocol is
   pre-registered — so there was no pool left to select. This task carried
   unstarted from week 5 *and* was stale on arrival; the lesson is that a
   carried-forward task should be re-read against the decisions taken since,
   not just executed.

   **What was actually open, and it is the better version.** §5.2 was written
   when the only model was the organizers' `dev-best.ckpt`. From week 7 we
   train our own models, so we choose the splits: the pool stays D4's 500 and
   the contamination is removed from the *training* side instead.
   `scripts/select_msa_eval_pool.py` → `run/msa_eval_pool.json`, carrying
   `holdout_train_ids` for week 7 to consume.

   **The numbers.** Measured on the consonantal skeleton against all 71,391
   train rows (29 shards memory-mapped, two columns projected, so the 14 GB of
   audio never pages in — **4.2 s at 138 MB peak RSS**, no GPU):

   | Set | Sentences also in train | Train rows | Raw-string match |
   |---|---|---|---|
   | Annotated block (500) | **140 = 28.0%** | 319 | **0** |
   | Eligible pool (2,092) | 510 = 24.4% | 1,157 | 0 |
   | Whole dev (2,542 skeletons) | 836 = **32.9%** | 2,075 | 0 |

   Holdout cost is **319 rows = 0.45% of train**, which is why this is worth
   doing rather than declaring. The pool-level row also costs D4's extension
   decision nothing to revisit later: 24.4% is already measured.

   **Three findings worth more than the deliverable.**

   - **The check everyone runs passes.** Train/dev id overlap is **zero**. An
     id-disjointness test certifies the split clean while 28% of the
     evaluation text sits in training.
   - **Raw-string matching finds 0 of 319, for *two* reasons.** D4's is the
     first — dev bare, train pre-vowelized (holdout median rate 0.571 vs
     train's 0.000; 180/319 pre-vowelized against a 32.3% base rate). The
     second was not predicted: among bare/bare pairs the strings differ only
     in **punctuation placement** — train `أحضرها لي.` vs dev `.أحضرها لي`,
     the same six characters with the full stop moved; also `ثمين ، لهذا` vs
     `ثمين، لهذا`. `skeleton` folds punctuation and collapses whitespace, so
     it sees through both. This retroactively justifies week 4's single pinned
     normalization contract better than anything measured at the time.
   - **32.9% vs Sadeed's 34.6%.** Nearly the same figure on a different
     corpus, which turns "we deduplicated" from diligence into a result.

   **Flagged, not built.** Skeleton identity is exact-match, so Sadeed's
   near-duplicate band (68.12% above 0.5 similarity) is neither counted nor
   held out. At 0.45% holdout cost the exact tier is cheap insurance; a
   similarity threshold is a parameter to defend rather than a fact to report.

   **One correction to this plan's own wording:** `msa_pool.skeleton` is a
   re-export — `skeleton` lives in `data/normalize.py`, pinned there precisely
   because three consumers must agree on it byte-for-byte. The reusable piece
   was `build_pool`'s grouping, now joined by `msa_pool.skeleton_overlap` for
   the cross-split direction (3 tests).

8. **Housekeeping, each small, each closing a known caveat.**

   - **Ask the organizers which vowelizer produced `tashkeel_sentence`.** One
     email. It determines whether D5's error-propagation headline is
     understated and whether D7 was a genuine choice or a coin-flip, and the
     5.00× vs 1.05–1.22× asymmetry means the answer is probably "an
     EO-derived one". Cheap, external latency, so send it on day one.
   - **Resolve why ONNX Runtime declines a visible GPU.** Job 224461 printed
     `SELECTED ['CPUExecutionProvider']` on a node with an RTX 2060 and
     `onnxruntime-gpu` 1.30 installed, which cost 3h14m in stage A. Week 6
     needs no GPU, but weeks 7+ do, and the diagnosis is cheaper now than
     under deadline.
   - **Administer annotator 2's qualification test.** Packet is built
     (`run/qualification/blank.txt` + `instructions.md`); §9's pass marks are
     pre-registered and must not be re-read before scoring. Annotator 1
     passed at bucket A 1.17% [0.72, 1.90] and bucket B 14.33% [10.78, 18.81]
     → flag-and-adjudicate. External latency, so send it on day one.
   - **D6** — whether `Iqra_TTS`'s augmented rows become a clearly-labelled
     synthetic detection sub-arm. `run/tts_detection.slurm` exists and the
     annotation cost is zero, so the only real question is whether a synthetic
     sub-arm helps or dilutes. If used, the synthetic fraction goes in
     abstract-adjacent text, not a footnote.
   - **Delete `run/msa_pilot_annotated copy.json`** — 6,224 bytes, untracked,
     believed deleted and not. Left for an explicit go-ahead since untracked
     deletion is unrecoverable.
   - **Re-check whether Sadeed has released weights.** Once, cheaply. It is
     D8's third re-opening condition: downloadable frontier-quality weights
     dissolve both the reproducibility and the RQ4 objections at a stroke and
     make it an ordinary fifth tool rather than an API condition. Nothing
     else about D8 should be re-litigated this week.
   - **Licence terms for `QuranMB.v2` / `Iqra_Extra_IS26`** — open since week
     2, and it gates a deliverable (plan contribution 5, the released
     gold-diacritized Common Voice slice). `Iqra_Extra_IS26` returned HTTP 401
     in week 5; an access request has latency we do not control.

## Carried forward

**[16 Oct 2026]** Week 7 is planned in [week-07.md](week-07.md), which takes
one structural decision worth naming here: **it runs the Qur'anic arm only**,
and defers the MSA arm's evaluation to weeks 8–9. The engineering consequence
lands in week 7 regardless — the models trained there are the ones week 8
evaluates, so both contamination holdouts (task 7's 319 MSA rows and the
Qur'anic 62 found alongside them) must be in the training manifest before the
first run starts, or they cost a retrain.

- **D4's extension decision** (~1,592 further eligible rows) stays deferred
  until the main block yields real per-utterance timings, which now depends on
  task 4. The seeded draw makes an extension a superset of the committed
  block, so this is a volume decision and not a sampling one. **[14 Oct 2026]**
  Task 4 is done and the timings are instrumented —
  `annotate_msa.py --status` prints the running median and the cumulative
  hours — so this is now waiting on annotation hours, not on code.
- **`scripts/verify_quranmb_labels.py`** — optional. Residual value is
  per-row, plus identifying the one utterance by which the `safikhan` join
  (1,642) differs from the published count (1,643).
- **Interspeech 2027 page limit and formatting rules**, once the call is
  published (plan §8).
- **The Phonikud propagation quote**, unverified and unusable. Ten minutes,
  low priority.

## Out of scope for week 6

- **The input-path row and the text-dependent arm.** Weeks 7–9, and the 2×2
  cannot be completed without them. Week 6 delivers the scoring column only.
- **The MSA label path.** Weeks 7–9, and gated on annotation.
- **RQ3's weight estimation, the synthetic corruption sweep, RQ4's abstention
  curves, and plan §4.2's feedback-validity number.** Weeks 10–12. Task 2's
  attribution check is one diagnostic number, deliberately not the instrument.
- **Revisiting the paper's framing**, including in response to a null on
  clause 1. Pre-committed in plan §5, and §5 exists precisely to stop a
  framing decision being made under the pressure of a disappointing result.
- **Partial diacritization and cross-lingual validation**, held back
  deliberately in plan §2.
