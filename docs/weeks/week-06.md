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

What did **not** advance: annotation has not started, because no tool writes
the pre-registered logging schema; annotator 2 has not sat the qualification
test; and S3PRL is still unpinned.

## Scope

The whole week is one measurement plus the gate's second clause, and the
measurement is cheap in compute and expensive in definitions. Nothing here
needs a GPU: the Qur'anic `C_auto` is 98 prompts of 4,442 characters total,
and week 5 measured `catt-eo` at 0.015 s/verse and `catt-ed` at 1.205 s/verse
over 6,236 verses — so all four tools over 98 prompts is seconds, not hours.

| what | blocks | blocked on |
|---|---|---|
| the `C_gold` definition (task 1) | the entire label-path column | nothing — a decision, not an experiment |
| the label-path column (task 2) | the block gate, clause 1 | task 1 |
| the power estimate (task 3) | the block gate, clause 2; D2 | task 2's per-utterance variance |
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

### Clause 1 — label-path effect measurable

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

### Clause 2 — interaction power estimated

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
   That also means **`data/normalize.py` is not touched**, so the
   byte-identical regeneration guard below has nothing to catch — which is
   the cheapest possible outcome and the reason this was worth measuring
   before writing any code. Still record it as a numbered decision in
   `docs/msa-arm.md`; the measurement is what makes it a decision rather than
   a default.

   **If the rule changes `data/normalize.py`, the MSA artifacts must
   regenerate byte-identically.** `run/msa_annotation_block.json` and
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
      model that consumed `C`. A prompt-free model cannot have correlated
      errors with `C_auto`. **So the sign may legitimately flip in weeks
      10–12 for the text-dependent arm, and that flip would be a result, not a
      contradiction.** Writing this down now stops it being read as an
      inconsistency later.
   2. **|ΔF1| is monotone in CE DER**: `catt-eo` < `catt-ed` < `shakkala` <
      `mishkal`. Three unrelated instruments already agree on that ordering
      (DER on three corpora, week-4 phoneme conservation, NAACL Table 1), so a
      fourth disagreeing is evidence about the metric and not about the tools.
      *Falsifier:* any inversion, which would be the first real evidence for
      RQ3's claim that DER is miscalibrated — i.e. the most interesting
      possible failure here.
   3. **The case-ending share of label damage is high on this arm — and that
      is a caveat, not a finding.** Qur'anic recitation *pronounces* case
      endings; tajwīd prescribes them. Plan §1's argument that case endings
      are "close to irrelevant to pronunciation teaching" is an **MSA, pausal
      argument**, and the Qur'anic arm is the worst available place to make
      it. *Falsifier:* a low CE share. Either way the write-up must carry the
      asymmetry explicitly, because an uncaveated case-ending result from this
      arm would overstate RQ3's headline and a reviewer who recites will catch
      it immediately.

3. **▲▲ Estimate the RQ5 interaction power — `scripts/power_rq5.py`.** The
   gate's second clause.

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

4. **▲▲ Write the annotation logging tool.** Not a week-6 deliverable in plan
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

5. **Pin the S3PRL checkout.** Carried since week 4, and week 5 said "do it
   this week" without it being done. The exposure is specific:
   `run/prepare_baseline.sh:81` clones `s3prl/s3prl.git` at whatever `main`
   serves that day, and S3PRL supplies the featurizer, the downstream model
   *and* the training loop. **Week 7 starts training new arms**, and after
   that an unpinned training loop contaminates results rather than just the
   record. Needs a commit chosen against a venv known to build, i.e. a cluster
   round-trip: `git -C run/s3prl_checkout rev-parse HEAD` on the cluster is
   the first step and takes seconds.

6. **Establish the XLS-R-300m fine-tune wall-clock cost and the week-7 work
   split.** Open since week 1 and, per week 5's carried-forward note, this
   **cannot slip past week 6** — week 7 is when the two arms converge and need
   an assigned owner. The timeline assumes several fine-tunes are affordable;
   that assumption has never been measured.

7. **Select the MSA evaluation pool for §5.2.** Carried from week 5 task 6,
   unstarted. CV-Ar rows disjoint from `Iqra_train`, deduped **by consonantal
   skeleton, not raw string** — D4 is the cautionary case, where raw-string
   dedup reported `Iqra_train` dev duplicate-free and missed 46 rows, 33 of
   whose 44 groups were the same verse once bare and once pre-vowelized.
   `msa_pool.skeleton` already does this; reuse it. Cite Sadeed's precedent
   while doing it (34.6% of the Fadel test set is fully present in the Abbad
   training set) — the discipline is standard practice the field frequently
   skips, and saying so costs one sentence.

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

- **D4's extension decision** (~1,592 further eligible rows) stays deferred
  until the main block yields real per-utterance timings, which now depends on
  task 4. The seeded draw makes an extension a superset of the committed
  block, so this is a volume decision and not a sampling one.
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
