# Week 6 insights

Dates: 26 Oct – 1 Nov 2026 nominal; running ~3 weeks early (see
[docs/weeks/week-06.md](../docs/weeks/week-06.md)). This is the first week whose
deliverable is a paper number rather than machinery.

### Experiment: the label-path column on the Qur'anic arm

#### Hypothesis

Three predictions, written into `docs/weeks/week-06.md` task 2 before the run,
each with its falsifier:

1. **ΔF1 is negative** — scoring against an automatically diacritized canonical
   *deflates* F1 — and the mechanism is **FA inflation**. Where `C_auto` is
   wrong and the learner was right, the position looks like a mispronunciation
   the system failed to flag, so `sub_nosub` rises and recall falls.
   *Falsifier:* a positive ΔF1, or a negative one whose dominant moving cell is
   not FA.
2. **|ΔF1| is monotone in CE DER**: `catt-eo` < `catt-ed` < `shakkala` <
   `mishkal`. *Falsifier:* any inversion.
3. **The case-ending share of label damage is high on this arm** — a caveat,
   not a finding, since Qur'anic recitation pronounces case endings.
   *Falsifier:* a low CE share.

#### Methodology

`A` (`annotation_phoneme_string`, per clip) and `P`
(`run/quranmb_predictions_official_ckpt.json`, the organizers' own checkpoint)
held fixed; only `C` swapped. Two legs, per **D9**
([docs/msa-arm.md](../docs/msa-arm.md) §3.3b):

- **Leg A** — `C_gold = phonetize(normalize(recovered_text))` against
  `C_auto = phonetize(diacritize(strip_diacritics(recovered_text)))`. Same
  phonetizer, same text, same convention both sides, so the only thing
  differing between cells is the diacritizer and ΔF1 is attributable to it.
- **Leg B** — `C = reference_phoneme_string`, the benchmark's own canonical, to
  confirm the published condition reproduces and to price our phonetizer's own
  disagreement separately rather than inside ΔF1.

Counts pooled (summed, then `compute_metrics` once — never averaged ratios).
95% CIs from a **cluster bootstrap over the 98 prompts**, 10,000 replicates,
paired draw (both conditions resampled on the same prompt indices, since they
differ only in `C`).

#### Configuration

`scripts/measure_label_path.py --write`, `run/label_path.slurm`, git
`ccc1e7d`. 98 prompts, 1,642 clips, 16.76 clips/prompt (7–27). Bootstrap seed
20261014. Artifact: `run/quranic_label_path.json`.

#### Results

**Leg B reproduces exactly: F1 = 0.4415** over all 1,642 clips, matching the
published figure and the week-3 gate. The harness has not drifted.

Leg A's gold cell: **F1 = 0.4824** (P 0.3821, R 0.6540). So **leg B − leg A =
−0.0408** — our phonetizer's disagreement with the uploader's is worth 4.1 F1
points, and this is the number that would have been silently folded into every
ΔF1 had candidate (i) been used as `C_gold`. It belongs in the limitations.

| tool | CE DER | F1 | ΔF1 | 95% CI | excludes 0 |
|---|---|---|---|---|---|
| `catt-eo` | 3.982 | 0.5088 | **+0.0265** | [+0.0040, +0.0467] | yes |
| `catt-ed` | 4.677 | 0.5303 | **+0.0480** | [+0.0225, +0.0735] | yes |
| `shakkala` | 9.207 | 0.6333 | **+0.1510** | [+0.1187, +0.1834] | yes |
| `mishkal` | 18.195 | 0.7642 | **+0.2818** | [+0.2446, +0.3185] | yes |

**Clause 1 of the week-6 gate: PASS.** All four CIs exclude zero — not merely
the worst tool, which is all the pre-registered criterion required — and the
point estimates are monotone in CE DER with no inversion.

Cell decomposition, gold → `C_auto`:

| tool | ΔTA | ΔFR | ΔFA | ΔTR | P | R |
|---|---|---|---|---|---|---|
| gold | — | — | — | — | 0.3821 | 0.6540 |
| `catt-eo` | −226 | −516 | +499 | +357 | 0.4317 | 0.6194 |
| `catt-ed` | −475 | −602 | +482 | +626 | 0.4538 | 0.6379 |
| `shakkala` | −1,577 | −938 | +552 | +2,463 | 0.5683 | 0.7152 |
| `mishkal` | −5,122 | −1,231 | +534 | +6,986 | 0.7162 | 0.8191 |

Attribution — is the damage bounded by the corruption that caused it?

| tool | corrupted sites × clips | \|Δ(FR+FA)\| | ratio | prompts over bound | artifact prompts |
|---|---|---|---|---|---|
| `catt-eo` | 1,308 | 227 | 0.174 | 2 | **0** |
| `catt-ed` | 1,587 | 346 | 0.218 | 3 | **0** |
| `shakkala` | 3,855 | 678 | 0.176 | 1 | **0** |
| `mishkal` | 9,097 | 979 | 0.108 | 0 | **0** |

Ratios are well under 1, so corruption is **transmitted, not amplified**, by
the three-way Needleman-Wunsch alignment, and **0 prompts show damage at zero
corrupted sites** for any tool — the stronger check, since it cannot be
explained by scaling. Between 0 and 3 individual prompts per tool do exceed the
bound locally; they are not enough to carry the effect but they are the right
place to look if the mechanism is ever questioned.

Prediction 3, text level: CE/noCE DER ratio is **1.46–1.68** across the four
tools, so roughly 40% of each tool's text-level error sits on case endings.
Phoneme-level share not measured (see the script's docstring: it needs a
word-aligned phonetizer, deliberately not built).

#### Observations

1. **Prediction 1 is falsified on both counts, and this is the week's real
   result.** ΔF1 is **positive for every tool**, and the dominant moving cell
   is **TR**, not FA. Automatic diacritization does not make this MDD system
   look worse — it makes it look **dramatically better**, by up to +0.28 F1
   absolute. `mishkal`, the worst diacritizer in the set, takes F1 from 0.4824
   to 0.7642, which would be a state-of-the-art-looking number produced
   entirely by corrupting the reference.

2. **The mechanism is clean and it is the opposite of noise.** In this metric
   TR counts positions where `C ≠ A` (the speaker deviated from the canonical)
   and the system's output matches `A`. Corrupting `C` *enlarges the set of
   positions where `C ≠ A`*. Every position where the diacritizer gets it wrong
   **and the model correctly predicted what the speaker actually said** converts
   from TA into TR — a true rejection, with correct diagnosis. So reference
   corruption does not merely add noise to the score: it **manufactures
   apparent detections out of accurate model outputs**. TA falls by 5,122 and
   TR rises by 6,986 for `mishkal`, which is that conversion visible in the
   cells. FR also falls (−1,231), raising precision from 0.3821 to 0.7162.

   FA does rise (+482 to +552), as prediction 1 said it would, but the rise is
   nearly constant across tools and is swamped an order of magnitude over by
   the TR gain. So the predicted mechanism exists and is real; it is simply not
   the dominant one.

3. **Week 6's plan argued this sign was impossible here, and the argument was
   wrong.** Plan §1 raises F1 inflation but attributes it to a model whose
   *own* errors correlate with the reference's errors, and the week-6 plan
   concluded "a prompt-free model cannot have correlated errors with `C_auto`.
   So the sign may legitimately flip in weeks 10–12 for the text-dependent
   arm." The inflation observed here needs **no** such correlation: it comes
   from the base rate of apparent mispronunciations rising, which flatters any
   detector that fires often. The prompt-free arm had no protection against it.
   That reasoning should be corrected in the plan rather than quietly dropped,
   and the weeks-10–12 expectation re-derived.

4. **Leg B independently validates the mechanism, and it was not designed to.**
   The gold cell scores **higher** than the anchor (0.4824 vs 0.4415), i.e. our
   phonetizer's disagreement with the uploader's canonical inflates F1 by
   +0.0408 — in the same direction, by the same mechanism, as every
   diacritizer's error does. Two independent sources of canonical disagreement
   (a tajwīd convention gap and four diacritizers) move F1 the same way, which
   is strong internal evidence that the effect is a property of the *metric
   under canonical corruption* and not of any one tool.

   It also retrospectively vindicates D9. That +0.0408 is **larger than
   `catt-eo`'s entire ΔF1 (+0.0265)**: had `reference_phoneme_string` been used
   as `C_gold`, the best tool's measured label-path effect would have been
   smaller than the confound bundled in with it, and the paper would have
   reported a diacritization effect that was mostly our own phonetizer.

5. **Monotonicity holds, so RQ3 gets no support from this run.** The ordering
   `catt-eo` < `catt-ed` < `shakkala` < `mishkal` is now agreed by a fourth
   independent instrument. An inversion would have been the first real evidence
   that DER is miscalibrated against downstream harm; there is none. Note the
   relation is strongly non-linear, though: CE DER rises 4.6× from `catt-eo` to
   `mishkal` (3.98 → 18.20) while ΔF1 rises 10.6× (+0.027 → +0.282), so DER
   under-predicts label-path damage at the bad end. That is a weaker and more
   interesting version of RQ3's claim and is worth carrying forward.

6. **Prediction 3 is confirmed and it is a caveat with teeth.** ~40% of each
   tool's text-level error is on case endings, on an arm where recitation
   *pronounces* them. Plan §1's argument that case endings are "close to
   irrelevant to pronunciation teaching" is an MSA pausal argument and must not
   be supported with a number from this arm. The asymmetry has to be stated
   explicitly in the write-up; a reviewer who recites will catch it.

7. **The direction of the headline has to change, and the framing does not.**
   Plan §5's pre-commitment binds on a *null*, and this is not a null — it is a
   large, clean, monotone effect whose sign is the opposite of the one assumed.
   The paper's claim shifts from "automatic diacritization corrupts MDD
   evaluation and depresses measured performance" to "automatic diacritization
   **inflates** measured MDD performance, by converting correct model outputs
   into apparent detections." That is a stronger result and a more alarming one
   for the field: a system evaluated against an auto-diacritized reference
   reports a score that is too good, so the error is in the direction nobody
   checks.

8. **An implementation bug, caught and recorded.** The attribution diagnostic
   first divided total damage by *unscaled* per-prompt corrupted sites, while
   its own docstring said sites were scaled by the prompt's clip count. One
   corrupted canonical is re-used by all ~16.8 clips reading that prompt, so
   the ratio was inflated ~16.8× and read 1.82–3.53, which says "realignment is
   amplifying corruption", when the scaled values are 0.108–0.218, which says
   "corruption is transmitted". The sign of the conclusion inverted on a
   missing multiplication. Fixed and the artifact regenerated; every other
   figure in this entry reproduced bit-for-bit across the two runs, which is
   the reason the bug was safe to fix after the fact. The qualitative check
   that actually carries the gate —
   `prompts_damaged_with_zero_corruption` is empty — was unaffected either way,
   because it does not depend on scaling.

#### Conclusion

Clause 1 of the week-6 gate passes on criteria fixed before the run: four of
four CIs exclude zero, monotone in DER, with no alignment artifact. The
label-path effect is not only measurable, it is large.

But the **sign is positive**, which falsifies the week's primary prediction and
inverts the paper's headline: on the Qur'anic prompt-free arm, scoring against
an automatically diacritized canonical *inflates* MDD F1 — +0.0265 for the best
tool in the set and +0.2818 for the worst — because corrupted canonicals
convert accurate model outputs into true rejections. The FA-deflation mechanism
the plan predicted is present but an order of magnitude smaller than the TR
inflation that dominates.

Open, in order: (a) correct plan §1/week-6's claim that a prompt-free model
cannot show inflation, since it just did; (b) task 3's power estimate now has
real per-prompt variance to use (per-prompt ΔF1 sd 0.089 / 0.117 / 0.151 /
0.180 across the four tools, n = 98); (c) decide whether the inflation framing
changes which RQ is cuttable, which is a week-12 gate question and should not
be pre-empted here.

### Experiment: the RQ5 interaction power estimate

#### Hypothesis

Clause 2 of the week-6 gate asks for a number, not a verdict, so there is no
prediction to falsify in the usual sense. What *is* pre-registered is the
decision the number triggers: **if the MDE exceeds the input-path main effect
CROTTC-IF reports, RQ5 is cut in November** rather than discovered to be
underpowered in December (`docs/weeks/week-06.md` clause 2, from plan §6). RQ5
is the designated cuttable RQ, so a cut is a planned outcome and not a failure.

Two things were expected going in, both from the plan rather than from
intuition: that the design effect from prompt clustering would be large (`C` is
constant within a prompt, so for a `C`-driven effect the effective n should be
nearer 98 than 1,642), and that the MDE would land in the same ballpark as the
measured label-path effects, since it is built from their variance.

#### Methodology

The estimand is a difference-in-differences:

    DiD = ΔF1(text-dependent arm) − ΔF1(prompt-free arm)

RQ5 is an interaction, not a main effect — the main effect is already settled
by CROTTC-IF — and interactions cost more power than the effects they sit on.
Week 6 has measured the right-hand term; the left-hand term needs the
text-dependent arm, which is weeks 7–9. So the deliverable is the **minimum
detectable DiD** at 80% power, α = 0.05 two-sided:

    MDE = (z_0.975 + z_0.80) · SE(DiD) = 2.8016 · SE(DiD)
    SE(DiD) = SE_PF · √(1 + k − 2ρ√k)

with `k` the assumed ratio of the text-dependent arm's per-cluster variance to
the prompt-free arm's (**1×, 1.5×, 2×**, as the clause requires) and `ρ` the
correlation between the arms' per-prompt deltas. Both arms are scored on the
**same 98 prompts against the same corrupted canonicals**, so they are paired
and positively correlated; `ρ = 0` discards that and is therefore the
conservative headline, with `ρ = 0.5` reported beside it to show what is being
given away. Every figure in the artifact is keyed by both parameters, because
an MDE quoted without its assumptions gets cited later as if it were measured.

`SE_PF` is **measured** — the 10,000-replicate paired cluster bootstrap over
the 98 prompts — and re-derived here from the stored per-prompt counts so the
two can be compared.

The naive comparator is **bounded, not measured**, and the direction is stated
because it matters. A true clip-level i.i.d. bootstrap needs per-clip counts,
which the label-path artifact deliberately does not store (1,642 count rows ×
5 conditions, for a diagnostic). Instead: a **clip-equivalent bootstrap** — 1,642
draws from the clip multiset, prompt *j* drawn with probability ∝ its *mⱼ*
clips, each draw contributing that prompt's counts scaled by 1/*mⱼ*. That is
the clip bootstrap under the approximation that every clip of a prompt equals
the prompt's mean clip, which **deletes within-prompt variance**. So the naive
SE reported is a *lower* bound and the design effect built from it an *upper*
bound. Not re-running the cluster job to get the exact figure was a deliberate
cheap-path call; the bound is signed, and a second, analytic handle is
available as a cross-check (for a purely `C`-driven effect the ICC is 1 and the
classical design effect is √(1 + (m̄ − 1)) = √16.76 = 4.09).

On Kadambi's mixed-effects template: the prompt bootstrap is the
design-effect equivalent of a random intercept for utterance, which is the
grouping factor that matters for a `C`-driven effect. The other factor,
**speaker, is not modelled** — the artifact has no per-clip counts, so speaker
cannot be recovered from it, and the 18 speakers are *crossed* with the 98
prompts rather than nested. A prompt-only bootstrap treats speaker variation as
within-cluster noise, which **inflates** the per-prompt variance it measures.
So this approximation also errs toward a larger MDE.

**Pinning the threshold, which was not evaluable as written.** Our notes had
recorded only the *endpoint* of CROTTC-IF's main effect (F1 → 40.52%), not the
baseline it drops from, so the pre-registered rule had no number on its
right-hand side. Resolved from the paper: §6.4.3 table 4(c) gives
**CROTTC-LLaMA 56.87% F1** and **the same system with canonical injection
40.52%**, on the same 900-utterance L2-ARCTIC test set — a **within-system
ablation**, one model, one corpus, the canonical conditioning switched on and
off, worth **16.35 F1 points**. The abstract's 71.77% CROTTC-IF headline is the
wrong comparison for a threshold: it is a *different* system (the
implicit-fusion fix), so it prices the repair rather than the damage.

#### Configuration

`scripts/power_rq5.py --write`, reading `run/quranic_label_path.json` only. No
network, no GPU, no re-running of diacritizers; it is arithmetic on the stored
per-prompt counts and runs in under a minute locally. 10,000 replicates, seed
20261014 (the same seed as the label-path run, deliberately). Artifact:
`run/rq5_power.json`.

#### Results

Internal consistency first, since everything rests on it:

| tool | SE_PF recomputed | SE_PF as measured in the label-path run | ratio |
|---|---|---|---|
| `catt-eo` | 0.01068 | 0.01081 | 0.99 |
| `catt-ed` | 0.01314 | 0.01304 | 1.01 |
| `shakkala` | 0.01619 | 0.01641 | 0.99 |
| `mishkal` | 0.01869 | 0.01897 | 0.98 |

Two bootstraps in two scripts agree to within 2%, so the stored SD is the
quantity it claims to be.

**The design effect, measured and predicted:**

| tool | SE cluster | SE naive (lower bound) | design effect (≤) |
|---|---|---|---|
| `catt-eo` | 0.01068 | 0.00260 | 4.11× |
| `catt-ed` | 0.01314 | 0.00327 | 4.01× |
| `shakkala` | 0.01619 | 0.00397 | 4.08× |
| `mishkal` | 0.01869 | 0.00453 | 4.13× |

Against the analytic prediction of **4.09×**. Four empirical values in
[4.01, 4.13] around an analytic 4.09, from two unrelated derivations. **Quoting
the naive MDE would have overstated the power by a factor of four.**

**The MDE grid** (Qur'anic, 98 prompt clusters, ρ = 0 unless noted):

| tool | k=1 | k=1.5 | k=2 | k=1.5, ρ=0.5 | MSA n=500, k=2 |
|---|---|---|---|---|---|
| `catt-eo` | 0.0423 | 0.0473 | 0.0518 | 0.0338 | 0.0192 |
| `catt-ed` | 0.0521 | 0.0582 | 0.0638 | 0.0416 | 0.0254 |
| `shakkala` | 0.0641 | 0.0717 | 0.0785 | 0.0512 | 0.0327 |
| `mishkal` | 0.0740 | 0.0828 | 0.0907 | 0.0591 | 0.0390 |

**Worst corner: MDE = 0.0907 F1** (mishkal's variance, k = 2, ρ = 0,
clustered) against a **0.1635** threshold. **Recommendation: KEEP RQ5.**

The MDE against each tool's *own* label-path effect — a harsher question the
clause did not ask:

| tool | \|ΔF1\| | MDE/\|ΔF1\| worst corner | at k=1.5, ρ=0.5 |
|---|---|---|---|
| `catt-eo` | 0.0265 | **1.96** | 1.28 |
| `catt-ed` | 0.0480 | **1.33** | 0.87 |
| `shakkala` | 0.1510 | 0.52 | 0.34 |
| `mishkal` | 0.2818 | 0.32 | 0.21 |

#### Observations

1. **RQ5 survives, but the margin is 1.80× and that is a pass, not comfort.**
   The pre-registered rule is satisfied and the honest statement of by how much
   is "slightly less than a factor of two at the worst corner, 2.77× at the
   corner the design more plausibly sits at". Recording the margin rather than
   just the verdict matters, because a 1.80× margin is one adverse surprise
   away from a cut and weeks 7–9 will produce the surprise if there is one.

2. **What makes the pass defensible is that every approximation points the same
   way — toward a larger MDE.** The design effect is an upper bound, speaker
   variance is folded into the per-prompt spread rather than modelled, ρ = 0
   discards pairing the design actually has, and k = 2 doubles a variance
   nobody has observed. Four conservative choices stacked, and it still passes.
   Had any of them gone the other way the recommendation would be worth less
   than the number attached to it.

3. **The design-effect cross-check is the most reassuring thing in the run, and
   it was not designed as one.** An empirical bootstrap and a textbook ICC
   formula have nothing in common beyond the data, and they land within 1% of
   each other (4.01–4.13 vs 4.09). That also validates the week-6 plan's claim
   that a `C`-driven effect has an effective n "far nearer 98 than 1,642":
   1,642/4.09² = 98.1. The effective n *is* the cluster count, to three
   significant figures.

4. **RQ5 is powered against the threshold but not against the good
   diacritizers, and that is a scoping decision rather than a cut.** For
   `shakkala` and `mishkal` the MDE is a third to a half of the main effect, so
   an interaction of plausible relative size is visible. For `catt-eo` the
   worst-corner MDE (0.0518) is **nearly twice the tool's entire label-path
   effect (0.0265)** — an interaction there would have to exceed the effect it
   modifies before this n could see it. So RQ5 gets answered on the
   high-corruption tools and the low-DER tools are reported as lacking the
   resolution to say. Writing that in now is cheaper than discovering it in
   December, which is the entire purpose of the clause.

5. **The MSA arm is better powered than the Qur'anic one, for a structural
   reason worth stating.** 500 independent utterances beat 98 clusters by
   √(500/98) = 2.26, and the Qur'anic arm's 1,642 clips buy almost nothing
   because 16.8 of them share each canonical. But the transfer is *optimistic*
   in one respect: a Qur'anic prompt's ΔF1 pools ~16.8 clips and is therefore
   less variable than a single MSA utterance's would be, so the real
   per-utterance sd is probably larger than the 0.089–0.180 used. The MSA row
   is labelled a projection in the artifact and should not be quoted as a
   measurement.

6. **The threshold was not evaluable when the clause was written, and nobody
   would have noticed.** The rule said "the input-path main effect CROTTC-IF
   reports"; our notes held only F1 → 40.52%, an endpoint. Reading 40.52 as the
   effect would have given a 0.4052 threshold and a trivially comfortable pass;
   differencing the abstract's 71.77% CROTTC-IF against 56.87% would have given
   0.1490 and a 1.64× margin. The correct pairing — 56.87 → 40.52, same system,
   same corpus, conditioning on and off — gives 0.1635 and 1.80×. Three
   plausible readings, three different margins, one of them inflated by 2.5×.
   A pre-registered rule needs its threshold pinned to a *table cell*, not to a
   paper, and this one was not.

7. **D2 is decided, and on the argument the rule did not anticipate.** Week 6's
   branch was "bound if the MDE is comfortably below the effects of interest,
   otherwise take the sensitivity row". The MDE is *not* comfortably below, so
   the stated branch points at the sensitivity row — but the sensitivity row
   would carry the pilot's 2.3%–19.8% deviation band, an 8.6× spread, which is
   several times wider than the 0.0907 MDE it is meant to qualify. It would
   swallow the interaction whole. So the term is **bounded**, on
   informativeness rather than on margin, with the row kept beside it as a
   cross-check. The bound itself is cheap and available on the arm where both
   `A` and `C_gold` exist: re-score the four Qur'anic label-path conditions with
   `A := C_gold` substituted, and the difference against the measured ΔF1 *is*
   the residual term — one extra scoring condition per tool on the existing
   harness. Scheduled weeks 10–11 with the MSA column it qualifies.

8. **The artifact is reconstructable from an artifact, which is the cheap path
   working.** This whole estimate reads `run/quranic_label_path.json` and
   touches no model, no network and no GPU. The one thing it could not do for
   want of stored per-clip counts — an exact naive SE — was replaced by a
   *signed* bound plus an analytic cross-check that agreed to 1%. That is the
   right trade: a cluster re-run to tighten 4.04 into 4.06 would have changed
   no decision in this file.

#### Conclusion

Clause 2 delivers: **MDE = 0.0907 F1** at the most adversarial corner of the
pre-registered sensitivity design, against a **0.1635** threshold pinned from a
within-system ablation. **RQ5 is kept**, by a factor of 1.80 at that corner and
2.77 at the more plausible one — a pass recorded with its margin rather than as
a verdict. The design effect from prompt clustering is **4.0–4.2×**, confirmed
analytically, so the effective n is the 98 clusters and not the 1,642 clips;
the naive figure would have overstated power fourfold.

The result the clause did not ask for is the one that shapes weeks 7–9: RQ5 has
resolution on `shakkala` and `mishkal` but **not on `catt-eo` or `catt-ed`**,
whose label-path effects are smaller than the MDE. RQ5 is therefore scoped to
high-corruption tools in the write-up rather than cut.

**D2 is decided**: bound the residual interaction term, keep the 2.3%/19.8% row
as a cross-check, obtain the bound by re-scoring the Qur'anic conditions with
`A := C_gold` in weeks 10–11.

Open, in order: (a) the threshold's provenance should be quoted as a table cell
in the paper, not as "CROTTC-IF reports", since three readings of that phrase
give margins spanning 2.5×; (b) weeks 7–9 must report the text-dependent arm's
per-prompt ΔF1 sd so `k` stops being an assumption — it is one number and the
whole estimate is keyed to it; (c) speaker is still unmodelled, and if the
week-12 gate wants Kadambi's full specification the label-path harness needs to
emit per-clip counts, which is a one-line change there and a re-run here.

### Experiment: the annotation logging tool

#### Hypothesis

Writing the tool for the pre-registered protocol is an engineering task, not a
measurement one: §6 fixes the schema, §4 fixes the role assignment, §3 fixes
the rules, so the code should be a transcription of the document with no
decisions left in it.

Weakly held, and stated so the failure is legible. The protocol was written
without a tool, so anywhere it is *silent* the code has to decide something,
and anywhere it is stated in prose the code has to pick one reading. Expected
falsifier: at least one place where the document admits two readings that
differ in the data they produce.

#### Methodology

Three pieces, split on how long each has to outlive the annotation session:

1. `src/arabic_mdd/data/annotation.py` — §4's role assignment by draw rank,
   §6's 16-field record, the four ingest checks, R4 site/violation detection,
   and the file-level report (§4's ≥45 crossover rule, D4's timings). Placed in
   `src` because §6 requires the checks "automated at ingest" and weeks 7–9
   re-read `C_gold` long after the tool stops running.
2. `scripts/prepare_annotation_drafts.py` → `run/msa_annotation_drafts.json`,
   the 500 `catt-eo` drafts, generated once and committed. §6 logs the
   `catt-tashkeel` version that produced `suggested`, which can only be a fact
   about an artifact; and §8's Δ is a within-utterance contrast, so a crossover
   row's draft must be byte-identical whenever it is shown.
3. `scripts/annotate_msa.py` — the CLI, with `--status` and `--validate` modes
   that run the same checks on a finished file.

Verification: `tests/data/test_annotation.py` quotes the protocol wherever the
two could drift (§4's role table row by row, §6's field list in order, §4's ≥45
rule, §3's 31 R4 positions), and the checks were run against the committed
500-row block rather than only against fixtures.

#### Configuration

- Block: `run/msa_annotation_block.json`, seed 20261001, 500 rows (committed).
- Draft tool: `catt-eo`, `catt-tashkeel==1.0.2`, pin checked at runtime per §2.
- `protocol_version`: `git hash-object` of the protocol file — the blob sha, not
  repo HEAD, since HEAD moves when anything in the repo changes and so would not
  answer "which version of the protocol was this row annotated under".
- Drafting cost: **3.8s for all 500 rows** on CPU (0.008 s/row).
- Tests: 37, `uv run pytest` 235 passed.

#### Results

Several measurements came out of what was supposed to be transcription.

| what was checked | result |
|---|---|
| R4 sites, implemented rule, on the frame's marked text | **31 sites in 29 rows** — §3's figure exactly |
| R4 sites, skeleton-only reading of the same prose | **386 sites in 262 rows** |
| drafts whose skeleton differs from the frame (§6 check) | **1 of 500** — rank 380, id 64376 |
| drafts whose Arabic word count differs | **0 of 500** |
| rows with source punctuation that `catt-eo` strips | **299 of 299** |
| rows the punctuation splice repairs | **299**, leaving **1** (rank 380) |
| block rows whose `sentence` already carries marks | **71 of 500** = §7's 59 flagged + 12 under the flag's 0.05 threshold |
| rows the *first* version of the splice corrupted | **8**, invisible to all four of §6's checks |
| frame rows above §7's stated 0.50 exclusion threshold | **0** — max stored `diacritization_rate` is 0.4848 |

#### Observations

1. **The hypothesis is falsified, by R4.** §3 describes R4 in prose, and two
   readings of that prose differ by more than 12×: the skeleton-only one ("a
   non-final word followed by one beginning `ال`") gives 386 candidate sites in
   262 of 500 rows, the marked-reference one ("the preceding word's final letter
   carries sukūn") gives 31 in 29. The second reproduces §3's own "31 positions
   in the 500-row block" exactly, which is what settles it — the protocol
   counted with the rule it did not fully write down. The wrong one would have
   warned on more than half the block, almost always on a legitimate R1 case
   ending, and been tuned out by week 2. That number is now a regression test,
   so the rule cannot drift back.
2. **The punctuation finding is the one that would have cost real data.**
   `catt-eo` strips punctuation from *every* row that has any, 299 of 500. None
   of §6's four checks fire on it, because `skeleton()` ignores punctuation — so
   this was invisible to the entire gate as designed. The damage is not the
   missing characters; it is that §7 says punctuation is left untouched, so the
   anchored arm would have produced `final` strings systematically differing
   from the blank arm's on something that is not diacritization, feeding
   directly into §8's within-utterance Δ. Discovered by reading one row of a
   smoke run, not by any check.
3. **The splice is exact, not heuristic, and only because of a property that
   was already measured.** The draft is the source's letters with marks
   inserted — `skeleton(draft) == skeleton(source)` on 499 of 500 rows — so a
   two-pointer merge is well-defined. On the one row where that does not hold it
   declines and leaves the model's output alone. 299 rows repaired, including
   rank 250's tatweel stretching, which is not punctuation but is equally not
   the model's to delete.
4. **The fix for finding 2 introduced a worse bug than finding 2, and the only
   reason it is not in the artifact is a check nobody specified.** The first
   version of the merge emitted *any* source character the draft had skipped.
   But **71 of the 500 rows already carry marks** (§7's 59 flagged
   `partially_vowelized`, plus 12 whose rate falls under the flag's 0.05
   threshold), and on those rows the model routinely marks a letter differently
   from the source. So a source mark the draft "skipped" was not a dropped
   character at all — it was a disagreement, and carrying it across stacked it
   on top of the draft's own: `طَاغُونُ` became `طَاغُونَُ`, two short vowels on one
   letter, a string neither the source nor the model proposed. **8 rows.**

   The part worth keeping is *why* it nearly shipped. All four of §6's ingest
   checks compare skeletons or word counts, and both are blind to diacritics by
   construction — a draft with two vowels on a letter passes every check the
   protocol specifies, and the annotator would have met it as text that simply
   cannot be right. It was found by printing three rows, which is the same way
   finding 2 was found. The script now refuses to write the artifact if any
   letter carries two mutually exclusive marks, and that check is fatal rather
   than a warning precisely because nothing downstream would catch it.

   §7 is what settles the merge's behaviour, and it was already written: the
   source's "existing marks have no authority". So on a disagreement the
   draft's mark stands and the annotator corrects it, which is their job.
5. **A claim in §7 was worth verifying and holds.** §7 says rows with
   vowelization rate ≥ 0.50 were excluded at the sampling frame. Two rows
   looked above it under an ad-hoc rate, but the frame's own stored
   `diacritization_rate` tops out at **0.4848** (rank 31) and nothing reaches
   0.50; the discrepancy was my letter-counting, not a frame leak. The 71-vs-59
   gap resolves the same way — the `partially_vowelized` flag fires at ≥ 0.05,
   and 71 − 12 = 59 exactly. No action, but it is the kind of claim that is
   cheap now and expensive to re-establish in week 9.
6. **Rank 380 is handed to the annotator failing the protocol's own ingest
   check**, and that is not a bug to fix. `catt-eo` deletes `Eddie`, both
   occurrences, exactly as the organizers' vowelizer does — §7 already puts that
   deletion rate at 100% and calls it "a real `C_auto` error and D5's subject".
   §7 also says leave Latin untouched. So on this row the annotator must restore
   it, which is a one-row instruction, and it is now carried in the artifact
   where the tool warns at that row rather than in someone's memory.
7. **One asymmetry accepted rather than fixed.** The anchored arm's R4 warnings
   are computed from the draft, the blank arm's from the frame's
   `tashkeel_sentence` (never shown). Both give 31 sites, but distributed over 29
   rows and 30 rows respectively — so one row warns in one arm and not the other.
   This is an asymmetry in the *warning*, not in the data or the schema, and
   equalising it would mean showing blank-arm annotators something derived from
   a marked text, which is the one thing the blank arm cannot see. Left as is,
   recorded here.
8. **One documented divergence from the week plan.** Task 4 says `suggested` and
   `final` are stored "per word"; §6 stores them as whole-utterance strings with
   `flagged_words` as indices. The protocol is the pre-registered artifact, so it
   wins, and nothing is lost — per-word override rate, pilot §6a's regret, is
   recoverable from the two strings by word index.
9. **The ingest gate gained a check the protocol implies but does not list.**
   §6's four checks do not include "a blank record must not carry a
   `suggested`", because the schema makes it obvious. `build_record` cannot
   produce one, but `validate` is the gate for files that reached ingest by any
   route, so it now reports a blinding breach explicitly: §8's Δ for that
   utterance is not noisy, it is gone.
10. **What the tool deliberately does not do.** No adjudication UI —
   `adjudicated` is written `null`. Adjudication needs both finished files side
   by side, happens once, and §5's point is that `final_before` keeps annotator
   accuracy separable from adjudicated accuracy, which is easier to get wrong
   when threaded through the annotation loop than when written as its own pass.

#### Conclusion

The tool exists and the three prohibitions are enforced structurally rather than
by discipline: order comes from the committed draw with no skip or jump,
`build_record` forces `suggested` to `None` on the blank arm and the renderer is
never handed the drafts mapping on those rows, and a skeleton-altering
submission cannot be saved — it loops back into the editor, with `unusable` as
the only exit. Annotation is no longer blocked on code; it is blocked on
annotator 2's qualification test, which is external latency.

The hypothesis that this was pure transcription was wrong, and usefully so. A
pre-registered protocol is prose, and prose has readings: §3's R4 admitted two
that differ 12-fold, and only the document's own count of 31 could decide
between them. The general lesson for the remaining pre-registered documents is
that **a protocol clause with a number attached is checkable and one without is
not** — §3's "31 positions" is what made R4 implementable, while §7's
punctuation clause, which carries no count, is exactly the one the tool
silently violated on 299 rows until a smoke run exposed it.

The sharper lesson is about the checks rather than the prose. §6 specifies four
ingest checks and all four compare skeletons or word counts, so **the entire
specified gate is blind to diacritics** — a strange property for the gate on a
diacritization reference. Both of this task's real bugs lived in that blind
spot: punctuation stripped from 299 rows, and then 8 rows carrying two vowels
on one letter, introduced by the fix for the first. Neither was caught by a
check; both were caught by printing a row. The generator now carries a fifth
check that §6 does not list, and it is fatal rather than advisory because
nothing downstream would reject what it catches. Weeks 7–9 should expect the
remaining undetected problems to be diacritic-shaped, for the same reason.

Still open: D4's extension decision is now waiting on annotation hours rather
than on code (`--status` prints the running median); the adjudication pass needs
writing when two files exist; and §7's remaining prose clauses — Latin, digits,
the two malformed ranks — are enforced as warnings, not as checks, because
unlike R4 none of them comes with a count to verify an implementation against.

### Experiment: pinning S3PRL

#### Hypothesis

The S3PRL checkout has been drifting since week 4, so pinning it will reveal
that the tree currently in use is not the tree the week-3 gate was measured on
— and the pin will have to be chosen as a repair rather than a record.

#### Methodology

Read `HEAD`, the dirty state, and the commit date off the cluster checkout that
produced the gate; pin `run/prepare_baseline.sh` to that SHA; then check
upstream to find out whether `main` had in fact moved. The order matters — the
pin had to come from the cluster rather than from upstream, because pinning to
today's `main` would pin a commit the reproduced baseline was never measured
against, which converts a passed gate into a claim about software we no longer
run.

#### Configuration

- Checkout: `run/s3prl_checkout` on `citrin`, read 16 Oct 2026.
- Installed with `pip install -e`, so the working tree *is* the library.
- Upstream `main` queried the same day via the GitHub API.

#### Results

| | |
|---|---|
| cluster `HEAD` | `ec8064b5889f81ca460fbe2c094ce576a6f120b7` |
| what that commit is | upstream's own **v0.4.18** release, committed **2025-06-13** |
| tracked modifications (`--porcelain -uno`) | **none** — the SHA describes the tree exactly |
| untracked content | 3 training expdirs (`mhubert147_per`, `hubert_base_per`, `smoke_mhubert_219336`) |
| upstream `main`, 16 Oct 2026 | **the same commit** — unmoved for 16 months |

#### Observations

1. **The hypothesis is falsified, and in the boring direction.** Nothing had
   drifted. Upstream `main` is still the June 2025 release commit, so a fresh
   clone today lands on the pin by itself, and the pin changes nothing about any
   result already obtained. The risk described in week-06.md task 5 — "clones at
   whatever `main` serves that day" — was a correct mechanism with a realized
   probability of zero.
2. **Which means the SHA was never the missing piece; the enforcement was.**
   Choosing a commit only protects the clone, and the clone is the one moment
   this was never going to go wrong. The states where the damage is silent are a
   `git pull` in that tree, a re-clone after upstream eventually moves, and a
   second machine — none of which the original task framing mentions. The script
   now re-verifies `HEAD` against the pin on **every** run and exits 1 on
   mismatch, which is the part that would actually have caught something.
3. **`pip install -e` makes the dirty-tree case different in kind**, so it
   warns rather than fails. A hand-edit in that tree is already live in every
   run that has happened; refusing to start the job does not undo that, whereas
   announcing it at job start puts it in the log next to the result. The check
   uses `-uno` deliberately — the three expdirs are untracked *outputs* living
   inside the dependency checkout, and a version of this check that flagged them
   would be noise on every single run, i.e. ignored by week 8.
4. **Verified rather than assumed.** The four paths (matching HEAD, mismatched
   HEAD, untracked expdirs present, dirty tracked file) were exercised against a
   scratch git repo with stage 2 extracted into a harness. Worth the five
   minutes: a fail-closed check that is wrong in the closed direction would
   block every training job in week 7, which is the one week that cannot absorb
   it.
5. **The expdirs are a latent hazard, noted and not acted on.** Training outputs
   are being written *inside* the dependency checkout, so a `git clean -fdx`
   there — the obvious thing to type when a pin mismatch is reported — deletes
   experiment results. The error message deliberately recommends
   `checkout --detach` and not `clean`. Moving expdirs out is a week-7 change,
   not a week-6 one.

#### Conclusion

Pinned at `ec8064b` (v0.4.18), with the pin enforced on every run rather than
only at clone time, and verified on all four branches of that check. Week 7 can
start training without the training loop being a moving target.

The honest summary is that this task was five weeks overdue and cost nothing,
because the thing it was protecting against had not happened and still has not.
That is not an argument for having left it — it is an argument for having
checked, since the cost of checking was one command and the cost of being wrong
was the whole second half of the project. What the delay did cost is visible in
the framing: the task was written as a race against drift, and the deliverable
that mattered turned out to be an invariant check that nobody had asked for.

Still open, same exposure, requested in the same round-trip and not returned:
**`transformers` is installed unpinned** (`prepare_baseline.sh`, now line 92).
It supplies the featurizer's model code, `hf_hubert_custom` imports it at module
load, and unlike S3PRL it releases often — so of the two, it is now the more
likely to move under a running experiment. One `pip freeze` closes it.

---

### Experiment: training contamination in the MSA evaluation pool

#### Hypothesis

Week-6 task 7 was written as "select the MSA evaluation pool: CV-Ar rows
disjoint from `Iqra_train`, deduped by consonantal skeleton". The hypothesis
behind it was D4's, scaled up: if raw-string dedup *within* `Iqra_train` dev
under-counted duplicates by 46 rows, then raw-string reasoning *across* splits
should under-count too, and the evaluation pool should be checked with the
skeleton key rather than certified by id-disjointness.

#### Methodology

Two parts, and the first was not a measurement.

**Re-reading the task against decisions taken since it was written.** The task
carried from week 5 unstarted. In the interim D4 decided *annotate dev, not
fresh Common Voice*, because `tashkeel_sentence` exists only in `Iqra_train`
and D5's whole measurement is `tashkeel_sentence` vs `C_gold` — fresh Common
Voice text has no `C_auto` to compare against. The 500-row block is drawn, the
drafts are committed, the protocol is pre-registered. So "CV-Ar rows disjoint
from `Iqra_train`" was not open, and executing the task as written would have
thrown away the arm's `C_auto` leg to fix a bias §5.2 itself calls "not fatal".

What *was* open: §5.2 was written when the only model was the organizers'
`dev-best.ckpt`. From week 7 we train our own, so we choose the splits.

**Measuring the contamination so it can be held out of training.** All 71,391
`Iqra_train` train rows keyed by `normalize.skeleton` — the same key
`build_pool` dedups with, reached through a new `msa_pool.skeleton_overlap`
for the cross-split direction. The same measurement repeated with raw-string
equality, as the control. Three nested evaluation sets, so D4's extension
decision (~1,592 further rows) is costed now rather than re-measured later.

#### Configuration

- `scripts/select_msa_eval_pool.py --write` → `run/msa_eval_pool.json`
- `src/arabic_mdd/data/msa_pool.py`: `skeleton_overlap`, 3 tests (18 total)
- Train shards read with `pa.memory_map` + `.select(["id", "sentence"])`, so
  the audio column is never paged in: **4.2 s, 138 MB peak RSS**, no GPU
- Key: `normalize.skeleton` — NFC, strip combining marks and tatweel, fold
  punctuation, collapse whitespace

#### Results

| Set | Sentences also in train | Train rows | Raw-string match |
|---|---|---|---|
| Annotated block (500) | **140 = 28.0%** | 319 | **0** |
| Eligible pool (2,092) | 510 = 24.4% | 1,157 | 0 |
| Whole dev (2,542 skeletons) | 836 = **32.9%** | 2,075 | 0 |

- Train/dev **utterance-id overlap: 0**
- Holdout: **319 train rows = 0.45% of train**, 140 distinct sentences
- Holdout diacritization rate: median **0.571**, 180/319 pre-vowelized,
  against a train base rate of 32.3% (dev's is 18.9%)

#### Observations

1. **The check everyone runs passes.** Train/dev id overlap is exactly zero.
   An id-disjointness test — the standard one — certifies this split clean
   while 28% of the evaluation text sits in the training set.

2. **Raw-string matching finds 0 of 319.** Not "fewer"; none. The control
   returned zero on all three sets.

3. **And for two reasons, only one of which was predicted.** D4's is the
   first: dev bare, train pre-vowelized. The second was not predicted and is
   worse, because it is so small — among the bare/bare pairs the strings
   differ only in **punctuation placement**:

   | | |
   |---|---|
   | train `01237` | `أحضرها لي.` |
   | dev `67397` | `.أحضرها لي` |

   Same six characters; the full stop moved from one end to the other, almost
   certainly a bidi storage artifact. Also `ثمين ، لهذا` vs `ثمين، لهذا` and
   `تفعله؟` vs `تفعله ؟`. `skeleton` folds punctuation and collapses
   whitespace (§4.1), so it sees through both. Raw-string equality sees
   through neither.

4. **This retroactively justifies week 4's normalization contract better than
   anything measured at the time.** The argument for pinning one contract and
   routing every consumer through it was defensive — don't let two
   implementations diverge. It turns out the contract is also the only reason
   the contamination is visible at all. A pipeline that compared text without
   it would have reported a clean split and been wrong by 28 points.

5. **32.9% vs Sadeed's 34.6%.** Sadeed reports 34.6% of the Fadel test set
   fully present in the Abbad training set. Our whole-dev figure is nearly the
   same number on a different corpus, which is worth more than the citation
   was supposed to be: it stops being "the field frequently skips this" and
   becomes "this is what you find when you look, twice, independently".

6. **The fix costs 0.45% of train.** That is the number that decided the
   approach. Holding 319 rows out is not a trade-off, so there is no
   defensible reason to declare the bias instead of removing it.

7. **The task was stale on arrival, and nothing flagged it.** It was carried
   forward verbatim through two week plans while a decision that foreclosed
   it was taken in between. Both documents describe the pool as something to
   *select*; by week 6 it had been selected, annotated and pre-registered.
   The cheap guard is to re-read a carried task against the decisions taken
   since it was written, before executing it — this is the second time this
   week that re-reading beat executing (task 6's "measure the wall clock" was
   already measured too).

8. **The overlap is enriched for Qur'anic verses.** The first examples that
   surfaced were all Qur'an (`حتى إذا فتحت يأجوج ومأجوج`...), which touches
   the open "no Qur'anic/classical predicate" item in D4: those rows are the
   ones most likely to appear twice in a crowdsourced corpus. Not acted on —
   the predicate still needs an authoritative full Qur'an text — but it is
   now a measured hint rather than a suspicion.

9. **The pool-level row was free and pre-empts a later argument.** 24.4% for
   the 2,092-row eligible pool means D4's extension, if taken, does not need
   this re-measured and cannot be accused of having been checked only after
   the numbers came in.

10. **The same key, pointed at the Qur'anic arm, found contamination there
    too — and that arm is the paper's spine.** Observation 8 made this worth
    one more query. Measured against `recovered_text` in
    `run/quranmb_verse_map.json`: **15 of 98 QuranMB.v2 prompts (236 of 1,642
    clips = 14.4%)** have their verse in `Iqra_train` **train**, 5 prompts /
    71 clips in **dev**, union **16 prompts / 254 clips = 15.5%** (4 prompts
    in both). The clean subset is 82 prompts / 1,388 clips.

    This is more serious than the MSA figure despite being half its size,
    because the arm it hits is the one week 6 measured the label path on and
    the one week 7 builds the text-dependent model for. A prompt-free model
    can only have memorized acoustics; **a text-dependent model consumes `C`
    directly**, so for next week's arm this is text leakage of the purest
    kind. Holding it out costs 62 train rows = 0.09%.

    Deliberately *not* acted on here. The week-3 gate reproduced a published
    number on all 98 prompts, and retroactively changing that denominator
    would destroy the one figure that certifies the pipeline. The decision —
    hold out at training, report both denominators — is
    [week-07.md](../docs/weeks/week-07.md) task 5, taken before the models
    that depend on it exist rather than after.

#### Conclusion

The evaluation pool is D4's annotated 500. §5.2's contamination is removed on
the training side — `run/msa_eval_pool.json` carries `holdout_train_ids`, 319
rows for week 7 to drop from both training and model selection. The
organizers' `dev-best.ckpt` keeps the declared limitation for any number
computed with it, since we cannot retrain that.

The deliverable is a 0.45% holdout. The finding is that an id-disjoint,
raw-string-disjoint pair of splits shared 28% of its text, and that the
smallest thing hiding it was the position of a full stop.

**Flagged, not built:** skeleton identity is exact-match, so Sadeed's
near-duplicate band (68.12% above 0.5 similarity) is neither counted nor held
out. At 0.45% cost the exact tier is cheap insurance; a similarity threshold
is a parameter to defend rather than a fact to report. Revisit only if a
reviewer asks.

**Handed to week 7:** the Qur'anic contamination of observation 10 — 16
prompts / 254 clips / 15.5% — which is the more consequential half of this
experiment and arrived as a by-product of it.
