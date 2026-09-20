# Paper Project Plan v2 — Reference-Aware Evaluation for Arabic MDD

Target: **Interspeech 2027**, São Paulo. Submission **9 February 2027**. Rewritten 16 September 2026 — **21 weeks** of runway.

Supersedes mdd-paper-project-plan.md v1. The experimental machinery there survives almost intact; what changes is the spine of the argument. Evidence for the change is in [novelty-check.md](novelty-check.md), which is now complete across all four search tiers.

Team: two students, A100 cluster access, supervisor to be recruited once results exist.

---

## 0. What changed and why

The v1 plan was a measurement paper: *nobody has measured what automatic diacritization costs Arabic MDD, so we will.* The novelty check says the gap is real — nothing measures it — but it also turned up two things that make that framing the weaker of the available options.

**The precedent problem.** Mathad et al. (Interspeech 2021) and Kadambi et al. (Interspeech 2024) already asked the structurally identical question about forced-alignment error in automated pronunciation scoring. They do not scoop the work, but they remove the broad version of the novelty claim. "Nobody has studied how upstream component error contaminates a pronunciation score" was true on 15 September and is false now. A measurement paper walks straight into *incremental relative to Kadambi*.

**The effect-size problem.** Both papers find the upstream effect minor to moderate, not dominant. A measurement paper whose headline is a magnitude is a paper betting on that magnitude. v1's fallback — "if the effect is small, pivot the framing" — is a plan to rewrite the paper in December, which is not a plan.

So v2 moves the centre of gravity off the magnitude and onto three things that hold whatever the magnitude turns out to be: a **decomposition** that separates two corruption mechanisms nobody has separated, a **metric** showing that the number the field uses to choose a diacritizer does not predict the harm that diacritizer causes, and a **mitigation** that recovers most of the loss. The magnitude becomes evidence inside that argument rather than the argument itself.

The single insight the whole rewrite rests on is in §1.

---

## 1. The claim

In Arabic MDD the canonical sequence `C` does two jobs at once. It is the model's input, in text-dependent systems. And it is the **scoring key** — whether a phoneme counts as mispronounced at all is decided by comparing the learner's actual production `A` against `C`. Every published system depends on the second role; only some depend on the first.

This dual role is what makes reference error in MDD categorically different from upstream error elsewhere in speech. A misaligned boundary moves the *measurement window*: the reference phonemes are still right, and the score degrades because the acoustics were read over the wrong interval. A wrong diacritic moves the *target label*: the system is asked to detect deviation from a phoneme that was never correct, so the ground truth itself shifts. Alignment error cannot make a correctly pronounced phoneme count as an error by definition. A wrong diacritic can, and does.

**Working title:** *When the Reference Is Wrong: Decoupling Label and Input Corruption in Arabic Mispronunciation Detection*

**One-sentence claim:** Because the canonical sequence is simultaneously the model's input and the scoring key, automatic diacritization corrupts Arabic MDD evaluation in two separable ways that no existing protocol distinguishes; separating them shows that the diacritizer metric the field selects on (DER) does not predict downstream harm, that the harm concentrates in a diacritic class that is irrelevant to pronunciation teaching, and that most of it is recoverable by letting the reference abstain where the diacritizer is unsure.

### Where the error enters, and why it can be decoupled

```
undiacritized MSA text → [diacritizer] → vowelized text → [phonetizer] → canonical C
audio                  → [MDD model]  → predicted sequence P
human annotation                                         → verbatim actual A

scoring aligns (C, A, P) → TA / TR / FA / FR → CD / ED
                ↑
        C enters here too — as the key, not just the input
```

**Label-path corruption.** Corrupt `C` and the error labels move. Phonemes the learner produced correctly are relabelled as errors the system failed to catch. This hits *every* architecture, including prompt-free ones, and it is invisible to anyone who does not think to look, because the corrupted key is also what you would score against.

**Input-path corruption.** Text-dependent systems additionally consume `C` at inference and are misled by it. CROTTC-IF has already shown that canonical conditioning is harmful even when `C` is *correct*, and that "canonical information can easily override the subtle acoustic information" — so a corrupted `C` has a named mechanism waiting for it.

Here is the part that is only possible with a text front-end. A reference text is a detachable object: you can hand the model one version and the scorer another. Alignment is not detachable — you cannot give a model one alignment and the scorer a different one — which is precisely why Kadambi measures a combined effect and we can measure two separated ones. **The decoupling is the contribution, and it is not available to the precedent.**

### The 2×2

| | Scored against `C_gold` | Scored against `C_auto` |
|---|---|---|
| **Model given `C_gold`** | Ceiling — the condition the field currently reports | **Label corruption alone** |
| **Model given `C_auto`** | **Input corruption alone** — deployment-facing primary metric | Naive deployment evaluation |

Four cells, each answering a distinct question. The top-left is what published numbers mean. The bottom-left is the honest deployment number: the system operates as it would in production while truth is held fixed, which answers *is the feedback on the learner's screen correct*. The top-right isolates measurement corruption with the model unimpaired. And the bottom-right is what a group would report if they evaluated on MSA with an automatic diacritizer and did not think about any of this.

The bottom-right cell yields what may be the most citable single number in the paper: **the direction and size of the bias in a naive automatic-reference evaluation.** Nobody knows the sign. Label corruption could inflate apparent F1 (the system's own errors coincide with reference errors and cancel) or deflate it. Every group that moves from Qur'anic to MSA evaluation will produce that number without knowing it is biased. Telling them which way, and by how much, is cheap for us and immediately useful to them.

Prompt-free systems never consume `C`, so for them the rows collapse and only the scoring column exists. That makes the prompt-free arm a clean corroborating control on label-path corruption rather than the bolt-on architectural comparison it was in v1. The old RQ4 gets absorbed into the identification strategy and becomes load-bearing instead of optional.

### The empirical result to hunt for

Unchanged from v1, but now it does different work. Case endings (*i'rab*) dominate Arabic diacritization error — CATT's own WikiNews numbers show error roughly doubling when they are counted: **DER 5.43% with CE vs 3.11% without, WER 22.13% vs 12.68%** (the EO variant; pin this, because a reviewer checking ED will find different numbers). But case endings are grammatically determined word-final vowels that MSA speakers drop in pausal form, and they are close to irrelevant to pronunciation teaching. CATT's authors say so themselves: "the presence or absence of the diacritic on the last letter mostly depends on grammatical rules."

In v1 this was a hypothesis the paper's strength depended on. In v2 it is the **existence proof for the weighted metric** in §2's RQ3: a diacritic class that contributes heavily to DER and little to pedagogically meaningful downstream harm is exactly a demonstration that DER is the wrong selection criterion. If the case-ending effect is large, the weighted metric has a headline. If it is small, the weighted metric has a null that still shows DER's components are not interchangeable. Either way the section survives.

It also has a use outside this paper: excluding case endings from scoring is the concrete answer to the open scoring-threshold question in `09-phase2-technical-architecture.md` §7 of the language-learning-brainstorming project, which is where this line of work started.

Independent corroboration arrived from outside Arabic, which is worth quoting in the introduction: the Koshur Kashmiri diacritizer paper states that "automatic metrics such as DER and WER penalize every mark mismatch equally, yet not all diacritic deviations are perceptually or linguistically significant" (§VII-B — not the abstract), reporting DERm 0.2012 against a 77.5% expert rating. Two languages, two unconnected groups, one conclusion is a much stronger opening than one assertion about Arabic.

---

## 2. Research questions

Five, ordered by how load-bearing they are. RQ1 is setup, RQ2–RQ4 are the paper, RQ5 is corroboration that can be cut.

**RQ1 — How good is diacritization on the text that actually feeds Arabic MDD?** Published DER/WER are on WikiNews and news corpora. Nobody reports them on Qur'anic text or Common Voice Arabic transcripts. Measure, split with and without case endings. Keep this compact — it is necessary setup, not a contribution, and the domain-gap evidence is partly free from the KSAA-2026 shared task (best DER 10.56% on speech dictation against CATT's 5.43% on news; check metric definitions match before putting the two in one sentence).

**RQ2 — How does the F1 loss decompose into label-path and input-path corruption, and which way does a naive automatic-reference evaluation bias reported scores?** The spine. Run the full 2×2 for the text-dependent arm and the scoring column for the prompt-free arms.

**RQ3 — Where does the loss concentrate, and does DER predict it?** Two decompositions along different axes, and both are needed. **By diacritic class** — case ending versus internal short vowel versus gemination versus vowel length — which is what the weighted metric in §4.1 is built from. And **by phonetic category** — emphatics (ط ص ض ظ) against their plain counterparts (ت س د ذ), gutturals (ع ح خ غ ق ه), gemination/shadda, vowel length. The first axis asks which diacritizer mistakes matter; the second asks which phonemes suffer. They are not the same question and the answers may not line up. Then: does a weighted metric predict MDD F1 loss better than raw DER? This is the contribution that gets cited by people who do not work on Arabic.

The phonetic-category axis is where Kadambi's statistical template earns its keep — their mixed-effects specification with phoneme position and phoneme type as predictors is a better frame than a per-category F1 table, and it handles the speaker and utterance clustering this data will have. There is also a prior worth citing: the Speech Communication 2024 aligner study finds stops and fricatives robust to upstream error while nasals, approximants and vowels are not, which is the same shape of result for a neighbouring component and predicts that vowel-bearing positions are where diacritization damage concentrates.

**RQ4 — Can the loss be recovered by letting the reference abstain?** Threshold diacritizer confidence, mark low-confidence positions as underspecified, exclude them from scoring, and plot coverage against F1. Converts the paper from diagnosis to remedy.

**RQ5 — Do text-dependent systems degrade worse than prompt-free ones?** Now partly answered by RQ2's structure. Kept as an explicit comparison because CROTTC-IF makes it a live debate, but it is the cuttable one. Note that CROTTC-IF has already settled the main effect — conditioning hurts even with a correct `C`, dropping LLM F1 to 40.52% — so what remains is an *interaction*: is the additional damage from corruption larger for text-dependent systems? Interactions need more power than main effects. Specify as a difference-in-differences and power it at the week-6 pilot, not at week 12.

### What is deliberately out of scope

Two good ideas are being held back, and writing down why is part of protecting the deadline. **Partial diacritization** — a reference where a human has hinted the ambiguous positions, which is the actual deployment condition given that real Arabic text is neither fully nor zero diacritized — is a future-work sentence and, if there is slack, one oracle curve showing the headroom. **Cross-lingual validation** on Hebrew or Kashmiri is a paper of its own. Neither goes in the four pages.

---

## 3. Experimental design

Almost all of v1's machinery is unchanged. The new work is in conditions and metrics, not in models, which is why v2 does not cost more weeks than v1.

### Systems

A range of architectures matters more than a peak, because the argument is that the effect is architecture-independent.

| System | Role | Notes |
|---|---|---|
| mHuBERT frozen + BiLSTM + CTC | Organizer baseline, prompt-free | Fully specified in the benchmark paper; reproducing it is the week-3 gate |
| XLS-R-300m + layer fusion + TCN + CTC | Strong prompt-free | Approximates the whu-iasp recipe; the "modern system" arm |
| Same backbone, canonical-sequence conditioning | Text-dependent | **You build this.** Required for the 2×2 and for RQ5 |
| Any public IqraEval checkpoints | Bonus | Check what teams released; do not depend on it |

Building the text-dependent arm yourself remains a feature: two of your own systems differing in exactly one property is cleaner evidence than two unrelated teams' submissions. It is now also a requirement rather than a nicety, because only a system you control can be fed `C_auto` while the scorer sees `C_gold`.

### Diacritizers

CATT (EO and ED, [github.com/abjadai/catt](https://github.com/abjadai/catt)) as the strong modern system, plus Shakkala and Mishkal or Farasa as weaker points, so degradation is plotted as a *curve* against diacritizer quality rather than reported as a single before/after pair. The CATT paper's Table 5 spans 5.4% to 19.6% CE DER across these tools, which is spread enough without adding a fourth.

Two notes. **Mishkal is the useful anomaly**: its CE/no-CE gap is far larger than anyone else's (15.2 → 8.6), meaning it is disproportionately bad at case endings specifically. That makes it a probe for RQ3 rather than merely a weak baseline — if the weighted metric is right, Mishkal should rank better under weighting than under raw DER, and that is a concrete falsifiable prediction to state in advance. And CATT must expose token-level probabilities for RQ4; confirm this in week 1, because RQ4 depends on it.

State plainly as a limitation that the in-house vowelizer used to build Iqra_train is not public, so the work characterises a class of tools, not that specific one.

### Data

| Split | Source | Gold diacritics | Purpose |
|---|---|---|---|
| Qur'anic arm | QuranMB.v2, 1,643 utts | Yes, by convention | Controlled: strip, re-diacritize, compare |
| MSA arm | Iqra_train dev (Common Voice Ar) | **No — you create them** | The realistic condition |

These are not two samples of one population and the paper should say so early. Qur'anic text is canonically diacritized and fixed, so `C_gold` is free there — the benchmark paper says QuranMB is "fully vowelized by design." The field's flagship benchmark is therefore dominated by the one Arabic domain in which this failure mode *cannot occur*, while deployment is MSA, where it always does. That is a natural experiment, and it is a stronger opening paragraph than a robustness motivation.

**Annotation protocol.** Unchanged and still the schedule's main risk. Target 300–500 utterances, correct-the-machine rather than from-scratch: run CATT, present output, correct it. Faster and less error-prone, but it anchors toward the tool under evaluation, so roughly 50 utterances must be double-annotated blind — both annotators, independently, without seeing CATT output — to measure inter-annotator agreement *and* anchoring. Report both. Pre-register the protocol in writing before starting.

Consider the NAACL 2024 audio-informed diacritic restoration method as the starting point instead of CATT: if audio-informed restoration beats text-only CATT on Common Voice, the correct-the-machine baseline is better and the anchoring bias points away from the system under test rather than toward it. Worth an hour in week 4 before annotating anything.

Report agreement on the clinical model rather than bare Cohen's kappa — Harf-Speech's design (multiple independent raters, Pearson plus ICC(2,1)) is the template, and matching it makes the sample size harder to attack.

The annotated slice is a **releasable artifact in its own right**: a gold-diacritized evaluation subset of Common Voice Arabic. Verify the licence position early (Common Voice is CC0, which should be permissive) before promising a release in the paper.

### Conditions

For each system × diacritizer: the 2×2 of §1 for the text-dependent arm, the scoring column for prompt-free arms, and a **synthetic corruption sweep** where diacritics are perturbed at controlled rates and by controlled class. The sweep does three jobs now rather than one — it separates *how much* error from *which kind*, it extrapolates to diacritizers better than any that exist, and it is the estimation set for RQ3's weights.

For constructing plausible perturbations, borrow the Phoneme Confusion Map construction from arXiv 2506.02080 (phonetic proximity, common L2 errors, phonological rules) and the noise-injection protocol from r-G2P (arXiv 2202.11194), which is the closest existing corruption-sweep design. Cite both as technique. Neither is a precedent for the claim — 2506.02080's substitutions synthesise *learner* errors, and r-G2P measures G2P output against itself with no downstream task — and the write-up must not imply otherwise.

---

## 4. The three new instruments

These are what make v2 a methods paper rather than an ablation. All three are definitions over data the v1 design already produces, which is why they are nearly free in schedule terms and expensive only in thinking.

### 4.1 Weighted diacritization error (RQ3)

**The claim.** DER treats every diacritic mismatch as equivalent. The field selects diacritizers on DER. If different diacritic classes cause systematically different downstream harm, then DER is miscalibrated for the purpose the field uses it for, and a weighted variant will predict MDD F1 loss better.

**Classes.** Case ending (word-final, syntactically determined), internal short vowel (fatḥa / ḍamma / kasra), shadda (gemination — phonemically contrastive in Arabic and heavily emphasised in teaching), sukūn, tanwīn, and vowel length. These are not equal in pronunciation-teaching terms and there is no reason to expect them equal in downstream terms.

**Estimation, and the trap to avoid.** Do **not** fit weights on three or four real diacritizers and present the fit as a model; with that many points it is not credible and a reviewer will say so. Instead: derive a priori weights from phonological reasoning, estimate empirical weights from the **synthetic sweep**, which supplies many controlled conditions, and then validate on the real diacritizers as a held-out set. Report rank correlation between predicted and actual F1 loss for weighted versus raw DER, with bootstrap CIs. State the falsifiable prediction in advance: Mishkal should improve in rank under weighting relative to raw DER, because its error mass is disproportionately in case endings.

**Why it travels.** This is the contribution a reviewer who does not read Arabic can use, and the Koshur paper shows the same complaint arising independently for Kashmiri. Frame it as abjad-general with Arabic as the worked instance.

### 4.2 Learner-facing feedback validity

**The claim.** The IQRA 2026 organisers wrote that the phoneme-to-diacritic mapping is unsolved and that "until this mapping problem is solved… the practical utility of these systems for learner-facing applications remains limited." We cannot solve their mapping problem. We can measure a harm nobody has: what fraction of the corrective feedback a deployed system emits is wrong *because the reference was wrong*.

**Mechanism.** A reference error at position *i* makes the system's notion of correct wrong at *i*. A learner who pronounced *i* correctly is flagged — a false rejection — and then told to produce a phoneme that is itself incorrect. That is pedagogically worse than a missed error: a missed error is a lost teaching moment, a corruption-induced false rejection actively teaches the wrong thing.

**Metric.** Corruption-attributable false rejection rate: of all FRs under the deployment condition (bottom-left cell), the fraction traceable to a `C_auto`/`C_gold` mismatch rather than to model error. One number, directly interpretable, computable from results you already have, and it lets the paper be framed as answering the organisers' own call rather than critiquing their benchmark — which §5 explains is the correct posture given who reviews this.

### 4.3 Selective scoring by diacritizer abstention (RQ4)

**The claim.** Most of the loss is recoverable without a better diacritizer, by declining to score positions where the diacritizer is unsure.

**Mechanism.** CATT is a transformer and exposes token-level probabilities. Threshold them; positions below threshold are marked underspecified and excluded from scoring (or scored permissively against a candidate set). Sweep the threshold and plot coverage against F1 — a risk-coverage curve, which is standard selective-prediction methodology and entirely absent from this literature.

**The honest risk, which is also the interesting result.** Diacritizers may be *confidently wrong* precisely where it matters. Case endings are syntactically determined, so a model with a good language prior may assign high confidence to a case ending it gets wrong, in which case confidence-based abstention will fail to catch exactly the error class that dominates. If that happens, RQ4 returns a negative — and it is a publishable negative with a clean mechanistic explanation, plus it strengthens RQ3's argument that case endings should be excluded from scoring *by rule* rather than by confidence. Both outcomes are usable. Decide the framing when the curve exists, not before.

---

## 5. Getting it accepted

Interspeech is roughly four pages plus references — verify against the 2027 call. That is tight, and v2 has more moving parts than v1, so the page budget has to be allocated deliberately rather than discovered in January.

**Page allocation, decided now.** The 2×2 decomposition and the weighted metric are the paper and get the space: one figure for the decomposition, one table or figure for the metric validation. Feedback validity is one number in the results and one sentence in the discussion. Abstention gets one small figure. RQ1 is a compact table. Partial diacritization is a future-work clause.

**Cut order, also decided now**, so that page-limit surgery in week 19 is mechanical rather than an argument: first RQ5 entirely; then the abstention *figure*, with RQ4's result surviving as a sentence and a number in the text; then RQ1's table down to two sentences. RQ2 and RQ3 are not cuttable — if they do not fit, the paper is over-scoped and something has gone wrong much earlier.

### Contributions as a reviewer will read them

1. **A decoupling protocol** that separates label-path from input-path reference corruption in MDD — possible because the reference is a text object, and therefore not available to prior upstream-error work.
2. **Evidence that DER is miscalibrated** as a diacritizer-selection criterion for speech pipelines, plus a weighted alternative that predicts downstream harm.
3. **The direction and magnitude of bias** in naive automatic-reference MDD evaluation — a correction factor every group moving to MSA evaluation needs.
4. **A mitigation** (selective scoring) and a deployment-facing harm metric (corruption-attributable false rejections) that answers the challenge organisers' stated open problem.
5. **A gold-diacritized Common Voice Arabic evaluation subset**, released.

Note what is no longer contribution 1: "first measurement of diacritization error propagation." It is still in the paper, but as evidence rather than as the headline, and that is the whole point of the rewrite.

### The Kadambi problem, and how the paper handles it

This is the main reviewer risk and it deserves explicit management rather than a defensive paragraph.

**Cite it early and generously.** Mathad 2021 and Kadambi 2024 belong in the first paragraph of related work, described accurately. A reviewer who finds them in a paper that does not cite them concludes the authors did not read the field. A reviewer who sees them cited prominently concludes the question type is established at this venue — which helps.

**Borrow the method openly.** Kadambi's ΔPLLR is the shape of our primary metric with one component substituted, and their mixed-effects specification with phoneme position and phoneme type as predictors is a better statistical frame for RQ3's decomposition than a bare per-category F1 breakdown, because it handles the speaker- and utterance-level clustering our data will have. Adopting a published Interspeech methodology makes the design much harder to call ad hoc. Read Kadambi before finalising the week-6 pilot.

**State the difference as a design property, not a rebuttal.** The distinguishing sentence is in §1 and it should appear in the introduction: alignment error moves the measurement window, a corrupted diacritic moves the target label, and only the latter is decouplable into two separately measurable paths. Write that text in week 1 while the reading is fresh.

**Do not inherit their prior uncritically, and do not fight it either.** They find upstream error's downstream impact minor to moderate. Quote that honestly. Then note that the finding is about window-shifting error, and that the case-ending analysis is the empirical test of whether it transfers: case endings are the class where the label moves without the pedagogical target moving at all.

### Pre-committed position on effect size

**Decided in advance, in writing, week 0.** If the measured degradation is small, the paper reports a small degradation and the contribution is the protocol, the metric and the bias correction. It does not become a different paper in December. This is the single most important structural difference from v1, whose fallback was a framing pivot under deadline pressure.

Be honest that a modest-effect paper is a harder sell at Interspeech than a dramatic one — I do not want to pretend otherwise. But a modest effect plus a protocol plus a miscalibrated-metric result is a real paper, whereas a modest effect alone is not, and v1 had no insurance.

### Anticipated objections, and the pre-emption for each

**"Obvious — of course a noisy reference hurts."** Still the most likely objection, but it now misses the target: the paper's claim is not that noise hurts, it is that the noise enters through two paths that the field's protocol conflates, and that the metric used to choose the upstream tool does not predict the harm. Neither is obvious. Keep the word "ablation" out of the abstract.

**"This is Kadambi with diacritics."** Handled above. The answer is the decouplability argument plus the metric contribution, and it goes in the introduction.

**"Qur'anic text is always diacritized, so this is artificial."** Exactly why the MSA arm exists, and v2 now makes this the opening frame rather than a defence: the benchmark is clean *because* it is Qur'anic, and that cleanliness does not transfer to deployment.

**"You didn't evaluate the winning system."** Show the effect across three-plus architectures and argue architecture-independence. Range beats peak. Cite Fusion-Aware's F1 0.7201 as the current ceiling and be explicit that the degradation is measured relative to systems you control.

**"Your annotated sample is small."** Inter-annotator agreement, anchoring measurement, bootstrap CIs, and the ICC(2,1) reporting model borrowed from Harf-Speech. 300–500 utterances with stated CIs is defensible; a bare point estimate is not.

**"Isn't this just diacritization evaluation?"** No — RQ1 is deliberately compact and the centre of gravity is RQ2–RQ4. If a draft reads like a diacritization paper, cut RQ1 further.

**"Your weighted metric is fitted on too few systems."** Pre-empted by design: weights estimated on the synthetic sweep, validated on held-out real diacritizers, with the Mishkal rank prediction stated in advance. Say this in the methods, not in the rebuttal.

### Reviewer-pool reality

Unchanged from v1 and still the most important practical constraint. El Kheir, Chowdhury, Ali, Meghanani and Shahin recur across both challenge editions, the benchmark paper and the review article. They will review this.

Two implications. Frame the work as *strengthening* their benchmark — supplying the deployment-realism analysis their own future-work calls for, and answering the phoneme-to-diacritic feedback problem they named as open. And cite them generously: the benchmark paper, both challenge overviews, the review, the shared-task metric paper.

This is also why the benchmark-validity critique stays out of the headline even though §3 now makes it available. "The benchmark is clean because it is Qur'anic" is a framing observation stated neutrally and used to motivate the MSA arm. It is not "the benchmark is invalid." That fight is winnable later from a stronger position; it is not winnable in a first paper reviewed by the people who built it.

---

## 6. Timeline to 9 February

Twenty-one weeks with deliberate slack. Weeks run Monday to Sunday. One structural change from v1: the text-dependent arm moves earlier, because in v2 it is required for the primary contribution rather than for an optional RQ.

| Weeks | Dates | Work | Gate |
|---|---|---|---|
| 0 | 15 – 20 Sep | **Complete.** Novelty check across all four tiers; verdict not scooped. Remaining: Arabic-language queries, manual citation traversal. Write the pre-committed effect-size position | **Passed** |
| 1–3 | 21 Sep – 11 Oct | Cluster setup; pull Iqra_train, Iqra_TTS, QuranMB.v2; implement the official hierarchical metric; reproduce the mHuBERT baseline. Read Kadambi and Mathad; draft the differentiation text | **F1 ≈ 0.4414 ± 0.02 or stop and debug** |
| 4–6 | 12 Oct – 1 Nov | Diacritizer pipeline, all tools; RQ1 measurements on both domains; confirm CATT exposes token probabilities; begin annotation; **run the full label-path column on the Qur'anic arm** | **Label-path effect measurable, and interaction power estimated** |
| 7–9 | 2 – 22 Nov | Finish annotation including the double-annotated subset; train XLS-R prompt-free; **build the text-dependent arm**; MSA-arm label path | Both arms trained, annotation closed |
| 10–12 | 23 Nov – 13 Dec | Full 2×2 on the text-dependent arm; RQ3 weight estimation on the sweep and validation on real diacritizers; RQ4 abstention curves; feedback-validity number | **Drop RQ5 if null — keep RQ2–RQ4** |
| 13–15 | 14 Dec – 3 Jan | First complete draft. Figures finalised. **Approach the professor with results in hand** | Full draft exists |
| 16–18 | 4 – 24 Jan | Supervisor feedback; the ablations a reviewer will demand; internal review | Revised draft |
| 19–21 | 25 Jan – 9 Feb | Polish, page-limit surgery, code and data release prep, submit | **Submit 9 Feb (Tue)** |

Three weeks of genuine slack sit across the second half. Protect it; something will break.

### The gates, and one that got considerably better

The **week-3 gate** is the real go/no-go and is unchanged. If a fully specified published baseline cannot be reproduced, nothing downstream is trustworthy, and October is a much better time to learn that than January.

The **week-6 gate is stronger in v2 than in v1**, and this is worth noticing because it removes the schedule's worst dependency. Qur'anic text is diacritized by convention, so the Qur'anic arm needs **no annotation at all** — strip the diacritics, re-diacritize, and the gold reference is already in hand. That means the entire label-path column, which is the spine of the paper, can be run by week 6 on the reproduced baseline with zero dependence on the annotation effort. v1's week-6 pilot was 100 utterances of hand-annotated MSA and a judgement call about whether to continue. v2's is a real measurement of the paper's central quantity, available early and cheaply. If the decomposition does not work there, it will not work anywhere, and you find out in November with fifteen weeks left.

Add one thing to this gate: a **power estimate for the RQ5 interaction** specifically, not just for the headline. CROTTC-IF has already answered the main-effect question, so what remains is an interaction, and interactions need more power than main effects. If the pilot says it is undetectable at 300–500 utterances, cut RQ5 in November rather than discovering it in December.

The **week-12 gate** protects the core. RQ5 is the cuttable one; a null there must not be allowed to eat the paper.

What the gates no longer do is decide the paper's framing. That is pre-committed in §5 and should not be revisited under deadline pressure.

---

## 7. Week 1 concretely

1. **Read Kadambi 2024, then Mathad 2021, and write the differentiation text immediately.** Two or three paragraphs distinguishing window-shifting from label-shifting corruption. This text goes in the introduction, in the related-work section and eventually in the rebuttal, and it is much easier to write while the reading is fresh than in January. §1 and §5 of this plan are a draft to work from.
2. **Write down the pre-committed effect-size position** and both students sign off on it. One paragraph. It exists to be pointed at in December.
3. Create the HF account, request access to the IqraEval organisation datasets, confirm licence terms on Iqra_Extra_IS26 and QuranMB.v2, and establish whether the v2 test labels are fully public or leaderboard-held. This blocks everything and has latency you do not control.
4. **Confirm CATT exposes token-level probabilities** through its public interface. RQ4 depends on it entirely. If it does not, either patch the inference code or substitute a diacritizer that does, and find out now rather than in December.
5. Confirm cluster allocation and queue limits, and establish what a single fine-tuning run of XLS-R-300m actually costs in wall-clock, because the timeline assumes several are affordable.
6. Resolve the QuranMB.v1/v2 overlap oddity from the deep dive — the leaderboard row matching a 2025 baseline digit-for-digit. If v2 is substantially v1, the experimental design is unaffected but the framing needs care.
7. Split the work. One owns the model and training pipeline, the other owns the diacritization pipeline and annotation. Both converge at week 7, which is earlier than v1 because the text-dependent arm moved forward.
8. Finish the two residual novelty items: the Arabic-language queries (تشكيل + تقييم النطق, كشف أخطاء النطق + التشكيل الآلي, plus Arabic university thesis repositories) and the manual Google Scholar "cited by" traversal on CATT, Halabi & Wald, **and Kadambi 2024** — whose citers are the likeliest place to find someone who has already ported that method to a text front-end. Roughly two hours total. Record results in [novelty-check.md](novelty-check.md).

---

## 8. Open items

- Interspeech 2027 page limit and formatting rules, once the call is published.
- Whether CATT exposes usable token-level confidence. Blocks RQ4. Week 1.
- Whether IQRA 2027 runs at Interspeech 2027. No third edition is announced as of 16 September 2026; re-check when the call appears. If it runs, entering it with this work is complementary, and a shared-task entry plus a paper is a strong package for supervisor recruitment.
- Licence terms for redistributing a diacritized derivative of Common Voice Arabic. CC0 should be permissive; verify before promising a release in the paper.
- Whether any IqraEval team released usable checkpoints.
- Whether a newer diacritizer has superseded CATT since 2024. One candidate surfaced in the novelty check — "Evaluating Arabic Diacritization Models: Self-hosted to Commercial" (2026) — but only as an OpenAlex record with no venue or text retrieved. Chase it before finalising the diacritizer set.
- Whether the NAACL 2024 audio-informed restoration method beats text-only CATT on Common Voice. Decides the annotation starting point. Week 4.
- The Phonikud propagation quote, currently unverified and unusable. Low priority, ten minutes, PDF only.

---

## 9. What remains of the novelty check

The full search plan that was §8 of v1 has been executed; the record, including the search log, prior-art ledger and quote bank, is in [novelty-check.md](novelty-check.md). Do not re-run it. Two items remain open and are item 8 of week 1: the Arabic-language queries, and the manual citation traversal that the APIs could not deliver.

One habit from that exercise is worth carrying forward. For every paper you read closely, read the limitations and future-work sections **before** the methods. That is where the quote bank came from, and every near-miss sentence you find — a paper that notices the gold-diacritics assumption and walks past it — is a free sentence in the introduction. There are almost certainly more of them in the full texts than targeted fetches surfaced.

---

## Sources

Every figure used above is verified in [novelty-check.md](novelty-check.md), which is now the provenance record for this project.

**Note on a broken chain.** v1 cited `12-arabic-mdd-research-deep-dive.md` as the source catalogue. That file is not in this project and the only copy on this machine is in the Trash — it appears to have been deleted when the work moved out of the language-learning-brainstorming project. Nothing in v2 depends on it, because the novelty check re-derived and independently verified every number, but if the deep dive contained reasoning that is not reproduced here it should be recovered from the Trash before it is emptied.

The specific figures used above:

- Baseline F1 = 0.4414 on QuranMB.v2, and the 68-phoneme Halabi inventory — [IQRA 2026](https://arxiv.org/abs/2603.29087), El Kheir et al. Also the source of the phoneme-to-diacritic mapping quote in §4.2.
- Hierarchical TA/TR/FA/FR and CD/ED metric definitions — [Iqra'Eval 2025](https://aclanthology.org/2025.arabicnlp-sharedtasks.61/), El Kheir et al.
- In-house vowelizer applied to Common Voice transcripts; QuranMB "fully vowelized by design" — [Towards a Unified Benchmark](https://arxiv.org/abs/2506.07722), El Kheir et al.
- CATT DER/WER with and without case endings (**EO variant**), the full 11-system Table 5 spread, and the grammatical-rules quote — [CATT](https://arxiv.org/abs/2407.03236), Alasmary et al.
- Canonical conditioning harmful with a correct reference; F1 drop to 40.52%; "canonical information can easily override the subtle acoustic information" — [CROTTC-IF](https://arxiv.org/abs/2604.22133), Geng et al.
- Current top system F1 = 0.7201 — [Fusion-Aware Two-Stage Framework](https://arxiv.org/abs/2606.24086).
- Clinical multi-rater validation design, Pearson plus ICC(2,1) — [Harf-Speech](https://arxiv.org/abs/2604.06191).
- Audio-informed diacritic restoration for speech corpora — [NAACL 2024](https://aclanthology.org/2024.naacl-long.233/).
- Speech-domain diacritization DER 10.56% / WER 34.47% — [Fine-Tashkeel at KSAA-2026](https://aclanthology.org/2026.osact-1.31/).
- Real Arabic text is partially, not fully or zero, diacritized — [Arabic Diacritics in the Wild](https://aclanthology.org/2024.acl-long.792/), Elgamal et al.

New in v2, from the Tier 3–4 pass:

- Forced-alignment error propagating into pronunciation scores; the two-stage contamination logic; the minor-role finding — [Mathad et al., Interspeech 2021](https://www.isca-archive.org/interspeech_2021/mathad21_interspeech.html).
- ΔPLLR; mixed-effects modelling with phoneme position and type; "care must be taken to distinguish between the impact of alignment error (a spurious signal) and true acoustic deviation"; moderate effect — [Kadambi et al., Interspeech 2024](https://www.isca-archive.org/interspeech_2024/kadambi24_interspeech.html).
- DER penalising every mark mismatch equally; DERm 0.2012 against 77.5% expert rating — [Koshur Diacritizer](https://arxiv.org/abs/2606.15883), Malik et al. **Quote is in §VII-B of the full text, not the abstract.**
- Nikud reflecting formal grammatical rules rather than spoken pronunciation; the diacritizer-bypass alternative — [ReNikud](https://arxiv.org/abs/2606.20179).
- Controlled noise injection for G2P robustness — [r-G2P](https://arxiv.org/abs/2202.11194).
- Phoneme Confusion Map construction for plausible substitutions — [Enhancing GOP with Phonological Knowledge](https://arxiv.org/abs/2506.02080). Synthesises *learner* errors, not reference errors.
- Upstream error weighting and downstream task impact — [Not All Errors Are Equal](https://arxiv.org/abs/2412.06332), [Deletions Are More Equal](https://arxiv.org/abs/1904.01684). Note both are Alzheimer's-detection papers; borrow the principle, not the mechanism.
- Interspeech 2027 deadline, 9 Feb 2027 — [mldeadlines](https://mldeadlines.com/conference/interspeech-2027/). Verify against the official call.
