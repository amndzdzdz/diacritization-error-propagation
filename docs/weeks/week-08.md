# Week 8 — take the three trained arms to QuranMB.v2

Plan reference: [docs/mdd-paper-project-plan-v2.md](../mdd-paper-project-plan-v2.md)
§1 (the 2×2, D10, D11), §2 RQ2, §3 Conditions, §6 weeks 7–9.
Previous week: [week-07.md](week-07.md). Insights: [insights/week-08.md](../../insights/week-08.md).

Week 7 ended with a validated instrument and no readings from it. Week 8 is the
week that takes the readings. Everything below exists to produce **one table**:
plan §1's 2×2, on the text-dependent arm, on QuranMB.v2.

---

## The structural decision this week takes first: **D12 — what the collapse number is allowed to do**

D11 pre-registered that the canonical-copy shortcut would be *measured, not
mitigated*, and set a falsifier at `canonical_preference > ~0.5` on QuranMB.v2.
It did not say what happens when the number arrives. That has to be settled
before the number exists, for the same reason D10 and D11 did: the input-path
|ΔF1| is the paper's headline, and the collapse number is the one quantity that
can be used to argue the headline away. A rule chosen after seeing it is not a
rule.

**And week 7 falsified the threshold's form, which is why this cannot just be
carried over.** On the dev split the conditioned arm reported
`canonical_preference = 1.0000` — and so did the **ablated** arm, which never
received a canonical and cannot causally prefer one
([insights/week-07.md](../../insights/week-07.md), third entry; n = 3
informative utterances). An absolute threshold of 0.5 would have "fired" on a
model that is provably not copying. The statistic has no absolute scale; it has
a scale only relative to a model that cannot copy.

**Decided: the decision rule is a difference from the ablated arm, on a minimum
sample, and it changes how the cell is *reported* rather than whether it is
reported.**

1. **The null is job 224627**, the `use_canonical=False` arm — same data, same
   steps, same initialization family, structurally unable to consume the
   canonical. Its `canonical_preference` on QuranMB.v2 *is* the chance level for
   this metric on this corpus. Report both arms' numbers side by side, always,
   and never the conditioned arm's alone.
2. **The falsifier is `canonical_preference(conditioned) −
   canonical_preference(ablated) > 0.15`**, evaluated only if
   `informative_utterances ≥ 200`. Below that count the comparison is declared
   underpowered and the cell is reported with the count stated, not with a
   verdict. 200 is chosen as the smallest number at which a 0.15 difference is
   not plausibly three utterances of noise; QuranMB.v2's ~12.4% phoneme-level
   canonical/annotation disagreement should put this in the high hundreds, and
   if it does not, that is itself the finding.
3. **If it fires, the bottom-left cell becomes an upper bound, not an
   estimate.** The wording is pre-committed: *"for the text-dependent arm the
   input-path effect is bounded above by X; a copy-rate diagnostic indicates the
   arm partially restates reference corruption, so X is not attributable to
   degraded conditioning alone."* The cell is still reported. What it is not
   allowed to become is the headline number.
4. **If it fires, the prompt-free scoring column carries RQ2 instead.** That
   column cannot collapse — prompt-free systems never consume `C` — and week 6
   already measured it (+0.0265 to +0.2818, every CI excluding zero). RQ2
   survives a collapsed conditioned arm; it is RQ5 and the *decomposition into
   two paths* that degrade. This is the one place the paper's structure has real
   redundancy and it should be said out loud before it is needed.
5. **Retraining with a mitigation stays refused**, per D11. Canonical dropout,
   reweighting the 52 h, and training on TTS alone all remain rejected: each is
   a knob that moves the headline, and a number produced by turning it cannot be
   defended as not having been tuned. If the arm has collapsed, the paper says
   so.

**What D12 is not.** It is not a prediction. The expectation is that the
conditioned arm does *not* collapse — the 47,439 Iqra_TTS rows where
`sentence = phoneme_mis ≠ canonical` are 40% of training and they punish copying
directly. D12 exists so that the 60% chance of a clean result and the 40% chance
of a messy one are reported by the same rule.

**[DECIDED on design grounds, pre-registered before any arm is evaluated on
QuranMB.v2.]** Mirrored into plan §1 as D12, directly under D11, where it also
marks D11's absolute `> ~0.5` falsifier superseded.

---

## Where week 7 left things

Three models, all 200k steps, all on holdout-filtered manifests:

| job | expdir | arm | best dev PER |
|---|---|---|---|
| 224615 | `xlsr300m_promptfree_heldout` | organizers' `ctc`, verbatim | 0.07918 |
| 224627 | `xlsr300m_mdd_promptfree` | `ctc_cond`, `use_canonical=False` | 0.07972 |
| 224626 | `xlsr300m_mdd_conditioned` | `ctc_cond`, `use_canonical=True` | 0.02703 |

Three facts from that week are load-bearing here:

- **The port is validated.** 0.07972 vs 0.07918 (+0.68% relative) means the
  `ctc_cond` fork reproduces the organizers' pipeline with conditioning off, so
  the only difference between 224626 and 224627 is the conditioning. Week 8
  inherits a trustworthy instrument and does not need to re-establish it.
- **The conditioned arm's 0.02703 is a leak.** Dev is held-out `Iqra_train`
  (`2069 canonical from sentence, 0 from file`), so the prompt contained the
  target. No conclusion about conditioning exists yet. This is the entire reason
  week 8 is necessary rather than a formality.
- **Both arms were trained on held-out data**, so *our* arms never saw the 16
  contaminated QuranMB prompts. The organizers' checkpoint — which produced
  week 6's `run/quranmb_predictions_official_ckpt.json` — did. That asymmetry is
  task 7 and it decides which denominator each comparison gets.

**The calendar.** Plan §6 puts weeks 7–9 at 2–22 Nov; week 7 closed 20 Oct. The
code side is running roughly two weeks ahead, and that lead should be spent
here, on the Qur'anic arm and the MSA label path. It buys nothing on the weeks
7–9 gate's other clause — **annotation closed** — because annotation is bounded
by a human annotator's hours and not by this repository. Treat the lead as
banked slack for weeks 10–12, which plan §6 already identifies as the crowded
stretch.

---

## Scope

Week 8 is a **measurement week with almost no new machinery**, and that is a
finding rather than an assumption: the week-7 harness already writes, at
evaluation time, the exact triple both of this week's instruments consume.

`scripts/baseline_reproduction/ctc_cond/expert.py` writes three files per
evaluation:

```
{split}-hyp.ark      the prediction            P
{split}-ref.ark      the CTC target            A  (what the speaker said)
{split}-canon.ark    the canonical             C  (what they should say)
```

`(C, A, P)` is precisely the signature of
`arabic_mdd.metrics.hierarchical.evaluate_utterance(canonical,
human_annotation, prediction)` **and** of
`arabic_mdd.models.collapse.collapse_report`. So the whole week reduces to:
put QuranMB.v2 behind that manifest interface, vary column three, and read the
`.ark` files.

**The one thing that makes this cheap is worth stating plainly: the 2×2's rows
cost GPU passes and its columns are free.** A row is *which canonical the model
received*, which is a tensor fed into `MddModel.forward` and therefore an
inference pass. A column is *which reference the prediction is scored against*,
which happens offline in Python against a `.ark` file already on disk. So:

- Conditioned arm: **5 passes** (C_gold + 4 diacritizers), per D10's
  inference-time swap.
- Each prompt-free arm: **1 pass** — they ignore column three, so all five
  variants must produce byte-identical hypotheses. Assert it on two variants
  rather than assuming it; it is a free end-to-end check that the canonical is
  actually routed where we think it is.
- Total **7 passes over 1,642 clips**, batched. Trivial next to a 200k-step run.

No new metric, no new model code, no retraining.

---

## Week 8 exit criterion

> **Plan §1's 2×2 is measured on the text-dependent arm on QuranMB.v2, with
> D11's collapse question answered under D12's rule, and every cell reported
> against a stated denominator.**

Four clauses, each independently checkable:

1. **All four cells exist**, from 5 conditioned passes and the offline scoring
   column, with cluster-bootstrap CIs resampling the 98 prompts (not the 1,642
   clips) — the same estimator week 6 used, reused rather than reimplemented.
2. **D11 is answered**: `canonical_preference` and `copy_rate` on QuranMB.v2 for
   both the conditioned and the ablated arm, with `informative_utterances`
   reported, and D12's rule applied to whatever comes back.
3. **The input-path row is interpretable**, which requires the conditioned arm's
   top-left (gold-`C`) F1 to be in the same league as the prompt-free arms'. See
   the failure branch.
4. **Denominators stated**: every number labelled either all-98-prompts or
   clean-82-prompts, with the cross-checkpoint comparisons on the clean subset.

Deliberately *not* in the criterion: RQ3's decomposition, the synthetic sweep,
RQ4's abstention curves. Plan §6 puts those at weeks 10–12 and the lead banked
above is not an argument for pulling them forward.

### The honest failure branch

**If the conditioned arm's gold-`C` F1 on QuranMB.v2 is far below the
prompt-free arms'**, the 2×2 is being computed on a weak model and its |ΔF1| is
not comparable to week 6's. This is a live risk and not a remote one: both
`ctc_cond` arms trained on Iqra_train + Iqra_TTS and QuranMB.v2 is a different
corpus with different speakers and recording conditions, so this is a
cross-corpus evaluation by construction.

Pre-registered threshold and response, before the number exists: if the
conditioned arm's gold-`C` F1 is **below 0.40** — i.e. more than ~0.08 under the
0.4824 week 6 measured with the organizers' checkpoint — then:

- (a) Report it, and report the ablated arm's F1 beside it, because the gap
  between 224626 and 224627 at gold `C` is the *conditioning benefit* and is a
  result in its own right whichever sign it has. CROTTC-IF found conditioning
  *hurts*; a negative gap here would corroborate a published finding rather than
  embarrass us.
- (b) Diagnose before reacting: compare 224627's QuranMB F1 against 224615's. If
  both `ctc_cond` arms are low and the verbatim arm is fine, suspect the port on
  this corpus despite the dev-PER agreement. If all three are low, it is the
  corpus shift and it applies to every arm equally, which leaves |ΔF1| readable
  as a within-arm contrast.
- (c) Do **not** fine-tune on QuranMB.v2 to fix it. The 98 prompts are the
  evaluation set; the holdout in week 7 exists to keep them unseen and spending
  that on a nicer absolute number would trade the paper's one clean measurement
  for cosmetics.

**If instead every arm looks healthy and D12 does not fire**, week 8 ends with
the paper's spine measured and weeks 10–12 become decomposition work on numbers
that already exist.

---

## Tasks

1. **Dump QuranMB.v2 behind the manifest interface.** One script,
   `scripts/build_quranmb_manifests.py`, run once: decode
   `safikhan/quran_mbv2_formatted` to 1,642 wavs under
   `$DATA_DIR/CV-Ar/quranmb/wav/` (inside the existing `corpus.path` so that
   setting does not have to change) and emit **five** three-column manifests
   sharing that directory.

   Column contract, which is where the correctness lives:

   - `path` — relative to `corpus.path`, as the other manifests are.
   - `sentence` — `annotation_phoneme_string`, the human annotation `A`. It is
     the CTC target, so it becomes `{split}-ref.ark`, which is what we want to
     score against. It is **not** shown to the model: the conditioned arm reads
     column three and nothing else. No leak, but say so in the docstring because
     "the target is the annotation" reads alarming at a glance.
   - `canonical` — the variant. `C_gold` for one manifest, `C_auto` for each of
     `catt-eo`, `catt-ed`, `shakkala`, `mishkal`.

   **`C_gold` must be leg A's, not leg B's, and this is the easiest thing in the
   week to get silently wrong.** Week 6 carried two golds:
   `reference_phoneme_string` (leg B, the published-F1 anchor) and
   `phonetize(normalize(recovered_text))` (leg A). `C_auto` goes through the
   phonetizer, so only leg A is a like-for-like comparison, and the string fed
   to the model must be the string it is scored against. Import the derivation
   from `scripts/measure_label_path.py` rather than re-deriving it — if those
   two ever disagree the 2×2 is measuring the phonetizer.

   Reuse `arabic_mdd.diacritizers.chunking.diacritize_long` for the four tools
   and record `.seams`; reuse `provenance()` into the output JSON.

2. **Check every canonical variant against the 68-token vocabulary, before any
   GPU time.** `arabic_mdd.data.canonical.count_unknown_tokens` already does
   this and `build_canonical_manifest.py` already has the right failure shape:
   fatal, with `--allow-unknown` as a deliberate override. Wire it into task 1.

   This is not ceremony. The tokenizer maps an out-of-vocabulary token to
   `<unk>` **silently**, and here it would corrupt the model's conditioning
   input *and* the reference in the same pass — a degraded 2×2 whose only
   symptom is a disappointing |ΔF1|. `run/phoneme_inventory_diff.json` cleared
   the four tools on Iqra_train dev (`out_of_inventory: {}`, 66–67 of 68 types
   used), which is evidence about Iqra and not about Qur'anic text; QuranMB's
   canonicals and annotations have never been checked against `sws_arabic.txt`.

3. **Generate a QuranMB evaluation config.** `prepare_baseline.sh` currently
   substitutes `path_to_tsv_file/dev/dev.tsv` into both the `dev:` and `test:`
   lines of `sws_cond.yaml`, which is why week 7's `-m evaluate` scored 2,069
   dev clips under `evaluate_split: test`.

   Cheapest correct change: a second, line-anchored `sed` that rewrites only the
   `test:` line, emitting `sws_cond_quranmb.yaml` alongside the training config.
   Do **not** edit the vendored `config/sws_cond.yaml` — it is kept as a
   verbatim reference and week 7's provenance story depends on that. Do not
   reach for `-o` overrides either; the value is a YAML list and a generated
   config is both clearer in the log and greppable afterwards.

4. **Evaluate the three arms, 7 passes.** `run/evaluate_quranmb.slurm`, taking
   `EXP_DIR` and `CANONICAL_VARIANT`, running
   `run_downstream.py -m evaluate -e ${EXP_DIR}/dev-best.ckpt` against the task-3
   config, and copying the three `.ark` files out to a per-variant directory
   under `run/quranmb_eval/` so that one pass cannot overwrite another's.

   Week 7's lesson applies literally: **run the manifest build and the vocab
   check on a login node before any `sbatch`.** Three queue round-trips last
   week were guards that were right about what to check and wrong about where.

   The prompt-free assertion from Scope belongs here: run 224627 and 224615 on
   two different canonical variants and assert the hypotheses are byte-identical.
   If they are not, the canonical is reaching the model in an arm that is
   supposed to ignore it, and every number this week is void.

5. **Convert `.ark` to the predictions JSON week 6 already consumes.** Tiny
   converter: `{split}-hyp.ark` → `{id: predicted_string}`, the shape of
   `run/quranmb_predictions_official_ckpt.json`. `measure_label_path.py` already
   takes `--predictions` (line 473), so the week-6 estimator — counts pooled per
   prompt, `compute_metrics` applied once, 10,000-replicate bootstrap resampling
   the 98 prompts — is reused by pointing a flag at a different file.

   **One change it does need: `--write` targets a hardcoded `OUT`**
   (`run/quranic_label_path.json`, line 134). Week 8 produces one of these per
   arm per denominator, so add `--output` and leave the default where it is;
   overwriting week 6's committed artifact would destroy the comparison this
   week exists to make.

6. **Close D11.** `scripts/report_collapse.py` already reads exactly the three
   `.ark` files task 4 produces, so this is a CLI invocation per arm on the
   gold-`C` pass, not a new script. Report `copy_rate`,
   `informative_utterances` and `canonical_preference` for the conditioned
   **and** ablated arms, and apply D12.

   Expect `informative_utterances` to be large here and interrogate it if it is
   not: it counts utterances where `C ≠ A`, and QuranMB.v2's human annotation
   disagrees with the canonical on ~12.4% of phonemes. Week 7's third insights
   entry flagged a tokenizer round-trip artifact (canonical decoded from the
   unpadded row list, reference from the padded batch tensor,
   `ctc_cond/expert.py:196`) that cost 3 spurious informative rows on dev. If
   QuranMB's count comes back implausibly *low*, that artifact is the first
   suspect and this is the measurement that makes it worth fixing.

7. **Derive the clean-prompt denominator and report both.**
   `run/training_holdout.json` records `contaminated_prompts: 16`,
   `clean_prompts: 82`, `clean_clips: 1388` — but it stores
   `contaminated_prompt_skeletons` as **texts**, not prompt indices, so the
   subset is derivable and not yet derived. Join them against
   `run/quranmb_verse_map.json`'s 98 indexed prompts, commit the index list, and
   report every cell on both denominators.

   The reason this is a task and not a footnote: *our* arms are clean on all
   1,642 clips because week 7's holdout removed those prompts from training,
   while the organizers' checkpoint behind week 6's numbers and the published
   0.4414 was trained on them. So **our own arms' numbers are honest on all 98
   prompts, and any comparison against week 6 or the published baseline belongs
   on the clean 82.** Getting this backwards would either throw away 15% of our
   evaluation set for no reason or compare a clean model against a contaminated
   one without saying so.

8. **Start the MSA-arm label path** (plan §6, weeks 7–9). Scoped deliberately as
   *start*: the MSA arm needs annotation to close, which is task 9's half of the
   week, and the bottom-right "naive deployment evaluation" cell of §1's 2×2 is
   the one cell that cannot be computed from Qur'anic data at all. What is
   available now without annotation is the pipeline: `run/msa_eval_pool.json`
   and `run/msa_der.json` exist, and the week-6 label-path machinery is
   corpus-agnostic given a `(C, A, P)` triple. Build to the point where closing
   annotation is the only remaining input, then stop.

   **Two corrections found while building it, both load-bearing.**

   *This clause's premise was wrong about `(C, A, P)`.* The MSA arm has no `A`:
   D1 (`docs/msa-arm.md:640`) decided the arm is label-path only and does not
   annotate it, and D2 (`:946`) carries `A := C_gold` as a *bounded* assumption
   instead, with the bound taken on the Qur'anic arm in weeks 10–11. So the
   week-6 driver is not reusable here, and `scripts/measure_label_path_msa.py`
   is a separate driver rather than a flag on the old one — the two arms differ
   in `C_gold`'s source, in `A`'s existence, in the phonetization convention
   (**pausal** here per the protocol's §3 R1–R4, prescriptive there), and in the
   cluster unit (one utterance here, 98 prompts there).

   *And `ΔF1` is not this arm's quantity.* Under `A := C_gold` the gold cell has
   `C == A` at every position, so `TR` and `FA` are **zero by construction** and
   its F1 is `_safe_div`'s 0, not a measurement. `ΔF1` against that measures the
   substitution, not the bias in a naive evaluation. The MSA arm reports the gold
   cell's **FRR** (pure model error on real MSA speech, against week 4's CV-Ar
   dev 0.0708), the auto cell's **manufactured TR** (apparent detections created
   by reference corruption — contribution 4, the one D1 explicitly preserved),
   and **ΔFRR** with a paired utterance bootstrap. F1 is still written to the
   artifact with a `f1_note` saying why it is not the number.

   This bears on **D2's own construction** and should be read before weeks 10–11
   spend time on it: D2 obtains the bound by "re-scoring the four label-path
   conditions a second time with `A := C_gold`" and differencing against the
   measured ΔF1. But under that substitution the Qur'anic gold cell also loses
   all its positives, so the "residual term" would be dominated by the
   structural absence of real positives rather than by the deviation base rate it
   is meant to price. Cheap to check — both `A` and `C_gold` exist on the
   Qur'anic arm, so it is one extra scoring condition — and **deliberately not
   checked here**, because D2 schedules it for weeks 10–11 and running it now
   would be pre-building. Named so the branch is decided on a number.

   **What landed.** `scripts/measure_label_path_msa.py`, two phases:
   phase 1 builds `C_auto` for five conditions over the committed 500-row frame
   and caches it to `run/msa_c_auto.json`; phase 2 reads
   `run/msa_c_gold_{a1,a2}.json` and scores. The five conditions are the four
   evaluated tools plus the organizers' `tashkeel_sentence` — which is not a
   fifth tool but **the naive-deployment condition itself**, the column that
   makes §1's bottom-right cell computable at all. Phase 1 owns the whole
   diacritizer runtime and every refusal path, and it has been **run to
   completion**: 499 rows × 5 conditions, no blank cell, 211 s of diacritizer
   time in total (catt-ed 0.161 s/row and shakkala 0.235 s/row dominate; the
   other three are ~0.014 s/row). That is cheap enough that `--refresh-cache`
   is not a decision anyone has to think about. `P` joins 1:1 from
   `run/cv_dev_predictions.json`; row `64376` is excluded as
   `normalize`-unfair (Latin script) rather than cleaned.
   Phase 2 is implemented and refuses with the count of rows still missing a
   `final`, and separately refuses to let file order pick between the 50
   crossover rows' two annotations. Closing annotation is the only remaining
   input, which is where the clause said to stop.

9. **Annotation, including the double-annotated subset** (plan §6 weeks 7–9
   gate, §7.7's owner split). **Owned jointly, outside this plan**, same as
   week-07 task 9. Noted because it is the clause of the weeks 7–9 gate that the
   two-week code lead does not advance, and because
   `annotate_msa.py --status` is where its burn-down lives.

10. **Insights.** One `insights/week-08.md` entry per experiment, written when
    the run finishes rather than at the end of the week (CLAUDE.md). The entries
    this week should land in this order, because each is a precondition for
    reading the next: the manifest build and vocabulary check; the prompt-free
    byte-identity assertion; the collapse measurement under D12; the 2×2 itself.

---

## Carried forward

Unchanged from week 7 unless noted:

- ~~**The byte-identical regeneration guard** names `data/normalize.py` while
  the U+0670 deletion lives at `phonetizer.py:179`.~~ **Done.**
  [week-06.md](week-06.md) now names `phonetizer.py:179` as the rule's home and
  scopes the guard to `data/normalize.py` and `msa_pool.py`'s `skeleton` — the
  two functions whose output a committed MSA artifact actually contains.
- **The `.ark` canonical/reference decode asymmetry** (`expert.py:196`) — 3
  spurious informative utterances on dev. Promoted from "logged, not chased" to
  a named suspect if task 6's `informative_utterances` looks low. **Diagnosis
  corrected while task 6 was built**: the mechanism is not a pad/unpad decode
  asymmetry but `encode`'s `split(" ")` against `data.py`'s bare `.split()`, so
  it only bites rows with repeated or leading whitespace — 0 of 1,642 QuranMB
  rows. It cannot explain a low informative count here.
- **ONNX Runtime declines a visible GPU** (job 224461). Week 8 needs GPU for 7
  short inference passes; if the diacritizers are re-run on cluster rather than
  locally this is in the way again.
- **D4's extension decision** (~1,592 further eligible rows) — annotation hours,
  not code.
- **Near-duplicate contamination** — exact skeleton matches only. Revisit only
  if a reviewer asks.
- **Expdirs live inside the S3PRL checkout.** Week 8 reads `dev-best.ckpt` from
  three of them, so a stray `git clean` in the pinned dependency now costs three
  200k-step runs rather than one.
- **`scripts/verify_quranmb_labels.py`** — optional; residual value is the one
  utterance by which the `safikhan` join (1,642) differs from the published
  count (1,643). Task 1 touches the same dataset, so this is cheap if it is ever
  going to be done.
- **Licence terms for `QuranMB.v2` / `Iqra_Extra_IS26`** — open since week 2,
  gates plan contribution 5.
- **Interspeech 2027 page limit and formatting rules**, once the call is
  published (plan §8).
- **`run/logs/label_path_rerun.log`** untracked — keep or delete, user's call.
- **The frontier-LLM fifth condition** (D8) — still declined; scheduled as one
  RQ1 row on MSA gold once annotation closes.

---

## Out of scope for week 8

- **RQ3's two decompositions, the synthetic corruption sweep, RQ4's abstention
  curves, and plan §4.2's feedback-validity number.** Weeks 10–12. The code lead
  banked above is slack for that stretch, not permission to start it early.
- **Retraining anything.** D10 fixed the rows as an inference-time swap and D12
  refuses mitigation retrains. The only circumstance that reopens training this
  week is task 4's byte-identity assertion failing, which would mean the arms
  are not what week 7 reported.
- **Fine-tuning or model selection on QuranMB.v2.** See the failure branch (c).
- **Revisiting the paper's framing.** Pre-committed in plan §5.
- **Partial diacritization and cross-lingual validation**, held back in plan §2.
