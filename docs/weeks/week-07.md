# Week 7 implementation plan

Dates: 2 – 8 Nov 2026 nominal. Opens the weeks 7–9 block in
[mdd-paper-project-plan-v2.md](../mdd-paper-project-plan-v2.md) §6 — "*Finish
annotation including the double-annotated subset; train XLS-R prompt-free;
**build the text-dependent arm**; MSA-arm label path*" — whose gate is "**both
arms trained, annotation closed**".

Week 6 closed on **16 Oct**, sixteen days early; week 5 closed twelve days
early and week 4 nine. Slack against plan §6 is now roughly four weeks. Week 7
is the first week where that slack is worth *spending* rather than banking,
and the reason is in the scope section: this week's critical path is GPU
wall-clock on a queue that takes one job at a time, so it is the first week
whose duration is set by something other than how fast we think.

See [week-06.md](week-06.md) for the eight closed tasks and
[insights/week-06.md](../../insights/week-06.md) for the five experiments.
[docs/msa-arm.md](../msa-arm.md) §5 (checkpoint and split hygiene) and §5.2.1
(the contamination holdout) are the sections this week acts on.

---

## The structural decision this week takes first: **the Qur'anic arm only**

Week 7 trains on `Iqra_train` and **evaluates on QuranMB.v2 only**. The MSA
arm's evaluation is weeks 8–9. This is a deliberate sequencing choice and not
a deferral of difficulty; five reasons, in descending order of how much they
matter.

1. **The Qur'anic arm needs no annotation.** `C_gold` is free there by
   convention (D9: our phonetizer's output on the recovered text). The MSA
   arm's label path is gated on annotation, which is gated on annotator 2's
   qualification test — external latency we do not control, and week-6 task
   8's first item for exactly that reason. Putting the MSA arm in week 7 makes
   two GPU-days of training wait on a packet in the post.

2. **The 2×2 only becomes a 2×2 on one corpus at a time.** Week 6 measured
   the **top-right** cell — label corruption with the model unimpaired — on
   the Qur'anic prompt-free arm. The rows (`model given C_gold` vs `model
   given C_auto`) need the text-dependent arm. If the rows were measured on
   MSA while the columns were measured on Qur'anic, the decomposition would
   not close: there would be four numbers but no 2×2. The Qur'anic arm is
   where the table first exists in full.

3. **Plan §6's own argument, one level up.** The week-6 gate was justified as
   "*if the decomposition does not work there, it will not work anywhere, and
   you find out in November with fifteen weeks left.*" The same logic applies
   to the input-path row. If the text-dependent arm fails to train, or if the
   input-path row is null on the controlled arm, we want that in hand before
   spending GPU-days on the arm that *also* depends on annotation quality.

4. **Failure isolation, which is the technical reason.** On the Qur'anic arm
   the domain is fixed, `C_gold` is gold by convention, and the prompt-free
   baseline is a reproduced published number. Any failure there is a code or
   design failure and nothing else. The MSA arm adds domain shift (§6),
   inventory questions (§4), an unknown mispronunciation base rate (§3.3), and
   human annotation error all at once — four candidate explanations for one
   disappointing number. Debugging a new architecture under those conditions
   is the worst available order.

5. **Budget.** One arm is two training runs. Two arms is the same two runs
   plus a second evaluation pass — cheap in principle, but see task 3: a
   single XLS-R run may not fit the current SLURM time limit, so the run
   count is the quantity under pressure this week.

**What this costs, stated plainly.** The weeks 7–9 gate says *both* arms
trained. Nothing about Qur'anic-first relaxes that; it reorders it. The MSA
arm's evaluation lands in weeks 8–9 as plan §6 has it, and this week's two
trained models are the ones it uses.

**The one thing Qur'anic-first must not do is cost a retrain.** The models
trained this week are the models the MSA arm evaluates in week 8. So the MSA
evaluation pool's holdout has to be in the training manifest **now**, in week
7, even though nothing this week evaluates on it (task 4). Getting this wrong
is the single most expensive mistake available this week: it is two more
training runs, which is most of a week of queue time, to fix a filter that
costs 0.45% of the training data to apply.

---

## Where week 6 left things

Eight tasks closed. Seven things set this week's shape.

1. **The label-path effect is real, and its sign is positive.** Clause 1
   passed: label corruption **inflates** F1, monotonically in CE DER, on all
   four tools, every 95% CI excluding zero — **+0.0265** (`catt-eo`) to
   **+0.2818** (`mishkal`) against a gold-`C` F1 of **0.4824**. The mechanism
   is not error cancellation: corrupting `C` enlarges the apparent
   mispronunciation set, so TA → TR conversions manufacture apparent
   detections out of accurate model outputs (`mishkal`: TA −5,122, TR +6,986,
   precision 0.38 → 0.72). **The bias in a naive evaluation runs in the
   direction nobody checks** — reported scores are too *good*, so the error
   does not announce itself.

   What this does to week 7: the input-path row now has a sign to be
   surprised by. There is no mechanical reason for the input path to inflate —
   a model handed a corrupted canonical should simply do worse — so the
   expected signature of the 2×2 is **a positive top-right and a negative
   bottom-left**, and the bottom-right is where they fight. If the input-path
   row also comes back positive, that is a bug hypothesis before it is a
   finding, and the alignment-artifact check from week-6 task 2 is the first
   thing to re-run.

2. **RQ5 is kept, at a factor of 1.80.** Worst-corner MDE **0.0907** F1
   against the pre-registered **0.1635** threshold. One consequence week 7
   must respect: RQ5 is powered against the *threshold*, not against each
   tool's own label-path effect. `MDE/|ΔF1|` is 0.32 (`mishkal`) and 0.52
   (`shakkala`) but **1.33 (`catt-ed`) and 1.96 (`catt-eo`)**. So the
   interaction is answerable on the high-corruption tools only. This does
   **not** reduce the tool count for RQ2 — the input-path row still needs all
   four — but it does mean a null on `catt-eo` is a resolution statement, not
   a result, and the write-up should say so in advance.

3. **`C_gold` on the Qur'anic arm is settled** (D9): our phonetizer's output
   on `recovered_text` in `run/quranmb_verse_map.json`, with the U+0670
   deletion rule retained. The text-dependent arm's conditioning input is this
   same object, which is what makes the rows and columns commensurable.

4. **S3PRL is pinned and the pin is enforced on every run** —
   `ec8064b5889f81ca460fbe2c094ce576a6f120b7` (v0.4.18), fatal mismatch check
   in `run/prepare_baseline.sh`. The training loop is no longer a moving
   target. **`transformers` is not pinned** and is now the likelier of the two
   to move; it is task 1 because it is a hard blocker on trusting anything
   trained this week.

5. **The annotation tool exists** and the blocker on the first annotated
   utterance is annotator 2's qualification test, not code. Annotation runs in
   parallel all week and belongs to the other owner (task 9).

6. **Splits that pass the standard check are contaminated.** Week-6 task 7:
   `Iqra_train` train/dev are id-disjoint *and* raw-string-disjoint, yet
   **28.0%** of the MSA annotation block's sentences occur in train. The
   holdout is `run/msa_eval_pool.json` → `holdout_train_ids`, 319 rows,
   **0.45%** of train. Task 4 applies it.

7. **And the Qur'anic arm is contaminated too** — found while closing task 7,
   not yet acted on. **15 of 98 QuranMB.v2 prompts (236 of 1,642 clips =
   14.4%)** have their verse text present in `Iqra_train` **train**; a further
   5 prompts / 71 clips are in dev (union 16 prompts / 254 clips = **15.5%**;
   4 prompts are in both). This is week 7's business and not week 6's
   because it bites hardest on the arm being built: a prompt-free model can
   only have memorized acoustics, but **a text-dependent model consumes `C`
   directly**, so for the arm this week builds it is text leakage of the
   purest kind. Task 5.

---

## Scope

The week is two training runs, one new model, and four decisions that must all
close before the first run starts.

**The critical path is queue time, and it is the first time that has been
true.** Runs go to `NvidiaAll` one at a time, and the sandbox these plans are
written in has no GPU at all, so nothing here can be dry-run at scale before
it is submitted.
The mHuBERT baseline took **24h** for 200k steps at batch 16. XLS-R-300m with
layer fusion and a TCN is a bigger upstream, a fused featurizer and a
different head — estimated 2–2.5× → **48–60h**. Three runs, serial, is
therefore **6–7.5 days minimum**, which does not fit the week: the exit
criterion asks for "trained or training" for exactly this reason.

**Superseded in part (18 Oct 2026): `NvidiaAll` has no walltime cap**, so the
"possibly over the per-job limit" half of this risk is gone and both training
jobs carry `-t 0`. Queue time remains the critical path; only the per-job wall
has been removed. Task 3's measurement is consequently optional and is not
being run — see the exit criterion's failure branch.

| what | blocks | blocked on |
|---|---|---|
| pin `transformers` (task 1) | trusting any trained checkpoint | one `pip freeze` from the cluster |
| 2×2 row semantics (task 2) | the run *count* | a decision, not a measurement |
| XLS-R step rate (task 3) | the SLURM time limit, and whether feature caching returns | a 200-step smoke run |
| the holdouts (task 4, task 5) | **both training runs** — getting this wrong costs a retrain | tasks 4/5 are cheap; sequencing is the whole risk |
| the text-dependent arm (task 7) | the input-path row, RQ5, the whole 2×2 | an architecture decision, then CPU work |
| XLS-R prompt-free (task 6) | the 2×2's prompt-free column on the strong system | tasks 1, 3, 4, 5 |

Tasks 1–5 are a day. Task 7 is CPU work that proceeds *while* task 6 occupies
the GPU. That parallelism is the only reason the week fits.

**Not this week:** the MSA arm's evaluation, the MSA label path, RQ3's
weights, RQ4's abstention curves, the synthetic corruption sweep, and the
feedback-validity number. All weeks 8–12 and all unchanged.

---

## Week 7 exit criterion

The weeks 7–9 gate ("both arms trained, annotation closed") is a block gate,
not a week gate. Week 7's own exit criterion, fixed here:

> **The Qur'anic arm's 2×2 is runnable: both XLS-R models trained or training
> on held-out data, the text-dependent arm implemented and smoke-tested
> end-to-end, and all four pre-training decisions closed and written down.**

**A fifth decision appeared while task 7 was built, and is closed too.** D11 —
the canonical-copy shortcut: on 79 of the 131 training hours the CTC target
*is* the canonical, so copying column three is a perfect solution available
without reading the audio, and a conditioned arm that collapses into an echo
would report `DER` dressed up as an input-path effect. Decided: measure it on
both arms, do not mitigate it in training. It belongs in this list because it
is pre-registered for the same reason D10 was — it bears on the headline
|ΔF1|, and after the first result is seen the choice can no longer be made
honestly.

Deliberately *not* "the 2×2 is measured". At 48–60h per run with one job at a
time, the input-path measurement lands in week 8 and claiming it for week 7
would be scheduling fiction. What week 7 must not do is reach Sunday with a
model trained on un-held-out data, or with the row semantics still undecided.

**The honest failure branch.** If task 3 says one XLS-R run exceeds the per-job
limit and cannot be raised, the fallbacks in preference order are: (a) fewer
steps with the step count reported, (b) feature caching (week-6 task 6 flagged
it and declined to build it — this is the condition under which it comes
back), (c) a smaller upstream, stated as a limitation. Taking (c) silently is
the one unacceptable outcome, because plan §3's argument is that the effect is
architecture-independent and that argument needs a second architecture that is
genuinely strong.

**This branch is dead as of 18 Oct 2026: `NvidiaAll` has no maximum
walltime.** Both training slurm files now carry `-t 0` explicitly rather than
`48:00:00`. Three consequences, because the branch was load-bearing in more
than one place:

- **Task 3's measurement becomes optional** and is not being run. It existed to
  size a limit that does not exist. What it *also* measured is now unverified:
  whether `batch_size: 16` fits this GPU. That failure is loud and arrives in
  the first minutes, so it costs a resubmission rather than a run.
- **Feature caching stays declined.** Its one re-opening condition was (b)
  above, and that condition can no longer fire. Week-6 task 6's refusal to
  build it now stands for the rest of the project rather than provisionally.
- **"Fewer steps" is no longer a fallback, so 200k is the number for every
  arm.** This matters for the comparison and not just the schedule: the arms
  are only comparable at equal steps, and the limit was the one thing that
  might have truncated one arm and not the other.

The cost of unlimited walltime is that a wedged job no longer dies on its own.
`save_step: 200` bounds the loss from a manual kill to 200 steps, which is the
reason that is tolerable.

---

## Tasks

1. **Pin `transformers`, and the rest of the venv with it. Day one, hard
   blocker.** S3PRL is pinned; `transformers` supplies the featurizer's model
   code, `hf_hubert_custom` imports it at module load, and it releases often.
   One `pip freeze` from the cluster venv closes it, and the same output lets
   `prepare_baseline.sh` pin `torch`/`torchaudio` while it is open. Enforce it
   the way task 5 enforced S3PRL — a fatal check on every run, not a one-time
   install — because the failure mode is silent drift between two runs that
   the paper will compare.

   This is a blocker rather than hygiene: a checkpoint trained against an
   unknown `transformers` cannot be re-evaluated later with confidence, and
   every number in the paper's spine comes off these two checkpoints.

   **DONE (17 Oct 2026).** `run/prepare_baseline.sh` now carries a `PIN_*`
   block beside `S3PRL_COMMIT` and a fatal verifier at the end of step 2:

   | package | pin | on mismatch |
   |---|---|---|
   | `torch` | 2.4.1 | **fatal** |
   | `torchaudio` | 2.4.1 | **fatal** |
   | `transformers` | 4.46.3 | **fatal** |
   | `numpy` | 1.24.4 | warning |
   | `safetensors` | 0.5.3 | warning |

   The split is not arbitrary. The first three are the ones a checkpoint is
   *bound to*: `transformers` supplies the upstream's model code, and
   `torch`/`torchaudio` fix the op semantics and the audio decode path, so a
   drift in any of them makes two checkpoints non-comparable without saying
   so. `numpy` and `safetensors` are transitive and load-bearing only for
   reading — a mismatch is worth knowing about but not worth refusing to
   train over, and making them fatal would turn a routine `pip` resolution
   into a blocked job.

   `pip install transformers` (unpinned) became
   `pip install "transformers==$PIN_TRANSFORMERS"`, so a fresh venv installs
   the pin rather than installing HEAD and then failing the check.

   **Provenance caveat, recorded rather than resolved.** The freeze was read
   off `danburit` on 17 Oct 2026. The S3PRL commit pin was read off `citrin`
   on 16 Oct 2026. Whether those are the same machine is **not verified**, so
   it is possible the versions above describe a venv that is not the one that
   produced the week-3 gate number. The check will say so loudly on the next
   run from any other host, which is the cheap way to find out: if it fires on
   the machine that trains the baseline, the pins get corrected to that
   machine's freeze and the week-3 reproduction is re-stated against the
   corrected ones.

2. **Decide the 2×2's row semantics: is `C_auto` swapped at inference, or
   trained on?** Worth roughly **nine GPU-days**, which is why it is a task
   and not a footnote. Inference-time swap = 1 text-dependent run.
   Training-time corruption = 1 clean run + one run per tool = 5.

   **Recommendation: inference-time swap, as primary.** Three reasons.

   - **It is what isolates the input path.** The 2×2's left column holds truth
     fixed and varies the model's input; the point is that *one thing* changes
     between the rows. Retraining changes the model as well as its input, so a
     training-time row measures "a model trained on corrupted canonicals"
     against "a model trained on clean ones" — a different and confounded
     contrast.
   - **It is the deployment story.** A deployed text-dependent MDD system was
     trained by its builders on whatever canonicals they had, and then meets a
     corrupted canonical for the utterance in front of it. That is an
     inference-time swap, exactly.
   - **It is the cheap one**, and the expensive alternative answers a question
     the paper does not ask.

   **The caveat, declared now rather than discovered in review.** A model
   trained only on clean `C` may be brittle to corrupted `C` in a way that
   *overstates* the input-path effect — a train/test mismatch masquerading as
   a corruption effect. The mitigation is one training-time run on `mishkal`
   only (the worst tool, so the largest available discrepancy) as a robustness
   check, scheduled **only if** task 3 leaves budget. If it does not, the
   caveat goes in §9 as a limitation with the direction of the bias named.
   Write the decision and the caveat down before training, so the robustness
   run cannot look post-hoc.

   **DONE (16 Oct 2026).** Written up as **D10** in
   [docs/mdd-paper-project-plan-v2.md](../mdd-paper-project-plan-v2.md) §1,
   beside the 2×2 table it disambiguates rather than here, because it is a
   standing design decision and not a week-7 task outcome. It records the
   recommendation above as decided, the train/test-mismatch caveat with the
   direction of the bias named, an explicit **falsifier for the decision
   itself** (the `mishkal` robustness run showing the effect vanish under
   training-time exposure), and the **expected 2×2 signature** — positive
   top-right, negative bottom-left — as a bug guard, so an input-path row that
   also inflates is a bug hypothesis before it is a finding.

3. **Measure the XLS-R step rate with a 200-step smoke run.** Replaces the
   48–60h estimate with a number. Three things to fix first:

   - `run/smoke_train.slurm` hardcodes `-u hf_hubert_custom`; parameterize the
     upstream name and the `-k` path.
   - **`hf_hubert_custom` cannot load XLS-R.** It builds a `HubertModel`.
     `hf_wav2vec2_custom(ckpt)` exists at the pinned commit and is the right
     entry point. Finding this at step 0 of a smoke run is the point of the
     smoke run.
   - Pin the XLS-R checkpoint the way mHuBERT was pinned (§5.1): resolve the
     repo at a sha, materialize into `run/upstreams/`, pass the **directory**
     to `-k`, so the sha lands in the checkpoint's own `Args.upstream_ckpt`.
     `from_pretrained` ignores `revision` on this path, so the directory name
     is the only pin that survives.

   Report steps/s, peak GPU memory, and the projected wall-clock for 200k
   steps. Then decide the time limit, the step count, or the fallback — see
   the exit criterion's failure branch.

   **Machinery DONE (16 Oct 2026); the measurement is DECLINED (18 Oct 2026)
   and the task is closed unmeasured.** `NvidiaAll` turns out to have no
   walltime cap, so the limit this was sizing does not exist. The machinery
   below is kept — it is the only cheap way to check that `batch_size: 16`
   fits before committing a long run, and the pin-pairing bug it found was
   worth the task on its own. Submit if you ever want the rate:

   ```bash
   UPSTREAM_PIN=xlsr300m STEPS=200 sbatch run/smoke_train.slurm
   ```

   What changed. `pin_upstream.py` now carries an `UPSTREAMS` registry instead
   of two module constants, and the registry owns the **`-u` name / `-k`
   directory pairing** — `mhubert147 → hf_hubert_custom`, `xlsr300m →
   hf_wav2vec2_custom` — because that is exactly the pairing the plan above
   warns you can get half right at the cost of a queue slot. No slurm file
   names an `hf_*` upstream any more; they name a pin, and prepare step 8
   writes `run/.upstream_<pin>.env` with both values. Per-pin, not the old
   single `.upstream_dir`: with two upstreams registered, "whatever ran last"
   is not a pin, and an XLS-R prepare followed by a baseline submit would have
   handed `hf_hubert_custom` an XLS-R directory.

   Four things verified against the pinned S3PRL commit and the Hub rather
   than assumed:

   - **`hf_wav2vec2_custom` exists at `ec8064b5` and is interface-identical**
     to `hf_hubert_custom`: `Wav2Vec2Model.from_pretrained(ckpt)`,
     `output_hidden_states=True`, `get_downsample_rates() == 320`. So swapping
     the upstream really does change only the upstream, which is what the
     comparison against the mHuBERT baseline needs.
   - **XLS-R-300m pins to `1a640f32ac3e39899438a2931f9924c02f080a54`** (Hub
     HEAD, 2026-10-16). Its full 25-commit history has exactly three
     non-README commits: the weights (2021-10-26) and one `Update config.json`
     (2021-11-03). Two years of metadata churn on frozen weights, so pinning
     to HEAD is safe in the same way §5.1's mHuBERT pin was.
   - **`facebook/wav2vec2-xls-r-300m` has no `model.safetensors`** — its file
     list is exactly `[config.json, preprocessor_config.json,
     pytorch_model.bin]` (~1.26 GB). The old `_weights_filename()` preferred
     safetensors globally, so it would have requested a file that does not
     exist, downloaded nothing, and tripped the completeness check. Now
     declared per repo.
   - **24 layers / 1024-dim, so 25 hidden states reach the featurizer**
     against mHuBERT's 13, and the downstream input dim doubles. **Nothing to
     configure**: s3prl sizes the weighted sum and the downstream input from a
     dummy forward pass, not from the config file. The 200-step run is the
     check that this is true rather than merely likely, which is most of why
     it is worth a queue slot before the 48-hour job.

   One non-finding, recorded so it is not re-investigated: `hf_wav2vec2`'s
   expert wraps the feature-extractor load in a bare `except:` that silently
   falls back to fetching `facebook/wav2vec2-base`. That would be an un-pinned
   network dependency, but it cannot fire here — we materialize
   `preprocessor_config.json` into the `-k` directory — and even if it did,
   the two configs differ only in `return_attention_mask`, which the expert
   passes explicitly on every call anyway.

4. **Apply the MSA holdout to the training manifest, before any run starts.**
   `run/msa_eval_pool.json` → `holdout_train_ids`, 319 rows = **0.45%** of
   train. Two rules:

   - **Both** XLS-R runs use the held-out manifest, even though nothing this
     week evaluates on MSA. The models trained this week are the models week 8
     evaluates, and re-deriving them later costs 4–5 days of queue.
   - The held-out rows come out of **model selection as well as training**.
     §5.2's whole complaint about `dev-best.ckpt` is selection exposure, so
     dropping them from the training loader and leaving them in the dev loader
     fixes nothing.
   - **The week-3 mHuBERT reproduction is not retrained.** It is a
     reproduction of a published number and must keep matching the
     organizers' setup. Where a table puts it next to our XLS-R runs, the
     difference in training data gets a footnote.

   **DONE (16 Oct 2026), together with task 5, and it needed a second key.**
   `scripts/build_training_holdout.py --write` → `run/training_holdout.json`:
   the union of both arms' holdouts in one artifact, because the manifest
   filter should have one input and not two. **379 train ids = 0.53%**, **505
   dev ids = 19.51%** (2,083 rows left for selection). The dev figure is the
   cost that matters and the plan above understated it by only quoting the
   train side: 500 of those 505 ids *are* the MSA evaluation pool, so holding
   them out of selection is not a rounding error, it is a fifth of the
   selection set. It is still the right trade — the alternative is selecting
   `dev-best` on the evaluation set.

   `arabic_mdd.data.holdout` + `scripts/apply_holdout.py` do the filtering,
   wired into `run/prepare_baseline.sh` as step 6/8 (15 tests). Two things
   that were not in the plan and are load-bearing:

   - **An id-only holdout has a hole, and it is silent.**
     `prepare_baseline.sh` step 4 merges Iqra_TTS into the *same* train
     directory, Iqra_TTS is synthesized **from `Iqra_train`'s phoneme refs**,
     and its files are named `audio_<speaker>_<idx>.wav` — so no id filter can
     reach a single TTS row. Filtering by id alone would drop a held-out
     utterance and leave a synthesized reading of the same text in training.
     The second key is the **transcript string**, which in this manifest *is*
     the phoneme sequence, so it compares like with like without needing the
     Arabic text the TSV does not carry: **517 distinct phoneme refs** from
     the 884 held-out ids. Identical phoneme sequences mean identical text, so
     a match there is never a false positive. Residual, flagged not fixed:
     Iqra_TTS's *augmented* rows carry `phoneme_mis`, so a deliberately
     corrupted reading of held-out text matches neither key. Iqra_TTS is not
     cached in the sandbox, so that residue is unmeasured.
   - **`APPLY_HOLDOUT` is mandatory, with no default.** Both wrong answers are
     invisible in a log — a contaminated "clean" model trains perfectly
     happily, and a held-out reproduction misses 0.4414 by an amount you then
     spend days debugging — so `prepare_baseline.sh` refuses to start until
     the run says which experiment it is. `train_baseline.slurm` hardcodes
     `0`; our runs pass `1`.

5. **Close the QuranMB contamination finding, and decide how it is
   reported.** Promote week 6's probe into a measured artifact alongside
   `run/msa_eval_pool.json`: 15/98 prompts and 236/1,642 clips in train, 5/98
   and 71 clips in dev, measured on the consonantal skeleton against
   `recovered_text`. Then two decisions.

   - **Training side**: hold the 62 train rows out, the same way as task 4.
     Cost is 0.09% of train, so combined with task 4 the holdout is under 0.6%
     and there is no trade-off to weigh. The 6 dev rows come out of model
     selection for the same reason task 4's do.
   - **Reporting side**: recommend reporting QuranMB **both ways** — all 98
     prompts as primary, for comparability with the published leaderboard and
     with our own week-3 gate number, and the uncontaminated **82 prompts /
     1,388 clips (84.5%)** as a robustness column. That denominator is the
     union of the train and dev overlaps (16 prompts, 254 clips, 15.5%; 4
     prompts appear in both), because §5.2's complaint is selection exposure
     as much as training exposure. The week-3 gate figure itself is
     **not** recomputed; it is a reproduction, and retroactively changing its
     denominator would destroy the one number that certifies the pipeline.

   Worth noticing *why* this surfaced: the overlap is enriched for Qur'anic
   verses, which is also the open "no Qur'anic/classical predicate" item in
   D4. It stays open — the predicate still needs an authoritative full Qur'an
   text — but it is now a measured hint rather than a suspicion.

   **DONE (16 Oct 2026).** Measured and committed inside
   `run/training_holdout.json` under `sources.quranmb`, alongside the MSA arm,
   with the per-prompt skeletons that matched so the 16 can be audited rather
   than trusted: 15/98 prompts and 236/1,642 clips in train, 5/98 and 71 clips
   in dev, **16 prompts / 254 clips / 15.5%** in the union, leaving the **82
   prompts / 1,388 clips (84.5%)** robustness denominator. The 62 train and 6
   dev rows are in the holdout the filter applies. The reporting decision
   stands as recommended above and is now the plan of record; nothing this
   week evaluates on QuranMB yet, so it is not yet exercised.

6. **Train XLS-R-300m prompt-free** — plan §3's "strong prompt-free", the
   modern-system arm, approximating the whu-iasp recipe: **XLS-R-300m + layer
   fusion + TCN + CTC**. Frozen upstream, matching the SUPERB paradigm the
   organizers' own checkpoint uses (`upstream_trainable = False`), so the
   comparison against the mHuBERT baseline varies the upstream and the head
   and not the training regime.

   Capture `Args`/`Config` with `inspect_s3prl_ckpt.py` and store it beside
   the results (§5.3). Week 3 is the proof this matters: the organizers' own
   README specified an upstream their own published checkpoint contradicts.

   The deliverable is the prompt-free arm of the Qur'anic 2×2 on a second
   architecture — which is what lets the paper claim the effect is not an
   artifact of one model.

   **Submittable (16 Oct 2026) — `sbatch run/train_xlsr_promptfree.slurm` —
   with one deliberate deviation from the recipe above, and it is the TCN.**

   ~~XLS-R-300m + layer fusion + TCN + CTC~~ → **XLS-R-300m + layer fusion +
   BiLSTM + CTC**, i.e. the organizers' `sws.yaml` run verbatim with only the
   upstream swapped. Layer fusion is unchanged — that *is* s3prl's learned
   weighted sum over hidden states, and it now spans 25 layers rather than 13.

   **Why.** s3prl's ctc downstream exposes exactly two heads, `RNNs` (the
   BiLSTM) and `Wav2Letter` (an undilated conv stack, not a TCN), reached via
   `eval(model_select)` over `expert.py`'s own import namespace. Adding a TCN
   means either editing a tracked file inside the sha-pinned checkout — which
   breaks the pin's provenance story and trips `prepare_baseline.sh`'s
   dirty-tree warning on every subsequent run — or materializing a forked
   `downstream/ctc_tcn/` directory, re-vendoring `data.py`, `expert.py` and
   `metric.py` in order to own 60 lines of model. That is a lot of surface for
   a component the paper never makes a claim about.

   **And the substitution is arguably better than the original.** Holding the
   head fixed makes the upstream the *single* variable between this run and
   the week-3 baseline, which is exactly what "the effect is not an artifact
   of one model" needs; the sentence above it, "varies the upstream and the
   head", was describing a weaker comparison than the one now available.

   **Escalation trigger, so this is not just a rationalization.** If this run
   does not beat the mHuBERT baseline's F1, the head becomes a live hypothesis
   and the forked `downstream/ctc_tcn/` is the next thing to build.

   Two things verified at the pinned s3prl commit rather than assumed, which
   is why **no config file was needed at all**:

   - `upstream/interfaces.py`'s `Featurizer` runs a dummy forward on 1 s of
     noise and takes `layer_num = len(hidden_states)` and `output_dim =
     feature.size(-1)` from the result. 25 layers and 1024 channels are
     *discovered*, not declared.
   - `downstream/ctc/expert.py` then does `nn.Linear(upstream_dim,
     project_dim)` with `project_dim` already 1024, so the BiLSTM sees a
     1024-dim input under either upstream and the head is literally identical
     in both runs.

   The job hardcodes `APPLY_HOLDOUT=1`, *fails* if the filter reports are
   missing, echoes the drop counts into its own log so the log alone answers
   "was this run contaminated?", and ends by writing
   `inspect_s3prl_ckpt.py`'s output to
   `run/xlsr_promptfree_provenance.txt` (§5.3). `BATCH_SIZE`/`ACCUM` exist as
   an OOM fallback and are documented to be used *together*, since halving the
   batch without doubling accumulation changes the recipe rather than only its
   memory footprint. The 48-hour limit is still a guess; task 3's 200-step
   smoke run is what replaces it with a number.

7. **Build the text-dependent arm.** `src/arabic_mdd/models/` is empty; this
   is the week's one genuinely new piece of modelling, and plan §3 calls it
   "**you build this** — required for the 2×2 and for RQ5".

   **The design constraint comes from the plan and it is strict.** Plan §3:
   "two of your own systems differing in exactly one property is cleaner
   evidence than two unrelated teams' submissions." So the two arms must be
   **one code path with the conditioning block ablated**, not two models.
   Same upstream, same layer fusion, same TCN, same CTC loss, same optimizer,
   same steps, same seed. The only difference is whether `C` is read.

   **Proposed mechanism**: a single cross-attention block over an embedding of
   the canonical phoneme sequence, inserted between the fused features and the
   TCN, with CTC unchanged. Reasons: it keeps the loss identical across arms
   (so F1 is comparable without an argument), it is the family the
   canonical-conditioning literature uses, and ablating it to identity
   recovers the prompt-free arm exactly. Alternatives considered and the
   reason against each: broadcasting a pooled canonical vector is too weak to
   produce the conditioning effect the design depends on; a seq2seq decoder
   changes the loss and breaks comparability; frame-level concatenation needs
   a monotonic alignment we would have to estimate, which adds a failure mode
   inside the thing being measured.

   Tests, per the workflow rule: the conditioning block ablated must reproduce
   the prompt-free forward pass bit-for-bit, and a shape/one-step-overfit test
   on a handful of utterances. Smoke-test end-to-end before it touches the
   queue — a 48h job that dies at hour 40 on a shape error costs two days.

   **DONE (17 Oct 2026), queue-ready and unqueued.** Both arms are submittable
   with `USE_CANONICAL=0/1 sbatch run/train_mdd_arm.slurm`; neither has run,
   because the GPU is the user's and three jobs do not fit the week. What
   follows is the model (built first), then the harness decision, then the
   data-side work the harness turned out to require.

   Three modules under `src/arabic_mdd/models/`, 51 tests, plus
   `scripts/smoke_mdd_arm.py`:

   - `vocab.py` — the 71-class output vocabulary, a deliberate
     re-implementation of s3prl's `WordTextEncoder` at `text.mode: 'word'`.
     Not a free choice: `<pad>`/`<eos>`/`<unk>` first so **blank is index 0 and
     is also pad**, the 68 phonemes in `sws_arabic.txt`'s **file order** (which
     is not sorted — sorting renumbers every class and makes our predictions a
     permutation of the baseline's), and `<eos>` appended to CTC targets, which
     is odd for CTC but is what the baseline trained with.
   - `conditioning.py` — the cross-attention block, as proposed above.
   - `mdd.py` — `MddConfig`/`MddModel`, defaulted to `sws.yaml` field for
     field, with the conditioner switched by `use_canonical`.

   **Two implementation decisions that the proposal above did not anticipate.**

   **The conditioner is constructed in *both* arms.** Building it only when
   `use_canonical` is set is tidier and would have quietly falsified the
   "exactly one property" claim: `nn.MultiheadAttention` and `nn.Embedding`
   draw from the global RNG at construction, so the ablated arm would consume
   less randomness and its *recurrent head* would initialize differently. The
   arms would then differ in the conditioning **and** the initialization, with
   nothing in any result to reveal it. Both arms now construct all three
   submodules; the ablated arm routes nothing through one of them and its
   parameters get no gradient. Cost: ~21 MB of dead weights in that
   checkpoint. A test asserts the head and projector weights are identical
   across arms under one seed.

   **The gate is zero-initialized, so the identity is exact rather than
   approximate.** The residual's output projection starts at exactly zero, so
   at step 0 the conditioned arm computes the *bitwise identical* function to
   the ablated one — which is both the bit-for-bit test this task asks for and
   a stronger experimental property than it was asked to be: the arms start
   from one point, so later divergence is learned and not inherited from a
   different initialization. Confirmed independently at full width by the smoke
   script, where both arms report the same step-0 loss to three decimals
   (26.890). Documented cost: the attention weights get no gradient at step 0,
   since it arrives through the gate; the gate's own gradient is nonzero, so
   the block unlocks after one step rather than staying stuck, and a test
   pins that.

   **The recurrent head is a narrow port of s3prl's `RNNs`, and the scope is
   what keeps it honest.** `src/` cannot import from a Python-3.8 checkout that
   only exists on the cluster, and an injectable-head abstraction would leave
   the arms untestable here. Under `sws.yaml` — `total_rate: 320` against a
   320 upstream rate, `sample_rate: [1, 1]` — **every time-downsampling branch
   in s3prl's head is dead code**, so the port omits them entirely and
   `MddConfig` *raises* on a configuration that would have needed them. What
   remains is two bidirectional LSTM layers with dropout and a projected tanh
   in s3prl's operation order, short enough to read against the original.

   **Verified at real width** (`uv run python scripts/smoke_mdd_arm.py`):
   51.56M shared parameters + 5.32M conditioner, logits `(16, 300, 71)`,
   2.8 GB RSS on CPU, and the loss falls 17.49 → 0.25 in 12 steps on a repeated
   batch, which is the cheap proof that the blank index, the packing and the
   CTC lengths are all wired right. The conditioning's compute overhead is
   **not** usefully measured here — the sandbox's shared CPU gives +12.9% at 3
   steps and −0.7% at 12, and a negative value is impossible by construction,
   so the script now refuses to let the figure be quoted below 10 steps. It
   needs a GPU.

   **One caveat the model carries permanently.** The arms differ in trainable
   capacity as well as in conditioning (56.89M vs 51.56M, +10.3%). This is
   inherent — conditioning cannot be added without parameters — and it is
   confined to the prompt-free *vs* text-dependent contrast. D10's comparison,
   which is the one the paper rests on, is *within* the text-dependent arm at
   identical capacity, so it is untouched. Stated here so the cross-arm
   contrast carries the caveat.

   **The harness: a thin `downstream/ctc_cond/` adapter, not our own loop.**
   The choice was between writing a training loop in `src/` and forking
   s3prl's `downstream/ctc/` to import `MddModel`. The adapter won on one
   argument: a loop of our own makes every difference from task 6's run
   ambiguous between "the conditioning did it" and "our optimizer/specaug/
   bucketing differs", and task 6's run is the control that says whether this
   port is sound. Reusing s3prl's runner keeps that control valid.

   The constraint accepted with it, and held: the adapter stays **glue**. Every
   judgment lives in `src/arabic_mdd/models/` where the tests are, and the fork
   has exactly three substantive differences from upstream `ctc/expert.py` —
   the projector moved inside `MddModel`, a `canonicals` positional argument
   threaded through `forward`, and a third `{split}-canon.ark` written at
   evaluation. `text.py` and `metric.py` are *not* forked; both downstreams
   read one vocabulary file.

   - `scripts/baseline_reproduction/ctc_cond/{__init__,data,expert}.py` —
     tracked template, materialized into the checkout by `prepare_baseline.sh`
     step 7b, which is untracked there by design.
   - `scripts/baseline_reproduction/config/sws_cond.yaml` — `sws.yaml` with a
     documented four-line diff.
   - `run/train_mdd_arm.slurm` — both arms from one file; `USE_CANONICAL` is
     **required**, because a conditioned run submitted with the switch off
     trains happily and fills the text-dependent row with a prompt-free
     result. s3prl's `-o` runs values through `eval`, so `use_canonical=True`
     arrives as a boolean rather than the (truthy) string `"True"`.
   - `scripts/smoke_ctc_cond_expert.py` — builds the real expert against a real
     checkout and runs `forward`/`backward`/`log_records` on random features,
     for both arms, with no GPU and no dataset. It confirmed at full width what
     the unit tests assert at toy width: identical step-0 loss (89.7893) and
     identical parameter counts, with `conditioner` gradients absent when
     ablated and present when not.

   **An unadvertised constraint this surfaced.** `src/arabic_mdd/models/` must
   stay importable under **Python 3.8**, because the expert imports `MddModel`
   from it inside s3prl's 3.8 venv. Four `zip(..., strict=True)` calls had to
   go (each was redundant — the lengths are checked immediately above, or the
   lists are built in lockstep); the ceiling is recorded in `pyproject.toml`'s
   per-file `B905` ignore and in `mdd.py`'s docstring, and enforced only by an
   import check in `prepare_baseline.sh` step 7b that catches `SyntaxError`
   and names the likely culprit. Nothing checks it in CI, which is the cheap
   end of that trade and the reason it is written down.

   **The data-side work the harness required, which was not in this task's
   plan.** The conditioned arm needs a third manifest column, and *where each
   row's canonical comes from* turned out to be a judgment, not plumbing:

   - `src/arabic_mdd/data/canonical.py` (+17 tests) — Iqra_train rows take
     their canonical from `sentence` (the two coincide, so the filesystem is
     never touched); Iqra_TTS rows must read a dumped file. A TTS row with no
     canonical is **dropped and reported**, never defaulted to `sentence` —
     that default would set canonical == phoneme_mis on exactly the 52 h that
     punish canonical-copying, yielding a manifest that trains, converges, and
     has the anti-collapse signal deleted. `train_mdd_arm.slurm` additionally
     hard-fails if `canonical_from_file == 0` on train.
   - `scripts/baseline_reproduction/dump_tts_canonicals.py` — Iqra_TTS's
     filename index is a *stream position*, not a dataset field, so the dump
     assumes stream order and **verifies it per row** against the on-disk
     transcript, aborting on the first disagreement. It streams with
     `remove_columns(["audio"])`, so 52 h of audio is never fetched.
   - `scripts/build_canonical_manifest.py` — runs **after** the holdout filter,
     on whichever manifest the run will train on, so the two cannot disagree
     about which rows exist. Both arms read the same three-column file and the
     prompt-free arm ignores column three: one fewer difference between the
     runs than pointing them at separate files would be.
   - `scripts/report_collapse.py` — reads the three `.ark` files and feeds
     `collapse_report`, joined on utterance id rather than line order. Run for
     both arms, non-fatally, because a 40-hour run must not be discarded by a
     reporting script. **Expect `canonical_preference: nan` on dev** and do not
     read it as a failure: dev is held-out Iqra_train, where the CTC target
     *is* the canonical for every row, so there is nothing to resolve either
     way. The decisive measurement is QuranMB.v2 in week 8 — see **D11**, which
     this work is what prompted.

   **Scheduling consequence, which changes the run count.** Because both arms
   come from this module, the 2×2's prompt-free row is *this* model with
   `use_canonical=False` — a **third** training run, not task 6's s3prl XLS-R
   run. Task 6's run keeps a separate job: the "strong prompt-free"
   modern-system anchor for the architecture-independence claim, and the
   control that says whether this port is sound, since a faulty port shows up
   as our ablated arm underperforming s3prl's at the same config. Three runs at
   one job at a time does not fit week 7; the ordering decision belongs to the
   owner split.

8. **Capture provenance on every run and write the insights entries as they
   land.** `inspect_s3prl_ckpt.py` output stored next to each checkpoint
   (§5.3). One `insights/week-07.md` entry per experiment, in the prescribed
   structure, written when the run finishes rather than at the end of the
   week. For a week whose units of work are two-day GPU jobs, retroactive
   write-ups lose the detail that makes them worth having.

9. **The owner split** (plan §7.7: "one owns the model and training pipeline,
   the other owns the diacritization pipeline and annotation. Both converge at
   week 7"). **Owned by us jointly, outside this plan.** Noted here because it
   is the half of week 7 with a hard deadline and because the two halves of
   this plan divide along exactly that line: tasks 1–7 are the model-and-
   training side, annotation and the qualification test are the other.

---

## Carried forward

- **The byte-identical regeneration guard** names `data/normalize.py`, but the
  U+0670 deletion it is guarding lives at `phonetizer.py:179`. Flagged in week
  6, unedited. One-line fix, no behaviour change, worth doing while nothing
  depends on it.
- **D4's extension decision** (~1,592 further eligible rows) — waiting on
  annotation hours, not on code. `annotate_msa.py --status` prints the running
  median and cumulative hours. The pool-level contamination figure (24.4%) is
  already measured, so an extension does not reopen task 7.
- **Near-duplicate contamination.** Both holdouts are exact skeleton matches,
  so Sadeed's 68.12%-above-0.5-similarity band is neither counted nor held
  out. At under 0.6% exact-match cost the exact tier is cheap insurance; a
  similarity threshold is a parameter to defend rather than a fact to report.
  Revisit only if a reviewer asks.
- **Expdirs live inside the S3PRL checkout** (`s3prl/mhubert147_per/` and
  friends, untracked). Harmless until someone runs `git clean` in a pinned
  dependency, which is why task 5's error message recommends
  `checkout --detach` instead. Moving them out is a week-7-sized change and
  belongs with task 1's `prepare_baseline.sh` edits.
- **`run/logs/label_path_rerun.log`** untracked — keep or delete, user's call.
- **ONNX Runtime declines a visible GPU** (job 224461, `SELECTED
  ['CPUExecutionProvider']` on a node with an RTX 2060). Cost 3h14m in week
  5's stage A. Week 6 needed no GPU; week 7 does, so this is now in the way.
- **`scripts/verify_quranmb_labels.py`** — optional. Residual value is
  per-row, plus identifying the one utterance by which the `safikhan` join
  (1,642) differs from the published count (1,643).
- **Licence terms for `QuranMB.v2` / `Iqra_Extra_IS26`** — open since week 2,
  gates plan contribution 5.
- **Interspeech 2027 page limit and formatting rules**, once the call is
  published (plan §8).

## Out of scope for week 7

- **The MSA arm's evaluation and the MSA label path.** Weeks 8–9. The
  structural decision at the top of this plan is precisely this deferral, and
  task 4 is what keeps it from costing a retrain.
- **RQ3's weight estimation, the synthetic corruption sweep, RQ4's abstention
  curves, and plan §4.2's feedback-validity number.** Weeks 10–12.
- **The frontier-LLM fifth condition.** Declined as a condition (D8) and
  scheduled as a single RQ1 row on MSA gold once annotation closes. None of
  D8's three re-opening conditions can fire this week.
- **Revisiting the paper's framing.** Pre-committed in plan §5. Week 6's
  positive label-path sign falsified one of its stated predictions and the
  framing still does not move; that is the section working as designed.
- **Partial diacritization and cross-lingual validation**, held back
  deliberately in plan §2.
