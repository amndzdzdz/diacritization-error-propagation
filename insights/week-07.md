# Week 7 insights

Dates: 2–8 Nov 2026 nominal; running ~4 weeks early (see
[docs/weeks/week-07.md](../docs/weeks/week-07.md)). This is the week the 2×2's
*rows* get a model to measure them with, on the Qur'anic arm only.

### Experiment: the contamination holdout, and the hole an id filter leaves

Week-07 tasks 4 and 5. Not an experiment in the sense of a hypothesis about
the world — it is machinery — but it had one empirical question inside it that
changed the design, so it is logged with the same structure.

#### Hypothesis

Week 6 found two independent overlaps between `Iqra_train` and our evaluation
pools, both invisible to id matching and to raw-string matching, both visible
to the consonantal skeleton:

- **MSA**: 28.0% of the 500 hand-annotated rows have their sentence in train
  (319 rows), and the 500 rows are themselves `Iqra_train` *dev* rows.
- **Qur'anic**: 16 of 98 QuranMB.v2 prompts (254 of 1,642 clips).

The working assumption going in was that **dropping the matching utterance ids
from the training manifest is sufficient**. The question this entry settles is
whether that assumption is true. It is not.

#### Methodology

1. `scripts/build_training_holdout.py` reads the 29 train Arrow shards by
   memory-mapping and projecting three columns (`id`, `sentence`,
   `phoneme_ref`), so the ~14 GB of audio never pages in, plus the cached dev
   text. It applies `msa_pool.skeleton_overlap` — the week-6 function — against
   (a) the 500 annotated sentences and (b) QuranMB.v2's `recovered_text`, and
   writes the **union** of both arms as one artifact.
2. Then I traced what the training loader actually reads, rather than what the
   dataset contains. `run/prepare_baseline.sh` step 5 generates the TSV that
   S3PRL's CTC downstream consumes, and **step 4 merges Iqra_TTS into the same
   train directory first**.
3. `arabic_mdd.data.holdout` + `scripts/apply_holdout.py` filter that TSV,
   wired into `prepare_baseline.sh` as step 6 of 8.

#### Configuration

- Key: `arabic_mdd.data.normalize.skeleton` (NFC → strip combining marks and
  tatweel → fold punctuation → collapse whitespace). Unchanged from week 6, so
  the artifact is a function of the same bytes the annotation block was drawn
  with.
- Artifact: `run/training_holdout.json`, with per-source provenance
  (`sources.msa_eval_pool`, `sources.quranmb`) rather than a bare id list, so
  a later reader can tell *why* a row was dropped.
- Filter keys: utterance id parsed from the manifest `path`
  (`audio_<id>.wav`), **and** the whitespace-normalized transcript column.
- Splits as measured: train 71,391 rows, dev 2,588 rows.

#### Results

| | train | dev |
|---|---|---|
| MSA arm | 319 rows (140 sentences) | 500 — *the pool itself* |
| Qur'anic arm | 62 rows | 6 rows |
| **union held out** | **379 = 0.53%** | **505 = 19.51%** |

Qur'anic contamination, by denominator: 15/98 prompts and 236/1,642 clips in
train, 5/98 and 71 clips in dev, **16 prompts / 254 clips / 15.5%** in the
union (4 prompts appear in both splits). Clean robustness denominator: **82
prompts / 1,388 clips = 84.5%**.

Second key: **517 distinct phoneme refs** across the 884 held-out ids.

Dev rows left for model selection: **2,083**.

#### Observations

1. **The two costs are not comparable, and quoting only the first flatters the
   decision.** 0.53% of train is free; nobody has to think about it. 19.51% of
   dev is a fifth of the model-selection set, and the week-07 plan as written
   quoted only the train side. The dev cost is real and is the one a reviewer
   would ask about. It is still the right trade, because the alternative is
   performing `dev-best` selection on the evaluation set — which is the exact
   objection `docs/msa-arm.md` §5.2 raises against the organizers' own
   published checkpoint. Paying 19.51% to not inherit that objection is cheap.
2. **An id-only holdout would have failed silently, and it took reading the
   setup script to see it.** `prepare_baseline.sh` step 4 copies Iqra_TTS's
   52 h into the same train directory; Iqra_TTS is *synthesized from
   `Iqra_train`'s phoneme refs*; and its filenames are
   `audio_<speaker>_<idx>.wav`, so **no id filter can reach a single TTS
   row**. The failure mode is the worst available: the holdout runs, reports
   379 rows dropped, and leaves a synthesized reading of the held-out text in
   training. Every log line looks correct.
3. **The right second key was already sitting in the manifest.** The TSV's
   `sentence` column is not Arabic text — `csv_to_tsv_with_transcripts.py`
   inlines `transcript_<id>.txt`, whose contents are `phoneme_ref`. So keying
   on the transcript string compares like with like, needs no text the TSV
   does not carry, and cannot produce a false positive: two utterances with
   identical phoneme sequences are the same text. Deliberately *not*
   `skeleton` here — on a phoneme string, stripping diacritics would conflate
   `b a` with `b i`, which is the distinction the whole task is about.
4. **Not fixed, and flagged rather than buried:** Iqra_TTS's *augmented* rows
   carry `phoneme_mis` (a deliberate mispronunciation) instead of
   `phoneme_ref`, so a corrupted synthesis of held-out text matches neither
   key. Iqra_TTS is not cached in the sandbox, so the size of that residue is
   unmeasured. It is a leak of *mispronounced* readings of held-out text,
   which is a weaker leak than a clean one, but it is a leak.
5. **`APPLY_HOLDOUT` has no default on purpose.** Both wrong answers are
   invisible: a contaminated "clean" model trains perfectly happily, and a
   held-out reproduction misses 0.4414 by an amount you then spend days
   debugging. So `prepare_baseline.sh` refuses to start until the run states
   which experiment it is. This is the second time this week's machinery has
   turned "pick a sensible default" into "refuse" — the first was the empty
   manifest check — and both times the reason is the same: the failure is
   cheap to detect now and expensive to detect later.
6. **The filter writes a sibling, not an in-place edit.** `train.heldout.tsv`
   next to `train.tsv`, so one 131 h data directory serves both the
   reproduction and the experiments without a re-download. The config is
   derived state: `prepare_baseline.sh` rewrites `sws.yaml` every run, so
   whichever manifest the current `APPLY_HOLDOUT` selects is the one wired in.
7. **The drop reports go to `run/`, not next to the TSV.** `run/data/` holds
   131 h of audio and is untracked; the drop counts are the only evidence a
   given run was actually held out, so they belong somewhere committable.
8. **Where this nearly went wrong.** The first version of the filter landed in
   `scripts/baseline_reproduction/`, next to the code it calls. That directory
   is `extend-exclude`d from ruff because it is *vendored* organizer code kept
   diffable against upstream — so a new file of ours there escapes linting
   entirely, and would have escaped tests too. Caught by the lint errors it
   raised before being excluded. The logic is now in `src` with tests, per
   CLAUDE.md, and the script is a ~90-line runner.

#### Conclusion

The holdout is built, measured, and in the training path, before any of this
week's runs start — which was the point, since the models trained this week
are the models week 8 evaluates on MSA, and re-deriving them later costs 4–5
days of queue.

The transferable finding is observation 2: **a contamination holdout has to be
keyed off what the training loader reads, not off what the dataset contains.**
The dataset-level answer (379 ids) was correct and insufficient, because a
setup script merged a second corpus derived from the first into the same
directory under different filenames. Nothing about the dataset card, the id
space, or the overlap measurement could have revealed that; only the 213 lines
of `prepare_baseline.sh` could.

Carried into the rest of the week: the filter has not yet run against a real
131 h manifest, only against a synthetic one built from the real key sets
(5 rows → 1, exercising both keys including a TTS-named row). The first
`APPLY_HOLDOUT=1` smoke run is what confirms the phoneme key actually fires on
merged TTS rows at scale, and the `dropped_by_phoneme_ref` count in
`run/holdout_applied_train.json` is the number to read.

### Experiment: pinning XLS-R-300m, and what the pin registry had to own

Week-07 task 3, the machinery half. The measurement half — steps/s, peak GPU
memory, projected wall clock — needs a GPU and is not in this sandbox.

#### Hypothesis

Task 3 asserted three things about swapping mHuBERT for XLS-R-300m, each of
which would cost a queue slot if wrong:

1. `hf_hubert_custom` **cannot** load XLS-R, because it builds a
   `HubertModel`; `hf_wav2vec2_custom` exists at the pinned S3PRL commit and
   is the right entry point.
2. XLS-R can be pinned by the same mechanism as mHuBERT (§5.1): resolve a sha,
   materialize a directory, pass the directory to `-k`, because
   `from_pretrained` ignores `revision` on this code path.
3. The 24-layer / 1024-dim upstream drops into the existing CTC config without
   editing it.

#### Methodology

Verified against sources rather than reasoned about: downloaded the S3PRL
tree at the pinned commit `ec8064b5` and read
`s3prl/upstream/hf_wav2vec2/{hubconf,expert}.py`; queried the Hub for
`facebook/wav2vec2-xls-r-300m`'s sha, full commit list, file list,
`config.json`, and `preprocessor_config.json`, and compared the last against
mHuBERT-147's and `facebook/wav2vec2-base`'s.

#### Configuration

- S3PRL pin: `ec8064b5889f81ca460fbe2c094ce576a6f120b7` (v0.4.18), unchanged.
- XLS-R pin: `facebook/wav2vec2-xls-r-300m` @
  `1a640f32ac3e39899438a2931f9924c02f080a54`, Hub HEAD on 2026-10-16.
- Registry: `UPSTREAMS` in `scripts/baseline_reproduction/pin_upstream.py`,
  replacing the two `MHUBERT_*` constants.
- Harness: `UPSTREAM_PIN=xlsr300m STEPS=200 sbatch run/smoke_train.slurm`.

#### Results

All three hypotheses confirmed, and one of them was almost wrong for a reason
nobody had considered.

1. `hf_wav2vec2_custom(ckpt)` exists at the pin and is **interface-identical**
   to `hf_hubert_custom`: `Wav2Vec2Model.from_pretrained(ckpt)`,
   `output_hidden_states=True`, `get_downsample_rates() == 320`, same
   `{"hidden_states": ...}` return. Confirmed, no surprise.
2. The pin mechanism transfers unchanged. XLS-R's 25-commit history has
   exactly three non-README commits — the weights (2021-10-26) and one
   `Update config.json` (2021-11-03) — so HEAD is two years of metadata churn
   on frozen weights, the same situation as mHuBERT's.
3. `num_hidden_layers: 24`, `hidden_size: 1024`, `do_stable_layer_norm: true`,
   `feat_extract_norm: layer`. So 25 hidden states instead of 13, and double
   the downstream input dim — and nothing to configure, because s3prl's
   featurizer sizes the weighted sum and the downstream input from a dummy
   forward pass rather than from the config file.

The near-miss: `facebook/wav2vec2-xls-r-300m` publishes **no**
`model.safetensors`. Its file list is exactly `[config.json,
preprocessor_config.json, pytorch_model.bin]`, ~1.26 GB.

#### Observations

1. **The registry's real job is not the sha, it is the pairing.** The sha was
   already parameterizable — `pin_upstream.py` has taken `--repo`/`--revision`
   since week 5. What it did not carry is that `-k <mHuBERT dir>` is only
   valid with `-u hf_hubert_custom` and `-k <XLS-R dir>` only with `-u
   hf_wav2vec2_custom`. That pairing was previously split across a constant in
   one file and a hardcoded flag in a slurm script, which is precisely the
   shape of thing that goes half-updated. It now lives in one dict, and no
   slurm file names an `hf_*` upstream at all: they name a pin, and read the
   `-u` name back out of `run/.upstream_<pin>.env`.
2. **`_weights_filename()` had a latent bug that only a second upstream could
   expose.** It preferred safetensors globally, on the reasoning that modern
   `transformers` prefers them — true, but about the *loader*, not about
   whether the repo has one. On XLS-R it would have asked
   `snapshot_download` for a nonexistent file, downloaded nothing, and tripped
   the completeness check. Loud failure, not silent, so the original design
   held up; but `has_safetensors` is now per repo because the condition always
   was.
3. **The single `.upstream_dir` file became a bug the moment a second upstream
   existed.** `train_baseline.slurm` hardcodes `-u hf_hubert_custom` and read
   `-k` from `cat run/.upstream_dir`, so an XLS-R prepare followed by a
   baseline submit would have fed `hf_hubert_custom` an XLS-R checkpoint. The
   file is now per pin, and the legacy one is still written for anything
   older that reads it.
4. **A thing that is not a problem, written down so it is not investigated
   twice.** `hf_wav2vec2/expert.py` wraps the feature-extractor load in a bare
   `except:` and silently falls back to downloading `facebook/wav2vec2-base`'s
   — an un-pinned network dependency inside a pinned run. It cannot fire here,
   because `preprocessor_config.json` is materialized into the `-k` directory;
   and if it did, the two configs differ only in `return_attention_mask`
   (`true` for XLS-R, `false` for wav2vec2-base and for mHuBERT-147), which
   the expert passes explicitly as `True` on every call regardless.
   `do_normalize` is `true` in all three.
5. **What is still unverified, and only a GPU can verify it.** Observation 3's
   "nothing to configure" is a claim about s3prl's dummy-forward sizing that I
   read in code and have not executed. That is most of the value of spending a
   queue slot on 200 steps before a 48-hour job.

#### Conclusion

The XLS-R arm is submittable. Everything that could be checked without a GPU
has been checked against the pinned source and the Hub rather than inferred,
and the `-u`/`-k` pairing — the one error in this area that costs a whole
queue slot and produces a confusing traceback — is now structurally
unavailable rather than merely documented.

The transferable observation is 1: when a second instance of something arrives,
the thing that breaks is rarely the parameter you remembered to parameterize.
The sha was already a flag; the *pairing* was the hidden constant, spread
across two files, and it only became visible because a second upstream forced
the question of which `-u` goes with which `-k`.

---

### Experiment: pinning the Python venv, and which pins are allowed to be fatal

#### Hypothesis

The week-7 plan calls this a day-one hard blocker rather than hygiene, on the
grounds that "a checkpoint trained against an unknown `transformers` cannot be
re-evaluated later with confidence." The hypothesis under that claim is
narrower and worth stating so it can be wrong: **S3PRL being pinned does not
pin the model code.** `hf_hubert_custom` and `hf_wav2vec2_custom` are thin
wrappers; the layer stack, the hidden-state list, and the forward semantics all
live in `transformers`, which S3PRL imports at module load and does not
constrain. So two runs at the same S3PRL sha can differ, and nothing in either
checkpoint would record that they did.

#### Methodology

Read the actual venv rather than resolve a dependency graph. One `pip freeze`
from the cluster's `run/s3prl_venv`, filtered to the five packages that could
change a number, then written into `prepare_baseline.sh` as defaults and
enforced on every run.

Enforcement, not installation, is the point. A one-time `pip install` of the
right versions leaves the next person's venv unconstrained; a check that runs
before every training job cannot be skipped by forgetting.

The verifier was tested by extraction rather than by running the whole script:
`awk` pulled the heredoc body out to a standalone file, which was then run with
a deliberately wrong `PIN_TORCH=9.9.9` (expect exit 1 with a `pip install`
remediation) and with every pin empty (expect exit 0, silent).

#### Configuration

- Freeze source: `run/s3prl_venv` on `danburit`, 17 Oct 2026.
- Fatal: `torch==2.4.1`, `torchaudio==2.4.1`, `transformers==4.46.3`.
- Advisory: `numpy==1.24.4`, `safetensors==0.5.3`.
- Every pin is `${PIN_X:-default}`, so an empty value disables that one check
  without editing the script.
- `pip install transformers` → `pip install "transformers==$PIN_TRANSFORMERS"`
  in step 2.

#### Results

The check passes trivially today (the pins were read off the venv they
describe), so the finding is not a number. Three substantive outcomes:

1. **The install was unpinned, not just the check.** Step 2 read
   `python -c "import transformers" || pip install transformers`. On a fresh
   venv that installs HEAD. Without the fix, the new verifier would have
   *created* the failure it then reports — install latest, immediately refuse
   to run.
2. **Two of the five pins should not be fatal**, and deciding which took the
   longer part of the task (below).
3. **A provenance gap surfaced that was invisible while there was one pin.**
   The S3PRL sha was read off `citrin` on 16 Oct; this freeze off `danburit` on
   17 Oct. Whether those are one machine is unverified.

#### Observations

1. **Fatal-by-default would have been the wrong instinct, and it took writing
   out the failure mode to see it.** The test for "fatal" is not "does this
   package matter" — all five matter. It is: *can a mismatch here change a
   reported number without changing anything a reader can see?* `transformers`
   can; it is the model code. `torch`/`torchaudio` can; they fix op semantics
   and the audio decode path. `numpy` and `safetensors` are read-side and
   transitive — a mismatch is worth printing but not worth converting a routine
   `pip` resolution into a blocked queue slot. Making them fatal would have
   produced exactly the outcome the check exists to prevent: someone commenting
   it out.
2. **An unpinned install plus a pinned check is worse than neither.** It fails
   on first run on a clean machine, for a reason that looks like the check being
   broken rather than the install being wrong. Enforcement and provisioning have
   to agree, and the obvious edit (add the check) touches only one of them.
3. **The remediation line is the part that keeps the check alive.** The verifier
   prints `pip install torch==2.4.1 ...` and says in words not to silence it.
   A check that reports a mismatch and leaves the reader to work out the fix
   gets deleted the first time it fires during a deadline.
4. **stdout and stderr interleave, and the fix is one line.** The first version
   printed its per-package report to stdout and its complaint to stderr, so an
   interactive run showed the remedy *before* the problem. `sys.stdout.flush()`
   before the stderr block. Cosmetic, except that confusing output is how a
   diagnostic loses the reader's trust.
5. **The provenance caveat is recorded, not resolved — deliberately.** I cannot
   check from here whether `danburit` and `citrin` are the same host. The honest
   options were to assert they are (unsupported) or to note the gap and let the
   check find out: if the pins are wrong for the machine that trains the
   baseline, the next run says so loudly, on the first run, for free. Writing
   "unverified" in the plan costs a sentence; asserting it and being wrong costs
   a week-3 reproduction that no longer means what it says.

#### Conclusion

Closed, and the day-one framing was right for a reason the plan states only in
passing: pinning S3PRL pinned the *harness*, not the *model*. The gap was one
line of shell (`|| pip install transformers`) and it sat under a sha pin that
made the venv look constrained.

The transferable observations are 1 and 2. A check's severity should be set by
whether a violation is *invisible downstream*, not by how important the package
feels — and adding enforcement without fixing provisioning builds a tripwire
across your own doorway.

---

### Experiment: building the two arms as one code path

#### Hypothesis

Plan §3 asks for "two of your own systems differing in exactly one property",
and week-07 task 7 restates it as a build constraint: the arms must be one code
path with a conditioning block ablated, not two models. The hypothesis worth
testing is that this is *achievable* rather than aspirational — that "one code
path" survives contact with an actual `nn.Module`, where a conditional
submodule is the obvious way to express an optional block.

Secondary hypothesis, from the task's test requirement: that the ablation can
be shown faithful *exactly* rather than within a tolerance. "Bit-for-bit" is
easy to write in a plan and is not automatic.

#### Methodology

Build the downstream from fused features onward, in `src/arabic_mdd/models/`,
with the arms switched by one config flag. Then attack the "exactly one
property" claim with tests designed to break it rather than to confirm it:
compare parameter tensors across arms, compare forward passes across arms, and
check that each guard actually guards something (for every "X must not affect
Y" test, a mirror test that X *does* affect Y when it should).

Verify at two scales. The unit tests use a toy width (dim 8, 4-unit LSTMs) so
they run in seconds; `scripts/smoke_mdd_arm.py` runs the same code at
`sws.yaml`'s real 1024 dims to catch anything that only appears at size.

#### Configuration

- `vocab.py`, `conditioning.py`, `mdd.py`; 51 tests in `tests/models/`.
- `MddConfig` defaults copied from `sws.yaml` field for field.
- Cross-attention: 8 heads at `project_dim` 1024, zero-initialized output gate,
  pre-norm on the query path, fixed sinusoidal positions.
- Real-width check: batch 16 x 300 frames (6 s at 50 fps), 40 canonical
  phonemes.

#### Results

- **51.56M shared parameters + 5.32M conditioner.** Logits `(16, 300, 71)`,
  2.8 GB RSS on CPU.
- **The step-0 loss is identical across arms to three decimals (26.890)** at
  full width, independently of the unit test that asserts it bitwise.
- **Loss 17.49 → 0.25 in 12 steps** on a repeated batch, both arms.
- **The timing comparison failed to produce a usable number**: +12.9% at 3
  steps, −0.7% at 12. A negative conditioning overhead is impossible by
  construction, so this is scheduler noise on a shared CPU.
- 304 tests pass overall; ruff clean.

#### Observations

1. **The "one code path" constraint nearly broke on something invisible, and it
   was not the forward pass.** The natural implementation builds the
   conditioner only when `use_canonical` is set. That is wrong, and wrong in a
   way no result would reveal: `nn.MultiheadAttention` and `nn.Embedding` draw
   from the global RNG when constructed, so the ablated arm consumes *less
   randomness* and its recurrent head — the part the arms are supposed to
   share — initializes differently. The arms would have differed in the
   conditioning **and** the initialization, and the paper would have attributed
   both to the conditioning. The fix is to construct the conditioner in both
   arms and route nothing through it in one, paying ~21 MB of dead checkpoint
   weights. The general shape of this: **a constraint stated over behaviour has
   to be enforced over construction too**, because initialization is behaviour.
2. **Zero-initializing the gate converted a test requirement into an
   experimental property.** The task asked for a bit-for-bit ablation test.
   Starting the residual's output projection at exactly zero delivers it —
   `frames + 0` is bitwise `frames` — but it also means the two arms are the
   same *function* at step 0, so a later divergence in their learning curves
   cannot be an artifact of landing in different basins. The test and the
   property are the same fact, which is the sign the design is right rather
   than merely tested.
3. **Every "does not affect" test needs a mirror, or it passes for the wrong
   reason.** `test_zero_gate_makes_the_block_an_exact_identity` would also pass
   if the gate were frozen, if the attention returned zeros, or if the
   canonical embedding were dead — i.e. if the text-dependent arm were the
   prompt-free arm with extra parameters, which is precisely the failure that
   would make the 2×2 measure nothing. So each identity test is paired with one
   that breaks it: the gate receives gradient, the block stops being an
   identity once the gate moves, canonical *content* changes the output, and
   after 10 steps swapping the canonical at inference changes the logits. The
   last of those is D10's operation, tested directly.
4. **The vocabulary was the one place with no design freedom, and it looked
   like the place with the most.** Index 0 is pad *and* the CTC blank; the 68
   phonemes are in `sws_arabic.txt`'s unsorted file order; `<eos>` is appended
   to CTC targets. All three are odd in isolation — a sorted vocabulary and a
   separate blank would be cleaner — and all three are forced, because the
   leaderboard scorer and two s3prl-trained checkpoints already use them.
   Sorting the 68 tokens would have renumbered every class and made our
   predictions a permutation of the right answer, which would have shown up as
   a catastrophically bad F1 and been debugged as a modelling failure.
   `phonetizer.SWS_ARABIC_INVENTORY` holds the same tokens as a `frozenset`,
   and the set is the wrong type here for exactly one reason: **the order is
   the index assignment**, and a set has no order.
5. **The riskiest part of porting s3prl's head turned out to be code that
   never runs.** The head has time-downsampling branches reachable from
   `sample_rate` and `sample_style`. Under `sws.yaml` — `total_rate: 320`
   against a 320 upstream rate, per-layer `sample_rate: [1, 1]` — none of them
   execute. So the port omits them and `MddConfig` raises on configurations
   that would need them, which shrinks the port to two LSTM layers plus a
   projected tanh. **Porting only the reachable subset, and failing loudly on
   the rest, is cheaper and safer than porting faithfully**: a faithful port of
   dead code is untested surface that looks tested.
6. **A measurement that cannot be trusted should say so in its own output.**
   The conditioning-overhead figure swung from +12.9% to −0.7% with the step
   count, and negative is impossible — the conditioned arm strictly does more
   work. Rather than drop the measurement or quote the favourable run, the
   script now prints a refusal below 10 steps or on a negative value. The
   number that would otherwise have reached a write-up is the one that is
   wrong.
7. **The honest gap: there is no training harness, so the arm cannot be
   submitted.** No feature extraction, no dataloader over the held-out
   manifests, no loop, no dev-best checkpointing. "Implemented and
   smoke-tested end-to-end" is true of the downstream from fused features
   onward and false of the pipeline, which means this task does not satisfy
   the week-7 exit criterion by itself. Recording it rather than letting
   "model built" read as "arm ready".
8. **Building the model changed the run count, which is a scheduling fact and
   not a modelling one.** Because both arms come from this module, the 2×2's
   prompt-free row is this model ablated — a third run, not task 6's s3prl
   XLS-R run. Task 6's run is still worth its slot, and for a reason that only
   became visible once the port existed: it is the **control on the port**. If
   our ablated arm underperforms s3prl's at the same config, the port is wrong,
   and without that run there would be no way to distinguish a bad port from a
   bad idea.

#### Conclusion

The modelling is done and the "exactly one property" claim is now enforced by
tests rather than asserted in a plan. The arm is not runnable: the training
harness is the remaining work, and the choice between our own loop and a thin
s3prl adapter that imports `MddModel` is deliberately left open rather than
taken by default, since the model's location and the trainer's location are
separable decisions and only the first has been made.

The transferable observations are 1 and 5. A claim about two systems differing
in one property is a claim about their *construction*, not just their forward
passes, and the RNG is part of construction. And when porting, port the
reachable subset and raise on the rest — faithfulness to unreachable code buys
nothing and hides how much is actually verified.

---

### Experiment: the training harness, and the data column it turned out to need

#### Hypothesis

The previous entry left one decision open: train the two arms with a loop of
our own, or fork s3prl's `downstream/ctc/` into a `ctc_cond/` that imports
`MddModel`. The hypothesis under test was that the fork could stay **glue** —
data wiring and a forward call — with every judgment remaining in
`src/arabic_mdd/models/` where the tests are. If a decision ended up having to
live in `expert.py`, the hypothesis was false and the right response was to
move that decision into `src/`, not to accept an untested fork.

#### Methodology

Decide first, on the identification argument rather than on effort. Then build
the fork and attack it the way the model was attacked: read the upstream source
for every assumption instead of inferring it from behaviour, and run the real
expert end-to-end without a GPU before anything reaches the queue.

Specifically: read `utility/helper.py`'s `override`, `downstream/runner.py`'s
downstream construction and train loop, and `downstream/ctc/text.py`'s decoder,
rather than guessing at the config and argument conventions. Then
`scripts/smoke_ctc_cond_expert.py` — real `sws_cond.yaml`, real
`DownstreamExpert`, real s3prl checkout, random features — for both arms.

#### Configuration

- Decision: the fork. `scripts/baseline_reproduction/ctc_cond/{__init__,data,expert}.py`,
  materialized into the checkout by `prepare_baseline.sh` step 7b.
- `config/sws_cond.yaml`: `sws.yaml` with a four-line diff (`*.cond.tsv`
  manifests, `use_canonical`, two conditioner knobs, `-d ctc_cond`).
- `run/train_mdd_arm.slurm`, one file for both arms, `USE_CANONICAL` required.
- New data side: `src/arabic_mdd/data/canonical.py` (+17 tests),
  `scripts/build_canonical_manifest.py`,
  `scripts/baseline_reproduction/dump_tts_canonicals.py`.
- New diagnostic plumbing: `scripts/report_collapse.py`.
- Smoke: batch 3, 120 frames, upstream dim 1024, upstream rate 320, seed 0.

#### Results

- **Both arms: `params=56,888,391 (conditioner 5,324,800)` and `loss=89.7893`**
  — identical to the digit (`89.789306640625` both times), now at the real
  expert's full width rather than the model smoke script's.
- **`grads={'conditioner': False, 'head': True}` ablated, `{'conditioner':
  True, 'head': True}` conditioned.** The ablation is exact *and* the
  conditioning is live; one number without the other proves nothing.
- All three `.ark` files written at `log_records`; `dev-canon.ark` line 1
  decodes to `'utt_0 yy n AA x'`, i.e. the canonical round-trips through the
  tokenizer rather than through an index dump.
- 342 tests pass; ruff check and format clean across 115 files.
- `build_canonical_manifest.py` verified end-to-end on fabricated data in both
  the success path and the missing-canonical refusal; `report_collapse.py`
  verified on synthetic `.ark` files (`copy_rate` 0.3333,
  `informative_utterances` 1, `canonical_preference` 0.0, including an
  empty-hypothesis row).

#### Observations

1. **The decision was settled by what task 6's run is *for*, not by effort.**
   A loop of our own is not much work. But task 6's s3prl XLS-R run is the
   control that says whether this port is sound, and that control only works if
   the ablated arm and task 6's run differ in the *head* alone. With our own
   loop, an underperforming ablated arm would be ambiguous between "the port is
   wrong" and "our optimizer/specaug/bucketing differs", and there would be no
   experiment that disambiguates them. Reusing s3prl's runner is what keeps a
   run we have already planned interpretable. **The cost of a reimplementation
   is not the writing; it is the control it destroys.**
2. **The fork imports the model instead of copying it, and that forced a
   Python-3.8 ceiling on `src/`.** S3PRL requires 3.8; the expert imports
   `MddConfig`/`MddModel` inside that venv, so the arms the cluster trains are
   the arms `tests/models/` tests. The price was four `zip(..., strict=True)`
   calls — a 3.10 keyword — each of which turned out to be redundant anyway
   (lengths checked immediately above, or lists built in lockstep). What is
   uncomfortable is that **nothing enforces the ceiling**: CI runs 3.12 and
   would never notice a `match` statement. The honest mitigation is cheap and
   partial — `prepare_baseline.sh` catches `SyntaxError` on import and names
   the likely culprit — and the constraint is written in three places rather
   than tested in one. Recording it as a known fragility is better than
   pretending a per-file lint ignore is a guard.
3. **The bug I would have paid for in queue time was a type, not a tensor.**
   I first wrote `self.tokenizer.decode(list(c))` for the canonical.
   `text.py`'s `idx_to_vocab` is a bare `self._vocab_list[idx]`, so a 0-dim
   tensor raises `TypeError` — but only in `log_records`, i.e. at the first
   *evaluation*, hours into a run. Reading the tokenizer's source found it in
   minutes; a smoke run that only exercised `forward` would not have. The fix
   normalizes to plain `int` once up front and uses the same list for the
   tensors and the decode. **Glue code's failures are type failures at
   boundaries, and they hide behind the parts that run early.**
4. **The data side was where the real judgment turned out to live, which is
   why it ended up in `src/` with tests.** The conditioned arm needs a third
   manifest column, and the question "where does this row's canonical come
   from?" has two different answers: Iqra_train rows can take it from
   `sentence` (the two strings coincide), Iqra_TTS rows must read a dumped
   file. The tempting shortcut is to default the second case to `sentence` when
   no file is found — and that default would set canonical == phoneme_mis on
   exactly the 52 h that punish canonical-copying. The result would be a
   manifest that trains, converges, scores well, and has had the anti-collapse
   signal deleted. So `build_canonical_rows` **drops and reports** instead, and
   `train_mdd_arm.slurm` hard-fails if `canonical_from_file == 0` on train.
   This is the entry's strongest instance of the general rule: the place a
   pipeline is most dangerous is where a missing value has a plausible
   substitute.
5. **The canonical's fetch key is a stream position, which is not a fact any
   dataset column records.** Iqra_TTS filenames are numbered by
   `enumerate(streaming dataset)`. Rejected: joining on `phoneme_mis` (not
   unique), recovering the canonical by nearest-neighbour against Iqra_train
   (introduces errors into the signal being measured), patching the downloader
   alone (prepare_baseline skips the download when audio already exists, so the
   patched loop would never run on the cluster's populated tree). The dump
   therefore *assumes* stream order and *verifies* it per row against the
   on-disk transcript, aborting on the first disagreement. **When a join key is
   an assumption rather than a value, the assumption has to be checked per row
   — otherwise a reordered stream fails on no row at all.**
6. **Both arms read the same three-column manifest, with the prompt-free arm
   ignoring column three.** Pointing the arms at different files would have
   been simpler to write and would have added a second difference between the
   runs. The manifest is also built *after* the holdout filter, from whichever
   TSV step 6 selected, so the holdout and the conditioning cannot disagree
   about which rows exist.
7. **Writing the collapse diagnostic is what revealed that it cannot be
   answered on dev**, and that discovery is worth more than the diagnostic. Dev
   is held-out Iqra_train, where the CTC target *is* `phoneme_ref` for every
   row, so canonical and target never disagree: `canonical_preference` is NaN
   by construction and `copy_rate` degenerates into exact-match accuracy. The
   temptation was to build a `test:` dataloader over the held-out TTS speaker
   to force an answer — a new bucketing CSV, a new TSV, a new sed placeholder,
   absolute paths — and the cheaper truth is that **QuranMB.v2 already supplies
   the informative triple in week 8 with no new infrastructure**, since
   `(canonical, human_annotation, prediction)` is exactly what
   `collapse_report` and `evaluate_utterance` already consume. Both the script
   and the slurm job now say so in their own output, so a NaN is read as
   expected rather than as a bug.
8. **The shortcut deserved a decision, so it got one: D11.** Collapse is not an
   implementation risk, it is a threat to the headline number — an arm that
   echoes its canonical reports the gap between `C_auto` and `C_gold`, which is
   a restatement of RQ1 wearing RQ2's clothes. Rejected the mitigations:
   canonical dropout trades the bias for its mirror image *and* introduces a
   free parameter with monotone influence on the reported effect; dropping
   Iqra_train or training on TTS alone breaks the one-property contrast. So:
   measure on both arms, declare the direction (collapse **inflates** the
   input-path |ΔF1|, same direction as D10's caveat), and pre-register the
   falsifier at `canonical_preference > ~0.5` on QuranMB.v2. **A knob that can
   tune the headline number is not a mitigation**, which is the whole reason
   this is a diagnostic.

#### Conclusion

The harness hypothesis held: the fork is glue, the three substantive
differences from upstream `ctc/expert.py` are enumerable (projector moved
inside the model, a `canonicals` argument, a third `.ark`), and nothing in it
decides anything. Both arms are now submittable from one file differing in one
value, and the full-width smoke run confirms they are the same function at step
0 with the conditioner gradient-live only when enabled. Observation 7's gap
from the previous entry is closed.

Two things this does *not* establish, stated so "queue-ready" is not read as
"validated": no arm has trained, and the collapse question is deferred by
construction to week 8's QuranMB.v2 evaluation rather than answered here.

The transferable observations are 1, 4 and 8. Prefer the implementation that
keeps an already-planned control interpretable, even when writing your own
would be faster. Treat any point where a missing value has a plausible
substitute as the most dangerous line in a pipeline. And when a risk can be
either measured or tuned away, measuring is the only option that cannot be
accused of having been chosen to suit the result.
