# Week 8 insights

Dates: 9–15 Nov 2026 nominal; running ~4 weeks early (see
[docs/weeks/week-08.md](../docs/weeks/week-08.md)). This is the week plan §1's
2×2 gets measured on the text-dependent arm, and the week D11's collapse
question gets an answer under D12's rule.

**Reading order, and why four of the five entries have no Results yet.** The
plan's task 10 fixes the order — manifest build and vocabulary check, then the
prompt-free byte-identity assertion, then collapse under D12, then the 2×2 —
because each is a precondition for reading the next. All four need a GPU and the
cluster's `$DATA_DIR`, so their Hypothesis / Methodology / Configuration are
written here **before** the runs, and Results / Observations / Conclusion are
filled when each job returns. That ordering is not bookkeeping: D12 is a
pre-registered rule, and a methodology written after the number arrives is not a
methodology. The one entry that is complete is the denominator derivation, which
needed no cluster at all; it is logged first because every later cell is
reported against it.

---

### Experiment: which 82 of the 98 prompts the organizers' checkpoint never saw

Week-08 task 7. Machinery with one empirical question inside it, which is where
the surprise was.

#### Hypothesis

Week 7 established that 16 of QuranMB.v2's 98 prompts (254 of 1,642 clips)
appear in `Iqra_train` by consonantal skeleton, and held them out of our own
arms' training manifests. So:

- **our** week-7 arms are clean on all 1,642 clips, and restricting them to a
  subset would discard 15% of the evaluation set for no reason;
- the **organizers'** checkpoint behind week 6's numbers and the published
  0.4414 was trained on those 16, so any comparison against it belongs on the
  clean 82.

`run/training_holdout.json` records the counts but stores
`contaminated_prompt_skeletons` as **texts**. The hypothesis under test is only
that the prompt indices are recoverable from the two committed artifacts, with
the same count falling out — i.e. that the join reproduces 16/82 exactly.

#### Methodology

`scripts/build_clean_prompt_subset.py` joins
`training_holdout.json:sources.quranmb.contaminated_prompt_skeletons` against
`quranmb_verse_map.json:prompts[].recovered_text`, keyed on the consonantal
skeleton, and writes the index list.

Deliberately a **join of two committed JSONs, not a re-run** of
`build_training_holdout.py`. That script reads the cluster-only `Iqra_train`
Arrow shards and rewrites the artifact the week-7 arms' manifests were already
filtered by; regenerating it to recover an index list it could have printed would
put the three 200k-step runs' provenance at risk to save nothing.

#### Configuration

- Key: `arabic_mdd.data.msa_pool.skeleton` — **imported**, not reimplemented.
  The same function `build_training_holdout.py` used. A second normalization
  rule here would silently under-match and the counts would still look
  plausible.
- Refuses on: a prompt with no `recovered_text`, any contaminated skeleton that
  matches no prompt, and any count other than 16/82/1388.
- Artifact: `run/quranmb_clean_prompts.json`, with a `which_denominator` field
  naming the condition under which each denominator is the right one, because
  the direction is the part that is easy to get backwards.

#### Results

Exact reproduction: **16 contaminated prompts / 254 clips**, **82 clean / 1,388
clips**, 98 / 1,642 total, **0 unmatched skeletons**.

Contaminated indices: `00016 00017 00020 00023 00037 00044 00050 00051 00052
00061 00068 00071 00076 00088 00092 00097`.

The finding that was not in the hypothesis: **two pairs of prompts have
byte-identical `recovered_text`** — `00009`/`00042` and `00073`/`00091`. The
clean 82 are therefore only **80 distinct texts**.

#### Observations

- The duplicate pairs matter because the cluster bootstrap resamples *prompts*
  as clusters, and two clusters drawn from the same text are not independent.
  The effect is small (2 pairs in 82) and in the direction of slightly
  understating the CI width.
- Recorded in the artifact's `duplicate_text_note` and **not corrected**:
  collapsing them would change the estimator week 6 already published CIs with,
  which would make this week's numbers non-comparable to last week's for a
  second-order reason.
- The verse map's own `skeleton_text` field is **not** `skeleton(recovered_text)`
  — only 4 of 98 are non-empty. Using it as the join key would have matched
  almost nothing and reported a near-empty contaminated set, which looks like
  good news. Caught only by importing the function instead of trusting the
  stored field.

#### Conclusion

Both denominators are now derivable and committed, with the direction stated in
the artifact rather than in someone's head: **all 98 for our own arms, clean 82
for anything compared against an organizer-trained checkpoint.**
`measure_label_path.py --clean-prompts-only` consumes the list, and labels its
output `clean_82` / `all_98` from the surviving prompt count rather than from
the flag — so a smoke run with `--limit 2` reports `clean_2` and cannot be
mistaken for the real thing.

---

### Experiment: the MSA arm's label path has no `A`, and therefore no ΔF1

Week-08 task 8. Logged as an experiment because the plan's premise for the task
turned out to be false, and finding that out *is* the result.

#### Hypothesis

Task 8 as written: "the week-6 label-path machinery is corpus-agnostic given a
`(C, A, P)` triple", so starting the MSA arm means pointing the existing driver
at MSA inputs and stopping where annotation blocks it.

#### Methodology

Traced each of the three inputs to the decision that owns it, rather than to
the code that would consume it.

#### Results

The premise does not hold, in two places.

**1. There is no `A` on this arm.** D1 (`docs/msa-arm.md:640`, 1 Oct 2026)
decided the MSA arm is label-path only and does *not* annotate `A`: "No
listening pass; annotation is text-only." D2 (`:946`, 14 Oct 2026) then decided
that `A := C_gold` is carried as a **bounded** assumption rather than as a
sensitivity row — the bound taken on the Qur'anic arm, where both quantities
exist, and scheduled for weeks 10–11.

**2. Under that substitution the gold cell has no positives, so ΔF1 is not a
measurement.** In the hierarchical metric `TR` and `FA` both require `C ≠ A`.
With `A := C_gold`, the gold cell has `C == A` everywhere, so both are **zero by
construction** and its F1 is `_safe_div`'s 0. Differencing an auto cell against
that measures the substitution, not the bias in a naive evaluation.

Three further differences from the Qur'anic arm, each of which would have been a
silent wrong answer in a shared driver:

| | Qur'anic arm | MSA arm |
|---|---|---|
| `C_gold` | `phonetize(recovered_text)`, D9 | human annotation, not yet closed |
| convention | prescriptive, as recited | **pausal** (protocol §3, R1–R4) |
| cluster | 98 prompts, ~16.8 clips each | one utterance = one clip = one `C` |

#### Observations

- The **pausal** convention is the one that would have been easiest to get
  wrong and hardest to notice. The annotators were instructed under R2 (the
  utterance-final mark is free) and R3 (tanwin is free), so phonetizing `C_auto`
  prescriptively would bill the convention difference to the diacritizer. That
  is D9's leg-A/leg-B mistake in a new place, where on the Qur'anic arm it was
  44 of 98 prompts of confound. `Phonetizer("pausal")` already exists, so this
  cost nothing once seen.
- The organizers' `tashkeel_sentence` is not a fifth tool, it is **the
  naive-deployment condition**: the column a group would score against with no
  thought given to any of this, which is what makes plan §1's bottom-right cell
  computable at all. It is carried in the frame, so it is free.
- The cluster unit being the utterance is not a weakening of the Qur'anic arm's
  prompt clustering — it is the same rule (`C` constant within a cluster)
  evaluated on a corpus where the cluster and the clip coincide.
- `P` joins 1:1: all 500 frame ids are present in `run/cv_dev_predictions.json`.
- 1 of the 500 rows (`64376`) is Latin-script and is excluded via `normalize`'s
  `fair` gate rather than cleaned — Latin letters are valid Buckwalter symbols,
  so the phonetizer invents phonemes for them instead of dropping them. 499 rows
  are scored.
- Phase 1 ran to completion on the full frame: 499 rows × 5 conditions, no blank
  cell, every condition's id set identical. 211 s of diacritizer time in total —
  catt-ed at 0.161 s/row and shakkala at 0.235 s/row dominate, the other three
  run at ~0.014 s/row. I had budgeted ~75 min from the week-5 DER runs; that
  estimate was off by a factor of 20, because the week-5 cost was per-*verse*
  chunking on long Qur'anic text and these CV-Ar sentences are short enough that
  0 of 499 rows chunked at all. Worth recording because it means the cache is a
  convenience, not a necessity, and re-running the whole column after an
  annotation fix is free.
- Smoke run on 8 rows with `tashkeel_sentence` substituted for `C_gold`: the
  `organizers-vowelizer` condition came out at exactly `ΔFRR = 0.0000`, which is
  the identity case behaving, and `catt-eo` also matched exactly — consistent
  with week 5's measured 98.65% position agreement between the two.

#### Conclusion

The MSA arm's reportable quantities are the gold cell's **FRR** (pure model
error on real MSA speech, read against week 4's CV-Ar dev 0.0708), the auto
cell's **manufactured TR** (apparent detections created by reference corruption
— contribution 4, the one D1 explicitly preserved), and **ΔFRR** with a paired
utterance bootstrap. F1 is written to the artifact with an `f1_note` so it
cannot be picked up without the reason it is not the number.

`scripts/measure_label_path_msa.py` is a separate driver, not a flag on the
week-6 one. Phase 1 (five `C_auto` conditions over the committed 500-row frame →
`run/msa_c_auto.json`) needs no human, owns the whole diacritizer runtime, and
**has been run** — the artifact is committed. Phase 2 is implemented and refuses with the count of rows still missing a
`final`, and separately refuses to let file order choose between the 50
crossover rows' two annotations. Closing annotation is the only remaining input.

**One thing deliberately not done.** D2's bound construction is now suspect for
the same reason as the above: under `A := C_gold` the *Qur'anic* gold cell also
loses all its positives, so the "residual term" it computes would be dominated
by the structural absence of real positives rather than by the deviation base
rate it is meant to price. It is cheap to check — one extra scoring condition on
an arm where both `A` and `C_gold` exist — and D2 schedules it for weeks 10–11,
so it is named in `docs/weeks/week-08.md` task 8 and left there rather than
pre-built.

---

### Experiment: the QuranMB.v2 manifests and the vocabulary check *(pending run)*

Week-08 tasks 1 and 2.

#### Hypothesis

All five canonical variants (`C_gold` plus `catt-eo`, `catt-ed`, `shakkala`,
`mishkal`) phonetize into the baseline's 68-token `sws_arabic.txt` vocabulary.
If a diacritizer's output introduces a token the model's symbol table lacks,
every clip in that variant's pass is scored against a canonical the model cannot
represent, and the row of the 2×2 is meaningless — which is why this is checked
*before* any GPU time rather than discovered in a log.

#### Methodology

`scripts/build_quranmb_manifests.py`, in this order:

1. Build all five canonical columns first (cheap, CPU, fails fast).
2. Decode `safikhan/quran_mbv2_formatted` to 1,642 16 kHz mono wavs under
   `$DATA_DIR/CV-Ar/quranmb/wav/`.
3. The D9 leg-A/leg-B tripwire.
4. The vocabulary check over `sentence` **and** all five canonical columns.
5. Emit five three-column manifests (`path`, `sentence` = annotation `A`,
   `canonical`).

`C_gold` is **leg A** — `phonetize(normalize(recovered_text))`, imported from
`measure_label_path.py`, never leg B's stored `reference_phoneme_string`. Two
legs that disagree is the week-6 finding; importing leg A means the two weeks
cannot drift apart.

#### Configuration

- `--allow-unknown` downgrades the vocabulary failure from fatal to a warning.
  It exists so that an unknown token is a *decision*, not a default.
- **96 of the 1,642 clips are stereo**, and s3prl's loader does
  `torchaudio.load(f).reshape(-1)`, which *interleaves* rather than downmixes —
  it would feed those clips as double-length, alternating-channel audio. The
  build script mean-downmixes to mono and refuses any clip that is not 16 kHz
  16-bit with 1 or 2 channels.
- Leg-A tripwire: `SystemExit` if leg A and leg B agree on *every* prompt, since
  that would mean the import silently picked up the stored string.

#### Results

*Pending — needs the cluster's `$DATA_DIR`. Login-node run, before any `sbatch`.*

#### Observations

*Pending.*

#### Conclusion

*Pending.*

---

### Experiment: the prompt-free arm ignores the canonical, byte for byte *(pending run)*

Week-08 task 4's assertion — the cheapest falsifier in the week.

#### Hypothesis

A `use_canonical=False` arm's hypotheses must be **byte-identical** across two
canonical variants, because the canonical never reaches its forward pass. If
they differ, the conditioning ablation is not an ablation and every number
produced this week is void.

Symmetrically, the conditioned arm's hypotheses must **differ** across variants.
If they do not, the input-path row of the 2×2 is exactly zero, and the first
suspect is that the manifest override never took effect rather than that the
model is robust.

#### Methodology

`run/evaluate_quranmb.slurm` copies `test-{hyp,ref,canon}.ark` into
`run/quranmb_eval/$EXP_DIR/$CANONICAL_VARIANT/`, then compares each new variant
against its already-present siblings:

- `canon.ark` identical across variants → **FATAL**: the swap did not happen.
- `hyp.ark` identical across variants → **required** for `use_canonical=False`
  (differing is FATAL), **warning** for `use_canonical=True`.

#### Configuration

- 7 passes over 1,642 clips: 5 conditioned (one per canonical variant) + 1 per
  prompt-free arm.
- The manifest is substituted with
  `-o config.downstream_expert.corpus.test=['<path>'],,args.expdir=<dir>`. The
  plan's line-anchored `sed` on a copied YAML is **not implementable**: under
  `-m evaluate -e <ckpt>` s3prl ignores `-c` entirely, and `config` is absent
  from `cannot_overwrite_args`, so the checkpoint's own config always wins. `-o`
  is the only override the runner honours. Verified against the pinned commit.
- `EVAL_EXPDIR` is a per-variant sibling directory, so the 7 passes cannot
  overwrite each other's `.ark` files — the way to lose the comparison silently.
- Row count asserted at 1,642. **2,069 is the week-7 failure signature** (the
  dev manifest, meaning the override did not take).

#### Results

*Pending.*

#### Observations

*Pending.*

#### Conclusion

*Pending.*

---

### Experiment: did the conditioned arm collapse into a canonical echo? *(pending run)*

Week-08 task 6. D11's question, answered under D12's rule.

#### Hypothesis

The conditioned arm did **not** collapse: the 47,439 `Iqra_TTS` rows where
`sentence = phoneme_mis ≠ canonical` are 40% of training hours and they punish
copying directly. D12 exists so that the clean and the messy outcome are
reported by the same rule, not because the messy one is expected.

#### Methodology

`scripts/report_collapse.py --split quranmb` on the **gold-`C`** pass, for the
ablated arm first and the conditioned arm second with
`--null <the ablated arm's report.json>`. Supplying `--null` makes the verdict a
computation that lands in the artifact, rather than a comparison a human makes
after reading two files — which is the whole point of pre-registering it.

#### Configuration

- D12's arithmetic lives in `arabic_mdd.models.collapse.d12_verdict`, under test,
  not in the script: `canonical_preference(conditioned) −
  canonical_preference(ablated) > 0.15`, evaluated only if
  `informative_utterances ≥ 200`. The power test is checked **before** the
  threshold, so an underpowered run can never report a firing falsifier.
  `tests/models/test_collapse.py` pins the exclusive threshold, the
  smaller-arm power rule, that NaN cannot fire, and that swapping the two
  arguments does not fire — an off-by-one in the comparison direction would
  otherwise flip the verdict invisibly.
- Null arm: job 224627, `xlsr300m_mdd_promptfree` (`use_canonical=False`).
- `informative_utterances` is expected in the high hundreds (QuranMB.v2's
  annotation disagrees with the canonical on ~12.4% of phonemes). If it comes
  back low, week 7's named suspect — the `.ark` decode asymmetry at
  `ctc_cond/expert.py:196` — is **not** the explanation: the mechanism is
  `encode`'s `split(" ")` against `data.py`'s bare `.split()`, which affects
  only rows with repeated or leading whitespace, and 0 of 1,642 QuranMB rows
  have any.

#### Results

*Pending.*

#### Observations

*Pending.*

#### Conclusion

*Pending.*

---

### Experiment: plan §1's 2×2 on the text-dependent arm *(pending run)*

Week-08 tasks 3–5, the week's exit criterion.

#### Hypothesis

Week 6 found the **label** path inflates F1. There is no mechanical reason for
the **input** path to inflate — corrupting what the model is *told* the learner
should say degrades its predictions, it does not manufacture true rejections.
So the signature of a correct implementation is a **positive top-right and a
negative bottom-left** (D10, recorded as a bug guard). An input-path row that
also comes back positive is a bug hypothesis before it is a finding.

#### Methodology

Rows = which canonical the model receives (one GPU pass each). Columns = which
reference the prediction is scored against (free, offline, over the `.ark`
files). `scripts/ark_to_predictions.py` converts `test-hyp.ark` to the
predictions JSON `measure_label_path.py --predictions` consumes, so the scoring
column reuses week 6's machinery unchanged rather than a second implementation
of the same metric.

#### Configuration

- Cluster bootstrap over the **98 prompts**, not the 1,642 clips: 10,000
  replicates, seed 20261014, paired draws, counts pooled per prompt before
  `compute_metrics` is applied once.
- Every cell reported against a stated denominator — `all_98` for our arms,
  `clean_82` for anything compared against week 6 or the published 0.4414
  (see the first entry).
- Empty hypotheses are kept as `""`. Dropping them would shrink the denominator
  and flatter the arm.
- **Pre-registered failure branch**: if the conditioned arm's gold-`C` F1 on
  QuranMB.v2 is below 0.40, it is reported beside the ablated arm's and
  diagnosed by comparing 224627 against 224615. There is **no** fine-tuning or
  model selection on QuranMB.v2 under any outcome.

#### Results

*Pending.*

#### Observations

*Pending.*

#### Conclusion

*Pending.*
