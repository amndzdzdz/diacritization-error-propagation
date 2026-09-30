# Week 4 — Diacritizer pipeline and phonetizer

Plan: [docs/weeks/week-04.md](../docs/weeks/week-04.md). Update this file as
each task below produces a result — do not wait until the end of the week.

## Logistics log

Track non-experimental tasks here as they close. One line each, dated.

- [x] 2026-09-23 — `docs/msa-arm.md` written: standing reference for the MSA
      arm. Consolidates the week-3 carried risks and adds three that review
      missed (mispronunciation base rate on read CV-Ar speech; the
      prescriptive-vs-pausal `C_gold` convention; dev-split contamination in
      the obvious choice of evaluation set). Contains the build order that
      weeks 4–9 follow.
- [x] Diacritizer wrappers (CATT EO/ED, Shakkala, Mishkal) under
      `src/arabic_mdd/diacritizers/`, with tests, 2026-09-28. Common
      `Diacritizer` interface (`base.py`) plus a `DIACRITIZERS` registry so
      the RQ1 sweep can iterate the set without knowing what backs each one.
      Every tool reports `provenance()` — the paper's methods needs the
      versions, and §5.1's pin-everything rule applies to diacritizers as
      much as to the upstream. Wrappers are deliberately thin: they do not
      normalize their input, because the normalization contract (§4.1) is a
      separate pinned step and burying it per-tool would make the tools
      incomparable and the contract invisible. Tests are offline throughout —
      each wrapper takes an injectable backend. **Farasa not wrapped**: plan
      §3 allows "Mishkal or Farasa", Mishkal is pure Python where `farasapy`
      shells out to Java, and RQ3's pre-registered rank prediction is about
      Mishkal. Three findings below.
- [x] **Shakkala cannot run in this project's environment**, 2026-09-28. It
      pins `tensorflow==2.9.3`, whose newest wheel is cp310, against Python
      3.12 — `uv` rejects it outright rather than degrading. Dropping it was
      not acceptable (plan §3 picks the tool set precisely because CATT's
      Table 5 spans 5.4–19.6% CE DER across CATT/Shakkala/Mishkal, and that
      spread is the x-axis of the RQ1 curve), so it runs **out-of-process**
      under its own `uv`-managed 3.10 interpreter — the same pattern
      `scripts/baseline_reproduction/` already uses for S3PRL's 3.8 venv.
      Protocol: JSON on stdin, JSON to a file, because Shakkala and
      TensorFlow both print to stdout on model load. The pin is made
      load-bearing rather than decorative: the worker reports its installed
      version and the wrapper raises if it is not 1.7. Shakkala also asserts
      above 315 input characters instead of truncating — harmless here, since
      the longest utterance in QuranMB.v2 is 118 chars and in `Iqra_train`
      191, so 0/74k rows are affected and no chunking is needed.
- [x] **Mishkal emits U+0001 and it silently merges words**, 2026-09-28.
      `TashkeelClass.tashkeel` substitutes U+0001 (its internal sentence
      separator) for sentence-final punctuation *and the space after it*:
      "ذهب الولد. ثم عاد" comes back as "ذَهَبَ الْوَلَدُ\x01ثُمَّ عَادٌ",
      which whitespace-splits into three words instead of four. Deleting the
      character would keep the corruption, so the wrapper restores it to a
      space. It also prepends a leading space unconditionally. Neither is a
      diacritization decision, and both would have reached the phonetizer.
      Worth noting as a pattern: this is the third time an artifact token has
      had to be handled at a tool boundary, after `<sil>` and the vowelizer's
      pause marks.
- [x] **The Shakkala worker was importing our own wrapper**, 2026-09-28 —
      caught by an end-to-end smoke run, not by the unit tests. The worker
      script lives next to `diacritizers/shakkala.py`, and Python puts a
      script's own directory first on `sys.path`, so its `import shakkala`
      resolved to the wrapper, which then tried to import `arabic_mdd` in an
      interpreter that does not have it. Fixed by dropping `sys.path[0]` in
      the worker. The lesson is about test design: every unit test passed
      against injected fakes while the real path was broken, so the wrappers
      were additionally run against all four real tools before being called
      done. There is now a regression test that executes the real worker
      script against a stub `shakkala` package, so the trap is covered
      offline without building a TensorFlow environment.
- [x] CATT token-probability extraction vendored behind a tested interface,
      2026-09-28. `CattDiacritizer.diacritize_batch_with_confidence` returns
      one `DiacriticConfidence(letter, diacritic, confidence, forced)` per
      input position, superseding week 1's EO-only probe in
      `scripts/check_catt_probabilities.py` (now rewritten as the real-weights
      check). RQ4 depends on this entirely. Two design points. First, the
      package version is **pinned** (`catt-tashkeel==1.0.2`, not `>=`) and
      re-checked at call time: the extraction rides on `_run_encoder`,
      `_run_decoder`, `_apply_space_mask` and `_prepare_batch_input`, which
      carry no compatibility promise, and week 1 flagged exactly this
      fragility when it deferred the production interface here. An upstream
      refactor now fails loudly instead of quietly returning misaligned
      confidences. Second, the `forced` flag: CATT hard-sets every space
      position to "no diacritic" via `_apply_space_mask`, so the confidence
      there describes a prediction that was discarded — **RQ4 must not
      threshold on those ~15% of positions**, and the flag makes that a typed
      property rather than a comment.
- [x] **The technique works on ED, but the two variants need different code**,
      2026-09-28. EO decodes in one shot, so `_run_decoder` hands back the
      full logit tensor and the probabilities are one softmax away. ED decodes
      autoregressively and keeps only `preds[:, -1, :]` each iteration, so
      there is no point at which the whole tensor exists — the loop itself has
      to be re-walked, accumulating the per-step distribution. Confirming ED
      separately was therefore not box-ticking. The non-obvious part is that
      the two paths nonetheless land on the *same* alignment: EO's logits
      index `input_ids[:, 1:-1]` while ED's step *i* produces the token at
      `target_ids[i+1]`, and both reduce to "position *k* describes input
      letter *k*" once `<BOS>` is dropped, which is what lets one assembly
      routine serve both. Verified against real weights on both checkpoints:
      `"".join(letter + diacritic)` reproduces `diacritize()` exactly, 3/3
      texts, both variants.
- [ ] Choose the RQ4 variant — deferred, with evidence. On the long news
      sentence ED is visibly *worse* than EO (`إِنْسِتْجَراٍم`, `يْمُكُن أْن`:
      shadda dropped, sukuun on a vowel-initial word), agreeing with EO on
      only 82/99 positions, while the two short sentences agree 100%. That
      contradicts CATT's own claim that ED is the more accurate variant, and
      three sentences cannot settle it — the RQ1 DER measurement in week 5
      does, on real corpora, so the choice waits for it. Both variants stay
      wrapped and both expose confidence, so nothing is blocked. Confidence
      spread looks usable for a risk–coverage curve either way (EO mean 0.957,
      min 0.477 on the long sentence).
- [ ] `utter-project/mHuBERT-147` pinned to a commit sha.
- [ ] 2026 diacritizer survey chased; diacritizer set finalised.
- [ ] Licence-terms confirmation for `QuranMB.v2`/`Iqra_Extra_IS26` —
      carried from week 2.
- [ ] Wall-clock cost of one XLS-R-300m fine-tune, and the week-7 work-split
      decision — carried from week 1–2. Cannot slip past week 6.

## Completed experiments

### Experiment: `Iqra_train` text-column inspection (MSA-arm step 1)

#### Hypothesis

Three things about the MSA arm were assumed rather than checked, and all
three are decidable by inspecting `IqraEval/Iqra_train`'s text columns —
no phonetizer, no annotation, no GPU:

1. That the phonetizer's only validation reference is QuranMB's 1,642
   Arabic strings, which exercise Qur'anic orthography only
   (`docs/msa-arm.md` §4).
2. That MSA text might emit phonemes outside the 68-token
   `sws_arabic.txt` vocab — risk (a), the failure mode that could
   manufacture the paper's headline result.
3. That `sentence` is the bare, undiacritized text the MSA arm feeds to
   CATT/Shakkala/Mishkal/Farasa.

#### Methodology

`scripts/inspect_iqra_train_text.py`. Reads only the text columns via
`pyarrow` column projection over HTTP, so the 12 GB of audio in these
parquet files is never fetched; rows cached as JSONL under `run/tmp/`.
Reports (1) the `phoneme_ref`/`phoneme_aug` token inventory against the
vocab, (2) diacritization rates for `sentence` vs `tashkeel_sentence`,
(3) consonantal-skeleton agreement before and after punctuation removal,
(4) non-Arabic character classes, (5) vowelizer content loss.

#### Configuration

`IqraEval/Iqra_train` at sha `02f9863`, `dev` split, 2,588 rows.
Vocab: `scripts/baseline_reproduction/vocab/sws_arabic.txt` (68 tokens).

#### Results

**A fourth column exists that changes the week-4 plan.** `Iqra_train`
ships `tashkeel_sentence` — the organizers' in-house vowelizer's output —
alongside `phoneme_ref`. It is fully diacritized in every row (mean
diacritization rate 0.783, zero rows below 0.20).

| Check | dev result |
|---|---|
| `phoneme_ref` tokens outside the 68-token vocab | **none** (66/68 used) |
| `phoneme_aug` tokens outside the vocab | **none** (identical set) |
| `phoneme_aug == phoneme_ref` | **2588/2588 (100%)** |
| `sentence` effectively undiacritized (rate < 0.05) | 1847/2588 (71.4%) |
| `sentence` heavily diacritized (rate ≥ 0.50) | 489/2588 (18.9%) |
| `sentence`/`tashkeel_sentence` same skeleton | 1353/2588 (52.3%) |
| …after also removing punctuation | **2580/2588 (99.7%)** |
| rows containing punctuation | 1231/2588 (47.6%) |

Vowelizer content loss, from the 8 rows that still disagree:

- **Latin script is silently dropped.** "أثبت استوديو Eddie عكس ذلك" →
  `tashkeel_sentence` has no "Eddie". Same for "Trans-Radio Press",
  "Diadem".
- **Rows carrying Qur'anic pause marks get truncated.** "فمن ثقلت موازينه
  فأولئك هم المفلحون" → `ۚ فَمَنْ ثَقُلَتْ`. Two of eight.
- **Superscript alef (U+0670) is rewritten as a literal alef.**
  "وَأَبْقَىٰ" → "وَأَبْقَىا".
- **Ordinary vowelizer errors.** "عَمَلٍ" for "عَمِلَ" (noun for verb);
  "تعذریننی" → "تَعَذَّرَ نَنْ" (word split wrongly).

#### Observations

The `tashkeel_sentence` → `phoneme_ref` pair is a **second phonetizer
round-trip reference, 45× larger than QuranMB's and covering MSA
orthography** — the exact gap `docs/msa-arm.md` §4 flagged as unaddressed.
73,979 pairs against 1,642.

Risk (a) looks much smaller than feared, at least in the dev slice: the
organizers phonetized MSA text into the 68-token vocab with zero
overflow. That is evidence, not proof — it constrains *their* phonetizer,
not ours, and it says nothing about what CATT/Shakkala/Mishkal/Farasa will
produce on text the organizers never processed.

`phoneme_aug` is identical to `phoneme_ref` in every dev row, so it
carries no mispronunciation signal and cannot stand in for `A`. This is
the first direct evidence for the base-rate problem in `docs/msa-arm.md`
§3.3: the MSA dev split ships **no annotation of actual production at
all**.

`sentence` is not the clean undiacritized input the arm assumed. It is
heterogeneous — 71% bare, 19% heavily diacritized — so the MSA arm needs
an explicit, pinned stripping step. And the vowelizer's own normalization
is *punctuation removal*, which accounts for essentially the entire
skeleton mismatch (52.3% → 99.7%). Any diacritizer we run must see the
same depunctuated text or the comparison is not like-for-like.

The content loss is the most consequential finding. `phoneme_ref` is the
CTC **training target**, so a dropped "Eddie" is audio the model was
trained to emit nothing for, and at evaluation it is a guaranteed
insertion against the reference. That is the same mechanical-false-
rejection mechanism as risk (a), already present in the organizers' own
data, and it is a real if small contamination of the validated baseline.

#### Conclusion

Three of the MSA arm's open questions move. The phonetizer gains a large
MSA-orthography round-trip reference (fold into the week-4 exit
criterion). Risk (a) is downgraded from "could manufacture the headline"
to "verify per diacritizer". The base-rate problem gains its first piece
of supporting evidence. A new item is added: the normalization contract
(punctuation, Latin script, U+0670) must be pinned before any diacritizer
runs, and the content-loss rate must be quantified on the train split
before deciding whether it affects the baseline materially.

#### Addendum — full train split (71,391 rows), same day

The train run finished and confirms every dev finding at 27× the scale,
tightens two numbers, and **corrects one**.

| Check | dev (2,588) | train (71,391) |
|---|---|---|
| `phoneme_ref` tokens outside the vocab | none (66/68 used) | **none** — 2,336,971 tokens, **68/68 types used** |
| `phoneme_aug == phoneme_ref` | 2588/2588 (100%) | **71376/71376 (100%)** |
| `tashkeel_sentence` mean diacritization rate | 0.783 | 0.785 (p05 0.660, p95 0.917) |
| `sentence` effectively bare (< 0.05) | 71.4% | 54.4% |
| `sentence` heavily diacritized (≥ 0.50) | 18.9% | **32.3%** |
| skeleton match | 52.3% | 49.7% |
| …after depunctuation | 99.7% | **99.5%** |
| rows with punctuation | 47.6% | 50.0% |
| rows losing >30% of words | — | **552 (0.8%)** |
| rows with Latin script | — | 21 (0.03%), **dropped in 21/21** |
| empty `phoneme_ref` | 0 | 15 |

Three things are new.

**`<<` and `gg` do occur.** The dev split used 66 of 68 vocab types; train
uses all 68. The inventory is exactly saturated — further evidence the
organizers' phonetizer targets this vocab precisely.

**Content loss is real but not material: 0.8%.** Median word-count ratio is
exactly 1.000, but 552 rows lose more than 30% of their words and the
minimum ratio is 0.000 — some rows are destroyed outright. The worst case
found is "إنَّ النَّفْسَ لَأَمَّارَةٌ بِالسُّوءِ" → `ۚ`, a whole utterance
reduced to a single pause mark. Too small to explain any headline effect;
large enough that these rows must be excluded from the round-trip's
denominator rather than scored as phonetizer failures.

**The U+0670 finding in the dev write-up above is wrong, and the correction
matters.** I recorded "superscript alef is rewritten as a literal alef",
inferred from 8 disagreeing dev rows. At scale the behaviour is
bidirectional: the vowelizer strips **every** U+0670 it is given (0 of 120
survive) and then **inserts its own on 2,738 rows (3.8%)** where the source
had none — "الرحمن" → "الرَّحْمَٰن", "موسى" → "مُوسَىٰ", "على" → "عَلَىٰ".
It is normalization-then-restoration under the vowelizer's own orthographic
convention, not a rewrite rule. The practical consequence inverts: U+0670
is a **live character in the phonetizer's input** at 3.8% of rows, rather
than something the vowelizer eliminates. `scripts/inspect_iqra_train_text.py`
now reports the two directions separately so this cannot be misread again.

Also worth pinning: a thin tail of non-MSA Arabic-script characters —
Farsi yeh (98 rows), keheh (36), and single instances of tcheh,
qaf-with-three-dots, heh-doachashmee, isolated-form ligatures, and one
HAMMER AND SICKLE (U+262D).

**Methodological note.** The dev-split U+0670 claim was drawn from eight
rows and stated as a rule. It was cheap to check at scale and it was wrong.
For the remaining normalization-contract items, quantify on train before
writing the rule down.

---

### Experiment: Word-final case-ending conventions across arms (MSA-arm step 2)

#### Hypothesis

`docs/msa-arm.md` §3.4 asserts that both arms must share a case-ending
convention, and warns that a prescriptive `C_gold` against pausal speech
would *manufacture* the case-ending effect RQ3 predicts. The assertion was
never checked. Null hypothesis: the two arms already agree.

#### Methodology

`scripts/compare_word_final_conventions.py`. Per-word alignment of text to
phonemes needs the week-4 phonetizer, so this uses two alignment-free
contrasts: the distribution of word-final marks in the Arabic text, and
the class (short vowel / long vowel / consonant) of the **utterance-final
phoneme**, cross-tabbed against the mark on the utterance's final word. A
marked case ending realized as a vowel means prescriptive; realized as the
bare consonant means pausal.

#### Configuration

MSA arm: `Iqra_train` dev, `tashkeel_sentence` vs `phoneme_ref`, 2,588
utterances. Qur'anic arm: `safikhan/quran_mbv2_formatted`,
`reference_arabic_string` vs `reference_phoneme_string` and vs
`annotation_phoneme_string`, 1,642 utterances.

#### Results

Utterance-final case endings, where the final word carries a case mark:

| Arm | n | realized as a vowel | dropped (pausal) |
|---|---|---|---|
| MSA — `Iqra_train` reference | 2,451 | **34.1%** | **65.9%** |
| Qur'anic — QuranMB reference | 1,425 | **80.1%** | 19.9% |
| Qur'anic — QuranMB annotation | 1,425 | 79.1% | 20.9% |

Utterance-final phoneme class:

| Arm | consonant | long vowel | short vowel |
|---|---|---|---|
| MSA reference | 66.2% | 25.0% | 8.8% |
| Qur'anic reference | 28.0% | 16.7% | 55.4% |

Word-final marks across *all* words are similar between arms (fatha
37–42%, kasra 19–20%, damma 14–16%, sukun 12–14%).

#### Observations

The two arms use **opposite** utterance-final conventions, and it is close
to an exact inversion: the MSA reference drops two thirds of its marked
case endings, the Qur'anic reference realizes four fifths of them. This is
waqf/pausal treatment in the organizers' MSA pipeline against recitation
convention in the Qur'anic data.

Mid-utterance the two look alike — the all-words mark distributions nearly
coincide — so the divergence is specifically at the pause, which is what a
pausal rule predicts and argues the effect is a genuine convention
difference rather than an artifact of the mark-counting heuristic.

The Qur'anic reference and annotation differ by only 1.0 point (80.1% vs
79.1%). The empirical pausal rate in careful recitation is therefore very
small, and `C_gold`/`A` disagreement at the utterance-final position is
not a meaningful error source on that arm.

Two caveats. `safikhan`'s `reference_arabic_string` is a *recovered* text
(`src/arabic_mdd/data/quranmb.py`); if it were recovered from the phoneme
string this contrast would be circular — the observed 80%, rather than
~100%, argues against full circularity but does not exclude partial. And
this measures the utterance-final position only, which is where pausal
rules bite hardest and is therefore the most favourable place to find a
difference.

#### Conclusion

§3.4's worry is confirmed and is not hypothetical. Cross-arm comparison of
case-ending effects is confounded as things stand, and RQ3's
case-ending claim would be measuring different things on the two arms.

This also cuts the other way and is useful: the organizers' MSA pipeline
already applies pausal treatment, so utterance-final case endings are
largely **absent from the MSA reference to begin with**. That gives the
"case endings inflate DER but cause little downstream harm" hypothesis a
concrete mechanism rather than an assumption.

Action: the convention must be chosen and pre-registered before annotation
(week 5), and the per-word version of this analysis re-run once the
phonetizer exists — this alignment-free version reaches only the final
word of each utterance.

---

### Experiment: MSA-arm mispronunciation base-rate screen (step 3)

#### Hypothesis

If Common Voice Arabic speakers rarely deviate from the prompt, then
`A == C` almost everywhere, the detection metric has no positives, and the
MSA arm cannot produce the 2×2's headline F1 figure
(`docs/msa-arm.md` §3.3). Running the validated checkpoint over dev audio
and scoring with `A := C` turns every model/reference disagreement into a
false rejection, giving an upper bound on what the model would be charged
if the base rate really were zero. Contrasting that against the model's
QuranMB FR rate (0.1241) should indicate whether there is headroom.

#### Methodology

Two stages on the cluster (`run/cv_dev_baserate.slurm`): Stage A runs
`mhubert147_per/dev-best.ckpt` over all `Iqra_train` dev audio and dumps
per-utterance wavs; Stage B scores with `A := C`, compares against the
QuranMB figure, and draws a stratified 50-utterance manifest (25 from the
high-disagreement tail, 25 uniform, seed 20261012) for the step-4 pilot.

#### Configuration

`IqraEval/Iqra_train` dev, 2,588 utterances, 2,588/2,588 ids matched.
Checkpoint: our own from-scratch retrain, not the organizers'.

#### Results

```
counts: MDDCounts(ta=77911, fr=5938, fa=0, tr=0, ...)
CV-Ar dev false-rejection rate : 0.0708
QuranMB false-rejection rate   : 0.1241
excess                         : -0.0533
```

Per-utterance disagreement: p10 0.000, p50 0.050, p90 0.194; 752/2,588
(29.1%) utterances have zero disagreement. `ta + fr = 83,849`, which
matches the dev `phoneme_ref` token count from step 1 exactly — a useful
end-to-end consistency check on the join and the scorer.

#### Observations

**The headline reading is that there is no headroom — and that reading is
not supportable.** My own script printed "no headroom for a meaningful
mispronunciation base rate; §3.3 option 2 is likely forced". That
conclusion overstates what this design can deliver, and the flaw is mine:
I wrote the in-domain caveat into the *positive* branch only. `dev-best.ckpt`
was model-**selected** on this very split (`docs/msa-arm.md` §5.2), so its
true error rate here is expected to sit well below the QuranMB figure
whether or not speakers ever deviate. A negative excess is therefore
equally consistent with:

1. no mispronunciations in CV-Ar dev; and
2. some mispronunciations, masked by a model that is simply much better on
   audio it was tuned against.

QuranMB is the wrong reference point for an in-domain split: different
domain, different speakers (learners vs fluent readers), different
acoustics, and out-of-domain for this checkpoint. The screen is
**inconclusive in the negative direction**, not negative.

The distribution is the more interesting output. 29.1% of utterances are
exactly clean and the median is 0.050, so disagreement is not uniformly
smeared — there is a real tail, which is what the enriched stratum targets.

#### Conclusion

**Verdict: inconclusive, by construction.** The probe ran correctly and
cheaply and its internal consistency checks pass, but its reference point
cannot separate the two hypotheses. Two corrective actions:

1. `scripts/cv_dev_baserate.py`'s negative branch now states the confound
   and reports "inconclusive" instead of forcing option 2.
2. A zero-base-rate control was built (`run/tts_control.slurm`):
   `Iqra_TTS`'s `original` rows are synthesized from `phoneme_ref`, so
   `A == C` by construction and their FR rate is the model's pure error
   rate on MSA phonetics. Read one-sidedly — `Iqra_TTS` is in-domain and
   synthetic, both of which push its rate down, so only
   `control FR >= 0.0708` is decisive.

**Premise check for that control** (800 sampled rows, before writing it):
after stripping `<sil>`, `phoneme_mis == phoneme_ref` in **500/500**
`original` rows and **3/300** `augmented` rows. The clean half is clean and
the mispronounced half is mispronounced, as needed. ~1/3 of the 46,851 rows
are `original` (~15.6k), interspersed in blocks rather than contiguous.

Two traps found while building it, both of which would have produced a
plausible-looking wrong number:

- **`<sil>` is not in the 68-token vocab** (`<` alone is — the glottal
  stop). Only `phoneme_ref` carries it, so comparing the two columns raw
  makes every clean row look 100% mispronounced; and leaving it in the
  scoring reference charges the CTC head a guaranteed false rejection per
  silence, inflating exactly the quantity being measured. Both stages strip
  it. `Iqra_train`'s `phoneme_ref` has no `<sil>`, which is why the CV-Ar
  probe needed no equivalent step.
- **`Iqra_TTS` is ordered by speaker**, so the `--limit` head-N subsample I
  first wrote would have measured one or two voices' idiosyncratic error
  rate rather than the model's. Now a seeded random shuffle, default 3,000
  rows (SE ≈ 0.002 on the FR rate, against a ~0.04 gap), with the speaker
  mix printed so any residual imbalance is visible.

Step 4's listening pilot moves from "confirmatory" to **critical path**. It
is the only unconfounded measurement available, and its 50 utterances and
audio are already prepared.

---

### Experiment: zero-base-rate control on `Iqra_TTS` `original` rows

#### Hypothesis

The step-3 screen was confounded: `dev-best.ckpt` is in-domain for CV-Ar
dev, so its low disagreement rate there cannot be attributed to an absent
mispronunciation base rate. Measuring the model's error rate where the base
rate is zero *by construction* should bound how much of the 0.0708 is model
error. Decisive only if `control FR ≥ 0.0708`.

#### Methodology

`run/tts_control.slurm`. Stage A runs the same checkpoint over a seeded
random subsample of `Iqra_TTS` rows with `label == original` (synthesized
from `phoneme_ref`, so `A == C` exactly), stripping `<sil>`; Stage B scores
with `A := C`, which here is an identity rather than an assumption.

#### Configuration

3,000 randomly subsampled `original` rows (seed 20261012), 115,304
phonemes, mean 38.4 per utterance. Same `mhubert147_per` checkpoint.

#### Results

```
counts: MDDCounts(ta=114539, fr=765, fa=0, tr=0, ...)
TTS-original false-rejection rate : 0.0066   <- pure model error
CV-Ar dev disagreement rate       : 0.0708
headroom for a real base rate     : +0.0642
```

#### Observations

**Verdict: inconclusive, as the one-sided design allowed for — and the
control turns out to be nearly vacuous as evidence.** At 0.0066 the model
makes 0.66 errors per 100 phonemes, roughly **10.7× lower than CV-Ar dev**
and 18.8× lower than QuranMB. The nominal headroom (0.0642) is 91% of the
dev disagreement, so the bound excludes almost nothing.

The low rate is itself informative, in an unflattering direction.
`Iqra_TTS` is ~7 synthetic voices, clean, consistent in prosody, and inside
the 131 h training set. A 0.66% error rate is close to memorization. The
control therefore measures the model at its easiest possible operating
point, and the 10.7× gap to real speech is readily explained by acoustics
alone — crowdsourced CV-Ar audio is noisy, multi-accent and variable —
without invoking any mispronunciations at all.

**What this closes.** It is now established that *no cheap proxy will
settle the base-rate question*. Both available reference points are
confounded in the same direction (QuranMB by domain, TTS by ease and
in-domain training), and there is no third one. Step 4 is not merely the
best option, it is the only one.

**What it opens.** Building the control surfaced that `Iqra_TTS`'s
**augmented** rows carry both `phoneme_ref` (`C`) and `phoneme_mis` (`A`),
differing in ~99% of rows. That is a complete `(C, A, P)` triple — the only
MSA-domain data in the project with genuine detection positives, and it
makes precision/recall/F1 computable on the MSA arm for the first time.

#### Conclusion

The control fails to constrain the base rate but succeeds at retiring the
proxy strategy. Two follow-ups: step 4 (unchanged, critical path), and
`run/tts_detection.slurm`, which scores the augmented rows as a real triple
to give the MSA arm a working metric path and an F1 comparable to the
Qur'anic arm's 0.4406 — as an upper bound and sanity check, since the
errors are injected and the audio synthetic.

---

### Experiment: Case-ending convention — the mismatch is in the published benchmark

#### Hypothesis

`docs/msa-arm.md` §3.4.1 identified a collision: `mhubert147_per` is trained
on `Iqra_train`'s `phoneme_ref`, which is **pausal** (65.9% of utterances end
on a consonant), but is evaluated on QuranMB.v2, which is **prescriptive**
(80.1% end on a realized case ending). A model taught to drop utterance-final
case endings, scored against a key that marks them, should be charged a
deletion at the end of nearly every utterance.

That predicts a specific error profile — high recall, depressed precision,
inflated FR — which is exactly the published baseline's profile
(P 0.3093 / R 0.7707 / FRR 0.1237). Prediction: the model's utterance-final
phoneme distribution is pausal while the reference's is prescriptive, and a
material share of the benchmark's false rejections is a convention artifact.

#### Methodology

`scripts/case_ending_impact.py`, two measurements of deliberately different
strength, on cached local files only (no GPU, no network):

1. **Final-phoneme class contrast** (strong, assumption-free). Classify the
   last token of `C`, `A` and `P` as short vowel / long vowel / consonant and
   compare the three distributions. Requires no alignment, so nothing about
   the metric's tie-breaking can influence it.
2. **Ablation estimate** (weak, indicative). Drop the final token from `C`,
   `A` and `P` on utterances where `C` ends in a short vowel, re-score, and
   report the delta. Removing a token shifts the alignment, so this is an
   order-of-magnitude estimate, not a measurement.

Run against **both** checkpoints, to separate "property of our training run"
from "property of the recipe".

#### Configuration

- Labels: `run/tmp/quranmb_text.jsonl` (cached official ground truth),
  1,642 utterances; 1642/1642 matched for both prediction files.
- Predictions: `run/quranmb_predictions.json` (our retrain, F1 0.4406) and
  `run/quranmb_predictions_official_ckpt.json` (organizers' released
  checkpoint, F1 0.4415).
- Metric: `arabic_mdd.metrics.hierarchical`, unchanged.

#### Results

Measurement 1 — utterance-final phoneme class:

| | `C` reference | `A` annotation | `P` ours | `P` organizers' |
|---|---|---|---|---|
| short vowel | 55.4% | 54.9% | 11.4% | 10.7% |
| long vowel | 16.7% | 16.4% | 21.3% | 22.8% |
| consonant | 28.0% | 28.7% | 67.3% | 66.5% |
| **vowel-final** | **72.0%** | **71.3%** | **32.7%** | **33.5%** |

`C − P` gap: **+39.3 points** (ours), **+38.6** (organizers').

Measurement 2 — ablation, 909/1642 (55.4%) utterances affected:

| | full | ablated | delta |
|---|---|---|---|
| F1 | 0.4406 | 0.4574 | **+0.0168** |
| precision | 0.3086 | 0.3211 | +0.0126 |
| recall | 0.7702 | 0.7944 | +0.0243 |
| FR rate | 0.1241 | 0.1196 | −0.0045 |

False rejections 6,180 → 5,858: **322 fewer, 5.2% of all FR**. On the
organizers' checkpoint, 6,207 → 5,868, 339 fewer, 5.5%.

#### Observations

1. **It is the recipe, not our run.** The two checkpoints agree to within
   0.7 points on every row of measurement 1. This is a property of the
   pausal CTC target that both share.
2. **The reference is not a prescriptive fiction.** `A` — the human
   annotation of what the reciters actually produced — tracks `C` to within
   0.7 points. The reciters do realize the case endings. The model does not
   predict them. The mismatch is model-side.
3. **It is waqf, not CTC end-truncation.** Generic truncation would depress
   every vowel-final class. Instead short-vowel-final collapses
   (55.4% → 11.4%) while long-vowel-final *rises* (16.7% → 21.3%). Dropping
   short final vowels and keeping long ones is precisely pausal form: the
   short vowel is the i'rab, the long vowel belongs to the stem. This is the
   observation that rules out the main rival explanation.
4. **The distributional fact is dramatic; the metric cost is not.** A
   39-point class gap converts to roughly +0.017 F1 and ~5% of FR. A
   class-count bound agrees: ~39% of 1,642 ≈ 640 utterances differ in final
   class ≈ 10% of FR, so 5–10% is the defensible range.

#### Conclusion

Hypothesis **confirmed in direction, over-stated in magnitude**. Before
running this I told the user the share might be ~27% of the benchmark's
false rejections. It is 5–10%. That is the second time this week a
qualitative reading got ahead of a measurement that was cheap to take (cf.
the U+0670 note), and the same rule applies: quantify before writing the
number down.

What survives is still worth reporting. The field's flagship Arabic MDD
baseline is **trained pausally and scored prescriptively**, a convention
mismatch visible in its own released checkpoint's output, accounting for
5–10% of its published false rejections. That is a methodological finding
about the benchmark, not only a nuisance parameter for us.

**Decision taken (23 Sep 2026): §3.4.1 option 4.** Keep both arms'
conventions; restrict RQ3's cross-arm claim to non-final positions; report
the utterance-final position separately; and publish the mismatch as a
finding. Options 1 and 2 both spend the 0.4414 reproduction anchor — the
project's hardest-won asset — on a secondary RQ. Option 4 costs one table
that now exists. MSA-arm `C_gold` therefore stays **pausal**, matching the
model; no re-derivation of QuranMB and no retrain on prescriptive targets.

The claim, as it should appear in the paper:

> The field's flagship Arabic MDD baseline is trained pausally and scored
> prescriptively. The mismatch is visible in the organizers' own released
> checkpoint — 72% of references end on a vowel, 33% of its predictions do —
> and accounts for 5–10% of its published false rejections. Cross-corpus MDD
> comparison in Arabic is silently confounded by case-ending convention.

Two guardrails on the write-up, both load-bearing: do not exceed the 5–10%
figure (the 39-point distributional gap is not the metric effect — that
conflation is exactly the error made above), and do not imply the benchmark
is invalidated (5–10% of FR is ~0.017 F1: material, not decisive).

---

### Experiment: 50-utterance listening pilot, MSA arm (step 4)

#### Hypothesis

Common Voice Arabic `dev` carries a non-trivial rate of genuine
mispronunciation. If it does (≥3% of words), the MSA arm can host a real
detection task; if it is ~0, the arm can only support a false-alarm study.
Pre-registered in `docs/annotation-protocol-pilot.md` §8 before any audio
was heard.

#### Methodology

Blinded manual annotation of the 50-utterance stratified sample
(`run/msa_pilot_sample.json`) via `scripts/pilot_annotate.py`, which hides
the model prediction, the disagreement score and the stratum. Verdict per
utterance (`clean`/`error`/`unusable`), plus per-word tags for errors. Only
the 25 `uniform_random` rows estimate a base rate; the 25
`high_disagreement` rows are diagnostic. Scored with
`scripts/pilot_baserate.py` (Clopper–Pearson exact intervals).

#### Configuration

- Annotator: 1 (the author), single pass, no second rater → no IAA
- `run/msa_pilot_annotated.json`, 50/50 complete
- Headline = **lenient** word-level rate on the uniform stratum, per §8
- `dialect`, `case`, `hesit`, `refbad` excluded from lenient

#### Results

Verdicts: 17 clean, 22 error, 11 unusable.

Base rate, uniform stratum, 18 usable utterances / 86 words. The headline
depends on where the `case` tag is placed, and the plausible definitions
straddle §8's 3% boundary — so all four are reported:

| definition | k/n | rate | 95% CI |
|---|---|---|---|
| A. count everything (strict) | 17/86 | 19.8% | [12.0%, 29.8%] |
| B. drop `refbad` only (case + dialect count) | 5/86 | 5.8% | [1.9%, 13.0%] |
| C. drop `refbad` + `dialect` (**case counts**) | 4/86 | 4.7% | [1.3%, 11.5%] |
| D. drop `case` too — **pre-registered headline** | 2/86 | 2.3% | [0.3%, 8.1%] |

Where the 17 strict words sit: `refbad` 12, `case` 2, `sub` 2,
`dialect`+`refbad` 1.

QuranMB FR rate for scale: 12.41%. Enriched stratum, diagnostic only:
3/93 = 3.2% under definition D.

Tag distribution over all usable rows (words): `refbad` 23, `case` 13,
`dialect` 2, `sub` 2, `word` 1, `del` 1, `unclear` 1.

Two further rates, neither of which the pilot was designed to measure:

| | k/n | rate | 95% CI |
|---|---|---|---|
| utterances where the **vowelizer** is wrong | 17/39 | 43.6% | [27.8%, 60.4%] |
| unusable (any reason) | 11/50 | 22.0% | [11.5%, 36.0%] |
| Qur'anic verse in the "MSA" corpus | 5/50 | 10.0% | [3.3%, 21.8%] |
| pre-vowelized source text | 10/50 | 20.0% | [10.0%, 33.7%] |

#### Observations

**The headline lands in the MARGINAL band and does not resolve it.** 2.3%
sits in §8's 1–3% band, but the CI reaches 8.1% — the `≥3%` band above is
not excluded, and neither is a rate below 1% at the bottom. The pilot was
powered for ~142 words (25 utterances); 7 uniform rows were unusable, so it
delivered 86. The design lost ~40% of its power to corpus quality, and the
pre-registered decision cannot be taken on this evidence alone.

**The `case` decision moves the answer across the decision boundary, and
that is the more informative result.** Counting dropped mid-sentence case
endings gives 4.7% (`≥3%`, proceed); excluding them gives 2.3%
(`1–3%`, marginal). §8 pre-committed that a strict/lenient split across
bands is itself the finding, and it is: **the MSA "mispronunciation" rate
is dominated by case-ending convention, not by mispronunciation.** The two
readings answer different questions — *did the human mispronounce?* (2.3%)
versus *will the detector see a mismatch?* (4.7%) — and only the second is
what an MDD system actually responds to, since it compares phonemes to `C`
and cannot tell a register choice from an error. Neither reading is
separable at n=86: every interval above spans 3%. This also ties the MSA
arm directly to the §3.4.1 case-ending finding on QuranMB, which is now a
cross-arm result rather than a Qur'anic-only one.

**The speaker is rarely the problem; the label path usually is.** Only 2
words in the entire uniform stratum are genuine mispronunciations (`sub`).
Against that, the automatic vowelizer is wrong on **43.6% of usable
utterances**. The ratio is roughly 10:1 in words (23 `refbad` vs 2 `sub`).
This is the project's central claim arriving from an experiment aimed at
something else — and it arrives much larger than the ~12% DER figures the
diacritizer literature reports, because utterance-level exposure compounds
per-word error.

**Three specimens worth keeping.** `72328`: ى written as ي, so the
vowelizer read *layl-ī* ("my night") for *Laylā* (the name); `phoneme_ref`
contains `l a y l ii`, so a correct speaker is scored as mispronouncing.
`69706`: genitive tanwin on the sentence subject (*Tarāmub-in*) plus a lost
shadda (*-iyan* for *-iyyan*), both inherited by the canonical. `65037`
(from step 3): passive verb + nominative object, 2 of 4 words wrong. All
three are false rejections manufactured by the reference, indistinguishable
in the metric from real mispronunciation.

**The corpus is contaminated for a two-arm design.** 10% of the sample is
Qur'anic recitation and 20% arrives already vowelized — in the corpus that
is supposed to be the *non*-Qur'anic arm. The four fully pre-vowelized rows
were excluded correctly (the vowelizer passes the input's own diacritics
through unchanged, so there is nothing to evaluate); the six partially
vowelized rows were scored. That split is principled, but it means ~20% of
the MSA arm does not exercise the diacritization path at all.

**The exclusions are not all pre-registered, and one direction favours us.**
§3 of the protocol defines `unusable` as audio/text defects; only 5 of the
11 exclusions are that. Five are Qur'anic (a domain criterion §1 already
implies, applied during annotation rather than before it) and one is a
fully-vowelized hadith. The pre-vowelized exclusion is defensible — on
those rows the pipeline passes the input's own diacritics through verbatim,
so the vowelizer never ran and cannot be credited with getting them right —
but it **raises** the headline: 43.6% excluded versus 39.5% if those rows
are included and counted correct. A post-hoc exclusion that inflates the
key statistic is exactly what a reviewer looks for, so both numbers travel
together from here. The conclusion is insensitive to it (~4 in 10 either
way, against 2 mispronounced words). Base-rate sensitivity for the one
debatable row is negligible: 2.33% → 2.17% if it returns clean. Full table:
`docs/annotation-protocol-pilot.md` §9a.

**Integrity checks, all passing.** 50/50 annotated, no duplicate ids, no
word assigned to two tag groups, no out-of-range indices, `error_words` in
sync with `errors`, every `unusable` carrying a note. The vowelizer does
not corrupt consonant skeletons (0/50 beyond punctuation). Three rows have
a word-count mismatch between plain and vowelized text so the bracket hints
did not align during annotation — all three are `clean`, so no word index
was affected.

#### Conclusion

MARGINAL, and the confidence interval does not permit the pre-registered
decision to be taken. Do **not** commit to option 1 on 86 words. The
`case` definition alone swings the answer from 2.3% to 4.7%, i.e. across
the threshold the decision hangs on, so the definition would have to be
settled before any scaling — and settling it is a modelling choice, not
something more annotation resolves.

What the pilot did establish, with more force than its own headline: the
MSA arm's dominant error source is the reference, not the speaker — 43.6%
of utterances versus 2 mispronounced words. That is an argument for
reframing the MSA arm as a false-alarm / label-noise study rather than
scaling the annotation to chase a detection base rate that may not exist.
Corpus contamination (10% Qur'anic, 20% pre-vowelized, 22% unusable) has to
be filtered before any larger annotation round, and filtering will shrink
the usable pool further.

Single annotator, one pass, no inter-annotator agreement. Every number here
is provisional on that.

---

## Planned experiments

Placeholders — fill in with the standard Hypothesis → Methodology →
Configuration → Results → Observations → Conclusion structure as each runs.

### Experiment: Phonetizer round-trip on QuranMB.v2

The week-4 exit criterion. `phonetize(reference_arabic_string)` against the
dataset's own `reference_phoneme_string`, 1,642 utterances, target ≥99%
exact sequence match with every mismatch explained.

### Experiment: Phoneme-inventory diff, MSA arm

Whether each diacritizer + phonetizer path emits phonemes outside the
68-token `sws_arabic.txt` vocab. Records the risk-(a) decision: extend the
vocab, or map/drop.

### Experiment: Audio-informed vs text-only diacritization on Common Voice

Decides the annotation starting point for week 5. One hour, per plan §Data.
