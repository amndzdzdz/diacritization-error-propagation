# Week 5 — Label path, RQ1, annotation start

Plan: [docs/weeks/week-05.md](../docs/weeks/week-05.md). Update this file as
each task below produces a result — do not wait until the end of the week.

## Logistics log

Track non-experimental tasks here as they close. One line each, dated.

- [x] 2026-10-03 — **WikiNews gold set located and licence resolved** (task 2,
      first half). `CAMeL-Lab/wild_diacritics` → `data/WikiNewsMax.tsv`, from
      "Arabic Diacritics in the Wild" (ACL 2024). Data **CC-BY-SA-4.0**, code
      MIT; the repo asks that work using it additionally cite Darwish et al.
      (2017) for the original WikiNews set. This closes the only OPEN item in
      [annotator-qualification.md](../docs/annotator-qualification.md) §5.
      Experiment below. Administering the test is *not* done and is the part
      that actually gates annotation.
- [x] 2026-10-03 — **Qualification size signed off at n = 20, and the packet is
      built** (task 2, second half, step 1 of 4). §6's correction was in the
      right direction: `scripts/build_qualification_test.py` (seed 20261003,
      drawn from the 293-segment band) realizes **1,259 bucket-A** and **326
      bucket-B** marked positions against §6's ~1,067/~202 targets — both met
      with margin, at 427 Arabic words instead of 70 segments' ~1,330. §7's
      three stale "70"s corrected to 20 so the committed doc matches the packet
      being executed. §9's pass marks deliberately untouched. Outputs in
      `run/qualification/`: `blank.txt` + `instructions.md` to send,
      `gold.json` withheld. **Still open: actually administering it.**
      Two incidental validations worth more than the packet:
      `strip_diacritics(original_gold)` reproduces the file's own `dediac`
      column on **all 16,163 word rows, 0 disagreements** — a third
      independent check on `normalize.py`, after the 500-row draw regression
      and the DER ordering agreement — and the earlier "558 segments" figure
      reconciles with a raw 596 (558 contain Arabic; 293 in band either way),
      so neither number was wrong.
- [ ] **Share-alike consequence to decide.** CC-BY-SA-4.0 is viral over
      derivatives. A qualification set redistributed as stripped-gold pairs is
      a derivative; plan §Data contribution 5 wants a releasable annotated
      artifact. The qualification material and the 500-row MSA block are
      separate corpora (`Iqra_train`/CV-Ar, not WikiNews), so the block itself
      is unaffected — but if qualification submissions ship alongside it they
      carry CC-BY-SA. Decide before publishing anything, not after.
- [ ] **Not vendored into the repo.** 825,282 bytes,
      `sha256:a9cc8f7725aa88529a582bbe655a522b384cbbd6b257fb8d72986d17edfdbeb2`,
      fetched 2026-10-03 from
      `raw.githubusercontent.com/CAMeL-Lab/wild_diacritics/main/data/WikiNewsMax.tsv`.
      Hash recorded here so the calibration is reproducible; committing the
      file is a licence decision, deliberately left open above.

## Completed experiments

### Experiment: four-diacritizer DER/WER sweep and the CATT variant decision (tasks 3, 8)

#### Hypothesis

Two questions, one run.

**Task 8:** which CATT model anchors the annotation. The rule was fixed in
writing *before* the measurement: choose on `ce_all` DER, ED only if it leads
EO by **> 0.3 absolute**, otherwise EO, with week 4's phoneme-conservation
tail as the declared tie-breaker. Prior from week 4: EO leads on conservation
(0.08% vs 1.04%), so expect EO, probably inside the margin — i.e. expect the
tie-breaker to do the work.

**Task 3's validation:** the plan's pre-registered sanity check — does the DER
ordering reproduce the week-4 conservation ordering and the NAACL Table 1
ordering? Two independent instruments already agree; a third disagreeing means
the new metric is wrong, not the field.

#### Methodology

`der(reference, tool(strip_diacritics(reference)))` for all four registry
tools, all four variants, micro-averaged over positions.

Two deliberate choices about the input:

- **The input is the stripped reference, not the dataset's `sentence` column.**
  A fifth of `sentence` arrives already vowelized (§4.1), and stripping the
  reference guarantees the tool is scored on the same word sequence it was
  asked about — so a skeleton mismatch means the tool *edited the text*, not
  that two columns disagree.
- **The reference is `tashkeel_sentence`, which is not gold.** It is the
  organizers' vowelizer's output, measured wrong on 43.6% of pilot utterances
  (§3.3a). So these are **agreement rates, not accuracy rates**, and the
  artifact records `reference_is_gold: false` rather than relying on anyone
  remembering.

#### Configuration

- `uv run python scripts/measure_der.py --split dev --write` → `run/msa_der.json`.
- `IqraEval/Iqra_train` dev, 2,588 rows with a non-empty reference.
  `ce_all`: 60,691 positions / 14,490 words.
- `catt-tashkeel==1.0.2` (pinned, runtime-checked), `shakkala==1.7` (via a
  Python 3.10 worker), `mishkal==0.4.1`. Provenance captured per tool inside
  the artifact.
- CPU only, 2026-10-03.

#### Results

**`catt-eo` is chosen, and the margin never came into play — EO leads by
4.556 absolute, 15× the 0.3 threshold.**

`ce_all` (the headline variant, CATT's own convention):

| tool | DER | WER | position errors |
|---|---|---|---|
| `catt-eo` | **1.346** | 4.852 | 817 / 60,691 |
| `catt-ed` | 5.902 | 18.923 | 3,582 / 60,691 |
| `shakkala` | 10.486 | 31.656 | 6,364 / 60,691 |
| `mishkal` | 23.035 | 62.160 | 13,980 / 60,691 |

All four variants, DER:

| tool | `ce_all` | `noce_all` | `ce_markedonly` | `noce_markedonly` |
|---|---|---|---|---|
| `catt-eo` | 1.346 | 0.996 | 1.580 | 1.146 |
| `catt-ed` | 5.902 | 4.353 | 6.835 | 4.807 |
| `shakkala` | 10.486 | 8.560 | 12.407 | 9.924 |
| `mishkal` | 23.035 | 16.941 | 27.062 | 19.075 |

**Sanity check passes.** The DER ordering `eo < ed < shakkala < mishkal` is
*exactly* week 4's conservation-tail ordering (0.08 < 1.04 < 1.39 < 5.14),
from an instrument sharing no machinery with DER, and preserves NAACL Table 1's
Mishkal-far-below-Shakkala. `mishkal` is 17× `catt-eo`, matching week 4's
"order of magnitude" prediction.

**Zero skeleton mismatches, all four tools, all 2,588 rows.**

#### Observations

1. **The prior was right in direction and badly wrong in magnitude.** Week 4's
   conservation tail separated EO and ED by 0.96 percentage points and I
   expected a near-tie on DER; the actual gap is 4.56 and ED's *WER is 3.9×*
   EO's. The two instruments agree on rank but not on how close the race is,
   because they measure different things — conservation detects a tool
   *declining to emit*, DER detects it emitting the *wrong* mark. ED emits
   confidently and wrongly, which conservation cannot see. **Recorded because
   the tie-breaker was designed for a race that did not happen**; if a future
   decision leans on the conservation tail as a proxy for accuracy, this is the
   counter-example.
2. **Mishkal's U+0001 word-merging, which justified refusing to align, did not
   fire once here.** Task 3's refusal-to-align design was argued from Mishkal
   merging words; on this corpus `skeleton_mismatch` is 0/2,588 for every tool.
   The justification was anticipatory and is so far unexercised. It is still
   the right default — a silent alignment would convert text-mangling into a
   slightly worse DER — but the honest statement is "the guard has not yet
   caught anything", not "it was necessary". The Qur'anic arm is where it is
   likely to bite.
3. **An internal consistency check the numbers pass.** `ce_all` positions minus
   `noce_all` positions is 60,691 − 46,201 = **14,490 = exactly the word
   count**. Dropping case endings must remove exactly one position per word,
   and it does. Cheap, and it would have caught an off-by-one in the
   word-final logic.
4. **`markedonly` is *harder* than `all` for every tool, not easier.** DER
   rises (1.346 → 1.580 for EO; 23.0 → 27.1 for Mishkal) while WER falls. The
   denominator shrinks faster than the error count, which means **errors
   concentrate on letters the reference chose to mark** — unsurprising in
   hindsight (an unmarked reference letter is usually one where nothing is
   predictable to get wrong) but it inverts the naive reading of "forgiving"
   variant. Any paper text calling `markedonly` the lenient variant would be
   wrong.
5. **Case endings carry a disproportionate share of the error.** They are
   14,490/60,691 = **23.9% of positions** but, for every tool, a far larger
   share of errors: `catt-eo` (817 − 460)/817 = **43.7%**, `catt-ed` 43.9%,
   `shakkala` 37.9%, `mishkal` 44.0%. Roughly 1.6–1.8× over-represented, and
   stable across a 17× spread in overall DER. This is independent support for
   D3's decision to score case endings separately in the qualification test,
   and it is why reporting a single case-endings-included DER hides where the
   difficulty actually is.
6. **`catt-eo`'s 98.65% position agreement with a reference known wrong on
   43.6% of utterances is not a contradiction, but it is not yet explained.**
   At WER 4.85% and 5.60 words/utterance, ~24% of utterances contain at least
   one EO/vowelizer disagreement (estimated, not measured — the script reports
   corpus rates, not a per-utterance histogram). So the vowelizer is wrong on
   43.6% while CATT disputes only ~24%: compatible in magnitude, and it
   implies CATT **agrees with the vowelizer at positions where the vowelizer is
   wrong**. Either they share error modes or the organizers' vowelizer is
   CATT-derived. This matters for D5 — if the two are correlated, `C_auto` vs
   `C_gold` will understate the error-propagation headline. Open; needs the
   per-utterance histogram and the annotated block.

#### Conclusion

**Task 8 closed: `catt-eo`, decisively, recorded as D7 in `docs/msa-arm.md`
with a date.** The pre-committed rule was applied unchanged and the outcome is
written down whichever way it went, so a reviewer asking "why EO?" gets an
answer that predates the number.

**Task 3's metric is validated against two independent instruments.** The
ordering agreement across DER, length conservation and an external paper's
table is the strongest available evidence that `metrics/der.py` measures what
it claims.

Not closed: observation 6's correlation between CATT and the organizers'
vowelizer is now the most consequential open question for D5, and it was not
visible before this run.

---

### Experiment: DER/WER metric — four variants, and what the metric refuses to do (task 3)

#### Hypothesis

Not a measurement, a construction, so the hypothesis is about the design:
**"DER" is not one number, and any single reported figure silently fixes two
free choices.** If the module exposes one function the paper will end up
comparing our numbers against published ones that made the other choice. The
claim under test is that the two axes are *separable and both matter* — i.e.
that the four variants disagree by more than rounding on real data.

#### Methodology

Two axes, fixed as a 2×2 rather than a flag soup:

1. **Case endings in or out.** `Variant.case_endings`. The field is
   deliberately *not* called `irab`: what we can actually detect is the mark on
   the last letter of a word, which is a positional proxy for i'rab, not i'rab.
   Naming it `irab` would promise a syntactic analysis the code does not do.
2. **Denominator.** `score_unmarked_reference`. `all` scores every letter;
   `markedonly` scores only letters the reference itself marks, which forgives
   an omission the reference also makes.

Alignment is **refused, not approximated.** If the hypothesis's consonantal
skeleton differs from the reference's word-for-word, the utterance is counted
in `skeleton_mismatch` and contributes *zero* positions. Rationale: Mishkal
emits U+0001 and merges words, so an edit-distance alignment would quietly
convert a tool that mangled the text into a tool with a slightly worse DER —
exactly the failure the paper is about. Better to report the count.

Scored marks are U+064B–U+0652 (harakat, tanwin, shadda, sukun) plus the
dagger alef U+0670. Marks are held as a **set** per letter, so
`<shadda><fatha>` and `<fatha><shadda>` are the same annotation — NFC does not
normalize that order, and tools disagree on it.

The second half of the task was a refactor: `strip_diacritics` and `skeleton`
lived in `msa_pool.py`, and the annotation block was drawn with them. DER needs
the same stripping step, so they moved to `data/normalize.py` and `msa_pool`
now re-exports them.

#### Configuration

- New: `src/arabic_mdd/metrics/der.py`, `tests/metrics/test_der.py` (26 tests).
- Moved into `data/normalize.py`: `DIACRITICS`, `is_arabic_letter`,
  `strip_diacritics`, `skeleton`.
- New: `tests/data/test_annotation_block_regression.py` (3 tests), which
  rebuilds the pool and the seeded draw from `run/iqra_dev_text.parquet` and
  compares field-for-field against the committed `run/msa_annotation_block.json`.
- Driver: `scripts/measure_der.py`. Smoke run `--tools catt-eo --limit 40`.

#### Results

The 2×2 is not decorative. On the first 40 dev rows with `catt-eo`:

| variant | DER | WER |
| --- | --- | --- |
| `ce_all` | 1.515 | 4.902 |
| `noce_all` | 1.223 | 2.956 |
| `ce_markedonly` | 1.637 | 3.922 |
| `noce_markedonly` | 1.331 | 2.463 |

Spread 1.22–1.64 DER and 2.46–4.90 WER on identical outputs. **The WER spread
is 2× — a factor larger than any inter-tool difference the paper expects to
report.** Zero skeleton mismatches for `catt-eo`, as expected for a tool that
only inserts marks.

The refactor is confirmed inert: the regression test reproduces
`pool_total` 2,588, `pool_eligible` 2,092, every exclusion and flag count, and
all 500 drawn rows **in order**, with the crossover sub-block still at ranks
0–49. Full suite 168 passed, ruff clean.

#### Observations

1. **One design bug, found by the Qur'anic test case.** U+06D6 (a Qur'anic
   annotation mark) is not in `SCORED_MARKS`, so the first `marked_letters`
   treated it as a *letter* — while `strip_diacritics` removes it. The two
   functions' letter sequences desynchronized, and an utterance whose skeleton
   check *passed* then raised `IndexError` on a length mismatch. Fixed by
   ignoring anything `strip_diacritics` removes that `SCORED_MARKS` does not
   cover, and locked by a parametrized invariant:
   `"".join(letters) == strip_diacritics(word)`. Worth recording because the
   bug was only reachable from the Qur'anic arm; an MSA-only test set would
   have shipped it.
2. **`Variant` has no default.** Constructing one requires naming both axes.
   This is the enforcement mechanism for the hypothesis above — there is no way
   to get "the" DER without a reviewer being able to see which one you asked
   for.
3. **`Score.__add__` raises across variants.** Micro-averaging pools positions;
   pooling two different definitions' positions is meaningless, so it is an
   error rather than a silent sum.
4. **Micro-averaged over positions, not averaged over utterances.** A 3-word
   utterance and a 30-word utterance are not equal evidence. This matters for
   the MSA corpus specifically, whose length distribution is wide (the pool's
   8–40-word eligibility band was itself a week-4 construction).
5. **`DIACRITICS` is deliberately wider than `SCORED_MARKS`.** Stripping should
   be aggressive — it defines the skeleton, and an unstripped exotic mark
   becomes a false skeleton mismatch. Scoring should be narrow. Collapsing the
   two sets would couple an input-cleaning decision to a metric definition.

#### Conclusion

DER/WER exists with both axes exposed and alignment refused. The four-way
spread on 40 rows is large enough that **reporting a bare "DER" in the paper
would be a reviewable defect**, which retroactively justifies the 2×2 over a
single default. The refactor is byte-identical on the drawn block, so the
annotation material is unaffected.

Open: the headline variant for RQ1-MSA is `ce_all` (CATT's own convention, so
our numbers are comparable to theirs), but `markedonly` is the honest variant
against a *vowelizer-output* reference, which is all `Iqra_train` offers before
the MSA block is annotated. Both are computed and stored; the choice of which
leads the table is a task-4/D5 decision, not this one.

---

### Experiment: full `Iqra_train` train-split phonetizer round-trip (task 9)

#### Hypothesis

Pure regression, no new judgment. Week 4 validated the phonetizer on
`Iqra_train` **dev** (2,588 rows, 99.85% exact) and explicitly flagged that
the 71,391-row train split had not been run. Expectation: a rate close to
99.85%, with no new failure class.

#### Methodology

`phonetize(tashkeel_sentence) == phoneme_ref`, pausal convention, exact
sequence match, over the whole train split. Then every mismatch classified by
cause rather than counted.

Running it required one fix: `load_iqra_train` did not drop the `audio`
column, so iterating train rows tried to decode waveforms and raised
`ImportError: ... install 'torchcodec'`. `quranmb.load_ground_truth` already
guards against this for the same reason; the guard is now in both.

#### Configuration

- `scripts/check_phonetizer_roundtrip.py --split train --skip-quranic`
  (`--split` / `--skip-quranic` added this week; dev remains the default, so
  the week-4 invocation is unchanged).
- `IqraEval/Iqra_train`, split `train`, 71,391 rows, local HF cache, 2026-10-03.

#### Results

**99.93% exact sequence match — 71,344 / 71,391. PASS against the ≥99%
criterion, and *better* than dev's 99.85%.** Token error rate 0.0000. No
out-of-inventory token produced.

All **47** mismatches are one phenomenon: **a shadda written on a long-vowel
carrier**, which is malformed Arabic orthography in the source text.

| sub-pattern | rows | example word |
|---|---|---|
| `<alef><shadda>` | 44 | `إِلاَّ` (25), `إِلَّاَّ` (11), `إِلّاَّ` (6), `طُلَّاَّبِهَا` (2) |
| `<alef maksura><shadda>` | 3 | `أَىَّ`, `أَىُّ` |

The 47 rows carry only **20 distinct sentences**; the rest are repeats. In
every case the correct spelling puts the shadda on the preceding consonant
(`إِلَّا`, lam+shadda, then alef), and the source instead attaches it to the
alef. We emit `< a` where `phoneme_ref` has `aa a`.

#### Observations

1. **No new failure class, and the one class is upstream.** The input is
   malformed, so neither output is "correct" — `phoneme_ref` reflects how the
   organizers' vowelizer resolved a shadda it should never have seen, and we
   resolve it differently. This is not a phonetizer defect to fix, and
   "fixing" it would mean reverse-engineering another tool's handling of
   invalid input. Recorded as a known 0.066% divergence instead.

2. **Train is cleaner than dev, which is worth not over-reading.** 99.93%
   versus 99.85% is 47 rows in 71,391 against 4 in 2,588. The difference is
   small-sample noise in dev's denominator, not evidence that train is better
   curated. The useful conclusion is only that the caveat is closed: no
   failure mode was hiding in the 96% of the corpus week 4 had not looked at.

3. **Two of the 500 annotation-block rows contain the malformed pattern** —
   rank 447 (id 70588, `سَأُعْطِيكَ أَيَّ شَيْءٍ إِلَّاَّ هَذَا`) and rank 475
   (id 73438). Both are in the main block, neither in the crossover sub-block
   (ranks 0–49). This is a concrete instruction gap for week-05 task 4: an
   annotator writing `C_gold` for rank 447 will either reproduce the malformed
   shadda or silently correct it, and the two choices give different phoneme
   strings. The protocol should say which — most defensibly, write correct
   orthography and record the deviation, since `C_gold` is supposed to be
   correct diacritization. 2 rows of 500 is small, but it is exactly the kind
   of unstated case that produces an unreproducible mid-stream decision, which
   §3.3a already had to declare once.

#### Conclusion

Caveat closed, criterion passed at 99.93%, no new failure class, and the
single class is an upstream orthography error with a bounded 0.066% footprint.
One loader bug fixed as a precondition. One protocol gap found and handed to
task 4.

### Experiment: QuranMB.v2 `ID` scheme — task 1 route (a)

#### Hypothesis

Task 1 route (a): if clip `ID`s carry surah:ayah references, verse recovery is
a lookup and the keystone task collapses to a verification step. Pre-declared
as fifteen minutes, to be run before writing any matcher.

#### Methodology

Read-only inspection of the cached `safikhan/quran_mbv2_formatted` split
(1,642 rows, audio column dropped — decoding needs `torchcodec`, which is not
installed and is not needed here). Decomposed `id` on `_`, tested each field
for functional dependence on `reference_phoneme_string` and
`reference_arabic_string`. No matcher written, no external text fetched.

#### Configuration

- `datasets.load_dataset("safikhan/quran_mbv2_formatted")["train"]`, local HF
  cache, 2026-10-03.
- Grouping compared on three candidate keys: `id` field A, `id` field B, and
  `B[:5]`.

#### Results

**Route (a) fails for its stated purpose.** `id` is
`AAAAA_BBBBB[suffix]` and contains no surah or ayah reference:

| field | meaning | evidence |
|---|---|---|
| A | speaker, 18 values `00000`–`00017` | 35–99 rows each; each speaker covers nearly the full prompt set |
| `B[:5]` | **flat prompt index, `00000`–`00097`** | determines `reference_phoneme_string` exactly: 0 of 98 map to >1 string |
| `B[5:]` | take/repeat suffix | empty on 1,142 rows, 1–3 chars on 500; 389 distinct of 500 |

396 of 1,143 (speaker, prompt) pairs have more than one row, so the suffix is
a retake counter. Verse recovery still needs route (b) or (c).

**The grouping unit is 98, not 96.** The three counts differ and the
differences are not rounding:

```
98 prompt indices  →  97 distinct phoneme strings  →  96 distinct Arabic strings
```

Both collapses are a single pair each, and they are different kinds of pair:

- **Prompts 73 and 91** share Arabic *and* phoneme string
  (`فَوَلِّ وَجْهَكَ شَطْرَ الْمَسْجِدِ الْحَرَامِ`). A genuine duplicate
  prompt.
- **Prompts 9 and 42** share Arabic
  (`وَكَذَٰلِكَ جَعَلْنَاكُمْ أُمَّةً وَسَطًا`) but have **different phoneme
  strings**: `... w a s a T A n aa` (prompt 9, n=8) versus
  `... w a s A T A n` (prompt 42, n=18). Same verse, two recitation
  realizations of the final word — the pausal/liaison distinction, on the
  Qur'anic arm.

#### Observations

1. **The 9/42 pair puts a hard ceiling on task 1's own validity criterion.**
   Task 1 declares `phonetize(recovered_verse) == reference_phoneme_string`
   the independent check that makes routes (a)/(b) genuine evidence rather
   than a reproduction. But `phonetize` is deterministic: one Arabic string
   yields one phoneme string. Prompts 9 and 42 have the same Arabic string and
   different phoneme strings, so **the check cannot pass for both, no matter
   how correct the recovery is.** Maximum achievable is 97/98. This is a
   property of the data, not of our pipeline, and it must be subtracted
   *before* the measurement rather than explained after — otherwise one
   guaranteed mismatch gets read as a recovery failure and sends someone
   debugging a matcher that is working.

2. **Deduplicating to 96 destroys the pausal signal.** Keying the verse map on
   Arabic text merges 9 with 42 and silently discards the fact that the
   corpus contains the same verse read two ways. That distinction is the
   Qur'anic-arm analogue of the §4 pausal convention, and week 4 already
   measured the convention as the single largest source of word-final
   disagreement. The map should therefore be keyed on **prompt index (98
   entries)**, recording the verse, with the two collision pairs annotated.
   Prompt index is also the safer key on its own merits: it is metadata, so
   it is not downstream of the uploader's phonemisation the way the Arabic
   column is.

3. **Independent corroboration of the pausal finding, from the other arm.**
   Task 2's WikiNews audit found liaison-vs-pausal to be the largest
   divergence class between two expert MSA golds (above). This is the same
   phenomenon appearing in Qur'anic recitation as two distinct prompts. Two
   unrelated corpora, same conclusion: the convention cannot be left to a
   judgment call, which is what week-05 task 4 already asserts and now has
   two-arm evidence for.

4. **Exit criterion 1's denominator needs restating.** It reads "≥95 of the 96
   distinct QuranMB.v2 sentences". On the corrected unit that is ≥95 of 98
   prompts, and given observation 1 the phonetize-based confirmation tops out
   at 97/98. Both numbers should be fixed before the measurement runs, since
   fixing them afterwards is indistinguishable from moving the goalposts.

5. **Speaker 16 has 99 rows against a 98-prompt set**, and speakers 3 and 12
   have 50 and 35. Uneven coverage is expected for retakes, but the 99 is
   worth a glance if the carried-forward 1,642-vs-1,643 published-count
   discrepancy is ever chased. Not chased here.

#### Conclusion

Route (a) is closed and negative: no verse reference in the `ID`. Task 1
remains the keystone and must proceed to route (b) — the organizers' published
prompt list — which is now a more attractive route than before, because a
published list would be keyed on prompt index and would confirm the 98-prompt
structure independently.

Two corrections fall out, both to be applied before any matcher is written:
the map is keyed on **98 prompt indices**, and the phonetize check's ceiling
is **97/98 by construction**.

### Experiment: WikiNews gold set acquisition and contract audit

#### Hypothesis

Task 2 assumed one artifact serving two purposes — qualification-test material
and RQ1 calibration against published machine DER. Three things had to hold
for that assumption to survive: the set is obtainable, its licence permits
this use, and its text is expressible under the §4.1 normalization contract.
A fourth was latent and unstated: that "the WikiNews gold" is a single
unambiguous reference. It is not, and that is the finding.

#### Methodology

A candidate was offered for identification: `Fumika/Wikinews-multilingual`
(HF). Rejected on inspection — a 33-language parallel news corpus for
cross-lingual sentence embeddings (Isono, Primer AI, arXiv 2305.14256,
CC-BY-2.5). Arabic is one of its 33 languages but the text is raw scraped
article prose with no gold vowelization. Same source website, unrelated
artifact, no diacritization annotation. Recorded because the name collision
is an easy trap and will recur.

The set task 2 actually needs is the hand-diacritized MSA benchmark of Darwish,
Mubarak & Abdelali (2017), which is what CATT's Table 5 reports against. It
has no canonical standalone distribution; it was located inside
`CAMeL-Lab/wild_diacritics`. The file was downloaded, parsed, and every
column passed through `arabic_mdd.data.normalize.unsupported_characters` —
i.e. audited against our own contract rather than taken on description.

#### Configuration

- Source: `CAMeL-Lab/wild_diacritics`, `data/WikiNewsMax.tsv`, branch `main`,
  fetched 2026-10-03. sha256 in the logistics log above.
- Audit tool: `normalize.unsupported_characters`, unmodified.
- Word counts restrict to tokens containing at least one character in
  `U+0620–U+064A`, `U+0670`, `U+0671`; punctuation, Latin and digit tokens are
  counted separately. Segments are delimited by blank rows.

#### Results

**Structure.** TSV, 16,774 body rows: 16,163 token rows, 604 blank
(segment separators), 7 literal `#` rows. Token rows split into **14,613
Arabic-bearing words** and 1,550 punctuation/Latin/digit tokens. 558 segments
contain Arabic; 1 / 8 / 19 / 26.2 / 208 for min / p25 / median / mean / max
Arabic words per segment. 293 segments fall in an 8–40-word band.

**Four columns, not one.**

| column | content | coverage |
|---|---|---|
| `dediac` | stripped input | all |
| `original_gold` | the Darwish et al. 2017 gold — **the reference CATT's 5.425/3.105 is measured against** | 14,613 / 14,613 |
| `max_gold` | maximal re-diacritization under the ACL 2024 standard | 14,613 / 14,613 |
| `max_gold_alt` | second legitimate reading where one exists | 359 (2.46%) |

**The two golds disagree on 544 of 14,613 Arabic words — 3.72%.** Not noise;
four systematic classes:

| class | example | note |
|---|---|---|
| liaison vs pausal case ending | `أَعْلَنَتْ` → `أَعْلَنَتِ`, `مِنْ` → `مِنَ` | our §4 pausal convention |
| dagger alif written explicitly | `هَذِهِ` → `هَٰذِهِ`, `بِذَلِكَ` → `بِذَٰلِكَ` | U+0670 |
| hamzat al-waṣl given its vowel | `اخْتِتَامُ` → `اِخْتِتَامُ` | |
| article alif given its vowel | `التَّاسِعَ` → `اَلتَّاسِعَ` | |

**Contract audit — passes.** `unsupported_characters` over all four columns
returns non-zero entries only on Latin letters, ASCII digits, and punctuation
(`.` `،` `(` `)` `:` `-` `/` `%` `؟` `!` `&` `+` `×`). Those are exactly the
§4.1.1 `UNFAIR` classes (`latin`, `digits`) plus punctuation, which §4.1.1
already routes downstream. No character outside the existing taxonomy appears.
`max_gold_alt` is all-zero, being Arabic-only by construction.

**U+0671 alef wasla: zero occurrences in all four columns.** U+0670 dagger
alif present: 37 in `original_gold`, 163 in `max_gold`, already covered by
`normalize.DAGGER_ALEF`. So §9's live falsifier does not fire here, as
expected for MSA — this set says nothing about the Qur'anic arm, exactly as
week 4's 99.85% MSA round-trip said nothing about it.

**Diacritic density: 58,551 marks in `original_gold`, 4.01 marks per Arabic
word.**

#### Observations

1. **Column choice is the definitional difference exit criterion 2 asks us to
   settle, and it is larger than the tolerance.** 3.72% word-level gold
   disagreement will move corpus DER by more than ±0.5 absolute. Calibration
   must therefore target **`original_gold`**: it is the reference CATT
   measured against, and against `max_gold` the ±0.5 gate would fail for a
   reason with nothing to do with our metric. This is the concrete form of
   plan §2's warning that CATT's and KSAA-2026's numbers may not share a
   definition — the mechanism is now named rather than hedged. Task 3's
   reported number must state the column.

2. **Independent corroboration that the pausal convention must be
   mechanical.** The largest divergence class is precisely the liaison/pausal
   case-ending choice that week-05 task 4 declares "not the annotator's
   problem" (6.52% of diacritic positions). Two expert annotation efforts on
   the same text differ systematically on it. That is external evidence for
   the design decision, obtained for free, and it belongs in the protocol's
   justification rather than as an assertion.

3. **`max_gold_alt` is evidence that genuinely ambiguous positions exist** —
   359 words, 2.46%, where expert annotators recorded two defensible readings
   (`التِّقْنِيَّةِ` / `التَّقْنِيَّةِ`; `وَالتِّلِفِزْيُونِ` /
   `وَالتِّلْفِزْيُونِ` — mostly loanword vocalization). Scoring those as
   disagreements would charge annotators for the language's own
   underdetermination and depress the measured agreement rate. §8's scoring
   and §9's adjudication routing should carry an acceptable-alternatives
   allowance. Cheap here, because the alternatives are enumerated for us on
   this set; on our own corpus they will not be.

4. **The sizing unit is wrong, and it is wrong by ~3.5×.** §6 derives "70
   utterances" from `Iqra_train` dev averaging 19.4 marks per utterance
   (2.89 case endings + 15.24 bucket A), targeting ~202 bucket-B and ~1,067
   bucket-A positions. A WikiNews segment is not an `Iqra_train` utterance: at
   4.01 marks/word and a 19-word median it carries **~76 marks**, about 4×.
   70 segments would yield on the order of 1,300 bucket-B and 4,000 bucket-A
   positions — far past what the power calculation asks for, at ~4× the
   annotator hours. Hitting §6's actual targets takes roughly **11 segments
   for bucket B and ~19 for bucket A**, so ~20 segments from the 293-segment
   8–40-word band satisfies both. §6's targets are a power calculation, not a
   pre-registered threshold, so re-deriving the count in the correct unit is
   legitimate — but it is a decision with a cost attached and is flagged for
   explicit sign-off, not silently applied.

5. **Register caveat, against §5's own criteria.** §5 asks for "short everyday
   prose, as close to Common Voice sentence register as the material allows."
   WikiNews is news prose with a 26.2-word mean and a 208-word maximum
   segment; the 8–40-word band exists and should be the sampling frame, but
   the register is still further from CV-Ar than the reserve option
   (`Iqra_train` rows carrying source marks) would be. The trade is deliberate
   and worth stating: WikiNews buys the calibration, which the reserve cannot,
   at the price of register fidelity. §5's criteria 2 and 3 are satisfied
   (MSA, not Qur'anic/hadith, no memorization risk); criterion 3's "as close
   to CV register as the material allows" is satisfied only by restricting to
   the band.

6. **A count discrepancy to carry, not to resolve now.** The literature
   describes WikiNews as ~18,300 words over 70 articles; we count 14,613
   Arabic-bearing tokens. The gap is plausibly tokenization (clitic splitting,
   punctuation handling) or a subset, and it matters only if the calibration
   misses. If task 3's DER lands outside ±0.5, this is the first thing to
   check before blaming the metric.

#### Conclusion

Task 2's acquisition half is closed: the set exists, CC-BY-SA-4.0 permits the
use, and it is contract-clean under `normalize`. Task 3's calibration target
is **`original_gold`**, stated in the API and in any reported number.

The unstated assumption did not survive. "The WikiNews gold" is two golds
differing on 3.72% of words, and task 2's premise that one artifact serves
both purposes is only half right: the **full** set is needed for calibration
(a corpus-level number must be computed over the corpus the published number
was computed over), while qualification needs a **~20-segment sample** from
the 8–40-word band. Different slices, different sizes, one acquisition. Task 2
should be read as two deliverables from here.

Nothing here touches the Qur'anic arm or the U+0671 falsifier, and task 1
remains the keystone and untouched.

### Experiment: annotator 1's qualification test — scored (task 2/§8)

#### Hypothesis

Pre-registered in `docs/annotator-qualification.md` §9 before any submission
existed, so this is a genuine prediction rather than a description:

1. **Bucket A ≤ 2%.** A literate native speaker should beat CATT's 3.105%
   comfortably on word-internal marks.
2. **Bucket B lands in 10–25%,** i.e. flag-and-adjudicate, because case
   endings need syntactic parsing and the back-of-envelope in §9 puts the
   machine at ~17.6% there.
3. **Overall DER below CATT EO's 5.425%** on the same corpus. This is the
   premise D1 and D5 rest on and it had never been measured
   apples-to-apples — the literature's human-vs-machine comparisons are
   across corpora.

#### Methodology

§7's procedure, unmodified. 20 WikiNews segments drawn by seed 20261003 from
the 293-segment 8–40-Arabic-word band, presented fully stripped of marks,
**from scratch rather than correct-the-machine** (§7 step 2 — deliberately
harder than the real job, so the bar is conservative). Gold never shown.
Scored by `scripts/score_qualification.py`, which implements §8 steps 1–5 and
deliberately refuses to print §9's thresholds so they cannot drift to fit the
result.

#### Configuration

- Packet: `run/qualification/blank.txt` + `instructions.md`, gold withheld in
  `gold.json` (gitignored).
- Submission preserved at `run/qualification/submissions/annotator_1.txt`.
- Reference: `original_gold`, with §8 step 3's `max_gold_alt` allowance.
- Scored positions are **marks**, not marked letters; §4's two free classes
  (tanwin, final word's final mark) removed first.
- Elapsed: 30 minutes, self-reported. 427 Arabic words.

#### Results

| bucket | scored | errors | rate | 95% Wilson |
|---|---|---|---|---|
| A internal | 1,363 | 16 | **1.17%** | [0.72, 1.90] |
| B case end | 293 | 42 | **14.33%** | [10.78, 18.81] |

20/20 items attempted. Zero skeleton alterations, zero word-count flags. The
`max_gold_alt` allowance was invoked on 2 of the 12 words that offer one.
Overall **3.50% DER** against CATT EO's **5.425%** on the same corpus.

All three hypotheses confirmed. Bucket A passes with the *entire* interval
below the bar, not merely the point estimate — which is the form of pass that
does not depend on the sample size argument in §6.

#### Observations

1. **The routing decision is determinate, so §6's extension does not fire.**
   [10.78, 18.81] is interior to the 10–25% band; it touches neither edge. I
   had earlier argued for extending, on the reasoning that bucket B was
   unresolved — that was wrong, and wrong for an avoidable reason: I was
   comparing the point estimate to the band edge instead of the interval. ~90
   minutes of annotator time saved.

2. **16 errors in bucket A is small enough to be read individually, and the
   profile is omission-heavy, not substitution-heavy** (§8 step 5's
   distinction earning its place). Omissions mean "didn't mark it"; a
   substitution-heavy profile would have meant "holds a different theory of
   the word", which is the worse failure and is absent.

3. **13 of bucket B's 42 errors are label-invariant** — a final sukun omitted
   on a foreign proper noun. Verified rather than assumed: phonetizing both
   forms gives the identical label sequence. These cost nothing downstream,
   which means the raw 14.33% overstates the damage.

4. **5 more are a convention mismatch I caused, and finding them refuted my
   own first inference.** I had guessed the 13 sukun cases and 5 liaison cases
   were the same phenomenon — both harmless. Testing it: `phonetize("مِنْ")`
   = `m i n` but `phonetize("مِنَ")` = `m i n a`. Liaison forms **are**
   label-impacting. The guess was wrong and the check caught it.

5. **That refutation exposes a false statement in our own protocol.** §3 told
   annotators they need not learn the pausal convention. True for the
   utterance's final word; **false for every other word**, whose final mark
   survives into the phoneme string. All five liaison errors match `max_gold`
   — the annotator wrote a defensible reading of an under-specified
   instruction. Corrected in §3, and **task 4's protocol must state the
   convention explicitly** or the main block inherits the same 1.7% of
   bucket-B positions as noise.

6. **Net of both non-skill classes, bucket B is 9.90% — which straddles the
   10% edge.** Recorded, and deliberately **not** used for routing. §9's bands
   are defined on the raw rate; re-deriving a route from a post-hoc subset is
   the threshold drift §1 exists to forbid. Noting the temptation because it
   was real: 9.90% would have promoted the annotator to unsupervised case
   endings, which is the more convenient answer.

7. **A side-finding with no bearing on the result: the phonetizer mangles
   loanwords.** `إِنْتَرْ → n i t`. Cause is `_halabi_phonetiser.py`'s
   `fixedWords` lookup at line 223, keyed on the *consonant skeleton*, so
   `إنتر` collides with `أنت` (`fixedWords['nt'] = 'n i1 t'`) regardless of
   diacritics. Invisible to `unsupported_characters`. Measured reach in
   `Iqra_train` dev: **zero** — all 21 colliding surfaces are the intended
   function words. So this is a WikiNews-register artifact, relevant to the
   calibration corpus and not to the arm.

8. **The error profile is itself evidence on the "did they use a tool?"
   question**, which §7 step 4 forbids but cannot enforce. A CATT user would
   reproduce CATT's errors and land near 5.43%, not 3.50%; and omitted final
   sukuns on foreign proper nouns are not a pattern CATT produces. The
   submission is consistent with unaided work.

#### Conclusion

Annotator 1 is **qualified for bucket A unsupervised and flag-and-adjudicate
on case endings**, by the thresholds as written. More importantly the arm's
load-bearing premise is now measured: a native non-specialist beats the
anchor 1.55× overall and 2.65× on the easy majority, on the same corpus with
the same reference. D1's decision to keep `C_gold` and D5's error-propagation
framing both survive.

Two things follow that are not about this annotator. The timing datum —
14.2 words/min ⇒ ~3.1 h for D4's 500-row block, ~12.9 h for the full
2,092-row pool, both upper bounds since the real job is
correct-the-machine — closes D4's deferred volume question favourably. And
§3's false pausal promise is a defect in a *pre-registered* document, found
only because the error analysis went word-by-word; it is now the main thing
task 4 must get right.

### Experiment: clip → verse recovery by consonant-skeleton match — task 1 route (c)

#### Hypothesis

Route (a) closed negative and route (b) was unreachable, so the pre-declared
fallback applies: the 98 QuranMB.v2 prompts can be located in an authoritative
fully-vowelized Qur'an by matching their consonant skeleton, and the located
spans will carry diacritics good enough to serve as the Qur'anic arm's
`C_gold`. Two predictions were written in the week plan *before* running and
are tested here rather than discovered: that the **edition choice** silently
decides whether §9's alef-wasla falsifier fires, and that `skeleton()` will
**fail on Uthmani orthography** because `is_arabic_letter` treats U+0671 as a
letter, so alef wasla survives stripping.

#### Methodology

Tanzil XML downloaded once and kept local (its copyright block asks that it not
be removed, so the text is gitignored and only its sha256 is committed). The
Qur'an is flattened to a single stream of words with (surah, ayah) tags; each
prompt's word-skeleton sequence is matched as an **exact contiguous span**.
Exact, not fuzzy: there is no threshold to tune and therefore nothing to
overfit. All distinct matching spans are kept as candidates.

**The primary candidate is the scripturally first, deliberately not the lowest
phoneme distance.** Selecting by closeness to `reference_phoneme_string` is the
uploader's own matcher, so it would deepen the circularity the task exists to
escape. Ambiguity is flagged for a human instead of resolved by the column
under suspicion.

#### Configuration

- `scripts/recover_quran_verses.py`, output `run/quranmb_verse_map.json`.
- Tanzil **`simple`** edition, XML, sha256 `d9b5bad94e472519…`, 6,236 verses,
  685,174 characters. `imlaei-simple` is a 404 on that endpoint.
- Prompts keyed on **prompt index (98)**, not Arabic text (96) — see the `ID`
  scheme experiment above.
- `Phonetizer("prescriptive")`, matching `reference_phoneme_string`'s
  convention.

#### Results

**Located 98/98.** Unique location 83/98; text unambiguous 96/98; phoneme
exact 54/98 against a ceiling of 97; `reference_arabic_string` agrees 98/98.

**Four earlier framings failed before skeleton-span matching worked**, and the
failures were diagnosed rather than patched around:

| approach | located |
|---|---|
| whole-verse phoneme lookup | 4–8 / 98 |
| nearest-verse Levenshtein | ~20 edits on length-35 — not a match at all |
| substring match on diacritized text | 58 / 98 |
| substring + trailing-token trim | 60 / 98 |
| **skeleton span on the word stream** | **98 / 98** |

The diagnosis that unlocked it: the prompts are **fragments**, not verses.

**Both of the week plan's predictions held, and the edition is the dominant
lever.** Uthmani located **16/98** with 80 outright non-matches; `simple`
located 98/98. Root cause exactly as predicted — U+0671 survives
`strip_diacritics` because `is_arabic_letter` counts it a letter, so every
verse written with alef wasla fails against a bare-alef query.

**The alef-wasla dodge was measured, not assumed** — the mistake week 4 caught
four separate times:

```
characters 685,174 | alef_wasla U+0671: 0 | dagger_alef U+0670: 3,218
quranic_marks U+06D6–U+06ED: 0 | unsupported_characters: {}
```

#### Observations

1. **The circularity worry is empirically clean but logically unchanged.**
   `reference_arabic_string` agrees on 98/98, so the uploader's recovery was
   correct and `match_distance > 0` is phoneme-level residual, not text error.
   This is still a *reproduction*: the same algorithm class, agreeing with
   itself. It raises confidence and proves nothing, and the week plan said so
   in advance.
2. **54/98 exact is not 54% correct.** The 44 inexact are dominated by one
   phonetizer disagreement — the dagger alef (U+0670), which `normalize()`
   drops on MSA evidence (177/177) but which QuranMB's reference realizes as a
   long vowel. `الرَّحْمَٰنِ` phonetizes `m a n` where the reference has
   `m aa n`, so the entire candidate set for prompt 00097 has a **distance
   floor of 1** that no recovery could remove.
3. **Route (b) is closed by evidence, not by effort.** `test_references` is
   ID + phonemes, `QuranMB.v2` is ID + audio, and `IqraEval/Iqra_Extra_IS26` —
   the place the week plan named — is a gated repo returning **HTTP 401**. The
   published join may exist; it is unreachable. Recorded as "unavailable to
   us", not "does not exist".
4. The 97/98 ceiling is structural: prompts 9 and 42 share one Arabic string
   against two phoneme strings, and `phonetize` is deterministic, so one must
   miss whatever is recovered. Subtract it *before* reading the exact count.

#### Conclusion

Route (c) delivers the mapping and the deliverables, and both written
predictions about orthography were confirmed rather than stumbled into. But it
cannot satisfy exit criterion 1's independence clause by construction, since it
consults `reference_arabic_string` to build the query. An instrument outside
both text columns is required, which is the next experiment.

### Experiment: human audio verification of the verse map — the independent half of task 1

#### Hypothesis

The shipped audio is the only artifact in QuranMB.v2 that is **not** downstream
of the uploader's phonemisation: it was recorded, not derived. If a human can
confirm that reciters are attempting the recovered words, route (c) is upgraded
from "a matcher we wrote agrees with a matcher they wrote" to "the recovered
text is what was read". Pre-declared pass mark: **≥95 of 98**.

#### Methodology

`scripts/verify_verse_map.py` plays clips and takes a verdict per prompt.
Three design choices carry the result:

- **Word identity is judged, not pronunciation accuracy.** QuranMB.v2 is a
  *mispronunciation* benchmark — `annotation_phoneme_string` deliberately
  differs from `reference_phoneme_string` — so judging pronunciation would
  reject correct recoveries and leak the benchmark's labels into a text
  decision.
- **Risk order, not prompt order**: ambiguous first, then multi-location, then
  non-exact, then the self-confirming remainder, so stopping early still buys
  the most information per minute.
- **The benchmark's own phoneme string is never displayed.** Picking a
  candidate by matching it would just rerun the uploader's matcher.

#### Configuration

- 98 prompts, ~17 takes each, 1,642 clips; `datasets.Audio(decode=False)` for
  raw WAV bytes with random access.
- Verdicts in `run/quranmb_verse_map_verified.json`; `--recheck` supersedes a
  verdict rather than overwriting it.

#### Results

**First pass: 93 confirmed, 5 `boundary_wrong` — every one a missing or extra
word at the span's *start*.** One systematic fault, not five independent ones.

**The matcher's core assumption is false.** It assumes a prompt is a contiguous
run of *whole Qur'anic words*. QuranMB.v2 trims and adds proclitics, and one
reference begins partway through a word:

| prompt | correction | d before → after |
|---|---|---|
| 00035 | add `إِنَّ` — 3:97 reads `فَإِنَّ`, the benchmark drops the `fa` | 5 → **3** |
| 00036 | add `وَ` → `وَإِنَّ` | 3 → **1** |
| 00069 | **remove** a leading `وَ` the text had and the audio does not | 3 → **1** |
| 00088 | add `يَغْشَى`; the reference begins `x $ aa` — truncated, and `خ` for `غ` | 3 → 3 |
| 00087 | **no change** — already d=0 | adding what was heard → 7 |

**After correction and re-listening: 98/98 confirmed**, each correction
re-heard on 2 takes.

**00087 resolved against the ear, on three independent grounds.** All 7
annotation strings are 10–11 phones beginning `r i s q A n aa`: the benchmark's
own annotators, hearing the same clips, did not transcribe the extra word
either. And `recovered_text` must stay in the frame `reference_phoneme_string`
defines, because that is the frame the labels are aligned to — adding a
7-phone word would shift every label for the prompt.

**A benchmark-wide property fell out of that adjudication.** Over all 1,642
clips:

```
duration ≈ 1.53s + 0.083s × n_phones      residual sd 1.05s
```

Every clip carries **~1.53s of material outside the annotated phone sequence**,
~3.3 words at the marginal rate. The apparent outliers (00087 at 2.46× the
median seconds-per-phone, 00097 at 2.08×) rank by *shortness*, not by defect —
they are the 2-word and 1-word prompts, where a fixed overhead dominates.

#### Observations

1. **The audio earned its cost on the first pass.** Four wrong spans were
   invisible to every text-column check, because `reference_arabic_string` is
   derived from `reference_phoneme_string` and carries the identical wrong
   boundary. No amount of agreement between those two columns could have
   surfaced this.
2. **The first diagnosis of the five failures was wrong, and wrong in an
   instructive way.** Prepending the *full* preceding Qur'anic word made `d`
   worse in every case, which looked like proof that the reciters were simply
   adding run-in context. The error was assuming the benchmark's spans respect
   Qur'anic word boundaries — the same assumption that produced the bug. Only
   reading the reference phoneme strings directly (`w a < i nn a …`) showed the
   conjunction was in the prompt and missing from our text. The annotator's
   ear was right and the first analysis of it was not.
3. **00036's corrected text is not a contiguous Qur'anic span at all**
   (`relocated: false` in the output), yet its reference phonemes begin
   `w a < i nn a` and every speaker recites it. That prompt is not verbatim
   scripture. Recorded, not smoothed over.
4. **A finding that belongs to week 7.** An MDD system fed whole clips sees
   ~1.5s of unlabelled audio per clip and will be charged for insertions that
   the benchmark never annotated. Whether to trim by forced alignment is a
   week-7 decision; the measurement is logged here because this is where it
   was taken.
5. Terminal rendering of diacritized Arabic defeated three successive fixes
   (isolating scripts per line, `arabic_reshaper` + `python-bidi`, bare-word
   display). A cell grid cannot join cursive letters or stack combining
   harakat; the working answer was an HTML sheet read in a browser, with the
   terminal reduced to audio and verdicts.

#### Conclusion

**98/98 confirmed against the pre-declared ≥95 of 98.** Exit criterion 1's
independence clause is unsatisfiable as written — every surviving route
consults a forbidden column — so the instrument was substituted and the
substitution recorded in the week plan, per task 1's own rule. The result is
reported as **validation by audio**, not as a validated route (c); anything
derived from the skeleton match alone still carries the reproduction caveat.

---

### Pre-registered predictions: the Qur'anic DER measurement

Written **before** `scripts/measure_quranic_der.py` was run, so that they can be
wrong. Three predictions, each with the thing that would falsify it.

1. **Every tool scores materially worse on the Qur'an than on its published
   WikiNews figure.** Qur'anic orthography (dagger alef, alef wasla, the
   pause marks) and Qur'anic morphology are out of domain for all four, none of
   which lists Qur'anic text in its training data. The 60-verse planning probe
   already put `catt-eo` at 6.219 `ce_all` against a published 5.425.
   *Falsified by:* any tool at or below its WikiNews rate.
2. **`catt-eo`'s MSA 1.346 does not transfer.** That number was *agreement with
   a vowelizer* on `Iqra_train`, not accuracy against gold; the Qur'anic number
   is accuracy against gold. The size of that gap is the week's real finding,
   because it is the first direct evidence that an agreement rate overstates a
   diacritizer's quality — which is the premise of the whole paper.
   *Falsified by:* a Qur'anic rate near 1.3.
3. **The 98-prompt subset is noisier than the corpus number but not
   systematically different** — 4,442 characters against ~685,000, so roughly
   a 12× wider interval, but centred in the same place. The subset is the text
   that actually carries this arm's labels, so a systematic difference would
   mean the corpus number cannot stand in for it.
   *Falsified by:* a subset/corpus gap larger than the between-tool gaps.

**Pre-declared sanity check on the metric, not on the tools:** the ordering must
come out `catt-eo < catt-ed < shakkala < mishkal`. Two unrelated instruments
already agree on it (DER on MSA; week-4 phoneme conservation). A third
instrument disagreeing is evidence that the metric is wrong, not that the tools
changed.

**Declared in advance, not chosen after seeing the numbers:** the headline keeps
dagger alef (U+0670) scored, consistent with the MSA and WikiNews numbers so the
three are comparable, and a **second row** reports the rate with U+0670 deleted
from the reference. The planning probe measured that choice at 0.85 DER absolute
on 60 verses — larger than the calibration tolerance — and the published
convention deletes it, so the choice has to be named rather than defaulted.
