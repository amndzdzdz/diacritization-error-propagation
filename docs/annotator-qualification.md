# Annotator qualification test — MSA arm `C_gold`

**Pre-registered. Fix this file and commit it before the test is
administered.** Thresholds chosen after seeing results are not thresholds.

Decision record: [docs/msa-arm.md](msa-arm.md) §3.3b D3.

## 1. Why this exists

The MSA arm's only remaining annotation is `C_gold` — the correct
diacritization of Common Voice Arabic transcripts (§3.3b D1 dropped the
listening pass). It is simultaneously the input to every MSA number and
plan contribution 5, a released artifact. All the arm's quality risk is
concentrated in it.

Two annotators are available. One is not a graduate-level Arabic
specialist. This test replaces the credential question — "is this person
qualified?" — with a measurement: *on text whose correct diacritization is
already known, how often is this person right?* The result decides what
each annotator is allowed to annotate unsupervised.

**Both annotators sit it, including the specialist.** A per-annotator
accuracy figure is only interpretable next to another one.

## 2. Why accuracy against gold, and not just agreement between annotators

Plan line 134 asks for inter-annotator agreement. Agreement is necessary
but it cannot detect the failure mode that matters here. The main block
uses *correct-the-machine*: the annotator is shown a diacritizer's output
and edits it. Two annotators who both tend to accept the machine's
suggestion will agree almost perfectly **and both be wrong in the same
places.** Agreement is blind to that by construction; accuracy against
gold is not.

So: report both, and treat the gold figure as the one that licenses the
release. Agreement is reported as a secondary number and is **not** a pass
criterion.

## 3. What the job actually is

Given an Arabic sentence with no vowel marks, write in the vowel marks
(*tashkeel*) — the short vowels, sukun, and shadda — so the sentence reads
as a competent speaker would read it aloud.

**The letters must not change.** Only marks are added. An item where the
consonant skeleton was altered is flagged, not silently scored.

Annotators do **not** need to learn the project's pausal convention. They
write fully marked Arabic; the phonetizer applies the convention
mechanically downstream (`phonetizer.py` `normalize`). Consequence worth
knowing: some positions cannot hurt us even if they are wrong (§4).

**CORRECTION, 2026-10-04 — the paragraph above is false for mid-sentence
words, and the error is load-bearing for task 4's protocol.** The pausal
convention is applied at the *end of the utterance* only. A non-final word's
final mark survives into the phoneme string, so the choice between a pausal
and a liaison form there is label-impacting. Measured on annotator 1's
submission: `phonetize("مِنْ")` yields `m i n` but `phonetize("مِنَ")` yields
`m i n a` — the same word, two different label sequences. Five of annotator
1's 42 bucket-B errors are exactly this, and all five match `max_gold` (the
liaison reading) rather than `original_gold` (the pausal one), so they are a
convention mismatch and not a skill deficit.

What survives of the paragraph: annotators need not learn the convention for
the **final word** of an utterance, which is the only place §4's "final
word's final mark" free class applies. Everywhere else they must be told
explicitly to write the pausal form. **Task 4's protocol must state this**;
this test did not, which is why the five errors are charged to the
instructions rather than to the annotator.

## 4. What is scored, and what is free

`C_gold` is pausal (§3.4.1), applied mechanically, so marks that the
convention strips are harmless. **[MEASURED]** over all 50,188 marks in
`Iqra_train` dev:

| class | marks | share | scored? |
|---|---|---|---|
| **A** word-internal | 36,410 | 72.55% | yes |
| **A** shadda | 3,030 | 6.04% | yes |
| **B** case ending (word-final mark, non-final word) | 7,474 | 14.89% | yes |
| tanwin | 1,754 | 3.49% | **no — stripped** |
| final word's final mark | 1,520 | 3.03% | **no — stripped** |

**93.48% of positions are scored.** Case endings are **15.93% of scored
positions** — the pausal convention does *not* relieve them meaningfully,
which was an early mistaken assumption. Tanwin is a small share of *marks*
even though it appears somewhere in 44.7% of *utterances*.

The two scored buckets are different jobs:

- **Bucket A — word-internal vowels and shadda.** "How is this word
  pronounced." Any literate native speaker should be strong here. 78.6% of
  scored positions.
- **Bucket B — case endings.** Grammatically determined word-final vowels
  (*i'rab*); requires parsing the sentence's syntax. A schooling skill, not
  a speaking skill. 15.9% of scored positions, and where the literature
  puts the error — CATT's WikiNews DER roughly doubles when case endings
  are counted (5.43% vs 3.11%).

**They are scored separately.** A single combined score is dominated by
bucket A and would pass a candidate who is weak at exactly the positions
§3.3a showed dominate this arm's findings.

## 5. Test material

Requirements:

1. Hand-diacritized by specialists, with the gold withheld from candidates.
2. Modern Standard Arabic. **Not Qur'anic, not hadith, not famous
   passages** — a devout or well-read native speaker may know them by
   heart, which tests recall rather than skill.
3. Short everyday prose, as close to Common Voice sentence register as the
   material allows.

**Recommended: the hand-diacritized WikiNews test set used by the
diacritization literature**, specifically because published machine DER
exists on the same data. That turns "the human beats the machine" into an
apples-to-apples claim rather than a cross-corpus one.

Rejected: Tashkeela (mostly classical/religious — memorization risk and
wrong register). Held in reserve: the ~20% of `Iqra_train` rows that
already carry marks in the source text — right domain and register, but
the marks' provenance is unverified; §3.3a excluded them for a different
reason and never checked whether they are correct. Would need auditing
first.

**RESOLVED 2026-10-03.** The set is `CAMeL-Lab/wild_diacritics` →
`data/WikiNewsMax.tsv` (ACL 2024 "Arabic Diacritics in the Wild"), data
**CC-BY-SA-4.0**, with Darwish et al. 2017 to be cited additionally.
Audited clean against the §4.1 normalization contract; measurements in
[insights/week-05.md](../insights/week-05.md). Two things it changes here:

- **Use `original_gold`, not `max_gold`.** The file ships both; they differ
  on 3.72% of Arabic words. `original_gold` is the Darwish 2017 reference
  that published machine DER is measured against, which is the entire reason
  this set was recommended.
- **Score with an acceptable-alternatives allowance.** `max_gold_alt` gives
  359 words (2.46%) a second defensible reading, mostly loanword
  vocalization. Counting those as errors or as disagreements charges the
  annotator for the language's own underdetermination. See §8.

Share-alike is viral over derivatives: a redistributed stripped-gold
qualification set, and anything shipped alongside it, inherits CC-BY-SA-4.0.
The 500-row MSA block is drawn from `Iqra_train`/CV-Ar and is unaffected.

## 6. Size

`Iqra_train` dev averages 19.4 marks/utterance: 2.89 case endings and
15.24 bucket-A marks.

**70 utterances** → ~202 bucket-B positions and ~1,067 bucket-A positions.

That resolves bucket A cleanly. It resolves bucket B only coarsely: 20
errors in 202 gives a 95% interval of roughly [6%, 15%]. So if bucket B
lands in the ambiguous 8–14% zone, **extend by a further 100 utterances
rather than deciding on the point estimate.** The test is text-only and
cheap; extending it is cheaper than mis-assigning the annotation work.

**The unit above is an `Iqra_train` utterance, and the chosen material is
not made of those.** A WikiNews segment carries ~76 marks (4.01 per Arabic
word, 19-word median) against an `Iqra_train` utterance's 19.4 — about 4×.
So 70 *segments* would deliver ~1,300 bucket-B and ~4,000 bucket-A
positions, overshooting the targets above at roughly 4× the annotator hours.
The targets, not the item count, are what this section actually fixes:
**~11 segments satisfies bucket B and ~19 satisfies bucket A, so ~20
segments**, sampled from the 293 segments in the 8–40-Arabic-word band
(restricting to that band is also what keeps §5's register criterion
satisfiable). The 70 here and the 70 articles in the WikiNews set are
unrelated numbers that happen to coincide — do not let them co-vary.

The extension rule is unchanged in substance: if bucket B lands in the
ambiguous zone, extend by enough segments to roughly triple the bucket-B
position count.

**SIGNED OFF 2026-10-03: n = 20. [MEASURED — the packet is built.]**
`scripts/build_qualification_test.py`, seed **20261003** (distinct from the
annotation block's 20261001), drawing from the 293-segment band. Realized,
rather than estimated:

| | drawn | §6 target |
|---|---|---|
| segments | 20 | — |
| Arabic words | 427 | — |
| **bucket A** marked positions | **1,259** | ~1,067 ✓ |
| **bucket B** marked positions | **326** | ~202 ✓ |

**The two counts above are in the wrong unit, corrected 2026-10-04.** The
builder counted marked *letters*; §4's table counts *marks*, so a letter
carrying `<shadda><fatha>` is two positions and not one. The builder also did
not apply §4's free classes before counting. Net effect: bucket B was
overstated (it included tanwin and the final word's final mark) and bucket A
understated (shadda was merged with its harakah). The realized **scored**
counts, from `scripts/score_qualification.py`, are:

| | scored | §6 target |
|---|---|---|
| **bucket A** | **1,363** | ~1,067 ✓ |
| **bucket B** | **293** | ~202 ✓ |

Both targets still clear, so the sign-off above is unaffected — but
`build_qualification_test.py`'s counts should not be quoted as position counts
anywhere. The scorer's are authoritative.

Both targets are met with margin, so the sizing argument above is confirmed in
the correct unit rather than assumed. Bucket B's 326 is comfortably past the
point where §6's "extend by 100 utterances" worry applies — at 326 positions a
20-error result gives a 95% interval of roughly [4%, 9%], which resolves
against the 10% band rather than straddling it. Counts are pre-pausal-filter;
§8 step 2 applies that at scoring and can only reduce bucket B.

12 of the 427 words carry a `max_gold_alt` second reading, so §8 step 3's
allowance is small but non-empty on this draw.

**Also measured while building it, and worth recording because it validates
our own code against an external reference:**
`strip_diacritics(original_gold)` reproduces the file's own `dediac` column on
**all 16,163 word rows, zero disagreements**. The packet's blank text is
therefore the text the gold was written for, and §8 step 1's skeleton check
cannot fire on our preprocessing. The builder refuses to emit a packet if this
ever stops holding.

## 7. Procedure

1. Strip all marks from the **20** drawn segments (§6; the "70" this step
   used to say was the pre-correction item count). Present as plain text.
2. **From scratch, not correct-the-machine.** The real job is
   correct-the-machine, so the test is deliberately harder than the job —
   it measures the person's own skill rather than their ability to spot a
   machine's errors. A conservative bar.
3. Both annotators receive the **same 20 items**, which yields an
   agreement number for free.
4. Worked alone. No discussion, no automatic diacritization tool, no
   looking up the source article. Gold never shown.
5. No time limit, but **record elapsed time** — it calibrates D4's volume
   target.
6. Keep the raw submissions. They are the evidence for the paper's
   reliability paragraph.

## 8. Scoring

1. Verify the consonant skeleton is unchanged; flag items where it is not.
2. Apply the pausal filter, then compare mark-by-mark against gold
   (`original_gold` — see §5).
3. **Accept `max_gold_alt` where it exists.** On the 359 words (2.46%) the
   source marks with a second defensible reading, either reading scores as
   correct. Without this, the test charges annotators for positions the
   language itself leaves open and depresses both the error rate and the
   agreement number — and the alternatives happen to be enumerated for us on
   this set, which will not be true on our own corpus. Report the count of
   positions where the allowance was invoked, so the effect on the measured
   rate is visible rather than buried.
4. Bucket each surviving position (A or B). Report error rate per bucket
   with 95% Wilson intervals.
5. Separate **omissions** (gold has a mark, annotator left none) from
   **substitutions** (wrong mark). They mean different things and an
   omission-heavy profile is a different problem from a
   substitution-heavy one.
6. Report inter-annotator agreement on the same items, as a secondary
   number — Pearson + ICC(2,1) per the plan's reporting model, not bare
   kappa.

## 9. Pass marks — fixed in advance

Calibration against the machine the annotation has to beat. From CATT's
WikiNews DER 5.43% with case endings and 3.11% without, and taking case
endings as ~16% of positions:

```
0.0543 = 0.84 x 0.0311 + 0.16 x B  ->  B ~ 17.6%
```

An estimate, not a measurement — the literature's "without case endings"
usually means excluding each word's last letter, and the 16% share is from
our corpus rather than WikiNews. Shown so it can be checked and corrected.

**Bucket A — gates participation.**

| | |
|---|---|
| ≤ 2% error | qualified for bucket A |
| > 2% error | **not suitable for this task** |

Not softened: bucket A is 78.6% of scored positions and the machine is at
~3.1%. An annotator who cannot beat the machine on the easy majority has
nothing to contribute to `C_gold`.

**Bucket B — calibrates the workload, does not gate participation.**

| | |
|---|---|
| ≤ 10% error | annotate case endings unsupervised; specialist spot-checks 10% |
| 10–25% error | annotate everything, **flag** every case ending they are unsure of; specialist adjudicates flagged positions only |
| > 25% error | annotate bucket A only; all case endings routed to the specialist |

Flag-and-adjudicate is the expected mode, not a punishment — see §6 on why
the test cannot finely resolve bucket B. It also uses each annotator where
they are strong and spends specialist time only where it is needed.

**Completion:** ≥95% of items attempted. Skipping hard items inflates
accuracy.

### 9.1 Annotator 1 — RESULT, measured 2026-10-04

20/20 items, 30 minutes, 427 Arabic words. Pass marks above were committed
before this was scored and are applied unchanged.

| bucket | scored | errors | rate | 95% Wilson | §9 outcome |
|---|---|---|---|---|---|
| A internal | 1,363 | 16 | **1.17%** | [0.72, 1.90] | **qualified** — whole interval under the 2% bar |
| B case end | 293 | 42 | **14.33%** | [10.78, 18.81] | **flag-and-adjudicate** — interval lies wholly inside 10–25% |

**No extension under §6.** The extension rule fires when bucket B straddles a
band edge; [10.78, 18.81] is interior to 10–25%, so the routing is already
determinate and ~90 further minutes of annotator time would change no action.

**The calibration this test was chosen to deliver.** Against CATT EO on the
same corpus and the same reference: annotator **3.50% DER vs 5.425%** overall,
and **1.17% vs 3.105%** on bucket A — 1.55× and 2.65× better respectively. The
human-beats-machine premise underlying D1 and D5 is therefore measured, not
assumed, and measured apples-to-apples rather than across corpora.

**Two systematic non-skill classes inside bucket B's 42 errors**, both
charged to the instructions:

- **13 label-invariant** — a final sukun omitted on a foreign proper noun.
  Verified by phonetizing both forms: the label sequence is identical, so
  these cannot propagate to any MDD number.
- **5 liaison-for-pausal** — see §3's correction. All five match `max_gold`.

Net of both, the label-impacting bucket-B rate is **9.90%**, which *does*
straddle the 10% edge. This is recorded for interest and is **not** used for
routing: §9's bands are defined on the raw measured rate, and re-deriving a
route from a post-hoc subset is exactly the threshold drift §1 forbids.

**Timing, for D4.** 14.2 words/min from scratch ⇒ a 500-row block is ~3.1 h
and the full 2,092-row pool ~12.9 h. The real job is correct-the-machine and
therefore faster, so these are upper bounds.

## 10. Anchoring — the other half of plan line 134

Separate from this test, and it concerns the **main block**, not the
qualification. Two annotators allow a crossover design that measures
anchoring and agreement at once:

- Block 1: annotator 1 sees the machine's suggestion, annotator 2 works
  from a blank page, same utterances.
- Block 2: swap roles on different utterances.

The suggested-vs-blank difference is the anchoring effect; the
same-utterance overlap is agreement. **This cannot be reconstructed after
the fact** — it has to be designed in before the main block starts, which
makes it a week-5 blocker alongside D4.

## 11. Reported in the paper

Per-annotator bucket A and bucket B accuracy with intervals; the agreement
number; the adjudication rate actually incurred; the anchoring effect from
§10; and the statement that annotator suitability was established by
measurement against gold rather than by credential.
