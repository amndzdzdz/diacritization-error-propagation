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

**OPEN**: confirm availability and licence of the chosen set before the
test runs. This is the only unresolved item in this protocol.

## 6. Size

`Iqra_train` dev averages 19.4 marks/utterance: 2.89 case endings and
15.24 bucket-A marks.

**70 utterances** → ~202 bucket-B positions and ~1,067 bucket-A positions.

That resolves bucket A cleanly. It resolves bucket B only coarsely: 20
errors in 202 gives a 95% interval of roughly [6%, 15%]. So if bucket B
lands in the ambiguous 8–14% zone, **extend by a further 100 utterances
rather than deciding on the point estimate.** The test is text-only and
cheap; extending it is cheaper than mis-assigning the annotation work.

## 7. Procedure

1. Strip all marks from the 70 gold sentences. Present as plain text.
2. **From scratch, not correct-the-machine.** The real job is
   correct-the-machine, so the test is deliberately harder than the job —
   it measures the person's own skill rather than their ability to spot a
   machine's errors. A conservative bar.
3. Both annotators receive the **same 70 items**, which yields an
   agreement number for free.
4. Worked alone. No discussion, no automatic diacritization tool, no
   looking up the source article. Gold never shown.
5. No time limit, but **record elapsed time** — it calibrates D4's volume
   target.
6. Keep the raw submissions. They are the evidence for the paper's
   reliability paragraph.

## 8. Scoring

1. Verify the consonant skeleton is unchanged; flag items where it is not.
2. Apply the pausal filter, then compare mark-by-mark against gold.
3. Bucket each surviving position (A or B). Report error rate per bucket
   with 95% Wilson intervals.
4. Separate **omissions** (gold has a mark, annotator left none) from
   **substitutions** (wrong mark). They mean different things and an
   omission-heavy profile is a different problem from a
   substitution-heavy one.
5. Report inter-annotator agreement on the same items, as a secondary
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
