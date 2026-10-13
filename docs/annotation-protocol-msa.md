# MSA-arm annotation protocol — `C_gold`, the 500-utterance block

**Pre-registered. Commit this file before the first utterance is annotated.**
`docs/msa-arm.md` §3.2 requires this in writing in advance, and not as a
formality: §4's crossover measures anchoring, and an anchoring measurement
whose procedure was fixed afterwards measures nothing.

Supersedes [annotation-protocol-pilot.md](annotation-protocol-pilot.md) for
the main block. That document governed a *listening* task; §3.3b D1 dropped
`A`, so everything in it about audio judgments, `dialect`/`hesit` tags and
strict-vs-lenient base rates does not carry over. What does carry over is
named where it is used: pilot §6a (store the machine's draft and the
annotator's final separately, per word) and §9a (filter the frame, don't
exclude per-utterance).

Decision records: §3.3b **D1** (text-only), **D3** (two annotators,
qualified), **D4** (500 rows, seeded draw), **D5** (what the block is for),
**D7** (the anchor is `catt-eo`), §3.4.1 option 4 (pausal). Qualification
results: [annotator-qualification.md](annotator-qualification.md) §9.1.

---

## 1. The job, the unit, and the frame

**The job.** Given an `Iqra_train` dev utterance and a machine draft of its
diacritization, produce `C_gold` — the **canonical** diacritization of the
written sentence. Canonical means *what the prompt text calls for*, not what
any speaker did. Annotators never open the audio (D1); there is no audio in
the task.

**The unit of work** is one utterance. **The unit of measurement** is one
mark (§3), which is why §2's edit counts and §4's agreement are computed per
position and not per utterance.

**The frame is already drawn and committed.** `run/msa_annotation_block.json`
— 500 rows, seed `20261001`, from the 2,092-row eligible pool, with
`crossover: true` on the first 50 in draw order. Do not re-draw, do not
substitute a row that looks hard, and do not skip one.

This is pilot §9a's lesson applied rather than repeated: Qur'anic and
pre-vowelized material was excluded at the *sampling frame* (D4's three
exclusion predicates), so no per-utterance exclusion judgment is delegated to
the annotator. The one remaining escape hatch is §5's `unusable`, and it is
deliberately narrow.

**Flagged rows are annotated, not skipped.** ~105 of the 500 carry a D4 flag
(59 partially vowelized, 47 short, 1 Latin-script, 3 word-count mismatch).
Flags are a results breakdown, never a deletion — a `word_count_mismatch` is
the vowelizer dropping a word, which is the arm's *subject*. Deleting those
rows would bias D5's headline **down**.

**Division of labour.** 450 non-crossover rows split 225/225 by draw-order
parity (annotator 1 takes even ranks, annotator 2 odd), so neither annotator
gets a contiguous — and therefore possibly atypical — stretch. All 50
crossover rows are done by **both** (§4). Per annotator that is ~1,450 words;
at annotator 1's measured 14.2 words/min *from scratch* it is ~1.7 h, and the
real task is correct-the-machine, so that is an upper bound. D4's ~1,592-row
extension is affordable on this evidence, but it stays a declared extension
and is decided after the block, not now.

## 2. The machine draft is `catt-eo` — and what that costs

Per D7, the draft shown in the anchored condition is **CATT encoder-only**,
`catt-tashkeel==1.0.2`, pin checked at runtime. Not the organizers'
`tashkeel_sentence`: that column is `C_auto`, the thing D5 measures `C_gold`
*against*, and seeding the reference with the object under test would make
D5's number mechanically small.

**The draft has no authority.** This is pilot §2a's rule, re-stated because
the tool changed and the rule did not. `catt-eo` disagrees with
`tashkeel_sentence` on 4.85% of words and is wrong somewhere in that
disagreement — which party is right at a given position is exactly what is
unknown. Treat the draft as a typing aid.

**Three honest costs, stated here so they are not discovered in review.**

1. **`catt-eo` is in the evaluated set, so anchoring points *toward* a system
   under test.** Plan line 136 wanted the opposite and §3.2 accepted this
   trade deliberately — the alternative (audio-informed restoration) buys an
   unmeasured bias *correlated with the dependent variable*, which is strictly
   worse. The bias here is measured, by §4.
2. **It is self-penalizing for the headline, which is why it is tolerable.**
   `catt-eo` agrees with `tashkeel_sentence` at 98.65% of positions while
   §3.3a found that column wrong on 43.6% of utterances, so the two share
   error modes. Every draft position an annotator accepts uncritically drags
   `C_gold` *toward* `C_auto` and **shrinks** D5's measured gap. The direction
   is unflattering to us, not flattering.
3. **But it is *not* self-penalizing for RQ1's per-tool table.** There,
   `catt-eo` is scored against a reference it helped write, so its DER is
   biased low relative to `shakkala`, `mishkal` and `farasa`, which had no
   such advantage. **The RQ1-MSA table must carry `catt-eo` with an explicit
   anchoring caveat**, and §4's blind arm supplies the only unanchored
   estimate we will have. This is a reporting obligation, pre-committed now.

**This reverses pilot §6a's exclusion rule, knowingly.** §6a committed that
the tool used as the starting point "must never enter the evaluated
diacritizer set", on circularity grounds. That rule was written when the
starting point was an LLM, which is *not* in the evaluated set and so could
be excluded at no cost. `catt-eo` cannot be — it is a headline system. The
rule is therefore replaced rather than quietly broken: the starting point may
be an evaluated tool, provided (a) the contamination is measured by a blind
arm and (b) the affected cell is flagged in the results table. Both are
committed above.

## 3. The pausal convention — stated explicitly, because the qualification test proved it cannot be left implicit

Annotator-qualification §3 told annotators they need not learn this
convention. **That was false and it cost five of annotator 1's 42 bucket-B
errors.** The convention is applied at the end of the *utterance* only, so a
non-final word's final mark survives into the phoneme string and is scored.
The rules below are therefore part of the job, not background.

`C_gold` is **pausal** (§3.4.1 option 4). Operationally, four rules:

| # | rule | status downstream |
|---|---|---|
| R1 | **Write the full, grammatically correct case ending (*i'rab*) on every word except the last.** | **scored** — survives into the phonemes |
| R2 | **The last word of the utterance: its final mark is free.** Write it or not. | stripped |
| R3 | **Tanwin is free, anywhere.** Write it or not; it is stripped mechanically. | stripped |
| R4 | **Never write a liaison (sandhi) vowel.** | **scored** |

**R4 is the one that bites, and it is the opposite of R1, which is why it
needs its own rule.** When a word that ends in sukun is followed by a word
beginning with `ال`, careful recitation inserts a helping vowel —
`مِنْ الْمَاءِ` read *mina l-māʾi*. **Do not write it.** Keep the sukun.

Verified against the organizers' own reference rather than asserted. Id
`63729`:

| | |
|---|---|
| `tashkeel_sentence` | `... كَأْسٍ مِنْ الْمَاءِ مَرَّتَيْنِ ...` — sukun kept |
| `canonical` | `... < s  m i n  l m aa < i  m a rr ...` — **no helping vowel** |

And the phonetizer agrees it is label-impacting: `phonetize("مِنْ")` → `m i n`
but `phonetize("مِنَ")` → `m i n a`. Same word, different label sequence. R4
is not a stylistic preference; getting it wrong writes a phoneme that is not
in the reference.

**Scale: 31 positions in the 500-row block, 1.49% of non-final words** (e.g.
`اخْتَارَتْ الْفُسْتَانَ`, `كَمْ السَّاعَةُ`, `لِمَنْ الدَّارُ`). Small, but it is
systematic rather than random, so left uninstructed it is a *bias* in 1.5% of
case-ending positions, not noise. Annotator 1 hit this at 5/427 words =
1.17%, which matches the frame's rate — further evidence it was the
instruction and not the annotator.

**R1 likewise verified**: id `63563` keeps the mid-sentence genitive
`السَّمَاوَاتِ` → `... t i ...`, and drops the final one
(`يَسْتَكْبِرُونَ` → `... r uu n`). R3: `سَدًّا` → `s a dd aa`, tanwin absent.

**The letters must not change.** Marks only. An utterance whose consonant
skeleton was altered is flagged and re-done, not silently scored — the same
check as qualification §8 step 1, and it is automated.

## 4. The crossover — the part that cannot be reconstructed afterwards

§3.2's blocker and the reason this document has a deadline. Design, on the 50
rows flagged `crossover: true`:

| draw rank | annotator 1 | annotator 2 |
|---|---|---|
| 0–24 | **anchored** (sees the `catt-eo` draft) | **blank page** |
| 25–49 | **blank page** | **anchored** |

Both annotators do all 50. That buys three things from one pass:

- **agreement** on all 50 utterances (not 25), reported as Pearson +
  ICC(2,1) per the clinical model, *not* bare kappa;
- **the anchoring effect** as a within-utterance contrast on all 50 —
  anchored-vs-blank distance to the `catt-eo` draft — with annotator
  identity balanced across the two halves, so an annotator-skill difference
  cannot masquerade as an anchoring effect;
- **an unanchored `catt-eo` DER** on 50 utterances, which is §2's cost-3
  correction factor.

**Blank page means blank.** No draft, no `tashkeel_sentence`, no
diacritization tool, no LLM, no looking up the source. The two annotators do
not discuss the crossover rows and do not see each other's files.

**Order.** The 50 crossover rows are annotated **first**, before any of the
450. Doing them last would measure anchoring in annotators who had already
spent hours learning `catt-eo`'s habits, which is a different and smaller
quantity than the one the paper will claim.

**Power, stated rather than discovered.** 50 rows is ~264 words and ~1,060
scored marks per arm. That is enough to detect a large anchoring effect and
**not** enough to certify a small one; a null result here is "no large effect
detected", never "no effect". Pilot §9's second lesson — bands set without a
minimum usable n — applies, so: if more than 5 of the 50 are `unusable`
(§5), extend the crossover with the next rows in draw order until 45 usable
remain, and record that this was done.

## 5. Flag-and-adjudicate, and the two escape hatches

Annotator 1's measured routing (qualification §9.1): bucket A **1.17%**
[0.72, 1.90] → unsupervised; bucket B **14.33%** [10.78, 18.81] → inside
§9's 10–25% band → **flag-and-adjudicate**. Annotator 2's routing is set by
the same pre-registered table once their submission is scored, and is not
negotiated.

So, per utterance:

1. **`flag`** — the annotator is unsure of a **case ending**. Record the word
   index. The annotator still writes their best guess; flagging does not mean
   leaving it blank. **The user adjudicates flagged positions**, and only
   those.
2. **`unusable`** — narrow, and narrower than the pilot's: empty text, text
   with no Arabic, or text so corrupt that no diacritization is defensible.
   **Not** "hard", **not** "ambiguous", **not** "the draft is bad". Requires a
   free-text reason. Excluded from numerator and denominator alike and
   reported as its own count.
3. **`ambiguous`** — the sentence genuinely admits two readings and the
   annotator can defend both. Record both. This is the WikiNews
   `max_gold_alt` situation, which occurred on 2.46% of words there, and the
   analogue must exist on our corpus even though nobody has enumerated it for
   us. Without this field the annotator is forced to pick one and the
   arbitrariness is invisible; with it, D5's DER can be reported both
   strictly and with the allowance.

**Adjudication is logged, not merged silently.** The pre-adjudication value
stays in the record (§6), so "how much did the adjudicator change" is a
number and the released artifact is auditable.

## 6. What is logged per row — the schema, fixed now

Pilot §6a's one unfixable regret was not storing the machine's draft and the
annotator's output separately, which made the override rate unrecoverable.
Its instruction for this block was explicit: *"store both, per word."* Hence:

```
id                 row id, from msa_annotation_block.json
rank               draw rank (joins back to the frame)
annotator          "a1" | "a2"
condition          "anchored" | "blank"
suggested          the catt-eo draft shown — null in the blank condition
final              the annotator's C_gold for the utterance
flagged_words      [word indices] — uncertain case endings (§5.1)
ambiguous          [{word index, alternative reading}] (§5.3)
status             "done" | "unusable"
unusable_reason    free text, required when status == "unusable"
note               free text, optional
started_at         ISO 8601
finished_at        ISO 8601
adjudicated        null, or {by, at, final_before, changes:[...]}
tool_version       catt-tashkeel version that produced `suggested`
protocol_version   this document's git sha
```

Five of these exist only to keep a measurement possible, and each one is
load-bearing:

- **`suggested` + `final`** → the override rate, per word, which is the
  anchoring measurement's raw material. `suggested` is stored even in the
  blank condition's *sibling* rows so the same distance metric applies to
  both arms; it is `null` only where nothing was shown.
- **`condition`** → without it the crossover is unanalysable after the fact.
- **`final_before`** inside `adjudicated` → separates annotator accuracy from
  adjudicated accuracy. Collapsing them would let adjudication quietly
  inflate the reliability paragraph.
- **`started_at`/`finished_at`** → D4's extension decision is explicitly
  deferred to real timings, and self-reported durations are not that.
- **`protocol_version`** → a mid-block change to this file must be visible in
  the data, not just in git.

**Automated at ingest, so the checks are not trusted to anyone's
discipline:** skeleton unchanged (§3), word count matches the frame, R4
violations counted and *reported rather than corrected* (they are evidence
about the protocol, and silently fixing them destroys that), and no row
written twice in the same condition by the same annotator.

## 7. Orthography, non-Arabic content, and malformed text

§4.1.1's normalization contract, restated as instructions rather than as a
specification. The annotator does less here than they might expect, and the
point of saying so is that improvisation is what breaks the contract.

| situation | the annotator does | why |
|---|---|---|
| **Latin script** in the sentence | **leave it untouched.** Do not transliterate, do not delete. | The vowelizer deletes it (100% rate), which is a real `C_auto` error and D5's subject. Deleting it too would hide that. 1 row of the 500 — rank 380, id 64376, two occurrences of `Eddie`. |
| **Digits** | leave untouched. Do not spell out. | Same seam. `digits` is an `UNFAIR` class. |
| **Punctuation** | leave as written. Do not add or remove. | Handled downstream; §4.1 showed 99.5% skeleton agreement once depunctuated. |
| **Partially pre-vowelized** text (59 rows) | **complete it, and correct it where it is wrong.** The existing marks have no authority. | Rate in [0.05, 0.50) was kept precisely so the arm sees it; rows ≥ 0.50 were excluded at the frame. |
| **Qur'anic pause marks** (U+06D6–U+06ED) | leave untouched. | Not diacritics; out of scope for the mark set. |

`UNFAIR` rows are annotated, flagged, excluded from the **phoneme-level**
denominator, and reported as their own line. The pool is the sampling frame,
`UNFAIR` is the evaluation denominator, and the flag joins them — do not
collapse the three.

**Malformed upstream orthography — the gap task 9a found.** The train-split
round-trip surfaced a shadda written on a long-vowel carrier, an
orthographic impossibility present in the source. **Two block rows are
affected: ranks 447 and 475.** Rule, so this is not improvised at 11pm:

> Diacritize the sentence as written, leaving the malformed cluster as it
> stands, and set `note` to `malformed_upstream`. Do **not** silently repair
> the skeleton — that would make the row's skeleton check fail for a reason
> unrelated to the annotator, and it would delete evidence about the corpus.

2 rows of 500. Reported as a count, not fixed.

## 8. The anchoring analysis plan — written before the data exists

Fixed now so the threshold cannot be chosen once the effect size is known.

**What is compared.** For each crossover utterance, for each scored mark
position (§3's R1/R4 survivors), whether the annotator's `final` agrees with
the `catt-eo` `suggested` value. Let `p_anchored` and `p_blank` be those
agreement rates. **The anchoring effect is `Δ = p_anchored − p_blank`**,
estimated with a paired test over utterances (utterance is the unit of
resampling; marks within an utterance are not independent) and reported with
a 95% interval.

Reported per bucket, never pooled: bucket A and bucket B separately, for the
same reason qualification §4 refuses to pool them — bucket A is ~79% of
positions and would swamp the case endings where §3.3a says the arm's
findings live.

**Pre-committed readings.**

| Δ (bucket B, where the effect is expected) | reading |
|---|---|
| **< 5 pp** | Anchoring is small. Correct-the-machine stands for the whole block as designed. |
| **5–15 pp** | Real but bounded. Block stands; D5's headline is reported **with** a stated direction of bias (it understates, per §2 cost 2) and the crossover's 50 blind utterances give the sensitivity. |
| **> 15 pp** | §9's falsifier fires. **Correct-the-machine is abandoned for case endings** — bucket B is re-annotated blank-page across the block — and the schedule cost is taken. Bucket A stays anchored. |

The asymmetry in that last row is deliberate: abandoning the anchor entirely
would cost roughly 2× the annotation hours to fix a bias that §3.3a localizes
to the positions bucket B isolates.

**Why Δ and not agreement-between-annotators.** Two anchored annotators who
both accept the draft agree perfectly and are both wrong — §3.3b D3's
argument. Δ is a *within-utterance, between-condition* contrast, so it sees
exactly the failure that inter-annotator agreement is blind to. The
agreement number is still reported (§4), as a secondary.

**A null result is not a clean bill of health.** §4's power note applies:
~1,060 scored marks per arm detects a large effect, not a small one.

## 9. The `A`-on-50 hedge — DECLINED for the crossover block

§3.3b D1's hedge suggests annotating `A` (verbatim production) on the
crossover sub-block at "negligible cost", lifting the pilot's base-rate n
from 86 words to ~136 utterances' worth for a limitations sentence.

**Not done here, and the reason is not cost.** The crossover's entire
validity rests on the blank-page arm being blank. `A` annotation requires
opening the audio; §3.2's third ground for rejecting audio-informed anchoring
is that audio evidence biases the reference *at exactly the positions a
speaker mispronounced* — correlated with the dependent variable. Putting
audio in front of an annotator on the same 50 utterances whose `C_gold` is
the anchoring measurement risks precisely that contamination, on the one
sub-block the arm cannot re-run.

So: `A` is not annotated on the crossover rows. If the base-rate sentence is
judged worth keeping, annotate `A` on a **disjoint** set — the pilot's
original 50, or rows from outside the 500 — as separate work, after `C_gold`
is complete. **This does not reopen D1**; the arm's deliverable is still the
label path only.

---

## Open, and deliberately not decided here

- **The annotation tool.** `scripts/pilot_annotate.py` is a listening
  interface and cannot be reused. The schema above is fixed; the thing that
  writes it is not built. This is the next piece of code the block needs.
- **Farasa.** Plan line 119 lists it in the evaluated set; it is not yet
  wrapped, so D7's table has four tools and not five. Does not block
  annotation — the anchor is decided — but it blocks the RQ1 table.
- **Which `C_gold` is released.** Pre- or post-adjudication, and under which
  licence. §7 of `msa-arm.md` owns this.
- **D2** — whether `A := C_gold` is an assumption or a measurement — remains
  open and is untouched by this document.
