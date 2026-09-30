# Pilot annotation protocol (step 4, 50 utterances)

**Pre-registered 2026-09-23, before any audio was heard.** Fixing the rules
after listening would let the base rate be chosen rather than measured, and
this number decides what the MSA arm can claim (`docs/msa-arm.md` §9).

Scope: the 50-utterance listening pilot only. The full annotation protocol
(weeks 5–8, 300–500 utterances, ~50 double-annotated) inherits these rules
and adds inter-annotator agreement; it is a separate document.

---

## 1. The question

**What fraction of Common Voice Arabic read speech contains genuine
mispronunciations?**

`docs/msa-arm.md` §3.3 needs this because the MSA arm has no detection
positives unless `A ≠ C` somewhere. Two cheap proxies were tried and both
came back non-binding — the CV-Ar dev screen is confounded by an in-domain
checkpoint, and the `Iqra_TTS` control is confounded by memorization. This
pilot is the only remaining instrument.

## 2. Annotate at the WORD level, not the utterance level

This is the single most important instruction, and it is a power result,
not a preference. For the 25 uniformly-sampled utterances:

| unit | n | 95% upper bound if zero errors are found |
|---|---|---|
| utterance | 25 | **12.0%** |
| **word** | **142** | **2.1%** |
| phoneme | 807 | 0.37% |

QuranMB's false-rejection rate is 12.41%. So a binary clean/not-clean
verdict per utterance could not distinguish "essentially no errors" from
"as error-rich as QuranMB" — it would reproduce exactly the inconclusive
outcome the two earlier proxies already gave. Marking **which words** are
wrong costs little extra effort and makes the pilot decisive.

Do not transcribe phonemes. Full verbatim phonemic transcription is the
week 5–8 `A` annotation; it is far too slow for 50 utterances and is not
needed to answer the base-rate question.

## 2a. The diacritics shown are NOT the gold standard

**Revised 2026-09-23, before any audio was heard**, after the first draft
got this wrong.

`tashkeel_sentence` is the output of the organizers' **in-house automatic
vowelizer**. It is not human-verified, and it is frequently wrong. This
paper's entire thesis is that automatic diacritization is error-prone and
that its errors propagate into MDD — so treating that same tool's output as
the pronunciation gold standard here would be self-defeating.

A worked example from this very sample, id `65037`:

| | |
|---|---|
| written | `أريد الذهاب إلى اليابان` — *"I want to go to Japan"* |
| vowelizer | `أُرِيدَ الذَّهَابُ` — ***urīda al-dhahābu*** |
| correct | ***urīdu al-dhahāba*** |

The vowelizer put the verb in the **passive** and its object in the
**nominative** instead of the accusative. Two of four words wrong. A speaker
who reads that sentence correctly would be scored 50% erroneous by an
annotator checking against the diacritics.

**So the question is not "does the audio match the diacritics".** It is:

> Is what you hear a **legitimate reading of the written (undiacritized)
> sentence**?

Undiacritized Arabic admits more than one valid vocalization. A competent
speaker resolves it from context, and their reading can differ from the
vowelizer's while being perfectly correct. It is the **speaker** who is
being measured here, not the tool.

The vowelized forms are still displayed, in brackets beside each word, for
two narrow purposes: identifying the intended word when audio is unclear,
and letting you record vowelizer errors. They carry no authority.

**Tag `refbad`** when the speaker is right and the vowelizer is wrong. It is
excluded from the base rate and reported separately — where it becomes a
free observation about the label path, which is RQ1's actual subject.

*Consequence for word numbering:* words are numbered on the **undiacritized**
sentence, because that is what the speaker read. The two strings do not
always align — 3 of these 50 utterances have different word counts, since
the vowelizer drops punctuation-only tokens and loses content outright in
~0.8% of the corpus (§4.1). Where they do not align, the vowelizer's output
is printed as one line instead of word-by-word.

## 3. Verdict per utterance

Exactly one of:

| verdict | meaning |
|---|---|
| `clean` | Every word is a recognisable, correct production of the written word |
| `error` | At least one word carries a genuine mispronunciation (list which) |
| `unusable` | Audio truncated, inaudible, wrong text, empty, or speech unrelated to the sentence |

`unusable` is excluded from both the numerator and the denominator, and is
reported separately — it is a data-quality figure, not a pronunciation one.

## 4. What counts as an error

### Counts

- **Substitution** — a different consonant or vowel than the written word
  requires (`tag: sub`). **Short vowels count.** `a`, `i`, `u`, `A`, `I`,
  `U` are all phonemes in the 68-token `sws_arabic.txt` inventory, so a
  wrong short vowel is a phoneme substitution — precisely the error class
  MDD exists to detect. It is not a minor or cosmetic deviation.

  The line to draw is **lexical vs grammatical**:

  | what changed | example | verdict |
  |---|---|---|
  | The word itself | `كَتَبَ` *kataba* "he wrote" read as `كُتِبَ` *kutiba* "it was written" | `error` + `sub` |
  | Only the grammatical ending, word intact | `الْكِتَابُ` *al-kitābu* read as *al-kitāba* mid-sentence | `error` + `case` |
  | The grammatical ending on the final word | utterance ends *al-kitāb* not *al-kitābu* | `clean` |

  Test: does the vowel change **which word** it is, or only its
  **grammatical role**? Mid-sentence case endings *are* present in the
  reference, so dropping one is a genuine `A ≠ C` deviation — but it is
  also an ordinary register choice in read MSA, so it is tagged `case` and
  excluded from the lenient rate rather than being silently counted as
  mispronunciation.
- **Deletion** — a written phoneme is absent (`tag: del`).
- **Insertion** — a phoneme not in the written word (`tag: ins`).
- **Wrong word** — a different word, a skipped word, or an added word
  (`tag: word`). Note it even though it is a reading error rather than an
  articulation error; it still makes `A ≠ C`.

### Does NOT count

- **A dropped case ending (i'rab) on the LAST word of the utterance.**
  Decided in `docs/msa-arm.md` §3.4.1 (option 4): the MSA arm's `C_gold` is
  **pausal**, matching the model's training target. A speaker who ends on
  `الكتاب` as *al-kitāb* rather than *al-kitābu* is producing the pausal
  form, which is correct under our convention. Verdict `clean`.

  This is visible in the reference itself. For id `66052`, `phoneme_ref`
  keeps the mid-sentence genitive (`الْغَيْثِ` → `l g A y ^ i`) but drops the
  final one (`عَقَقَ` → `E A q A q`, no trailing `a`).
- **A reading that differs from the vowelizer but is legitimate.** See §2a.
  The speaker is the subject of this measurement, not the tool. Tag
  `refbad` to record it.
- **Hesitation, restart, self-correction, filled pause.** If the word is
  ultimately produced correctly, the utterance is `clean` for our purposes;
  tag `hesit` and note it. These matter for ASR, not for phoneme-level MDD.
- **Speech rate, prosody, intonation, stress placement.** Out of scope —
  the metric is segmental.
- **Recording quality** that still leaves the word identifiable.

### The judgement call: dialectal realization

Common Voice Arabic is read MSA spoken by speakers with regional accents.
Systematic dialectal substitutions are common:

| written | frequent realization |
|---|---|
| ث /θ/ | /t/ or /s/ |
| ذ /ð/ | /d/ or /z/ |
| ظ /ðˤ/ | /zˤ/ |
| ق /q/ | /ɡ/ or /ʔ/ |
| ج /d͡ʒ/ | /ʒ/ or /ɡ/ |

Counting these as errors could push the base rate toward 100% and make it
meaningless; not counting them could hide real positives. **So do not
decide now — record and separate.**

Mark the word as an error **and** add `tag: dialect`. The scoring script
then reports two rates:

- **strict** — every marked word counts;
- **lenient** — words tagged `dialect` (and `case`, `hesit`, `refbad`) are
  excluded.

Both go in the write-up. This is the whole reason for tagging: it lets the
base rate be recomputed under either definition without re-listening.

### Tags attach to words, not to utterances

**Revised 2026-09-23, before any audio was heard.** The first draft recorded
one tag set per utterance, and the scorer excluded a word from the lenient
rate only if *every* tag on its utterance was excludable.

That is wrong in the one configuration this corpus produces constantly. Read
MSA drops mid-sentence i'rab routinely, so an utterance can carry five `case`
words beside a single genuine `sub`. Under per-utterance tagging the set
`{sub, case}` is not a subset of the excluded set, so **all six** words enter
the lenient numerator instead of one — and on a 12-word sentence that is 50%
rather than 8.3%. The error is not symmetric: it inflates the rate, i.e. it
pushes the result toward §8's `≥ 3%` band, the outcome that suits the paper.

So tags attach to the **word**. You enter word numbers, then the tags for
those words, repeating until every marked word is described.
`pilot_annotate.py` stores the result under `errors` as `(words, tags)`
groups — words sharing a tag set collapse into one group — and writes
flattened `error_words`/`tags` fields for review only.

**A word can carry more than one tag, and they are independent facts.** The
speaker mispronounces a word *and* the vowelizer got that same word wrong:
that is `sub` + `refbad` on one word, and both must be recorded. Re-entering
a word adds tags to it rather than replacing them, so the second fact can be
noticed after the first has been entered — which is how it usually happens.

The scorer drops a word from the lenient rate only when **every** tag on it
is excludable. Verified:

| tags on the word | strict | lenient |
|---|---|---|
| `sub` + `refbad` — speaker wrong, vowelizer also wrong | 1 | **1** |
| `refbad` alone — speaker right, vowelizer wrong | 1 | 0 |
| `case` + `refbad` — ending only, vowelizer also wrong | 1 | 0 |

The first row is the one that matters: flagging the vowelizer must never
launder a genuine mispronunciation out of the base rate.

## 5. Blinding — mandatory

`run/msa_pilot_sample.json` contains a `prediction` field: the model's
output. **Do not look at it while judging.** If you do, your verdicts
anchor to the model and the base rate measures agreement with the model
rather than the speaker's production — which is precisely the confound
(`docs/msa-arm.md` §3.2, anchoring) that already invalidated the CV-Ar dev
screen.

Also hidden: `disagreement` and `stratum`. Knowing an utterance was drawn
from the high-disagreement pool is a direct prompt to find something wrong
in it.

`scripts/pilot_annotate.py` enforces all of this — it shows only the audio
path and the text, and presents the 50 in a fixed shuffled order so the two
strata interleave. Use it rather than editing the JSON by hand.

### Deferring a hard utterance

`s` postpones the current utterance; `j` postpones it and jumps to a random
other one. Both send it to the back of the queue, so it **returns in the same
session** — postponed, never dropped.

That distinction is the whole point. An utterance you cannot decide is, by
construction, more likely to contain something wrong than one you dispatch in
ten seconds. Letting those fall out of the sample would bias the base rate
**down**, toward §8's `< 1%` falsification band, and the loss would be
invisible in the output. The tool therefore warns when every remaining
utterance has been deferred without a verdict, lists any left undecided at
exit, and records `n_deferrals` per row so difficulty can be cross-checked
against the error marks afterwards.

Deferring does not compromise blinding — the strata stay hidden — but it does
mean the annotation order is no longer the fixed shuffle. That is why the
order is recorded rather than assumed.

## 6. Procedure

1. Listen to the whole utterance once, at normal speed, before judging.
2. Read the diacritized text (`tashkeel_sentence`) — that is the reference
   production. The undiacritized `sentence` is shown for context only.
3. Re-listen as often as needed. Slowing down is allowed.
4. Decide the verdict; if `error`, give the word numbers and their tags, one
   group per error type (§4).
5. Write a short free-text note for anything surprising, and **always** for
   `unusable`.
6. If genuinely unsure after three listens, mark `error` with tag `unclear`
   and say so in the note. The scoring script reports these separately so
   an ambiguous result stays visible rather than being silently resolved.

`?` at any prompt reprints the decision rules. Use it freely — holding §4 in
your head for 50 utterances is how strictness drifts.

**On the difficulty of the tag call.** It is smaller than it looks. `dialect`,
`case`, `hesit` and `refbad` are all excluded from the lenient rate (§7), and
both rates are reported. A word given the wrong tag therefore moves between
two published numbers; it does not corrupt either. What must be right is only
the `clean`/`error` split and the word indices. The tag vocabulary exists to
let the rate be *recomputed* under a different definition later, not to be
adjudicated perfectly at annotation time.

Expect roughly 45–75 minutes. Do it in one or two sittings, not five —
drift in strictness across sessions is a real effect, and with n=25 in the
uniform stratum a few drifting judgements move the headline.

### 6a. Declared deviation: the procedure actually followed

**Recorded 2026-09-26, after annotation.** §6 above was written as a
listen-first loop against `tashkeel_sentence`. The annotator did something
different and better, and the paper must describe what was done rather than
what was registered.

The four steps, as executed on all 50 utterances:

1. **Read the undiacritized `sentence` and form an independent judgement**
   of the correct vocalization, before seeing any machine output.
2. **Obtain a candidate diacritization from an LLM (Claude)**, together with
   the positions it flagged as admitting more than one valid reading.
3. **Correct that candidate** into `C_gold` using the annotator's own
   step-1 judgement.
4. **Only then listen**, and mark deviations against the now-corrected
   `C_gold`, treating the step-2 multi-reading positions as legitimately
   ambiguous.

Three things follow, and each needs a sentence in the paper.

**This is the correct-the-machine design the plan asked for, not a departure
from it.** Plan line 134 specifies "correct-the-machine rather than
from-scratch: run CATT, present output, correct it", and flags that it
"anchors toward the tool under evaluation". Plan line 136 proposes using a
*different* diacritizer as the starting point precisely so "the anchoring
bias points away from the system under test rather than toward it". Using an
LLM rather than CATT does exactly that. Step 1 — forming the judgement before
seeing machine output — is a further anchoring mitigation the plan did not
require.

**The LLM used in step 2 must never enter the evaluated diacritizer set.**
`C_gold` is partly LLM-derived, so scoring an LLM against it would be
circular. The evaluated set is fixed as CATT (EO and ED), Shakkala, Mishkal
and Farasa (plan line 119), and this exclusion is committed here rather than
decided later.

**The override rate was not recorded, and should have been.** Keeping the
step-1 judgement and the step-2 candidate as separate fields would make
"how often did the annotator overrule the machine" a direct measurement of
the anchoring the plan wanted ~50 double-blind utterances to estimate — from
a single annotator. The pilot cannot produce that number retrospectively.
**For the 300–500 block: store both, per word.**

*Note also that §6 step 2 as written ("that is the reference production")
contradicts §2a, which strips `tashkeel_sentence` of any authority. §2a
governs; the §6 wording is an editing error, and the procedure above is what
§6 should have said.*

## 7. Reporting

`scripts/pilot_baserate.py` handles this. Two things it does that hand
computation gets wrong:

- **Stratum reweighting.** The 50 are **not** a random sample. 25 were
  drawn from the 100 highest-disagreement utterances specifically to find
  errors if any exist. Only the 25 `uniform_random` rows estimate the base
  rate; averaging all 50 overstates it, badly.
- **Confidence intervals.** At these sample sizes the point estimate alone
  is not reportable.

The `high_disagreement` stratum answers a different and still useful
question: *when the model and the reference disagree, is the speaker or the
model wrong?* That is a precision diagnostic, and it feeds §3.4.1's
convention finding — not the base rate.

## 8. Pre-committed reading of the outcome

Fixed in advance so the result cannot be reinterpreted to suit the paper.
Rates are **word-level, lenient** (dialect/case/hesitation excluded), on the
uniform stratum:

| outcome | reading |
|---|---|
| **≥ 3%** | A real base rate exists. §3.3 option 1 proceeds: annotate 300–500 utterances and measure RQ1 on the MSA arm directly. |
| **1–3%** | Marginal. Option 1 is viable only with a larger annotation set; cost it explicitly before committing, and consider enriching the pool rather than sampling uniformly. |
| **< 1%** | Near-zero base rate — §9's falsification condition fires. The MSA arm falls back to option 2 (label path only); the headline F1-bias number comes from the Qur'anic arm's `C_auto` condition; `Iqra_TTS`'s augmented rows supply injected positives (option 3). |

If the strict and lenient rates fall in different bands, that is itself the
finding: it means the MSA base rate is dominated by dialectal realization,
and the paper must say which definition it uses and why.

## 9. Outcome (24 Sep 2026) — closed

Ran as specified: 50/50 blind, single annotator, one pass.
`run/msa_pilot_annotated.json`. Analysis `insights/week-04.md`; standing
reference `docs/msa-arm.md` §3.3a.

**The last clause of §8 is what fired, and it fired on `case`, not on
`dialect`.** The bands did not select a row:

| definition | rate | 95% CI | band |
|---|---|---|---|
| count everything | 19.8% | [12.0%, 29.8%] | ≥3% |
| drop `refbad` only | 5.8% | [1.9%, 13.0%] | ≥3% |
| drop `refbad` + `dialect` (case counts) | 4.7% | [1.3%, 11.5%] | ≥3% |
| lenient — the §8 headline | 2.3% | [0.3%, 8.1%] | 1–3% |

Every interval spans 3%. The `case` definition alone moves the point
estimate across the boundary the decision hangs on, so **no row of §8 can be
taken**, and the §8 escape clause applies verbatim: the paper must state
which definition it uses and why. That is a modelling decision (it belongs
with msa-arm.md §3.4.1 option 4), not something a larger n resolves.

Two things §8 did not anticipate:

- **Power.** The bands were set assuming 25 uniform utterances (~142
  words). Seven were unusable; 86 words arrived. A protocol that
  pre-commits to bands should also pre-commit to a minimum usable n, or
  state what happens when attrition undercuts it. It did neither.
- **The by-product outgrew the headline.** `refbad` — added to §4 late, as
  bookkeeping so vowelizer errors would not contaminate the speaker rate —
  returned **43.6%** of usable utterances [27.8%, 60.4%], against 2
  mispronounced words. The pilot's most consequential number is one it was
  not designed to measure and had no pre-committed reading for.

**Not fixed:** single annotator, no inter-annotator agreement. Plan line 134
requires ~50 double-annotated blind; until that exists every number above is
provisional, including the 43.6%.

**Two deviations are declared: §6a (the procedure actually followed,
LLM-assisted correct-the-machine) and §9a (exclusions beyond the §3
wording).** Both were decided during annotation rather than before it.

### 9a. Declared deviation: exclusions beyond the §3 wording

§3 defines `unusable` as "audio truncated, inaudible, wrong text, empty, or
speech unrelated to the sentence". Eleven rows were excluded; the grounds
were not all that:

| ground | n | ids | within §3? |
|---|---|---|---|
| audio/text defect | 5 | 63566, 63842, 66052, 71116, 71747 | yes |
| Qur'anic verse | 5 | 64076, 66665, 71266, 73417, 73793 | no — **domain** criterion (§1: the arm is defined as non-Qur'anic) |
| pre-vowelized, not Qur'anic | 1 | 69711 | no — but the text is a hadith, so the domain ground applies |

The domain exclusions are correct and would have been pre-registered had
the contamination been anticipated; §1 already defines the arm as the
non-Qur'anic condition, so Qur'anic rows were never in scope. They are
recorded here as a deviation anyway, because the decision was taken during
annotation rather than before it.

**The pre-vowelized exclusion moves the headline in the flattering
direction and must always be reported with its sensitivity.** On those rows
the pipeline passes the input's own diacritics through verbatim, so the
vowelizer never ran:

| | excluded (as reported) | included, counted "vowelizer correct" |
|---|---|---|
| vowelizer error rate | **43.6%** [27.8, 60.4] | 39.5% [25.0, 55.6] |

Excluding is the defensible choice — the denominator for "how often does
the vowelizer err" is rows where it operated — but it *raises* the number
the paper leans on, which is the pattern reviewers scrutinise. State both.
The conclusion is insensitive to it: ~4 in 10 either way, against 2
mispronounced words.

Base-rate sensitivity for `69711` is negligible: 2.33% as reported, 2.17%
if it returns clean, 8.70% only under the implausible assumption that all
six of its words were mispronounced.

**For the 300–500 block:** filtering Qur'anic and pre-vowelized rows becomes
a *sampling-frame* decision applied before annotation (week-04.md carried
forward, item 1), not a per-utterance verdict. That removes the deviation
rather than repeating it.
