# Week 5 implementation plan

Dates: 19–25 Oct 2026 nominal. Continues the week 4–6 block in
[mdd-paper-project-plan-v2.md](../mdd-paper-project-plan-v2.md) §6, of which
this week owes: "**RQ1 measurements on both domains**; **begin annotation**".
The block gate — "label-path effect measurable, and interaction power
estimated" — is week 6's, and everything below is chosen for whether it
unblocks that gate.

See [week-04.md](week-04.md) for the eight closed tasks and
[insights/week-04.md](../../insights/week-04.md) for the eleven experiments.
[docs/msa-arm.md](../msa-arm.md) is the standing reference; §3.3b (D1–D6),
§4.1.1 (the normalization contract), §8 (build order) and §9 (falsifiers) are
the sections this week acts on.

**Calendar note.** Week 4 closed on **3 Oct**, nine days before its nominal
start. That slack is real but it is not free capacity for week 7 — the
longest pole in the project is annotation, which is gated on a qualification
test that is gated on an external artifact (task 2). Spend the slack on
starting that chain earlier, not on pulling model work forward.

## Where week 4 left things

The text side of the pipeline exists and is measured. Four diacritizers
behind one interface (`catt-eo`, `catt-ed`, `shakkala`, `mishkal`, all CPU),
per-position confidence extraction for both CATT variants, a phonetizer
validated at **99.85%** round-trip on `Iqra_train` dev, and the §4.1
normalization contract pinned in `arabic_mdd.data.normalize`.

Four things from week 4 set this week's shape.

1. **The anchor is fixed.** `C_gold` is anchored on text-only CATT (§3.2,
   decided 3 Oct). Audio-informed restoration was rejected on three grounds,
   the load-bearing one being that conditioning on the audio biases the
   anchor in a direction *correlated with the dependent variable*. So nothing
   about the annotation design is still waiting on evidence — it is waiting
   on being written down, which is task 4.

2. **The annotation block is drawn and committed.** 500 utterances, seed
   `20261001`, from a 2,092-row pool, in `run/msa_annotation_block.json`
   (§3.3b D4). 14,754 phoneme positions, 2,638 words, ~105 flagged rows
   (21.0%). The crossover sub-block is the first 50 in draw order. ~1,592
   further eligible rows are declared as an extension, decided on the
   qualification test's measured timings.

3. **Risk (a) is retired, and a different one is live.** All four tools stay
   inside the 68-token inventory on all 2,588 dev utterances, emit no
   character the phonetiser silently eats, and conserve phoneme count to ≤1%
   (`catt-eo` 0.9998, `catt-ed` 0.9955, `shakkala` 0.9984, `mishkal` 0.9900).
   What replaced it is **alef wasla U+0671** on the *Qur'anic* arm — absent
   from the Buckwalter map, so `ٱلْحَمْدُ` silently loses `< a`, and present in 0
   MSA dev rows, which is why a 99.85% MSA round-trip says nothing about it.
   It is §9's live falsifier and it fires, if it fires, inside task 1.

4. **Two numbers to check RQ1 against before trusting it.** The per-utterance
   conservation tail (>10% short: 0.08% / 1.04% / 1.39% / **5.14%**) ranks the
   four tools from an instrument unrelated to DER, and the NAACL paper's
   Table 1 ordering (Mishkal 21.79 vs Shakkala 6.85) agrees with it. If RQ1's
   DER ordering disagrees with both, the metric is wrong, not the tools.

## Scope

Week 5 is where the project stops being able to defer things. Four items are
each a hard precondition for something in week 6 or later, and three of the
four are blocked on something outside the repository:

| what | blocks | blocked on |
|---|---|---|
| clip → verse recovery | the label path on **both** arms; D1 itself | an authoritative Qur'an text edition (free) |
| WikiNews gold set | annotator qualification → annotation; RQ1 metric calibration | availability + licence (**unconfirmed**) |
| crossover pre-registration | the whole 500-row block | nothing — just has to be written first |
| RQ1 metric module | RQ1, D5, the EO/ED choice, the frontier decision | nothing |

Provisional split of the remainder of the block, revising week 4's table:

| Week | Work |
|---|---|
| **5** | Clip → verse recovery; RQ1 on both domains; annotation begins under a pre-registered protocol |
| 6 | Label-path column on the Qur'anic arm; RQ5 interaction power estimate (**block gate**) |

Not this week: any model training, the text-dependent arm, the MSA label
path (weeks 7–9).

## Week 5 exit criterion

> **The label path is unblocked on at least one arm, and annotation has
> started under a protocol committed before the first annotated utterance.**

Three falsifiable parts. All three are pass/fail by criteria fixed *here*,
before the measurements run.

1. **Clip → verse recovery succeeds on the Qur'anic arm.** ≥95 of the 96
   distinct QuranMB.v2 sentences mapped to a verse span in an authoritative
   fully-vowelized text, **by a criterion that consults neither
   `reference_phoneme_string` nor `reference_arabic_string`** — see task 1 on
   why that clause is the whole test. Plus: `normalize.unsupported_characters`
   run over the recovered text, with every non-zero entry either eliminated by
   a recorded decision or declared as a label-path exclusion.

   > **[13 Oct 2026] RESTATED — denominator, instrument, and audit scope.**
   > The criterion above is kept verbatim as written on 1 Oct. Three parts of
   > it are wrong or unsatisfiable as stated, and the replacements are fixed
   > *here*, before the pass/fail call, not after seeing it.
   >
   > **(i) Denominator 96 → 98.** "96 distinct sentences" counted distinct
   > *Arabic strings*. That unit is wrong twice over: prompts 9 and 42 share
   > one Arabic string but have **different** phoneme strings (one pausal, one
   > liaison word-final), so counting by text merges two genuinely different
   > prompts and discards the distinction the project is about; 73 and 91 are
   > a true duplicate. `scripts/recover_quran_verses.py` keys on **prompt
   > index**, which is metadata and therefore — unlike the Arabic column —
   > not downstream of the uploader's phonemisation. 98 is also the *harder*
   > denominator, so this is not a relaxation. The threshold scales: **≥95 of
   > 98**. Note the separate **97/98 ceiling** that applies to
   > `phonetize(recovered) == reference_phoneme_string`, not to mapping: 9 and
   > 42 share one Arabic string against two phoneme strings, so one of them
   > must miss whatever text is recovered. Mapping itself has no ceiling.
   >
   > **(ii) "Consults neither column" is unsatisfiable; substituted, not
   > dropped.** Task 1's own rule is "record the substitution; do not quietly
   > relax the criterion", so: route **(a)** (ID scheme) closed *negative* and
   > is already recorded above. Route **(b)** (published prompt list) is now
   > closed *by evidence*: `test_references` carries ID + phonemes only,
   > `QuranMB.v2` carries ID + audio only, and `IqraEval/Iqra_Extra_IS26` is
   > gated (HTTP 401). Only route **(c)** remains, and (c) matches the
   > skeleton of `reference_arabic_string` — it consults exactly the column
   > the clause forbids. No criterion of the written form can be met.
   >
   > The substituted instrument is **human verification against the audio**
   > (`scripts/verify_verse_map.py`). The audio shipped with the benchmark,
   > was not generated from either text column, and is the only thing in the
   > distribution that is not downstream of the uploader's phonemisation. It
   > satisfies the *intent* of the clause — independence — which the written
   > form expressed as a property of the matcher. This is a weaker guarantee
   > in one respect (a human, not an algorithm) and a stronger one in another
   > (it tests the uploader's recovery, not just reproduces it), and the
   > result is reported as **validation by audio**, not as a validated route
   > (c). The §4.3 "reproduction, not validation" caveat still binds anything
   > derived from (c) *alone*.
   >
   > **(iii) The character audit must run over the recovered text, not the
   > corpus.** As first implemented, `recover_quran_verses.py` audited the
   > whole Tanzil corpus. That is a superset of the recovered spans *only*
   > while every recovered text is a verbatim corpus span — which four of them
   > are not, since `AUDIO_OVERRIDES` writes spans the Qur'an does not contain
   > contiguously (00035, 00036, 00069, 00088). Audited directly over the 98
   > recovered texts (4,442 characters): `unsupported_characters` is **{}**,
   > including all four overrides. The corpus audit is kept as well, since it
   > is what tests §9's alef-wasla falsifier against the edition.

2. **RQ1 has at least one number on genuinely gold text**, i.e. the Qur'anic
   domain, with the metric calibrated to reproduce CATT's published WikiNews
   DER — **EO: 5.425 with case endings, 3.105 without**, Table 5 of arXiv
   2407.03236v3; **ED: 5.963 / 3.631**, so the target is variant-specific and
   the calibration must name which — to within **±0.5 absolute**, or the gap
   explained as a definitional difference and the definition we use stated
   explicitly. Plan §2 already warns that the KSAA-2026 and CATT
   numbers may not share a definition; this is where that gets settled rather
   than hedged in prose. **The calibration must also name the gold column**:
   task 2 found the WikiNews distribution ships two golds differing on 3.72%
   of Arabic words, and only `original_gold` is the reference CATT measured
   against. Against `max_gold` this criterion would fail for a reason
   unrelated to our metric.

   > **[14 Oct 2026] MET — both clauses.** Jobs 224460 and 224461 on
   > `enderbit`; artifacts `run/wikinews_der.json` and `run/quranic_der.json`;
   > write-up in `insights/week-05.md`.
   >
   > **Gold-text clause.** `scripts/measure_quranic_der.py` scores all four
   > tools against two genuinely gold references — the 6,236 verses of the
   > Tanzil `simple` edition (sha256 `d9b5bad9…f47f`, 330,709 letter
   > positions) and the 98 recovered QuranMB prompt texts (4,442 characters),
   > reported separately. Both carry `reference_is_gold: true`, which is the
   > field that distinguishes them from `run/msa_der.json`, whose reference is
   > itself a vowelizer's output. Headline, published convention, case endings
   > included: `catt-eo` **4.599** on the corpus and **3.982** on the 98
   > prompts. The pre-declared ordering `eo < ed < shakkala < mishkal` holds on
   > both references and in both conventions.
   >
   > **Calibration clause — the branch that fired is neither of the two the
   > clause anticipated.** The clause offered ±0.5 agreement *or* an explained
   > definitional difference in prose. What `scripts/calibrate_der.py` leg A
   > delivers is stronger than both: scoring CATT's own recorded outputs
   > against their own gold file reproduces **all sixteen Table-5 figures to
   > within 0.0005** — the table's printing precision — for
   > `eo_mlm_ns_epoch_193`, `ed_mlm_ns_epoch_178`, Shakkala and Mishkal. The
   > variant is named per row, the gold column is `original_gold` as required,
   > and the two conventions differ by a *measured* offset (+1.04 for
   > `catt-eo` to +4.66 for `mishkal` on CE DER) rather than by an argued one.
   >
   > The ±0.5 band was pre-registered and is reported unwidened. Leg B, our
   > invocation of the four tools on CAMeL-Lab's 558-segment
   > `WikiNewsMax.tsv`: `catt-eo` **−0.221** (inside), `catt-ed` **−0.523**
   > (outside, by 0.023), `shakkala` −0.062, `mishkal` −0.179. All four deltas
   > are negative, which is the signature the decisive/diagnostic split exists
   > to detect — a bad invocation moves one row, a corpus difference moves all
   > four the same way, and leg B's 558 segments are not CATT's 400-line file.
   > The same-corpus test is leg A and it is exact, so `catt-ed`'s 0.023 is a
   > property of the corpus substitution and not of our metric. Stated plainly
   > because the band's whole purpose is to not be moved after the fact.
   >
   > **Both conventions are reported side by side in every artifact**, and the
   > consequence binds the rest of the project: no rate of ours is comparable
   > to a published DER unless scored by `metrics/der_published.py`.

3. **`docs/annotation-protocol-msa.md` is committed, with the crossover
   assignment fixed, before the first utterance of the 500 is annotated.**
   Verifiable from git history, which is the point.

**The failure branch, declared now.** If part 1 fails, that is the week-5
finding and it escalates the same day: §3.3b "What would reverse D1" says
both arms lose the label path, which is a larger problem than the MSA arm's
scope and must not be carried silently into week 6. Parts 2 and 3 do not
depend on part 1 and continue regardless.

If part 2's external gold set turns out to be unavailable (task 2), the
calibration clause is void and part 2 reduces to "RQ1 measured on the
Qur'anic domain with the metric definition stated and a self-consistency
check against the week-4 conservation ranking." Record the substitution;
do not quietly relax the criterion.

## Tasks

Ordered by what blocks what, not by size.

1. **▲▲ Clip → verse recovery probe** — the keystone. Promoted from chore to
   precondition by D1 (§3.3b): the Qur'anic arm's label path needs
   trustworthy diacritized Arabic to strip and re-diacritize, and the only
   text column that exists (`reference_arabic_string`) is circular — the
   uploader built it by phonemising the Qur'an and matching against
   `reference_phoneme_string`, shipping the residual as `match_distance`,
   which our own edit distances reproduce row for row.

   Cheap: 1,642 rows collapse to **96 distinct sentences**.

   **Try the three routes in this order, and stop at the first that works.**

   a. **Does `ID` encode surah:ayah?** Inspect the id scheme first. If the
      clip ids carry verse references, recovery is a lookup and the rest of
      this task is a verification step. This is fifteen minutes and it may
      delete the task; do it before writing a matcher.
   b. **Do the organizers publish the prompt list?** The Iqra'Eval test set
      was read from a fixed set of verses; `IqraEval/Iqra_Extra_IS26` and the
      shared-task materials are the places to look. A published join is
      better evidence than a matcher we wrote.
   c. **Consonant-skeleton match against an authoritative text.** Only if (a)
      and (b) fail.

   **The circularity trap, stated so it is not walked into twice.** The
   obvious validation — "does the recovered verse agree with
   `reference_arabic_string`?" — reproduces `match_distance` and measures the
   uploader's recovery, exactly as week 4's void exit criterion did. The
   independent check is available only for routes (a) and (b), where the
   mapping came from metadata rather than from the phonemes: then
   `phonetize(recovered_verse) == reference_phoneme_string` is a genuine test,
   and agreement is evidence. Under route (c) the matcher is *the same
   algorithm the uploader ran*, so neither check is independent and the result
   must be reported as a reproduction, not a validation. Prefer (a)/(b)
   accordingly, and if only (c) is available say so in the write-up.

   **Choose the script edition deliberately.** Tanzil-style distributions ship
   Uthmani and Imlaei orthographies. Uthmani carries alef wasla (U+0671) and
   the Qur'anic pause marks (U+06D6–U+06ED); Imlaei largely does not. So the
   edition choice silently decides whether §9's falsifier fires. Picking
   Imlaei to dodge it is defensible — but then **measure that the dodge
   worked** (`unsupported_characters` over the whole chosen text, expecting
   zero U+0671) rather than assuming it, which is the mistake week 4 caught
   four separate times. Record the edition, its source and a content hash.

   **Re-validate `skeleton()` on Qur'anic orthography before using it as the
   match key.** `msa_pool.skeleton` was built and measured on MSA only, and
   `is_arabic_letter` treats U+0671 as a letter — so alef wasla *survives*
   stripping, and any verse the Uthmani text writes with U+0671 where our
   query text has a bare alef will fail to match. That is a concrete
   prediction; check it rather than discovering it as an unexplained
   mismatch rate.

   Deliverables: `scripts/recover_quran_verses.py`, the mapping committed as
   `run/quranmb_verse_map.json` (with the route used and the edition hash
   recorded in it), and an `unsupported_characters` report over the recovered
   text.

   > **STATUS 2026-10-13 — DONE via route (c) + an audio pass. Substitution
   > recorded, not quietly taken.**
   >
   > **Routes (a) and (b) are both closed, for different reasons.** (a) was
   > inspected first as instructed and closed *negative*: the id scheme carries
   > a speaker index and a prompt index, no verse reference. (b) is closed by
   > evidence rather than by effort: `test_references` carries ID + phoneme
   > string only, the `QuranMB.v2` repo carries ID + audio only, and
   > `IqraEval/Iqra_Extra_IS26` — named above as the place to look — is a
   > **gated repo returning HTTP 401**. So the published join may well exist
   > and is simply not reachable; that is recorded as "unavailable to us", not
   > as "does not exist".
   >
   > **Route (c) therefore ran, and the paragraph above binds**: the matcher is
   > the same algorithm class the uploader ran, so 98/98 agreement with
   > `reference_arabic_string` is a *reproduction*. It is reported as such.
   > This is also why the §4.1 restatement rules the written independence
   > clause unsatisfiable — (c) consults the forbidden column by construction.
   >
   > **The substitution is an instrument, not a weaker threshold.** Independence
   > was restored by `scripts/verify_verse_map.py`: a human listening to the
   > shipped audio, which was not generated from either text column. Judged on
   > **word identity, not pronunciation accuracy** — QuranMB.v2 is a
   > mispronunciation benchmark, so judging pronunciation would reject correct
   > recoveries and leak the benchmark's own labels into a text decision.
   > Result: **98/98 confirmed**, after 5 corrections the audio caught and the
   > text columns could not.
   >
   > **What the audio bought, stated concretely, because this is the argument
   > for having done it.** The matcher assumed a prompt is a contiguous run of
   > *whole Qur'anic words*. QuranMB.v2 trims and adds proclitics relative to
   > the Qur'an, and one prompt's reference begins partway through a word. Four
   > spans were wrong (00035, 00036, 00069, 00088) and no text-column check
   > could have found them — `reference_arabic_string` is derived from
   > `reference_phoneme_string`, so it carries the identical wrong boundary.
   > A fifth (00087) was *correct* and the audio was the anomaly. All five are
   > recorded in `AUDIO_OVERRIDES` with what was heard and why, and every
   > entry keeps `skeleton_text` beside `recovered_text`, so each correction is
   > auditable rather than absorbed into a tuned matcher.
   >
   > **One finding that belongs to week 7, logged here because it was measured
   > here.** Over all 1,642 clips, `duration ≈ 1.53s + 0.083s × n_phones`.
   > Every clip carries ~1.53s of material outside the annotated phone
   > sequence — ~3.3 words at the marginal rate — and 00087 proves it is
   > sometimes audible speech. The benchmark's labels do not describe it, so an
   > MDD model fed whole clips will be charged for insertions that are really
   > unlabelled audio.

2. **▲ Acquire the WikiNews gold set — and check its licence.**
   **Acquisition closed 2026-10-03**; see
   [insights/week-05.md](../../insights/week-05.md). The set is
   `CAMeL-Lab/wild_diacritics` → `data/WikiNewsMax.tsv` (ACL 2024 "Arabic
   Diacritics in the Wild"), data **CC-BY-SA-4.0**, cite Darwish et al. 2017
   additionally. This resolved the only unresolved item in
   [annotator-qualification.md](../annotator-qualification.md) §5. It gates two
   otherwise-unrelated things, which is why it was this high:

   - the **qualification test material**, and therefore the annotation block;
     and
   - the **RQ1 metric calibration** in task 3, since published machine DER
     exists on the same data.

   **These are two slices, not one.** Calibration needs the **full** corpus —
   a corpus-level DER must be computed over the corpus the published number
   was computed over. Qualification needs a **~20-segment sample** from the
   293 segments in the 8–40-Arabic-word band. The original plan conflated
   them.

   Three corrections to what this task assumed:

   - **The "70" is not shared.** WikiNews is 70 *articles* (~18,300 words in
     the literature; 14,613 Arabic tokens as we count them), not 70 sentences.
     §6's "70 utterances" comes independently from its own mark-count power
     calculation. The two numbers coinciding is a coincidence, and treating it
     as causal would make them wrongly co-vary.
   - **The sizing unit is wrong by ~3.5×.** §6 targets ~202 bucket-B and
     ~1,067 bucket-A positions, derived from `Iqra_train` dev at 19.4
     marks/utterance. A WikiNews segment carries ~76 marks (4.01/word, 19-word
     median), so 70 segments overshoots to ~1,300 / ~4,000 positions at ~4×
     the annotator hours. ~20 segments hits both targets. **Needs explicit
     sign-off before administering** — it is a re-derivation in the correct
     unit, not a relaxation of a pre-registered threshold, but it has a cost
     attached.
   - **Calibration targets `original_gold`, and task 3 must name the column.**
     The file ships both the Darwish 2017 gold and the ACL 2024 maximal gold;
     they differ on **3.72% of Arabic words**, which exceeds the ±0.5 DER
     tolerance. `original_gold` is what CATT's 5.425/3.105 was measured
     against. This is plan §2's definitional warning, now with a named
     mechanism.

   If the licence turns out to block the use after all, fall back to §5's
   reserve — the ~20% of `Iqra_train` rows carrying marks in the source
   text — and **audit the marks' provenance first**, which §3.3a never did.
   That fallback is weaker in a specific way worth writing down: it has no
   published machine DER on it, so it buys the qualification test but not the
   calibration, and task 3's exit clause changes accordingly.

   **Still open, and this is the half that gates annotation: administer the
   test** (§7) and score it (§8). Both annotators, same items, from scratch
   rather than correct-the-machine, buckets A and B scored separately, elapsed
   time recorded — the timings are what set D4's extension decision. Do not
   re-read §9's pass marks before scoring; they are pre-registered. Carry one
   addition into §8: `max_gold_alt` enumerates 359 words (2.46%) with two
   defensible readings, so scoring needs an **acceptable-alternatives
   allowance** or it charges annotators for the language's own
   underdetermination.

3. **RQ1: the DER/WER module, and measurements on both domains.**

   New module, `src/arabic_mdd/metrics/der.py`, with tests. Two things it must
   get right, both of which are definitional rather than algorithmic:

   - **Variants are explicit parameters, not a default.** With and without
     case endings; and whether letters that carry *no* diacritic in the gold
     count in the denominator. Published numbers differ on both axes and plan
     §2 flags exactly this when putting CATT's 5.43 next to KSAA-2026's 10.56.
     Name each variant in the API so a reported number cannot be ambiguous
     about which it is.
   - **The stripping step is pinned and shared.** RQ1's input is gold text
     with diacritics removed, and the clip→verse matcher needs the same
     skeleton function. Promote `strip_diacritics` / `skeleton` out of
     `msa_pool.py` (a sampling-frame module) into `normalize.py`, re-exporting
     from `msa_pool` — and **regression-test that the committed 500-row draw
     is byte-identical afterwards**. If that refactor changes the block, the
     block silently stops being the one D4 decided, which is the worst
     available outcome for a one-line move.

   Then measure, per tool, on both domains:

   - **Qur'anic.** Gold is genuinely gold: the Qur'an's vowelization is
     authoritative. Strip, re-diacritize, score. Report at corpus level over
     the chosen edition, and separately over the QuranMB subset once task 1
     delivers the map. This is RQ1's one clean number this week.
   - **MSA deployment corpus, with a caveat that must be in the number's
     label.** On `Iqra_train` dev the only reference available now is
     `tashkeel_sentence`, which is the organizers' **vowelizer's output, not
     gold** — week 4 measured it wrong on 43.6% of pilot utterances. So DER
     against it is *agreement with the organizers' pipeline*, not accuracy,
     and reporting it as accuracy would invert the paper's own finding.
     Report it under that name. RQ1-MSA proper is computed against `C_gold`
     when the block is annotated (task 7's D5), and the two numbers together
     are more interesting than either alone.

   Sanity check before reporting: does the DER ordering match week 4's
   conservation tail (mishkal worst by an order of magnitude) and the NAACL
   Table 1 ordering? Two independent instruments already agree; a third
   disagreeing means the metric is wrong.

   **STATUS 2026-10-14 — DONE, both halves.** The Qur'anic half is measured
   and calibrated; details below the 10-03 entry it supersedes.

   - Module: `src/arabic_mdd/metrics/der.py` + `tests/metrics/test_der.py`
     (26 tests). Four named variants, no default; alignment **refused**
     (`skeleton_mismatch`) rather than approximated.
   - Refactor: `strip_diacritics` / `skeleton` / `is_arabic_letter` /
     `DIACRITICS` promoted to `data/normalize.py`, re-exported from
     `msa_pool`. `tests/data/test_annotation_block_regression.py` rebuilds the
     pool and the seeded draw and confirms the committed 500-row block is
     reproduced field-for-field, in order. **The block is unchanged.**
   - MSA measurement: `run/msa_der.json`, all four tools × four variants,
     labelled `reference_is_gold: false`. `catt-eo` 1.346 `ce_all` DER.
   - **Sanity check PASSES on both instruments** — DER ordering
     `eo < ed < shakkala < mishkal` is exactly the conservation-tail ordering,
     and Mishkal stays far below Shakkala as in NAACL Table 1. Mishkal is 17×
     `catt-eo`.
   - **Qur'anic half DONE 2026-10-14** (job 224461), unblocking what the
     10-03 entry below recorded as blocked. `run/quranic_der.json`,
     `reference_is_gold: true`: all four tools over the full 6,236-verse
     Tanzil `simple` edition plus the 98-prompt QuranMB subset, two
     conventions, `sampled: false`, 0 chunking seams. Published convention CE
     DER: `catt-eo` **4.599**, `catt-ed` 4.897, `shakkala` 9.448, `mishkal`
     18.499. Ordering check **HOLDS**, so the third instrument agrees with the
     two that already did.
   - **Calibration DONE** (`run/wikinews_der.json`). `metrics/der_published.py`
     reproduces **all sixteen** of CATT's Table-5 figures to three decimals,
     on two platforms. Our convention's offset is therefore measured
     (+1.04 `catt-eo` to +4.66 `mishkal` on CE DER), not assumed. Every rate
     this project reports must now name its convention.
   - **The finding that outranks the Qur'anic rate itself:** `catt-eo`'s MSA
     1.346 was an *agreement* rate and understated its error ~5× against gold
     (6.736 Qur'an, 6.346 WikiNews), while the other three tools put the same
     comparison at only 1.05–1.22×. See task 8's amended status — this bears
     directly on D7.
   - Two of three pre-registered predictions **failed**, both informatively.
     Scored in full in `insights/week-05.md`.

4. **▲ Pre-register the main-block annotation protocol** —
   `docs/annotation-protocol-msa.md`, committed **before** any of the 500 is
   annotated. §10 of the qualification doc is explicit that the crossover
   **cannot be reconstructed after the fact**, which makes this the one task
   here with an irreversible deadline rather than a cost.

   Must fix, concretely enough to execute without a judgment call:

   - **Correct-the-machine on the text-only CATT anchor** (§3.2), with the
     variant named once task 3 decides it.
   - **The crossover assignment**, by draw rank, written out: which annotator
     sees the suggestion and which starts from a blank page, on which ranks,
     and the swap. The first 50 in draw order are the sub-block (§3.3b D4).
   - **The pausal convention is not the annotator's problem.** They write
     fully marked Arabic; `phonetizer.normalize` applies the convention
     mechanically. Say so, and carry §4's table of which positions are
     therefore free (6.52% of diacritic positions, harmless by construction).
   - **The §4.1.1 contract, as instructions.** Leave Latin script untouched;
     punctuation is handled downstream; the `UNFAIR` classes
     (`latin`, `digits`, `no_msa_equivalent`) are excluded from the
     phoneme-level denominator and reported as their own line — 25 rows of
     71,391 at corpus level, **1 row of the 500** (rank 380, id 64376). The
     pool is the sampling frame; `UNFAIR` is the evaluation denominator; the
     flag joins them.
   - **Adjudication routing**, taken from task 2's bucket-B result via §9's
     pre-registered table. Flag-and-adjudicate is the expected mode.
   - **The analysis plan for the anchoring effect**, written before the data
     exists: what is compared, on what positions, and what magnitude would
     force correct-the-machine to be abandoned for the affected subset (§9's
     last falsifier).
   - **The `A`-on-50 hedge** (§3.3b): annotate `A` on the crossover sub-block
     only, at negligible cost, so the base-rate estimate keeps accumulating
     for a limitations sentence at n=136 rather than n=86. This does not
     reopen D1 and the protocol should say that explicitly.

   **STATUS 2026-10-04 — DONE. `docs/annotation-protocol-msa.md` is written,
   9 sections, pre-registered and uncommitted-to-data.** Every bullet above
   is covered; four were answered differently from how they are posed here,
   and the differences matter more than the coverage:

   - **The pausal bullet above is wrong, and it was wrong in the
     qualification doc too.** "The pausal convention is not the annotator's
     problem" holds only for the utterance's *final* word. A non-final word's
     final mark survives into the phoneme string, and so does the
     liaison/sandhi choice before `ال`. Measured: `canonical` for id 63729
     has no helping vowel in `مِنْ الْمَاءِ`, and `phonetize("مِنْ")` ≠
     `phonetize("مِنَ")`. **31 positions in the block, 1.49% of non-final
     words.** The protocol states four explicit rules (R1–R4) instead of the
     reassurance. Also corrected in `annotator-qualification.md` §3 and
     `msa-arm.md` §3.4.1.
   - **The crossover is both annotators on all 50**, roles swapped at rank
     25, not 25 utterances each. Same annotator-hours, but it yields
     agreement on 50 rather than 25 and balances annotator identity against
     condition, so a skill difference cannot be read as anchoring. It also
     runs **first**, before any of the 450 — run last it would measure
     anchoring in annotators already fluent in `catt-eo`'s habits.
   - **The anchoring analysis plan has numeric bands** (Δ < 5pp / 5–15pp /
     > 15pp on bucket B), and the >15pp branch abandons correct-the-machine
     for **case endings only**, not for the block. A full abandonment costs
     ~2× the hours to fix a bias §3.3a localizes to bucket B.
   - **The `A`-on-50 hedge is DECLINED**, and not on cost grounds. `A` needs
     audio; §3.2's own argument against audio-informed anchoring is that
     audio evidence corrupts the reference at exactly the mispronounced
     positions. Spending that risk on the one sub-block that cannot be re-run
     is a bad trade. Deferred to a disjoint set.

   Two further commitments the bullets did not ask for: the per-row **logging
   schema** is fixed (pilot §6a's unrecoverable-override-rate regret, closed
   — `suggested` and `final` stored separately per word, plus `final_before`
   for adjudication), and anchoring on an *evaluated* tool is acknowledged to
   bias **RQ1's `catt-eo` cell** in the flattering direction even though it is
   self-penalizing for D5. That cell now carries a pre-committed caveat.
   Pilot §6a's "starting-point tool must never enter the evaluated set" is
   formally replaced rather than quietly broken.

   Still blocking the first annotated utterance: **no tool writes the
   schema.** `pilot_annotate.py` is a listening interface and cannot be
   reused.

5. **Decide the frontier fifth condition** — handed to week 5 by task 6 of
   week 4, and it has a real deadline: week 6's label-path column multiplies
   by the number of conditions, so deciding after that work starts means
   redoing it.

   The question is whether to add one frontier system so the 2×2 brackets the
   frontier rather than only the deployed range, answering "frontier
   diacritizers are at ~1.4% DER, is the premise obsolete?" with a measurement
   instead of an argument.

   **The evidence is weaker than week 4's summary of it, and the correction
   matters here.** Sadeed's 5.25/3.11 is reported against **CATT ED**
   (5.963/3.631). Against **CATT EO** — the variant we would actually
   deploy, at 5.425/3.105 — Sadeed wins by 0.17 with case endings and
   **loses by 0.005 without**, i.e. is tied. So "a genuine successor beats
   CATT" is not what the numbers say; Sadeed has no located weight release
   anyway. What survives as a real threat is the LLM line: LREC 2026 finds
   off-the-shelf LLMs beating specialized diacritizers, and Sadeed's own
   table puts Claude-3-7-Sonnet at **1.39** DER — a 4× gap, not a 3% one.
   If a frontier condition is added, it should be an LLM for that reason,
   and the Sadeed comparison should not be cited as motivation for it.

   Decide on three stated criteria, not on appeal:

   - **Reproducibility.** An API model cannot be pinned to a sha, which is the
     actual objection — §5.1 pins everything else across both arms. A frontier
     condition would be the only unpinnable element in the design. If it is
     added, it is added with a dated snapshot of inputs and outputs committed
     as the artifact, and the limitation stated.
   - **RQ4 compatibility.** Frontier APIs do not expose per-position
     confidence, so such a condition can serve RQ1/RQ2 but **cannot** serve
     RQ4's abstention curve. CATT stays primary for that reason regardless of
     the outcome here.
   - **Cost**, in money and in week-6 schedule.

   Cheapest sequencing: run it through task 3's RQ1 corpora first — that is a
   small number of API calls — and decide on its measured DER on *our*
   domains rather than on WikiNews. Record the decision in `docs/msa-arm.md`
   as a numbered decision, whichever way it goes, so a reviewer question has
   an answer with a date on it.

   **STATUS 2026-10-14 — DONE. No frontier fifth condition; the evaluated set
   stays at four. Recorded as D8 in [docs/msa-arm.md](../msa-arm.md).** The
   three criteria were applied in the order of weight stated there, and the
   **RQ4 incompatibility is the disqualifying one**: a frontier API exposes no
   per-position confidence, so the condition could serve RQ1/RQ2 but could not
   appear in RQ4's abstention curve — the paper's remedy and contribution 4.

   **The sequencing above could not be executed as written, and the
   substitution is recorded rather than quietly taken.** It assumes our
   domains can support an interpretable frontier DER. Task 3 measured them and
   they cannot, yet:

   - **Qur'anic** is gold and available, but the Qur'an is in any web-scale
     pretraining corpus *with its diacritics*, so a frontier DER there
     substantially measures memorization. Not speculation — it is the extreme
     of a gradient this week measured on our own four tools, where the three
     trained models all **improve** on the Qur'an relative to their published
     WikiNews figures (−0.530 to −1.066) and only rule-based Mishkal degrades
     (+3.253). That is prediction 1's falsification pointing at a second
     consequence.
   - **MSA** has no gold until the 500-row block is annotated, and its only
     diacritized column is `tashkeel_sentence` — which task 8's amendment
     shows is very likely `catt-eo`-derived. A frontier tool scored against it
     would yield agreement with a competitor's output.

   So the decision is **declined as a condition, scheduled as a measurement**:
   one RQ1 row on the MSA pool against `C_gold` once annotation closes, which
   lands *before* the weeks-10–12 2×2. That timing is what makes declining
   cheap rather than final, and D8 carries three declared re-opening
   conditions.

   One week-4 claim is retired in passing. Claude-3-7-Sonnet's **1.39** against
   CATT EO's **5.425** is a 4.04 absolute gap measured on a different corpus in
   a third convention, and our own WikiNews → Qur'an movement for a *fixed*
   tool spans −1.066 to +3.253. It is not safely a tool-quality gap — which is
   an argument for measuring it on a corpus we control, and against citing it
   as established.

6. **▲ Select the MSA evaluation pool for §5.2.** CV-Ar rows disjoint from
   `Iqra_train`, deduped **by consonantal skeleton, not raw string** — §3.3b
   D4 is the cautionary case: raw-string dedup reports `Iqra_train` dev as
   duplicate-free and misses 46 rows, 33 of whose 44 groups are the same verse
   once bare and once pre-vowelized. `msa_pool.py` already does skeleton
   dedup within dev; reuse it rather than writing a second one, which is
   another reason task 3 promotes the helper.

   Cite Sadeed's precedent while doing it: 34.6% of the Fadel test set is
   fully present in the Abbad training set (68.12% above 0.5 similarity). The
   discipline is standard practice that the field frequently skips, and saying
   so costs one sentence.

7. **Close D5, and take D2 and D6** (§3.3b "Still open").

   - **D5 — recommended yes, and task 3 makes it nearly free.** Promote the
     43.6% vowelizer error rate from a single-annotator listening judgment to
     an objective `tashkeel_sentence` vs `C_gold` DER/WER over the annotated
     set. The pilot's figure is retained as the pilot's own, with its 39.5%
     sensitivity. External sanity range from week 4 task 5: the NAACL paper's
     Table 3 puts text-only diacritizers at **19–36% DER on MSA speech**, so a
     D5 number far outside that band is a bug before it is a finding.
   - **D2** — whether `A := C_gold` is stated as an assumption with the
     2.3%/19.8% band as a sensitivity row, or the residual interaction term is
     bounded explicitly. The measurement is a delta (FR under `C_auto` − FR
     under `C_gold`, same `P`), so real mispronunciations appear in both terms
     and cancel to first order. This is a question about how much rigour to
     spend, so decide it against the week-6 power estimate rather than in the
     abstract — but write down which way it will be decided and on what.
   - **D6** — whether `Iqra_TTS`'s augmented rows become a clearly-labelled
     synthetic detection sub-arm (§3.3 option 3). `run/tts_detection.slurm`
     exists and the annotation cost is zero, so the only real question is
     whether a synthetic sub-arm helps or dilutes. If used, the synthetic
     fraction goes in abstract-adjacent text, not a footnote.

8. **Choose the CATT variant, by the rule fixed now.** Week 4 deferred EO vs
   ED to this week's DER measurement, having seen ED visibly worse than EO on
   one long sentence — against CATT's own claim, on evidence too thin to act
   on. Pre-commit the decision rule *before* looking at task 3's output:
   **choose on DER with case endings on the MSA deployment corpus; tie-break
   (within 0.3 absolute) on the week-4 conservation tail**, where EO leads
   0.08% to 1.04%.

   Note that three independent lines already point at EO — CATT's own Table 5
   (5.425/3.105 against ED's 5.963/3.631), the conservation tail, and week 4's
   one-sentence observation — so the live question is really whether ED wins
   *on our domains* by enough to overturn that. State the rule as such: ED is
   chosen only if it leads by more than 0.3 absolute, otherwise EO. Whichever
   is chosen, `catt-tashkeel` stays pinned
   `==1.0.2` with the runtime check, because the confidence path rides on
   private internals.

   **STATUS 2026-10-03 — DONE. `catt-eo`, recorded as D7 in `docs/msa-arm.md`.**
   The rule was applied unchanged and **the margin never came into play**: EO
   leads by **4.556** absolute on `ce_all` (1.346 vs 5.902), 15× the 0.3
   threshold, so the conservation tie-breaker was not needed. The fourth
   independent line now agrees with the other three. Pin confirmed:
   `catt-tashkeel==1.0.2` in `pyproject.toml`, `SUPPORTED_CATT_VERSION` checked
   at runtime, and the version is also written per-tool into
   `run/msa_der.json`'s `provenance`.

   One thing this did **not** settle, and it is the run's most consequential
   finding: `catt-eo` agrees with a reference known wrong on 43.6% of
   utterances at 98.65% of positions, which implies the two share error modes
   (or the organizers' vowelizer is CATT-derived). If `C_auto` and CATT are
   correlated, D5's error-propagation headline is **understated**. See
   `insights/week-05.md`, observation 6.

   **AMENDED 2026-10-14 — D7 stands; the word "decisively" does not.** Gold
   text (`run/quranic_der.json`, `run/wikinews_der.json`) shows the 4.556
   margin above was an artifact of the reference, not a property of the tools:

   | reference | `catt-eo` | `catt-ed` | gap |
   |---|---|---|---|
   | MSA agreement (`tashkeel_sentence`) | 1.346 | 5.902 | **+4.556** |
   | WikiNews gold | 6.346 | 6.649 | +0.303 |
   | Qur'an gold | 6.736 | 7.121 | +0.386 |

   EO still leads on **both** gold references, so the decision does not
   change and the pre-committed rule was applied honestly. But the margin
   shrinks ~12×, to roughly the 0.3 threshold the rule set — meaning the
   tie-breaker **would** have come into play had this reference been used. The
   claim to retire is that EO won by a wide margin; it won narrowly.

   The 10-03 ambiguity above is also now largely resolved in favour of its
   second branch. Agreement-rate inflation is **5.00× for `catt-eo`** but only
   1.05–1.22× for the other three tools, and a generically shared error mode
   would inflate all four. `tashkeel_sentence` looks EO-derived.
   **Open, and cheap to close: ask the organizers which vowelizer produced
   `tashkeel_sentence`.** That single fact determines whether D5's headline is
   understated and whether D7 is a genuine choice or a coin-flip.

9. **Housekeeping, each small, each closing a known caveat.**

   - **Round-trip the full `Iqra_train` train split** (71,391 rows). Week 4
     validated dev (2,588) and explicitly flagged that the train split had not
     been run. Pure regression, no new judgment, closes the caveat.

     **DONE 2026-10-03 — 99.93% (71,344/71,391), PASS, better than dev's
     99.85%; TER 0.0000, no out-of-inventory token.** All 47 mismatches are
     one upstream phenomenon (a shadda written on a long-vowel carrier),
     20 distinct sentences. Required one fix: `load_iqra_train` now drops the
     `audio` column, mirroring `quranmb.load_ground_truth`, or iterating train
     rows raises `ImportError: ... install 'torchcodec'`. **Two of the 500
     drawn block rows are affected (ranks 447 and 475)** — a task-4 protocol
     gap, since the protocol does not say what an annotator does with text
     that is orthographically malformed upstream.
   - **`quranmb.py` does not read `match_distance`** even though the column is
     what proved the text circular. Expose it on `MDDGroundTruth` so the
     circularity is visible in the loader rather than only in a week-4 doc.

     **DONE 2026-10-03.** `match_distance: int` on `MDDGroundTruth`, parsed in
     `parse_ground_truth_rows`, plus a `prompt_index` property. Verified
     against the real data: `match_type` is *exactly* `match_distance == 0`
     (904 exact / 0 with nonzero distance; 738 fuzzy / 0 with zero distance),
     so the string column carries no information the integer does not.
   - **Delete `run/msa_pilot_annotated copy.json`** — still untracked on disk
     (6,224 bytes), already out of git. Mentioned because it was believed
     deleted and is not.

     **NOT DONE** — untracked, so deletion is unrecoverable; left for an
     explicit go-ahead.

## Carried forward

Unchanged from week 4 unless noted; listed so nothing is lost between
sessions.

- **Pin the S3PRL checkout.** `run/prepare_baseline.sh:81` clones
  `s3prl/s3prl.git` at whatever `main` serves that day, and S3PRL supplies the
  featurizer, the downstream model *and* the training loop — a strictly larger
  reproducibility exposure than the upstream weights were. Needs a commit
  chosen against a venv known to build, i.e. a cluster round-trip, which is
  why it has not been done in-session. **Do it this week**: week 7 starts
  training new arms, and after that an unpinned training loop contaminates
  results rather than just the record.
- **Licence terms for `QuranMB.v2` / `Iqra_Extra_IS26`** — open since week 2.
  Common Voice is CC0 and the annotated slice is a releasable artifact in its
  own right (plan §Data, contribution 5), so this gates a deliverable. Task 1
  route (b) touches `Iqra_Extra_IS26` anyway; resolve it in the same pass.
- **Wall-clock cost of one XLS-R-300m fine-tune, and the week-7 work split** —
  open since week 1. Week 7 is when the two arms converge and need an assigned
  owner, so this **cannot slip past week 6**.
- **`scripts/verify_quranmb_labels.py`** — optional. The residual value is
  per-row, plus identifying the one utterance by which the `safikhan` join
  (1,642) differs from the published count (1,643).
- **D4's extension decision** (~1,592 further eligible rows) is deliberately
  deferred until task 2 yields real per-utterance timings. The seeded draw
  makes an extension a superset of the committed block, so this is a volume
  decision and not a sampling one.

## Out of scope for week 5

- **The label-path column itself** and the RQ5 interaction power estimate —
  week 6, and the gate. Week 5 only has to make them possible.
- **Any model training**: the XLS-R prompt-free arm, the text-dependent arm
  and the MSA label path are weeks 7–9. The validated `mhubert147_per` recipe
  stands.
- **Reopening D1.** It is conditional on task 1 and on nothing else. A
  disappointing base rate, a slow annotator or a tempting audio-informed
  estimator are not reasons; §3.3b lists the two that are.
- **Revisiting the paper's framing**, pre-committed in plan §5.
- **Partial diacritization and cross-lingual validation**, held back
  deliberately in plan §2.
