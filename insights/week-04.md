# Week 4 — Diacritizer pipeline and phonetizer

Plan: [docs/weeks/week-04.md](../docs/weeks/week-04.md). Update this file as
each task below produces a result — do not wait until the end of the week.

## Logistics log

Track non-experimental tasks here as they close. One line each, dated.

- [ ] Diacritizer wrappers (CATT EO/ED, Shakkala, Mishkal or Farasa) under
      `src/arabic_mdd/diacritizers/`, with tests.
- [ ] CATT token-probability extraction vendored behind a tested interface;
      technique confirmed on the ED checkpoint, not just EO.
- [ ] `utter-project/mHuBERT-147` pinned to a commit sha.
- [ ] 2026 diacritizer survey chased; diacritizer set finalised.
- [ ] Licence-terms confirmation for `QuranMB.v2`/`Iqra_Extra_IS26` —
      carried from week 2.
- [ ] Wall-clock cost of one XLS-R-300m fine-tune, and the week-7 work-split
      decision — carried from week 1–2. Cannot slip past week 6.

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
