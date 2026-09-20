# Week 1 — Setup and access

Plan: [docs/weeks/week-01.md](../docs/weeks/week-01.md). Update this file as
each task below produces a result — do not wait until the end of the week.

## Logistics log

Track non-experimental tasks here as they close. One line each, dated.

- [ ] Dataset access requested (HF account, IqraEval org, `QuranMB.v2`,
      `Iqra_Extra_IS26` licence terms, v2 test label availability)
- [ ] Cluster allocation and queue limits confirmed; XLS-R-300m fine-tune
      wall-clock estimate recorded
- [ ] QuranMB v1/v2 overlap resolved
- [ ] Work split agreed and recorded
- [ ] Differentiation text drafted (`docs/differentiation-kadambi-mathad.md`)
- [ ] Effect-size position drafted and signed off
      (`docs/effect-size-position.md`)
- [ ] Residual novelty-check queries run, results in `docs/novelty-check.md`

## Experiment: CATT token-level probability exposure

### Hypothesis

CATT's public inference interface exposes per-token (per-diacritic)
confidence scores or logits, sufficient to build the confidence threshold
that RQ4 (selective scoring by abstention) depends on.

### Methodology

Inspect the CATT public interface
([github.com/abjadai/catt](https://github.com/abjadai/catt)): run inference
on a sample sentence and check whether the returned output includes
token-level probabilities/logits, or whether they are reachable with a small
patch to the inference code. If not exposed and not easily patchable,
evaluate a substitute diacritizer from the Week 1–3 tool set (Shakkala,
Mishkal, Farasa) for the same property.

### Configuration

- CATT checkpoint/variant: TBD (EO vs ED)
- Sample input(s): TBD
- Environment: TBD

### Results

_Pending — fill in after running the check._

### Observations

_Pending._

### Conclusion

_Pending. If probabilities are not exposed, record the decision (patch vs.
substitute) and update_ [docs/weeks/week-01.md](../docs/weeks/week-01.md)
_task 4 accordingly._
