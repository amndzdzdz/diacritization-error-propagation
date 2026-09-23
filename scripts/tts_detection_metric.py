"""The MSA arm's detection metric on real positives -- `Iqra_TTS` augmented rows.

**Why this exists.** Every MSA-arm measurement so far has had to fake `A`.
`scripts/cv_dev_baserate.py` set `A := C` because Common Voice ships no
verbatim-production annotation, which zeroes TR and FA by construction and
makes precision, recall and F1 undefined as quantities of interest. The
zero-base-rate control had the same shape. Neither produces a detection
number.

`Iqra_TTS`'s **augmented** rows do. They carry both:

* `phoneme_ref` -- the canonical sequence, `C`
* `phoneme_mis` -- the sequence actually synthesized, `A`

and they differ (mis != ref in ~99% of augmented rows, measured). So
`(C, A, P)` is complete and the hierarchical metric can be computed for
real: TA/TR/FA/FR, precision, recall, F1, plus the diagnosis split. This is
the **only MSA-domain data in the project with genuine detection
positives**, and the organizers built it for exactly this reason.

**What it is and is not.** This is `docs/msa-arm.md` §3.3 option 3
(controlled injection) evaluated on data that already exists. It gives the
MSA arm:

* a working end-to-end check that the metric behaves on MSA phonetics;
* an MSA-side F1 directly comparable to the Qur'anic arm's 0.4406; and
* a fallback headline number if the real-speech base rate turns out to be
  too low to support option 1.

It does **not** measure real speech. The mispronunciations are injected,
the audio is synthetic, and `Iqra_TTS` is inside the 131 h training set --
so treat the F1 as an upper bound and a sanity check, not as the paper's
result. It also cannot settle the base-rate question; only step 4 can.

Run with:
    uv run python scripts/tts_detection_metric.py run/tts_augmented_predictions.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from arabic_mdd.data.phonemes import parse_phoneme_sequence
from arabic_mdd.metrics.hierarchical import aggregate, compute_metrics, evaluate_utterance

# The Qur'anic arm's numbers from our own retrain (insights/week-03.md).
QURANMB_F1 = 0.4406
QURANMB_PRECISION = 0.3093
QURANMB_RECALL = 0.7707
QURANMB_FR_RATE = 0.1241

# See tts_control_inference.DROP_TOKENS -- `<sil>` is not in the 68-token
# vocab, and only `phoneme_ref` carries it. Repeated here so the two stages
# cannot silently disagree.
DROP_TOKENS = {"<sil>", "sil", "<unk>"}


def _tokens(raw: str | None) -> list[str]:
    return [tok for tok in (raw or "").split() if tok not in DROP_TOKENS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records_json")
    args = parser.parse_args()

    records: dict[str, dict[str, str]] = json.loads(Path(args.records_json).read_text())
    print(f"{len(records)} utterances loaded from {args.records_json}")

    counts_list = []
    skipped = 0
    identical = 0
    speakers: Counter[str] = Counter()
    for rec in records.values():
        canonical = parse_phoneme_sequence(" ".join(_tokens(rec.get("phoneme_ref"))))
        annotation = parse_phoneme_sequence(" ".join(_tokens(rec.get("phoneme_mis"))))
        prediction = parse_phoneme_sequence(rec.get("prediction") or "")
        if not canonical or not annotation:
            skipped += 1
            continue
        if canonical == annotation:
            # No injected error in this row -- it contributes only negatives.
            # Counted and reported rather than dropped, because excluding it
            # would inflate precision.
            identical += 1
        counts_list.append(evaluate_utterance(canonical, annotation, prediction))
        speakers[rec.get("speaker") or "?"] += 1

    if skipped:
        print(f"  skipped {skipped} rows missing a reference or annotation")
    print(f"  rows where A == C (no injected error): {identical}/{len(counts_list)}")
    print(f"  speaker mix: {dict(speakers.most_common())}")

    totals = aggregate(counts_list)
    metrics = compute_metrics(totals)

    print("\n" + "=" * 72)
    print("MSA DETECTION METRIC  --  Iqra_TTS augmented rows, real (C, A, P) triple")
    print("=" * 72)
    print(f"  utterances scored: {len(counts_list)}")
    print(f"  counts: {totals}")

    print(f"\n  {'':<14s}{'MSA (TTS aug)':>15s}{'Qur.anic (QuranMB)':>22s}")
    rows = (
        ("F1", metrics.f1, QURANMB_F1),
        ("precision", metrics.precision, QURANMB_PRECISION),
        ("recall", metrics.recall, QURANMB_RECALL),
        ("FR rate", metrics.frr, QURANMB_FR_RATE),
    )
    for name, mine, theirs in rows:
        print(f"  {name:<14s}{mine:>15.4f}{theirs:>22.4f}")
    print(f"  {'FA rate':<14s}{metrics.far:>15.4f}{'--':>22s}")
    print(f"  {'DER':<14s}{metrics.der:>15.4f}{'--':>22s}")

    print("\n  READ: unlike every previous MSA measurement, TR and FA are nonzero")
    print("  here, so precision/recall/F1 are real quantities rather than")
    print("  artifacts of setting A := C. This is the MSA arm's metric path")
    print("  working end to end on MSA phonetics.")
    print("\n  CAVEAT, and it is a large one: the mispronunciations are INJECTED,")
    print("  the audio is SYNTHETIC, and Iqra_TTS sits inside the 131 h training")
    print("  set. Read this F1 as an upper bound and a sanity check -- not as the")
    print("  paper's MSA result, and not as evidence about real speakers.")


if __name__ == "__main__":
    main()
