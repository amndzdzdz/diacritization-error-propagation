"""Score the zero-base-rate control and re-read the CV-Ar probe against it.

Stage B of the control described in
`scripts/baseline_reproduction/tts_control_inference.py`. See that module's
docstring for why the CV-Ar dev probe alone could not settle
`docs/msa-arm.md` §3.3.

In one line: `Iqra_TTS`'s `original` rows were synthesized from
`phoneme_ref`, so `A == C` is an identity rather than an assumption, and
the false-rejection rate measured here is the model's **pure error rate**
on MSA phonetics with a guaranteed zero mispronunciation base rate.

The comparison is one-sided on purpose (again, see that docstring): only
`control FR >= CV-Ar FR` settles anything, because this control is both
in-domain and acoustically easy. This script prints the verdict in those
terms rather than as a symmetric test.

Run with:
    uv run python scripts/tts_control_baserate.py run/tts_control_predictions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arabic_mdd.data.phonemes import parse_phoneme_sequence
from arabic_mdd.metrics.hierarchical import aggregate, compute_metrics, evaluate_utterance

# Measured by run/cv_dev_baserate.slurm on `Iqra_train` dev, 2,588
# utterances (insights/week-04.md). The quantity this control interprets.
CV_DEV_FR_RATE = 0.0708
QURANMB_FR_RATE = 0.1241

# Stage A already strips these (see tts_control_inference.DROP_TOKENS), but
# repeat it here so the two stages cannot silently disagree: `<sil>` is not
# in the 68-token vocab, so any that survived into the reference would be
# scored as guaranteed false rejections and would inflate precisely the
# number this control exists to measure. Stripping twice is a no-op;
# stripping zero times is a wrong answer that still looks plausible.
DROP_TOKENS = {"<sil>", "sil", "<unk>"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records_json")
    parser.add_argument(
        "--cv-dev-frr",
        type=float,
        default=CV_DEV_FR_RATE,
        help="The CV-Ar dev disagreement rate this control is read against.",
    )
    args = parser.parse_args()

    records: dict[str, dict[str, str]] = json.loads(Path(args.records_json).read_text())
    print(f"{len(records)} control utterances loaded from {args.records_json}")

    counts_list = []
    skipped = 0
    stripped_here = 0
    for rec in records.values():
        raw_ref = (rec.get("phoneme_ref") or "").split()
        ref_tokens = [tok for tok in raw_ref if tok not in DROP_TOKENS]
        if len(ref_tokens) != len(raw_ref):
            stripped_here += 1
        canonical = parse_phoneme_sequence(" ".join(ref_tokens))
        prediction = parse_phoneme_sequence(rec.get("prediction") or "")
        if not canonical:
            skipped += 1
            continue
        # A == C by construction here, so this is an identity, not the
        # "assume every speaker perfect" approximation the CV-Ar probe made.
        counts_list.append(evaluate_utterance(canonical, canonical, prediction))
    if skipped:
        print(f"  skipped {skipped} rows with an empty reference")
    if stripped_here:
        print(f"  stripped {DROP_TOKENS} from {stripped_here} references at this stage")

    totals = aggregate(counts_list)
    metrics = compute_metrics(totals)

    print("\n" + "=" * 72)
    print("ZERO-BASE-RATE CONTROL  --  Iqra_TTS `original` rows (A == C by construction)")
    print("=" * 72)
    print(f"  utterances scored: {len(counts_list)}")
    print(f"  counts: {totals}")
    label = "TTS-original false-rejection rate"
    print(f"\n  {label:<36s}: {metrics.frr:.4f}   <- pure model error")
    print(f"  {'CV-Ar dev disagreement rate':<36s}: {args.cv_dev_frr:.4f}")
    print(f"  {'QuranMB false-rejection rate':<36s}: {QURANMB_FR_RATE:.4f}")
    headroom = args.cv_dev_frr - metrics.frr
    print(f"\n  {'headroom for a real base rate':<36s}: {headroom:+.4f}")
    print()

    if metrics.frr >= args.cv_dev_frr:
        print("  VERDICT: DECISIVE, negative. With a guaranteed zero base rate -- and")
        print("  with both in-domain training and clean synthetic audio working in")
        print("  its favour -- the model still errs at least as much as it disagrees")
        print("  on CV-Ar dev. The dev disagreement is fully accounted for by model")
        print("  error, leaving no room for real mispronunciations.")
        print("  => docs/msa-arm.md §3.3 option 2 (label-path only) is forced.")
    else:
        print("  VERDICT: INCONCLUSIVE. The model errs less on clean TTS than it")
        print("  disagrees on CV-Ar dev, so there IS nominal headroom -- but this")
        print("  control is in-domain and synthetic, and real speech is harder for")
        print("  reasons that have nothing to do with mispronunciation. The headroom")
        print("  above is an UPPER BOUND on the base rate, not an estimate of it.")
        print("  => step 4's listening pilot is the only way to resolve this.")

    print("\n  Neither outcome replaces step 4 on the positive side: a control can")
    print("  bound the base rate from above, but only human listening can confirm")
    print("  that any given utterance actually contains a mispronunciation.")


if __name__ == "__main__":
    main()
