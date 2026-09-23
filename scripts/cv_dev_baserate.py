"""Estimate the MSA-arm mispronunciation base rate, and draw the pilot sample.

Stage B of the probe described in `docs/msa-arm.md` §3.3 -- the largest
open design question in the MSA arm.

**The problem.** The hierarchical metric only counts a detection *positive*
where `A != C`: the speaker actually mispronounced something. QuranMB.v2
supplies those positives (learners reciting). Common Voice Arabic is read
speech by fluent readers, so `A` may equal `C` almost everywhere -- in which
case TR and FA go to ~0, precision collapses, and F1 stops meaning
anything. The metric does not fail loudly in that case; it returns a number,
and the number is uninformative. Circumstantial evidence that the
organizers hit this themselves: `Iqra_train` ships no annotation column at
all (only `phoneme_ref`), while `Iqra_TTS` ships `phoneme_mis` -- synthetic
speech with *injected* mispronunciations.

**The probe.** Run the validated baseline over the dev audio (Stage A) and
align its output `P` against `phoneme_ref` (`C`). Disagreements are model
errors UNION real mispronunciations. Scoring with `A := C` -- "assume every
speaker was perfect" -- turns every disagreement into a false rejection, so
the resulting FR rate is an upper bound on what the model would be charged
if the base rate really were zero. Contrast that against the model's FR
rate on QuranMB (0.1241, `insights/week-03.md`):

* FR rate at or below the QuranMB figure => disagreement is fully explained
  by model error. There is no room for a meaningful base rate, and
  `docs/msa-arm.md` §3.3 option 2 is forced.
* FR rate materially above it => the excess is candidate mispronunciations,
  and the arm can work as designed.

This is a **screen, not a proof**: the two rates are not measured on the
same acoustics, and dev is in-domain for this checkpoint (`§5.2`), which
biases the rate downward. It narrows the question enough to aim 50
utterances of human listening at it, which is what the second half of this
script prepares.

**The sample.** Writes a stratified manifest for the step-4 listening
pilot: half drawn uniformly at random (an unbiased base-rate estimate, once
reweighted) and half from the high-disagreement tail (enriched, so the
pilot actually encounters errors if any exist). Pure random sampling at a
low base rate would need far more than 50 utterances to see anything.

Run with:
    uv run python scripts/cv_dev_baserate.py run/cv_dev_predictions.json
    uv run python scripts/cv_dev_baserate.py run/cv_dev_predictions.json \
        --dump-audio run/pilot_audio
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

from arabic_mdd.data.phonemes import parse_phoneme_sequence
from arabic_mdd.metrics.hierarchical import aggregate, compute_metrics, evaluate_utterance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_iqra_train_text import fetch_split, load_rows  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Our own from-scratch retrain's QuranMB numbers (insights/week-03.md).
# The reference point this probe is read against.
QURANMB_FR_RATE = 0.1241
QURANMB_F1 = 0.4406


def _per(canonical: list[str], prediction: list[str]) -> float:
    """Per-utterance disagreement, as a fraction of canonical length.

    Uses the same `A := C` trick as the aggregate: FR / (TA + FR).
    """
    counts = evaluate_utterance(canonical, canonical, prediction)
    denominator = counts.ta + counts.fr
    return counts.fr / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_json")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument(
        "--manifest",
        default="run/msa_pilot_sample.json",
        help="Where to write the step-4 listening-pilot manifest.",
    )
    parser.add_argument(
        "--dump-audio",
        default=None,
        help="Collect the sampled utterances' wavs into this directory, for the pilot.",
    )
    parser.add_argument(
        "--audio-source",
        default="run/cv_dev_audio",
        help="Where Stage A's --save_audio_dir wrote its wavs.",
    )
    args = parser.parse_args()

    predictions: dict[str, str] = json.loads(Path(args.predictions_json).read_text())

    fetch_split(args.split)
    rows = {str(r["id"]): r for r in load_rows(args.split)}

    matched = [rid for rid in predictions if rid in rows]
    print(f"{len(matched)}/{len(predictions)} predictions matched an `{args.split}` id")
    if not matched:
        raise SystemExit("No predictions matched -- check that Stage A's ids line up.")

    empty = sum(1 for rid in matched if not predictions[rid].strip())
    if empty:
        print(f"WARNING: {empty} predictions are empty strings")

    # ---- Aggregate: what would the model be charged if nobody erred? ----
    per_utterance: list[tuple[str, float, int]] = []
    counts_list = []
    for rid in matched:
        canonical = parse_phoneme_sequence(rows[rid].get("phoneme_ref") or "")
        prediction = parse_phoneme_sequence(predictions[rid])
        if not canonical:
            continue
        counts_list.append(evaluate_utterance(canonical, canonical, prediction))
        per_utterance.append((rid, _per(canonical, prediction), len(canonical)))

    totals = aggregate(counts_list)
    metrics = compute_metrics(totals)

    print("\n" + "=" * 72)
    print("BASE-RATE PROBE  --  scored with A := C ('assume every speaker perfect')")
    print("=" * 72)
    print(f"  utterances scored: {len(counts_list)}")
    print(f"  counts: {totals}")
    label = f"CV-Ar {args.split} false-rejection rate"
    print(f"\n  {label:<34s}: {metrics.frr:.4f}")
    print(f"  {'QuranMB false-rejection rate':<34s}: {QURANMB_FR_RATE:.4f}  (insights/week-03.md)")
    excess = metrics.frr - QURANMB_FR_RATE
    print(f"  {'excess':<34s}: {excess:+.4f}")
    print()
    if excess <= 0:
        print("  READ -- and read this narrowly. Disagreement is BELOW the QuranMB")
        print("  error rate, but that is also exactly what an in-domain checkpoint")
        print("  produces whether or not a base rate exists: `dev-best.ckpt` was")
        print("  model-SELECTED on this very split (docs/msa-arm.md §5.2), so its")
        print("  true error rate here is expected to be well under the QuranMB")
        print("  figure. A negative excess is therefore CONSISTENT WITH both")
        print("  'no mispronunciations' and 'some mispronunciations, masked by a")
        print("  model that is simply much better on in-domain audio'.")
        print("  => the screen is INCONCLUSIVE in the negative direction. It does")
        print("     not force docs/msa-arm.md §3.3 option 2; it fails to rule it in")
        print("     or out. Resolve with a zero-base-rate control on comparable")
        print("     audio (Iqra_TTS clean rows), then step 4.")
    else:
        print("  READ: disagreement exceeds the model's known error rate. The excess")
        print("  is candidate real mispronunciation -- but dev is in-domain for this")
        print("  checkpoint, so the model's true error rate here should be LOWER than")
        print("  the QuranMB figure, making this an understatement of the excess.")
        print("  This direction IS informative: the confound works against it.")
    print("  Either way this is a screen, not a measurement. Step 4 measures it.")

    per_values = sorted(p for _, p, _ in per_utterance)
    if per_values:
        n = len(per_values)
        print("\n  per-utterance disagreement distribution:")
        for label, idx in (("p10", n // 10), ("p50", n // 2), ("p90", 9 * n // 10)):
            print(f"    {label}: {per_values[idx]:.3f}")
        clean = sum(1 for p in per_values if p == 0.0)
        print(f"    utterances with zero disagreement: {clean}/{n} ({100 * clean / n:.1f}%)")

    # ---- Stratified sample for the human pilot ----
    rng = random.Random(args.seed)
    per_utterance.sort(key=lambda item: -item[1])
    half = args.sample_size // 2
    tail_pool = per_utterance[: max(half * 4, 1)]
    enriched = rng.sample(tail_pool, min(half, len(tail_pool)))
    enriched_ids = {rid for rid, _, _ in enriched}
    remaining = [item for item in per_utterance if item[0] not in enriched_ids]
    uniform = rng.sample(remaining, min(args.sample_size - len(enriched), len(remaining)))

    manifest: list[dict[str, Any]] = []
    for stratum, items in (("high_disagreement", enriched), ("uniform_random", uniform)):
        for rid, per, length in items:
            row = rows[rid]
            manifest.append(
                {
                    "id": rid,
                    "stratum": stratum,
                    "disagreement": round(per, 4),
                    "n_canonical_phonemes": length,
                    "sentence": row.get("sentence"),
                    "tashkeel_sentence": row.get("tashkeel_sentence"),
                    "phoneme_ref": row.get("phoneme_ref"),
                    "prediction": predictions[rid],
                    # Filled in by the annotator in step 4:
                    "annotator_verdict": None,
                    "annotator_notes": None,
                }
            )

    manifest_path = REPO_ROOT / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n  wrote {len(manifest)} sampled utterances to {args.manifest}")
    print(
        "  strata: "
        f"{sum(1 for m in manifest if m['stratum'] == 'high_disagreement')} high-disagreement, "
        f"{sum(1 for m in manifest if m['stratum'] == 'uniform_random')} uniform"
    )
    print("\n  STEP 4: listen to each, set `annotator_verdict` to one of")
    print("    'clean'      -- read as written")
    print("    'deviation'  -- audible departure from the prompt (note where)")
    print("    'unusable'   -- truncated / inaudible / wrong audio")
    print("  Reweight by stratum before reporting the base rate: the uniform")
    print("  stratum is the unbiased estimate; the enriched one is for finding")
    print("  out what deviations look like, not for estimating how common they are.")

    if args.dump_audio:
        _collect_audio([m["id"] for m in manifest], Path(args.audio_source), Path(args.dump_audio))


def _collect_audio(ids: list[str], source_dir: Path, out_dir: Path) -> None:
    """Copy the sampled utterances' wavs out of Stage A's audio dump.

    Deliberately a copy rather than a decode: decoding an HF Audio feature
    in this env needs `torchcodec`, which is not installed and should not be
    added for one listening pilot. Stage A already decodes every utterance,
    so `cv_dev_inference.py --save_audio_dir` keeps the wavs and this just
    selects from them.
    """
    if not source_dir.is_dir():
        print(f"  (no audio at {source_dir} -- rerun Stage A with --save_audio_dir to get wavs)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for rid in ids:
        source = source_dir / f"{rid}.wav"
        if source.exists():
            shutil.copy2(source, out_dir / f"{rid}.wav")
            written += 1
    print(f"  copied {written}/{len(ids)} wav files to {out_dir}")


if __name__ == "__main__":
    main()
