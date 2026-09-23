"""How much of the benchmark's false-rejection rate is a case-ending artifact?

`docs/msa-arm.md` §3.4.1 establishes a collision: `mhubert147_per` is
trained on `Iqra_train`'s `phoneme_ref`, which is **pausal** (65.9% of
utterances end on a consonant), but QuranMB.v2 -- the benchmark it is
evaluated on -- is **prescriptive** (80.1% of utterances end on a realized
case ending). A model taught to drop utterance-final case endings, scored
against a reference that marks them, will be charged a deletion at the end
of essentially every utterance.

That predicts a specific, checkable error profile: **high recall, depressed
precision, inflated FR** -- which is exactly the published baseline's
profile (P 0.3093 / R 0.7707 / FRR 0.1237). If the prediction holds, a
meaningful share of the field's flagship benchmark's false rejections are a
convention artifact rather than model error, and the MSA arm must not
inherit the same mismatch.

Two measurements, deliberately of different strength:

1. **Final-phoneme class contrast** (strong, assumption-free). Compare the
   utterance-final phoneme class of `C`, `A` and `P`. If `P` ends on a
   consonant far more often than `C` does, the model is applying pausal
   treatment to a prescriptive reference. Needs no alignment.
2. **Ablation estimate** (weaker, indicative). Re-score with the final
   token removed from `C`, `A` and `P` on utterances where `C` ends in a
   short vowel, and report the delta. Crude -- removing a token shifts the
   alignment -- so read the magnitude as an order-of-magnitude estimate,
   not a measurement.

Runs entirely on cached local files; no GPU, no network.

    uv run python scripts/case_ending_impact.py run/quranmb_predictions.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from arabic_mdd.data.phonemes import parse_phoneme_sequence
from arabic_mdd.metrics.hierarchical import aggregate, compute_metrics, evaluate_utterance

REPO_ROOT = Path(__file__).resolve().parent.parent
QURANMB_CACHE = REPO_ROOT / "run" / "tmp" / "quranmb_text.jsonl"

SHORT_VOWELS = {"a", "i", "u", "A", "I", "U"}
LONG_VOWELS = {"aa", "ii", "uu", "AA", "II", "UU"}


def classify(token: str) -> str:
    if token in LONG_VOWELS:
        return "long vowel"
    if token in SHORT_VOWELS:
        return "short vowel"
    return "consonant"


def _table(title: str, counts: Counter[str], total: int) -> None:
    print(f"\n  {title}  (n={total})")
    for name in ("short vowel", "long vowel", "consonant"):
        c = counts.get(name, 0)
        pct = 100 * c / total if total else 0.0
        print(f"    {name:14s} {pct:5.1f}%  {'#' * int(pct / 2.5)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_json")
    args = parser.parse_args()

    if not QURANMB_CACHE.exists():
        raise SystemExit(
            f"No cache at {QURANMB_CACHE}.\n"
            "Run: uv run python scripts/compare_word_final_conventions.py"
        )
    rows = {}
    with QURANMB_CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["id"])] = row

    predictions: dict[str, str] = json.loads(Path(args.predictions_json).read_text())
    matched = [rid for rid in predictions if rid in rows]
    print(f"{len(matched)}/{len(predictions)} predictions matched a QuranMB id")
    if not matched:
        raise SystemExit("No predictions matched -- check the id format.")

    # ---- 1. Final-phoneme class contrast ----
    fin_c: Counter[str] = Counter()
    fin_a: Counter[str] = Counter()
    fin_p: Counter[str] = Counter()
    triples = []
    for rid in matched:
        row = rows[rid]
        c = parse_phoneme_sequence(row.get("reference_phoneme_string") or "")
        a = parse_phoneme_sequence(row.get("annotation_phoneme_string") or "")
        p = parse_phoneme_sequence(predictions[rid])
        if not c or not a or not p:
            continue
        triples.append((c, a, p))
        fin_c[classify(c[-1])] += 1
        fin_a[classify(a[-1])] += 1
        fin_p[classify(p[-1])] += 1

    n = len(triples)
    print("\n" + "=" * 72)
    print("1. UTTERANCE-FINAL PHONEME CLASS  --  does the model apply pausal treatment?")
    print("=" * 72)
    _table("C  reference (the scoring key)", fin_c, n)
    _table("A  human annotation (what reciters said)", fin_a, n)
    _table("P  model prediction", fin_p, n)

    c_vowel = 100 * (fin_c["short vowel"] + fin_c["long vowel"]) / n
    a_vowel = 100 * (fin_a["short vowel"] + fin_a["long vowel"]) / n
    p_vowel = 100 * (fin_p["short vowel"] + fin_p["long vowel"]) / n
    print("\n  ends on a VOWEL:")
    print(f"    C {c_vowel:5.1f}%     A {a_vowel:5.1f}%     P {p_vowel:5.1f}%")
    print(f"    C - P gap: {c_vowel - p_vowel:+5.1f} points")
    if c_vowel - p_vowel > 10:
        print("\n  READ: the model ends on a consonant far more often than the")
        print("  reference does, while the human annotation tracks the reference.")
        print("  That is pausal treatment applied to a prescriptive key -- the")
        print("  §3.4.1 mismatch, visible in the benchmark itself.")
    else:
        print("\n  READ: no substantial pausal/prescriptive gap between C and P.")
        print("  The §3.4.1 mismatch does not show up in the model's output.")

    # ---- 2. Ablation estimate ----
    full = aggregate([evaluate_utterance(c, a, p) for c, a, p in triples])
    ablated = []
    touched = 0
    for c, a, p in triples:
        if c and classify(c[-1]) == "short vowel":
            touched += 1
            c2, a2, p2 = c[:-1], a[:-1] or a, p[:-1] or p
        else:
            c2, a2, p2 = c, a, p
        if c2 and a2 and p2:
            ablated.append(evaluate_utterance(c2, a2, p2))
    abl = aggregate(ablated)

    mf, ma = compute_metrics(full), compute_metrics(abl)
    print("\n" + "=" * 72)
    print("2. ABLATION ESTIMATE  --  drop the final token where C ends in a short vowel")
    print("=" * 72)
    print(f"  utterances affected: {touched}/{n} ({100 * touched / n:.1f}%)")
    print(f"\n  {'':<12s}{'full':>10s}{'ablated':>10s}{'delta':>10s}")
    for name, x, y in (
        ("F1", mf.f1, ma.f1),
        ("precision", mf.precision, ma.precision),
        ("recall", mf.recall, ma.recall),
        ("FR rate", mf.frr, ma.frr),
    ):
        print(f"  {name:<12s}{x:>10.4f}{y:>10.4f}{y - x:>+10.4f}")
    fr_drop = full.fr - abl.fr
    share = 100 * fr_drop / full.fr if full.fr else 0.0
    print(f"\n  false rejections: {full.fr} -> {abl.fr}  ({fr_drop} fewer, {share:.1f}% of all FR)")
    print("\n  CAVEAT: removing a token also shifts the alignment, so this over- or")
    print("  under-states the true final-position contribution. Order of magnitude")
    print("  only. Measurement 1 is the assumption-free result.")


if __name__ == "__main__":
    main()
