"""Check the QuranMB.v2 labels we score against are the organizers' own.

Step 2 of the week-3 gate debugging order (docs/weeks/week-03.md: "check data
prep ... for drift from the official recipe").

`src/arabic_mdd/data/quranmb.py` reads ground truth from
`safikhan/quran_mbv2_formatted`, a public third-party dataset that pre-joins
`IqraEval/QuranMB.v2` audio with the gated `IqraEval/IqraEval_Test_GT`
labels (see week 2's logistics log). The official leaderboard instead scores
against `01Yassine/labels_test`. Nothing so far has confirmed the
third-party join reproduces the official label strings exactly.

This matters because 738/1642 of the third-party rows are `match_type:
fuzzy`. Scoring them alone gives a much lower F1 than the `exact` rows
(0.3647 vs 0.4459). Per-utterance edit distances say that is the model
finding those clips harder rather than the labels being wrong -- canonical
and annotation stay *closer* to each other on fuzzy rows (0.072 vs 0.081)
while the model sits further from both -- but that is also exactly what you
would see if both label strings had drifted together from the audio. The two
cases are indistinguishable without the official labels, so: fetch them and
diff.

Requires access to the gated `IqraEval/IqraEval_Test_GT` and an HF token:
    export HF_TOKEN=hf_...
    uv run python scripts/verify_quranmb_labels.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from arabic_mdd.data.phonemes import parse_phoneme_sequence
from arabic_mdd.data.quranmb import load_ground_truth
from arabic_mdd.metrics.hierarchical import aggregate, compute_metrics, evaluate_utterance

OFFICIAL_GT_DATASET = "IqraEval/IqraEval_Test_GT"
GATE_F1 = 0.4414
GATE_TOLERANCE = 0.02


def _pick(row: dict, *candidates: str) -> str:
    """Return the first present column name, so this survives schema drift.

    The gated dataset is documented as `ID`/`Reference_phn`/`Annotation_phn`,
    but week 2 already hit one case of a vendored script assuming column
    names the published dataset no longer used -- so don't assume here.
    """
    for name in candidates:
        if name in row:
            return name
    raise SystemExit(
        f"None of {candidates} found in {OFFICIAL_GT_DATASET}. Columns present: {sorted(row)}"
    )


def load_official_ground_truth(split: str) -> dict[str, tuple[list[str], list[str]]]:
    """Load `id -> (canonical, human_annotation)` from the gated official dataset."""
    import datasets

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print("WARNING: no HF_TOKEN set; the gated dataset load will likely fail.\n")

    ds = datasets.load_dataset(OFFICIAL_GT_DATASET, split=split)
    if "audio" in ds.column_names:
        ds = ds.remove_columns("audio")

    first = ds[0]
    id_col = _pick(first, "ID", "id")
    ref_col = _pick(first, "Reference_phn", "reference_phn", "reference_phoneme_string")
    ann_col = _pick(first, "Annotation_phn", "annotation_phn", "annotation_phoneme_string")
    print(f"official columns -> id={id_col!r} canonical={ref_col!r} annotation={ann_col!r}")

    return {
        str(row[id_col]): (
            parse_phoneme_sequence(row[ref_col]),
            parse_phoneme_sequence(row[ann_col]),
        )
        for row in ds
    }


def main(predictions_path: str, split: str) -> None:
    ours = load_ground_truth()
    official = load_official_ground_truth(split)

    print(f"\nours: {len(ours)} utterances   official: {len(official)} utterances")
    shared = sorted(set(ours) & set(official))
    print(f"shared ids: {len(shared)}")
    if only_ours := sorted(set(ours) - set(official))[:5]:
        print(f"  ids only in ours (first 5): {only_ours}")
    if only_official := sorted(set(official) - set(ours))[:5]:
        print(f"  ids only in official (first 5): {only_official}")
    if not shared:
        raise SystemExit("No shared ids -- id formats differ between the two sources.")

    # --- label diff, overall and split by the third-party join's match_type ---
    per_type: dict[str, Counter] = {}
    canonical_mismatches: list[str] = []
    annotation_mismatches: list[str] = []
    for example_id in shared:
        mine = ours[example_id]
        ref_official, ann_official = official[example_id]
        tally = per_type.setdefault(mine.match_type, Counter())
        tally["n"] += 1
        if mine.canonical == ref_official:
            tally["canonical_ok"] += 1
        else:
            canonical_mismatches.append(example_id)
        if mine.human_annotation == ann_official:
            tally["annotation_ok"] += 1
        else:
            annotation_mismatches.append(example_id)

    print(f"\n{'match_type':<12}{'n':>7}{'canonical ok':>15}{'annotation ok':>15}")
    print("-" * 49)
    for match_type, tally in sorted(per_type.items()):
        print(
            f"{match_type:<12}{tally['n']:>7}"
            f"{tally['canonical_ok']:>10} ({tally['canonical_ok'] / tally['n']:>5.1%})"
            f"{tally['annotation_ok']:>10} ({tally['annotation_ok'] / tally['n']:>5.1%})"
        )
    print(
        f"\ntotal mismatches: canonical {len(canonical_mismatches)}, "
        f"annotation {len(annotation_mismatches)}"
    )

    for label, ids in (("canonical", canonical_mismatches), ("annotation", annotation_mismatches)):
        for example_id in ids[:3]:
            mine = ours[example_id]
            theirs = official[example_id][0 if label == "canonical" else 1]
            got = mine.canonical if label == "canonical" else mine.human_annotation
            print(f"\n  {label} mismatch {example_id} (match_type={mine.match_type})")
            print(f"    ours     : {' '.join(got)[:150]}")
            print(f"    official : {' '.join(theirs)[:150]}")

    # --- the number that actually matters: re-score under official labels ---
    predictions_file = Path(predictions_path)
    if not predictions_file.exists():
        print(f"\n(skipping re-scoring: {predictions_path} not found)")
        return

    predictions: dict[str, str] = json.loads(predictions_file.read_text())
    scored = [i for i in shared if i in predictions]
    print(f"\n=== Re-scoring {predictions_path} on {len(scored)} utterances ===")
    for source, getter in (
        ("ours (safikhan join)", lambda i: (ours[i].canonical, ours[i].human_annotation)),
        ("official (gated)", lambda i: official[i]),
    ):
        counts = aggregate(
            [evaluate_utterance(*getter(i), parse_phoneme_sequence(predictions[i])) for i in scored]
        )
        m = compute_metrics(counts)
        lo, hi = GATE_F1 - GATE_TOLERANCE, GATE_F1 + GATE_TOLERANCE
        verdict = "PASS" if lo <= m.f1 <= hi else "FAIL"
        print(
            f"{source:<22} F1={m.f1:.4f}  P={m.precision:.4f}  R={m.recall:.4f}  "
            f"FRR={m.frr:.4f}  -> GATE {verdict}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="run/quranmb_predictions.json")
    parser.add_argument("--split", default="test", help="Split of the gated dataset to load.")
    args = parser.parse_args()
    main(args.predictions, args.split)
