"""Confirm CATT's token-level confidence on the real checkpoints, EO *and* ED.

Week 4 task 2 / plan §7 item 4. Week 1 established feasibility on EO only,
through undocumented internals, and deferred the production interface to week
4; that interface now lives in `arabic_mdd.diacritizers.catt` and is covered
by `tests/diacritizers/test_catt_confidence.py`. Those tests fake the ONNX
sessions, which is what keeps them offline — so they prove the *alignment*
logic but cannot prove the real checkpoints behave as assumed.

This script closes that gap, and specifically discharges the task's "confirm
the same technique works on the **ED** checkpoint" requirement: ED decodes
autoregressively and keeps only the last step's logits, so its extraction
path shares no code with EO's one-shot decode and has to be verified against
the real weights separately.

The check that matters is the reconstruction identity — joining
`letter + diacritic` over the returned positions must equal `diacritize`'s
output. Anything misaligned shows up there rather than as a plausible-looking
but wrong confidence.

Downloads the ONNX archives on first run (EO 72 MB, ED 86 MB).

Run with: uv run python scripts/check_catt_probabilities.py
"""

import numpy as np

from arabic_mdd.diacritizers.catt import VARIANTS, CattDiacritizer

SAMPLE_TEXTS = [
    "وقالت مجلة نيوزويك الأمريكية التحديث الجديد ل إنستجرام "
    "يمكن أن يساهم في إيقاف وكشف الحسابات المزيفة",
    "ذهب الولد إلى المدرسة",
    "بسم الله الرحمن الرحيم",
]


def main() -> None:
    for variant in VARIANTS:
        print(f"=== catt-{variant} ===")
        diacritizer = CattDiacritizer(variant=variant)

        baseline = diacritizer.diacritize_batch(SAMPLE_TEXTS)
        results = diacritizer.diacritize_batch_with_confidence(SAMPLE_TEXTS)

        for text, expected, positions in zip(SAMPLE_TEXTS, baseline, results, strict=True):
            reconstructed = "".join(p.letter + p.diacritic for p in positions)
            print(f"  input          : {text}")
            print(f"  diacritize()   : {expected}")
            print(f"  reconstructed  : {reconstructed}")
            print(f"  identical      : {reconstructed == expected}")

            scored = np.array([p.confidence for p in positions if not p.forced])
            print(
                f"  confidence     : mean {scored.mean():.4f}  "
                f"min {scored.min():.4f}  max {scored.max():.4f}  "
                f"(over {len(scored)} unforced of {len(positions)} positions)"
            )
            print()

        print("  lowest-confidence positions:")
        flat = [p for positions in results for p in positions if not p.forced]
        for position in sorted(flat, key=lambda p: p.confidence)[:10]:
            print(f"    {position.letter}{position.diacritic}  {position.confidence:.4f}")
        print()

    print("=== EO vs ED agreement ===")
    per_variant = {
        variant: CattDiacritizer(variant=variant).diacritize_batch_with_confidence(SAMPLE_TEXTS)
        for variant in VARIANTS
    }
    paired = zip(SAMPLE_TEXTS, per_variant["eo"], per_variant["ed"], strict=True)
    for text, eo_row, ed_row in paired:
        agreed = sum(a.diacritic == b.diacritic for a, b in zip(eo_row, ed_row, strict=True))
        print(f"  {agreed}/{len(eo_row)} positions agree :: {text}")


if __name__ == "__main__":
    main()
