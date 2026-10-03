"""Week 4 task 4: does the *diacritizer* path stay inside the 68-token inventory?

`docs/msa-arm.md` §4. The task-3 round-trip validated the **reference** path
(`tashkeel_sentence` -> `phoneme_ref`, 0 out-of-inventory tokens). This checks
the path the MSA arm actually runs:

    sentence -> normalize -> strip diacritics -> diacritizer -> phonetizer

and diffs the result against `sws_arabic.txt`. Any phoneme the CTC head cannot
emit is a guaranteed false rejection for a mechanical reason, which in the
metric is indistinguishable from the paper's own claim.

**Three diffs, not one.** The phoneme-level diff alone would give a false
all-clear, because the vendored phonetiser cannot emit an unknown token — it
emits *nothing* for a character it has no rule for (see
`arabic_mdd.data.normalize`). So this script also reports:

* **character-level**: characters in each diacritizer's output that the
  phonetiser has no rule for, i.e. phonemes lost silently;
* **conservation**: utterances where the diacritizer path yields a very
  different phoneme count from the reference, which is what a silent drop
  actually looks like downstream.

Reads the projected-column cache `run/iqra_dev_text.parquet`; all four tools
run on CPU here (CATT via ONNX, Shakkala out-of-process under its own 3.10).

Run with:
    uv run python scripts/check_phoneme_inventory.py --limit 200
    uv run python scripts/check_phoneme_inventory.py            # all 2,588
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from arabic_mdd.data.msa_pool import strip_diacritics
from arabic_mdd.data.normalize import normalize, unsupported_characters
from arabic_mdd.data.phonetizer import SWS_ARABIC_INVENTORY, Phonetizer
from arabic_mdd.diacritizers import DIACRITIZERS, build

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / "run" / "iqra_dev_text.parquet"
OUT = REPO_ROOT / "run" / "phoneme_inventory_diff.json"


def _describe(ch: str) -> str:
    return f"U+{ord(ch):04X} {ch!r} {unicodedata.name(ch, '?')}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="first N utterances only")
    parser.add_argument("--write", action="store_true", help=f"write {OUT.name}")
    args = parser.parse_args()

    rows = pq.read_table(
        CACHE, columns=["id", "sentence", "tashkeel_sentence", "phoneme_ref"]
    ).to_pylist()
    if args.limit:
        rows = rows[: args.limit]

    phonetizer = Phonetizer(convention="pausal")

    # --- the normalization contract, measured on the real input ------------
    contract: Counter[str] = Counter()
    dropped_chars: Counter[str] = Counter()
    for row in rows:
        result = normalize(row["sentence"])
        contract.update(result.flags)
        dropped_chars.update(result.dropped)
        if not result.fair:
            contract["UNFAIR(excluded)"] += 1

    print(f"=== normalization contract over {len(rows)} CV-Ar transcripts ===")
    for flag, count in contract.most_common():
        print(f"  {flag:22}: {count:5} rows ({count / len(rows):6.2%})")
    if dropped_chars:
        print("  characters dropped outright:")
        for ch, count in dropped_chars.most_common():
            print(f"    {_describe(ch)} x{count}")
    else:
        print("  characters dropped outright: none")

    # The reference path, for comparison. This is what task 3 validated.
    inputs = [normalize(row["sentence"]) for row in rows]
    bare = [strip_diacritics(n.text) for n in inputs]
    reference = {row["id"]: row["phoneme_ref"].split() for row in rows}

    report: dict[str, dict[str, object]] = {}

    for name in DIACRITIZERS:
        tool = build(name)
        diacritized = tool.diacritize_batch(bare)

        oov: Counter[str] = Counter()
        lost_chars: Counter[str] = Counter()
        emitted: Counter[str] = Counter()
        shortfall = 0
        total_predicted = 0
        total_reference = 0

        for row, text in zip(rows, diacritized, strict=True):
            lost_chars.update(unsupported_characters(text))
            phonemes = phonetizer.phonetize(text)
            emitted.update(phonemes)
            oov.update(t for t in phonemes if t not in SWS_ARABIC_INVENTORY)
            ref = reference[row["id"]]
            total_predicted += len(phonemes)
            total_reference += len(ref)
            # A silent drop shows up as a short sequence, not as a bad token.
            if len(ref) and len(phonemes) < 0.9 * len(ref):
                shortfall += 1

        print(f"\n=== {name} ===")
        print(f"  phoneme types emitted   : {len(emitted)}/68 of the inventory")
        print(f"  OUT-OF-INVENTORY tokens : {dict(oov) if oov else 'none'}")
        if lost_chars:
            print("  characters the phonetiser has NO RULE for (silently dropped):")
            for ch, count in lost_chars.most_common(12):
                print(f"    {_describe(ch)} x{count}")
        else:
            print("  characters with no phonetiser rule: none")
        print(
            f"  length vs reference     : {total_predicted} predicted / "
            f"{total_reference} reference = {total_predicted / total_reference:.4f}"
        )
        print(f"  utterances >10% short   : {shortfall} ({shortfall / len(rows):.2%})")

        report[name] = {
            "provenance": tool.provenance(),
            "phoneme_types_used": len(emitted),
            "out_of_inventory": dict(oov),
            "unsupported_characters": {f"U+{ord(c):04X}": n for c, n in lost_chars.items()},
            "predicted_over_reference": total_predicted / total_reference,
            "utterances_more_than_10pct_short": shortfall,
        }

    if args.write:
        payload = {
            "source": "IqraEval/Iqra_train",
            "split": "dev",
            "utterances": len(rows),
            "decision": "docs/msa-arm.md 4",
            "normalization_contract": dict(contract),
            "characters_dropped": {f"U+{ord(c):04X}": n for c, n in dropped_chars.items()},
            "tools": report,
        }
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
