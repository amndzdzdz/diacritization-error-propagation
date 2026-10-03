"""Draw the MSA arm's `C_gold` annotation block, reproducibly — `docs/msa-arm.md` §3.3b D4.

Reports the sampling frame (how many rows are eligible and what each filter
removed) and writes the drawn block to JSON. The filter predicates live in
`arabic_mdd.data.msa_pool` and are tested; this script is only the runner and
the report.

Reads the projected-column cache `run/iqra_dev_text.parquet` (0.35 MB) rather
than the dataset itself, whose rows carry 77 MB of audio this decision does
not need.

Run with:
    uv run python scripts/draw_annotation_block.py                # report only
    uv run python scripts/draw_annotation_block.py --write        # also write the block
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from arabic_mdd.data.msa_pool import (
    DIGITS,
    LATIN,
    PARTIALLY_VOWELIZED,
    SHORT,
    WORD_COUNT_MISMATCH,
    build_pool,
    draw_block,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / "run" / "iqra_dev_text.parquet"
OUT = REPO_ROOT / "run" / "msa_annotation_block.json"

# Pinned by §3.3b D4. Changing either of these re-draws the block, so they are
# constants here and not command-line options.
BLOCK_SIZE = 500
SEED = 20261001

# The crossover-anchoring sub-block (D3 §10): the first `CROSSOVER` rows of the
# drawn block, in draw order. Taken from inside the block rather than drawn
# separately so the reliability estimate describes the block that was actually
# annotated.
CROSSOVER = 50


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help=f"write {OUT.name}")
    args = parser.parse_args()

    table = pq.read_table(CACHE, columns=["id", "sentence", "tashkeel_sentence", "phoneme_ref"])
    rows = table.to_pylist()
    pool = build_pool((r["id"], r["sentence"], r["tashkeel_sentence"]) for r in rows)
    phonemes = {r["id"]: len(r["phoneme_ref"].split()) for r in rows}

    eligible = pool.eligible
    print(f"=== sampling frame: Iqra_train dev ({pool.total} rows) ===")
    print(
        f"  eligible              : {len(eligible)}/{pool.total} = {len(eligible) / pool.total:.1%}"
    )
    print("  excluded by")
    for flag, count in pool.exclusion_counts().items():
        print(f"    {flag:<20}: {count} ({count / pool.total:.1%})")
    print("  kept and flagged (share of eligible)")
    for flag, count in pool.flag_counts().items():
        print(f"    {flag:<20}: {count} ({count / len(eligible):.1%})")

    words = sum(len(c.sentence.split()) for c in eligible)
    positions = sum(phonemes[c.id] for c in eligible)
    print(f"  pool size             : {positions} phoneme positions, {words} words")
    print(
        f"                          {positions / len(eligible):.1f} and "
        f"{words / len(eligible):.1f} per utterance"
    )

    block = draw_block(pool, BLOCK_SIZE, seed=SEED)
    block_positions = sum(phonemes[c.id] for c in block)
    block_words = sum(len(c.sentence.split()) for c in block)
    flagged = sum(1 for c in block if c.flags)
    print(f"\n=== block: {BLOCK_SIZE} utterances, seed {SEED} ===")
    print(f"  {block_positions} phoneme positions, {block_words} words")
    print(f"  carrying at least one flag: {flagged} ({flagged / len(block):.1%})")
    for flag in (PARTIALLY_VOWELIZED, WORD_COUNT_MISMATCH, SHORT, LATIN, DIGITS):
        count = sum(1 for c in block if flag in c.flags)
        print(f"    {flag:<20}: {count}")
    print(f"  crossover sub-block   : first {CROSSOVER} in draw order")

    if not args.write:
        print(f"\n(dry run — pass --write to produce {OUT.name})")
        return

    payload = {
        "source": "IqraEval/Iqra_train",
        "split": "dev",
        "decision": "docs/msa-arm.md 3.3b D4",
        "seed": SEED,
        "block_size": BLOCK_SIZE,
        "crossover_size": CROSSOVER,
        "pool_total": pool.total,
        "pool_eligible": len(eligible),
        "exclusion_counts": pool.exclusion_counts(),
        "flag_counts": pool.flag_counts(),
        # The Qur'anic/classical predicate is deliberately absent: detecting
        # those rows needs an authoritative full Qur'an text, which is the same
        # artifact as the clip -> verse recovery D1 is conditional on. Recorded
        # as false so a later block cannot be mistaken for this one.
        "quranic_filter_applied": False,
        "block": [
            {
                "id": c.id,
                "rank": rank,
                "crossover": rank < CROSSOVER,
                "sentence": c.sentence,
                "tashkeel_sentence": c.tashkeel_sentence,
                "diacritization_rate": c.rate,
                "flags": sorted(c.flags),
            }
            for rank, c in enumerate(block)
        ],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
