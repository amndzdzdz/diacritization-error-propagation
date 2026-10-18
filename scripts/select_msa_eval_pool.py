"""Select the MSA evaluation pool and measure its training contamination — §5.2.

Week-6 task 7, carried from week 5. §5.2's own "cleanest fix" — draw the
evaluation utterances from Common Voice rows `Iqra_train` never touched — was
foreclosed by D4 ("Annotate dev, not fresh Common Voice"): `tashkeel_sentence`
exists only in `Iqra_train`, and D5's whole measurement is
`tashkeel_sentence` vs `C_gold`, so fresh Common Voice text has no `C_auto` to
compare against. The 500-row block is drawn, its drafts are committed, and the
protocol is pre-registered.

So the pool is the annotated block, and §5.2's contamination is handled on the
*training* side instead: from week 7 we train our own models, so we choose the
splits. This script produces the holdout — the training rows that must not be
seen — and the number that says how much it matters.

Reads the 29 train shards from the HF cache by memory-mapping each one and
projecting two columns, so the 14 GB of audio is never paged in.

Run with:
    uv run python scripts/select_msa_eval_pool.py                # report only
    uv run python scripts/select_msa_eval_pool.py --write        # also write the artifact
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from arabic_mdd.data.msa_pool import build_pool, skeleton, skeleton_overlap

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_CACHE = REPO_ROOT / "run" / "iqra_dev_text.parquet"
BLOCK = REPO_ROOT / "run" / "msa_annotation_block.json"
OUT = REPO_ROOT / "run" / "msa_eval_pool.json"

TRAIN_SHARDS = (
    "~/.cache/huggingface/datasets/IqraEval___iqra_train/default/0.0.0/*/iqra_train-train-*.arrow"
)


def train_rows() -> list[tuple[str, str]]:
    """`(id, sentence)` for every train row, without touching the audio column."""
    shards = sorted(glob.glob(str(Path(TRAIN_SHARDS).expanduser())))
    if not shards:
        raise SystemExit(f"no train shards under {TRAIN_SHARDS} — is the dataset cached?")
    rows: list[tuple[str, str]] = []
    for shard in shards:
        with pa.memory_map(shard, "r") as src:
            table = pa.ipc.open_stream(src).read_all().select(["id", "sentence"])
            ids = table.column("id").to_pylist()
            sentences = table.column("sentence").to_pylist()
            rows.extend(zip(ids, sentences, strict=True))
            del table
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help=f"write {OUT.name}")
    args = parser.parse_args()

    block = json.loads(BLOCK.read_text(encoding="utf-8"))["block"]
    dev = pq.read_table(DEV_CACHE, columns=["id", "sentence", "tashkeel_sentence"]).to_pylist()
    pool = build_pool((r["id"], r["sentence"], r["tashkeel_sentence"]) for r in dev)
    eligible = pool.eligible

    train = train_rows()
    print("=== splits ===")
    print(f"  Iqra_train train      : {len(train)} rows")
    print(f"  Iqra_train dev        : {len(dev)} rows, {len(eligible)} eligible (D4)")
    print(f"  evaluation pool       : {len(block)} annotated rows (D4's block)")
    dev_ids = {r["id"] for r in dev}
    shared_ids = {i for i, _ in train} & dev_ids
    print(f"  train/dev id overlap  : {len(shared_ids)}  <- the check that passes")

    # The three nested sets, so the extension decision (~1,592 further rows) is
    # costed now rather than re-measured later.
    sets = {
        "block_500": [(r["id"], r["sentence"]) for r in block],
        "eligible_pool_2092": [(c.id, c.sentence) for c in eligible],
        "dev_all": [(r["id"], r["sentence"]) for r in dev],
    }
    report: dict[str, dict[str, object]] = {}
    print("\n=== contamination: dev sentences that also occur in train ===")
    for name, rows in sets.items():
        keys = {skeleton(s) for _, s in rows}
        overlap = skeleton_overlap(train, against=[s for _, s in rows])
        copies = sum(len(v) for v in overlap.values())
        raw = {s for _, s in rows}
        raw_hits = sum(1 for _, s in train if s in raw)
        report[name] = {
            "rows": len(rows),
            "distinct_skeletons": len(keys),
            "skeletons_also_in_train": len(overlap),
            "train_rows_sharing_a_skeleton": copies,
            "train_rows_sharing_a_raw_string": raw_hits,
        }
        print(
            f"  {name:<20}: {len(overlap)}/{len(keys)} skeletons = "
            f"{len(overlap) / len(keys):.1%}, {copies} train rows"
            f"  (raw-string match finds {raw_hits})"
        )

    holdout = skeleton_overlap(train, against=[r["sentence"] for r in block])
    drop = sorted(i for ids in holdout.values() for i in ids)
    print("\n=== holdout for week 7 ===")
    print(f"  train rows to drop    : {len(drop)} ({len(drop) / len(train):.2%} of train)")
    print(f"  distinct sentences    : {len(holdout)}")

    if not args.write:
        print(f"\n(dry run — pass --write to produce {OUT.name})")
        return

    payload = {
        "decision": "docs/msa-arm.md 5.2",
        "pool_source": "run/msa_annotation_block.json",
        "pool_rows": len(block),
        "pool_ids": [r["id"] for r in block],
        "train_rows": len(train),
        "train_dev_id_overlap": len(shared_ids),
        "contamination": report,
        "sadeed_precedent": (
            "34.6% of the Fadel test set is fully present in the Abbad training set"
        ),
        # What week 7 must consume. Keyed by skeleton so a reader can see *why*
        # each id is dropped, and so the list survives an id renumbering.
        "holdout_train_ids": drop,
        "holdout_by_skeleton": {k: sorted(v) for k, v in sorted(holdout.items())},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
