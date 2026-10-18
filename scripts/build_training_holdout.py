"""Build the training/selection holdout for our own models — week-07 tasks 4 and 5.

Week 6 established that `Iqra_train`'s splits are id-disjoint *and*
raw-string-disjoint while still sharing text, and that the consonantal
skeleton is the only key that sees it (`docs/msa-arm.md` §5.2.1). This script
turns that into the one artifact the training manifest filter consumes.

Two independent contaminations, one mechanism:

- **MSA arm** (§5.2.1). The 500 annotated rows are the MSA evaluation pool.
  319 train rows share a skeleton with them, and the 500 rows are themselves
  `Iqra_train` *dev* rows — so they must leave the dev loader too, or
  `dev-best` selection is performed on the evaluation set, which is the exact
  complaint §5.2 raises against the organizers' checkpoint.
- **Qur'anic arm** (week-07 task 5). 16 of 98 QuranMB.v2 prompts (254 of
  1,642 clips) have their verse text in `Iqra_train`. This bites hardest on
  the text-dependent arm, which consumes `C` directly.

**What this does not do.** It does not touch the week-3 baseline
reproduction. That run reproduced a published number against the organizers'
full training set, and changing its inputs would destroy the one figure that
certifies the pipeline. The filter is opt-in per run for that reason.

Run with:
    uv run python scripts/build_training_holdout.py                # report only
    uv run python scripts/build_training_holdout.py --write        # also write the artifact
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from arabic_mdd.data.msa_pool import skeleton, skeleton_overlap

REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_CACHE = REPO_ROOT / "run" / "iqra_dev_text.parquet"
BLOCK = REPO_ROOT / "run" / "msa_annotation_block.json"
VERSE_MAP = REPO_ROOT / "run" / "quranmb_verse_map.json"
OUT = REPO_ROOT / "run" / "training_holdout.json"

TRAIN_SHARDS = (
    "~/.cache/huggingface/datasets/IqraEval___iqra_train/default/0.0.0/*/iqra_train-train-*.arrow"
)


def _split_rows(pattern: str) -> list[tuple[str, str, str]]:
    """`(id, sentence, phoneme_ref)` from Arrow shards, without paging in the audio."""
    shards = sorted(glob.glob(str(Path(pattern).expanduser())))
    if not shards:
        raise SystemExit(f"no shards under {pattern} — is the dataset cached?")
    rows: list[tuple[str, str, str]] = []
    for shard in shards:
        with pa.memory_map(shard, "r") as src:
            columns = ["id", "sentence", "phoneme_ref"]
            table = pa.ipc.open_stream(src).read_all().select(columns)
            rows.extend(zip(*(table.column(c).to_pylist() for c in columns), strict=True))
            del table
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help=f"write {OUT.name}")
    args = parser.parse_args()

    train = _split_rows(TRAIN_SHARDS)
    dev = [
        (r["id"], r["sentence"], r["phoneme_ref"])
        for r in pq.read_table(DEV_CACHE, columns=["id", "sentence", "phoneme_ref"]).to_pylist()
    ]
    print(f"splits: train {len(train)} rows, dev {len(dev)} rows")
    phoneme_of = {row_id: phonemes for row_id, _, phonemes in (*train, *dev)}

    # --- MSA arm (docs/msa-arm.md 5.2.1) -----------------------------------
    block = json.loads(BLOCK.read_text(encoding="utf-8"))["block"]
    block_ids = [r["id"] for r in block]
    train_keys = [(row_id, text) for row_id, text, _ in train]
    dev_keys = [(row_id, text) for row_id, text, _ in dev]
    msa_train = skeleton_overlap(train_keys, against=[r["sentence"] for r in block])
    msa_train_ids = sorted(i for ids in msa_train.values() for i in ids)
    print("\n=== MSA arm: the 500-row annotated evaluation pool ===")
    print(
        f"  train rows sharing a skeleton : {len(msa_train_ids)} across {len(msa_train)} sentences"
    )
    print(f"  dev rows that ARE the pool    : {len(block_ids)}")

    # --- Qur'anic arm (week-07 task 5) -------------------------------------
    prompts = json.loads(VERSE_MAP.read_text(encoding="utf-8"))["prompts"]
    verses = {r["recovered_text"]: r for r in prompts if r.get("recovered_text")}
    by_skeleton = {skeleton(text): row for text, row in verses.items()}
    total_clips = sum(r["n_clips"] for r in prompts)

    qur_train = skeleton_overlap(train_keys, against=list(verses))
    qur_dev = skeleton_overlap(dev_keys, against=list(verses))
    qur_train_ids = sorted(i for ids in qur_train.values() for i in ids)
    qur_dev_ids = sorted(i for ids in qur_dev.values() for i in ids)
    hit = set(qur_train) | set(qur_dev)
    hit_clips = sum(by_skeleton[k]["n_clips"] for k in hit)
    print("\n=== Qur'anic arm: QuranMB.v2 prompt text inside Iqra_train ===")
    for name, keys in (("train", set(qur_train)), ("dev", set(qur_dev)), ("union", hit)):
        clips = sum(by_skeleton[k]["n_clips"] for k in keys)
        print(
            f"  in {name:<6}: {len(keys):>2}/{len(prompts)} prompts, "
            f"{clips:>4}/{total_clips} clips = {clips / total_clips:.1%}"
        )
    print(
        f"  clean eval subset            : {len(prompts) - len(hit)} prompts, "
        f"{total_clips - hit_clips} clips"
    )
    print(f"  train rows to drop           : {len(qur_train_ids)}")
    print(f"  dev rows to drop             : {len(qur_dev_ids)}")

    # --- the union, which is what the filter consumes ----------------------
    hold_train = sorted(set(msa_train_ids) | set(qur_train_ids))
    hold_dev = sorted(set(block_ids) | set(qur_dev_ids))
    # The second key, and it is not redundant. `prepare_baseline.sh` step 4
    # merges Iqra_TTS's 52 h into the same train directory, and Iqra_TTS is
    # *synthesized from Iqra_train's phoneme refs* — so an id-only filter would
    # drop a held-out utterance and leave a synthesized reading of the same
    # text in training, which defeats the holdout silently. TTS files are named
    # `audio_<speaker>_<idx>.wav`, so their ids cannot collide with
    # `Iqra_train`'s five-digit ones and no id filter can reach them.
    #
    # The phoneme string is the right key: the TSV's transcript column *is* the
    # phoneme sequence, so this compares like with like without needing the
    # Arabic text, which the TSV does not carry. Two utterances with identical
    # phoneme sequences are the same text, so a match is never a false positive.
    hold_phonemes = sorted({phoneme_of[i] for i in (*hold_train, *hold_dev) if phoneme_of.get(i)})
    print("\n=== holdout the training manifest filter applies ===")
    print(f"  train : {len(hold_train)}/{len(train)} = {len(hold_train) / len(train):.2%}")
    print(
        f"  dev   : {len(hold_dev)}/{len(dev)} = {len(hold_dev) / len(dev):.2%}"
        "  <- mostly the eval pool itself; selection must not see it"
    )
    print(f"  dev rows left for model selection: {len(dev) - len(hold_dev)}")
    print(
        f"  phoneme strings : {len(hold_phonemes)} distinct"
        "  <- second key, reaches the merged Iqra_TTS rows"
    )

    if not args.write:
        print(f"\n(dry run — pass --write to produce {OUT.name})")
        return

    payload = {
        "purpose": (
            "utterance ids to drop from training AND model selection for our own runs, "
            "so neither evaluation pool is contaminated"
        ),
        "generated_by": "scripts/build_training_holdout.py",
        "key": "consonantal skeleton — arabic_mdd.data.normalize.skeleton",
        "key_rationale": (
            "train/dev are id-disjoint and raw-string-disjoint yet share text; only the "
            "skeleton sees it. See docs/msa-arm.md 5.2.1."
        ),
        "does_not_apply_to": (
            "the week-3 mHuBERT baseline reproduction, which must keep matching the "
            "organizers' full training set"
        ),
        "splits": {"train": len(train), "dev": len(dev)},
        "sources": {
            "msa_eval_pool": {
                "decision": "docs/msa-arm.md 5.2.1",
                "eval_set": "run/msa_annotation_block.json — 500 annotated rows",
                "train_ids": msa_train_ids,
                "dev_ids": sorted(block_ids),
                "dev_ids_note": (
                    "the evaluation pool itself; held out of selection, not of evaluation"
                ),
            },
            "quranmb": {
                "decision": "docs/weeks/week-07.md task 5",
                "eval_set": "run/quranmb_verse_map.json — 98 prompts, 1642 clips",
                "contaminated_prompts": len(hit),
                "contaminated_clips": hit_clips,
                "clean_prompts": len(prompts) - len(hit),
                "clean_clips": total_clips - hit_clips,
                "train_ids": qur_train_ids,
                "dev_ids": qur_dev_ids,
                "contaminated_prompt_skeletons": sorted(hit),
            },
        },
        "holdout_train_ids": hold_train,
        "holdout_dev_ids": hold_dev,
        # Second key. Iqra_TTS is merged into the train split and is synthesized
        # from Iqra_train's phoneme refs, and its filenames carry speaker names
        # rather than Iqra_train ids — so no id filter can reach it and an
        # id-only holdout would leave synthesized readings of the held-out text
        # in training. The TSV's transcript column is the phoneme sequence, so
        # this key compares like with like.
        "holdout_phoneme_refs": hold_phonemes,
        "counts": {
            "train_dropped": len(hold_train),
            "train_fraction": len(hold_train) / len(train),
            "dev_dropped": len(hold_dev),
            "dev_fraction": len(hold_dev) / len(dev),
            "dev_remaining_for_selection": len(dev) - len(hold_dev),
            "distinct_phoneme_refs": len(hold_phonemes),
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
