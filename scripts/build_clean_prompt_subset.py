"""Name the 82 uncontaminated QuranMB.v2 prompts -- week-08 task 7.

`run/training_holdout.json` already *counts* them (`contaminated_prompts: 16`,
`clean_prompts: 82`, `clean_clips: 1388`), but it records the contaminated
prompts as `contaminated_prompt_skeletons` -- consonantal skeletons of the verse
text. Nothing downstream can select clips with a skeleton: a clip id is
`<speaker>_<prompt>`, so every per-denominator report needs the **prompt
indices**. This script performs the join that `build_training_holdout.py`
already performed internally and discarded, and writes it down.

Re-deriving rather than re-running that script is deliberate: it reads the
71,391-row Iqra_train shards, which live on the cluster, and it rewrites
`run/training_holdout.json` -- the artifact the week-7 arms' training manifests
were filtered by. Regenerating that file to recover a projection of itself would
put the holdout of three 200k-step runs at risk for a lookup. Both inputs here
are committed JSON, so this runs anywhere and is pure bookkeeping. The counts in
`training_holdout.json` are the cross-check: if the join does not reproduce
16/82 prompts and 254/1388 clips exactly, one of the two files has moved and
this script refuses.

## Why two denominators exist at all, which is the part worth getting right

Our three week-7 arms were trained on holdout-filtered manifests, so they never
saw any of the 98 prompts. **Their numbers are honest on all 1,642 clips.** The
organizers' checkpoint -- which produced `run/quranmb_predictions_official_ckpt.json`
and the published 0.4414 -- was trained on the unfiltered set and did see the 16.
So the subset is not a cleanliness upgrade to apply everywhere; it is the only
footing on which *our* arm and *their* checkpoint can be compared. Applying it
to our own arms throws away 15% of the evaluation set for nothing; omitting it
from a cross-checkpoint comparison compares a clean model against a contaminated
one silently. Hence: every cell reported on both, with the comparison cells read
off the clean 82.

    uv run python scripts/build_clean_prompt_subset.py --write
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from arabic_mdd.data.msa_pool import skeleton

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOLDOUT = REPO_ROOT / "run" / "training_holdout.json"
DEFAULT_MAP = REPO_ROOT / "run" / "quranmb_verse_map.json"
OUT = REPO_ROOT / "run" / "quranmb_clean_prompts.json"

#: `training_holdout.json`'s own counts, restated as assertions. Drift in either
#: input file has to stop the build rather than quietly renumber a denominator
#: the paper reports against.
EXPECTED = {"contaminated_prompts": 16, "clean_prompts": 82, "clean_clips": 1388}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--verse-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--write", action="store_true", help=f"write {OUT.name}")
    args = parser.parse_args()

    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))["sources"]["quranmb"]
    contaminated_skeletons = set(holdout["contaminated_prompt_skeletons"])
    prompts = json.loads(args.verse_map.read_text(encoding="utf-8"))["prompts"]

    # `skeleton` is imported, not reimplemented: it is the same key
    # `build_training_holdout.py` used to produce the skeletons being joined
    # against, and a second normalization rule here would silently under-match.
    keyed = {p["prompt"]: skeleton(p["recovered_text"]) for p in prompts}
    clips = {p["prompt"]: p["n_clips"] for p in prompts}
    missing_text = sorted(p["prompt"] for p in prompts if not p.get("recovered_text"))
    if missing_text:
        raise SystemExit(
            f"prompts {missing_text} carry no recovered_text, so they cannot be keyed. "
            "Week 5 recovered all 98; this is drift in the verse map."
        )

    contaminated = sorted(p for p, key in keyed.items() if key in contaminated_skeletons)
    clean = sorted(p for p in keyed if p not in set(contaminated))
    unmatched = sorted(contaminated_skeletons - set(keyed.values()))

    print("=== week-08 task 7: the clean-prompt denominator ===")
    print(f"  prompts            : {len(keyed)}")
    print(
        f"  contaminated       : {len(contaminated)} prompts, "
        f"{sum(clips[p] for p in contaminated)} clips"
    )
    print(f"  clean              : {len(clean)} prompts, {sum(clips[p] for p in clean)} clips")
    print(f"  contaminated index : {contaminated}")

    if unmatched:
        raise SystemExit(
            f"{len(unmatched)} contaminated skeletons match no prompt in "
            f"{args.verse_map.name}. The two files were built from different verse "
            "recoveries, so the denominator cannot be trusted."
        )
    found = {
        "contaminated_prompts": len(contaminated),
        "clean_prompts": len(clean),
        "clean_clips": sum(clips[p] for p in clean),
    }
    if found != EXPECTED:
        raise SystemExit(
            f"join gives {found}, but {args.holdout.name} records {EXPECTED}. One of the "
            "two inputs has changed since week 7; reconcile before reporting against "
            "either denominator."
        )

    # Not a defect and not fixable here, but it has to be on the record because
    # the bootstrap's resampling unit is the prompt: two pairs of prompts carry
    # *identical* verse text, so the 82 clean prompts are 80 distinct texts and
    # the clusters are not quite independent. Too small to correct for (it would
    # mean collapsing clusters and changing week 6's estimator), large enough to
    # state.
    by_text: dict[str, list[str]] = defaultdict(list)
    for prompt, key in keyed.items():
        by_text[key].append(prompt)
    duplicates = sorted(sorted(group) for group in by_text.values() if len(group) > 1)
    print(f"  prompt pairs sharing a text: {duplicates}")

    payload = {
        "purpose": (
            "QuranMB.v2 prompt indices split by whether the organizers' training set "
            "contained the prompt text — the two denominators week 8 reports against"
        ),
        "generated_by": "scripts/build_clean_prompt_subset.py",
        "decision": "docs/weeks/week-08.md task 7",
        "key": "consonantal skeleton — arabic_mdd.data.msa_pool.skeleton",
        "join": (
            f"{args.holdout.name}:sources.quranmb.contaminated_prompt_skeletons "
            f"against {args.verse_map.name}:prompts[].recovered_text"
        ),
        "which_denominator": {
            "all_98": (
                "our own week-7 arms — trained on holdout-filtered manifests, so clean "
                "on every clip; using the subset would discard 15% of the evaluation "
                "set for no reason"
            ),
            "clean_82": (
                "any comparison against run/quranmb_predictions_official_ckpt.json, the "
                "published 0.4414, or any other organizer-trained checkpoint, which saw "
                "the 16"
            ),
        },
        "clean_prompts": clean,
        "contaminated_prompts": contaminated,
        "counts": {
            **found,
            "contaminated_clips": sum(clips[p] for p in contaminated),
            "total_prompts": len(keyed),
            "total_clips": sum(clips.values()),
        },
        "clips_per_prompt": clips,
        "duplicate_text_prompt_groups": duplicates,
        "duplicate_text_note": (
            "prompts with byte-identical recovered_text. The cluster bootstrap resamples "
            "prompts, so these clusters are not independent: the clean 82 are 80 distinct "
            "texts. Stated, not corrected — collapsing them would change week 6's "
            "estimator and the CIs it already published."
        ),
    }

    if not args.write:
        print(f"\n(dry run — pass --write to produce {args.output})")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
