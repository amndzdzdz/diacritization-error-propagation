"""`{split}-hyp.ark` -> the predictions JSON week 6's estimator already reads.

Week-08 task 5. The whole point of this file is that it is 60 lines: week 6's
`measure_label_path.py` takes `--predictions <json>` and does the rest -- counts
pooled per prompt, `compute_metrics` applied once, a 10,000-replicate cluster
bootstrap over the 98 prompts. Point that flag at the output of this converter
and the entire scoring column of plan section 1's 2x2 is computed by the
estimator that produced week 6's numbers, with no second implementation to keep
in agreement.

    uv run python scripts/ark_to_predictions.py \
        --ark run/quranmb_eval/conditioned/gold/quranmb-hyp.ark \
        --output run/quranmb_predictions_conditioned_gold.json

**The join key is free, and that is the only subtle thing here.**
`ctc_cond/data.py` keys each row by `str(path).split("/")[-1].split(".")[0]`,
i.e. the wav basename, and `build_quranmb_manifests.py` names every wav
`<clip_id>.wav`. So an `.ark` key *is* a QuranMB clip id, and
`measure_label_path.py` joins predictions to `load_ground_truth()` on exactly
that. Nothing needs remapping -- but `--expect` is here because nothing would
*complain* if it did: a silently short prediction file would be scored over
whatever clips happened to join, and `scored_clips` intersects rather than
asserts. 1,642 is the number.

An empty hypothesis line is kept as an empty string rather than dropped. CTC
emitting nothing but blanks is a real prediction -- the model declining to
commit -- and every phoneme of the annotation is then a missed rejection. A
dropped row would instead leave that clip out of the denominator, which flatters
the arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_collapse import read_ark  # noqa: E402

#: QuranMB.v2 as `safikhan/quran_mbv2_formatted` carries 1,642 clips. The
#: published count is 1,643; the one-utterance difference is week-2 carried
#: forward and is a property of the upload, not of this conversion.
EXPECTED_CLIPS = 1642


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ark", type=Path, required=True, help="a {split}-hyp.ark file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expect",
        type=int,
        default=EXPECTED_CLIPS,
        help="refuse unless the ark holds exactly this many utterances; 0 disables",
    )
    args = parser.parse_args()

    if not args.ark.is_file():
        raise SystemExit(f"{args.ark} not found")
    rows = read_ark(args.ark)
    if args.expect and len(rows) != args.expect:
        raise SystemExit(
            f"{args.ark.name} holds {len(rows)} utterances, expected {args.expect}. "
            "The commonest cause is an evaluation pass that re-scored the dev split "
            "because the test manifest override did not take -- check the `-o "
            "config.downstream_expert.corpus.test` line in run/evaluate_quranmb.slurm "
            "before trusting anything downstream."
        )

    predictions = {key: " ".join(tokens) for key, tokens in sorted(rows.items())}
    empty = [key for key, text in predictions.items() if not text]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{args.ark.name}: {len(predictions)} utterances, {len(empty)} empty hypotheses")
    if empty:
        print(f"  empty, e.g. {empty[:5]} — kept, not dropped (see the module docstring)")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
