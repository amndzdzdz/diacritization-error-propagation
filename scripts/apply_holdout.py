"""Rewrite an S3PRL CTC manifest with the contaminated utterances removed.

Week-07 tasks 4 and 5. `scripts/build_training_holdout.py` decides *what* is
contaminated; `arabic_mdd.data.holdout` decides *how* a manifest row matches;
this file is only the plumbing between them and the filesystem.

Called from `run/prepare_baseline.sh` step 5b, after the TSVs are generated and
before step 6 substitutes their paths into `sws.yaml`. It writes a sibling
`*.heldout.tsv` rather than editing in place, so the unfiltered manifest stays
on disk and the week-3 reproduction remains runnable from the same data
directory without re-downloading 131 h of audio.

Run with:
    uv run python scripts/apply_holdout.py \
        --manifest run/data/CV-Ar/train/train.tsv --split train
    uv run python scripts/apply_holdout.py \
        --manifest run/data/CV-Ar/dev/dev.tsv --split dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arabic_mdd.data.holdout import filter_rows, load_holdout, read_manifest, write_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HOLDOUT = REPO_ROOT / "run" / "training_holdout.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="TSV to filter")
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "dev"),
        help="which holdout id list to apply; the phoneme-ref key is shared",
    )
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument(
        "--output",
        type=Path,
        help="defaults to the manifest with a .heldout.tsv suffix, alongside it",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="where to write the JSON audit trail; defaults to <output>.report.json",
    )
    args = parser.parse_args()

    out = args.output or args.manifest.with_suffix(".heldout.tsv")
    report_path = args.report or out.with_suffix(".report.json")

    payload = json.loads(args.holdout.read_text(encoding="utf-8"))
    holdout = load_holdout(payload, split=args.split)
    header, rows = read_manifest(args.manifest)
    result = filter_rows(rows, holdout)

    report = {
        "manifest_in": str(args.manifest),
        "manifest_out": str(out),
        "split": args.split,
        "holdout": str(args.holdout),
        "holdout_keys": {
            "ids": len(holdout.ids),
            "phoneme_refs": len(holdout.phoneme_refs),
        },
        **result.report(),
    }
    write_manifest(out, header, result.kept)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    dropped = report["dropped_total"]
    print(
        f"{args.manifest.name}: {result.rows_in} -> {len(result.kept)} rows "
        f"({dropped} dropped = {dropped / max(result.rows_in, 1):.2%}; "
        f"{len(result.dropped_by_id)} by id, "
        f"{len(result.dropped_by_phoneme_ref)} by phoneme ref)"
    )
    if result.unparseable_paths:
        # Not fatal: Iqra_TTS paths are expected to miss the id pattern, and the
        # phoneme key is what covers them. Loud because a *large* count means
        # the filter is keying off something this manifest does not contain.
        print(
            f"  WARNING: {len(result.unparseable_paths)} paths did not match "
            f"audio_<id>.wav, e.g. {result.unparseable_paths[0]!r} — these were "
            "matched on the phoneme key only"
        )
    print(f"wrote {out}\nwrote {report_path}")


if __name__ == "__main__":
    main()
