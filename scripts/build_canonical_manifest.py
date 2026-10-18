"""Add the `canonical` column the text-dependent arm conditions on.

Week-07 task 7. `arabic_mdd.data.canonical` decides *where* each row's canonical
comes from and refuses to guess; this file is the plumbing between it and the
filesystem, in the same shape as `scripts/apply_holdout.py`.

Runs **after** the holdout filter in `run/prepare_baseline.sh`, on whichever
manifest the run will actually train on, so the holdout and the conditioning
cannot disagree about which rows exist. Writes a sibling `*.cond.tsv`; the
two-column manifest stays on disk untouched, so the week-3 reproduction and
task 6's prompt-free run keep reading exactly the file they always did.

    uv run python scripts/build_canonical_manifest.py \
        --manifest run/data/CV-Ar/train/train.heldout.tsv \
        --data-root run/data/CV-Ar \
        --vocab scripts/baseline_reproduction/vocab/sws_arabic.txt

**It will fail on a fresh data directory, and that is correct.** Iqra_TTS rows
have no canonical on disk until
`scripts/baseline_reproduction/dump_tts_canonicals.py` writes one. The
alternative -- defaulting those rows to their `sentence` -- would silently set
canonical == phoneme_mis on the only 52 h that punish canonical-copying. See
`arabic_mdd.data.canonical` and `arabic_mdd.models.collapse`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from arabic_mdd.data.canonical import (
    CONDITIONED_COLUMNS,
    build_canonical_rows,
    count_unknown_tokens,
)
from arabic_mdd.data.holdout import read_manifest, write_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOCAB = REPO_ROOT / "scripts" / "baseline_reproduction" / "vocab" / "sws_arabic.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="two-column TSV to extend")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="the corpus root the manifest's paths are relative to (…/CV-Ar)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="defaults to the manifest with a .cond.tsv suffix, alongside it",
    )
    parser.add_argument("--report", type=Path, help="JSON audit trail; defaults next to --output")
    parser.add_argument(
        "--vocab",
        type=Path,
        default=DEFAULT_VOCAB,
        help="one token per line; canonical tokens outside it are counted and refused",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help=(
            "proceed despite out-of-vocabulary canonical tokens. They become <unk>, "
            "so the conditioning signal is degraded in a way that only shows up as a "
            "disappointing F1 -- record the reason in insights/ if you use this."
        ),
    )
    args = parser.parse_args()

    out = args.output or args.manifest.with_suffix(".cond.tsv")
    report_path = args.report or out.with_suffix(".report.json")
    root = args.data_root

    def read_canonical(relative: str) -> str | None:
        path = root / relative
        return path.read_text(encoding="utf-8") if path.is_file() else None

    _, rows = read_manifest(args.manifest)
    build = build_canonical_rows(rows, read_canonical)

    vocabulary = frozenset(args.vocab.read_text(encoding="utf-8").split())
    unknown_count, unknown_examples = count_unknown_tokens(build.rows, vocabulary)

    report = {
        "manifest_in": str(args.manifest),
        "manifest_out": str(out),
        "data_root": str(root),
        "vocab": str(args.vocab),
        "vocab_size": len(vocabulary),
        "unknown_canonical_tokens": unknown_count,
        "unknown_canonical_examples": unknown_examples[:20],
        **build.report(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{args.manifest.name}: {len(build.rows)} rows "
        f"({build.from_sentence} canonical from sentence, "
        f"{build.from_file} from file, {build.passed_through} already present)"
    )
    print(f"wrote {report_path}")
    # The summary above goes to stdout and the complaints below to stderr.
    # Flush first, or an interactive run prints the remedy before the problem.
    sys.stdout.flush()

    if build.missing_canonical:
        print(
            f"ERROR: {len(build.missing_canonical)} rows have no canonical on disk, "
            f"e.g. {build.missing_canonical[0]!r}, expected under {root}.",
            file=sys.stderr,
        )
        print(
            "These are Iqra_TTS rows, whose `sentence` is phoneme_mis rather than the\n"
            "canonical. Dump their phoneme_ref first, with --output_dir pointing at the\n"
            "Iqra_TTS download directory (NOT --data-root above -- the dumper writes\n"
            "into the TTS tree, and prepare_baseline.sh step 4 merges it in):\n"
            "  python scripts/baseline_reproduction/dump_tts_canonicals.py \\\n"
            "      --output_dir <the Iqra_TTS directory>\n"
            "Nothing here will substitute `sentence` for them: that would delete the\n"
            "only 52 h of training signal that punishes copying the canonical.",
            file=sys.stderr,
        )
        sys.exit(1)

    if unknown_count and not args.allow_unknown:
        print(
            f"ERROR: {unknown_count} canonical tokens are outside {args.vocab.name} "
            f"({len(vocabulary)} tokens): {unknown_examples[:10]}.",
            file=sys.stderr,
        )
        print(
            "The tokenizer maps these to <unk> without complaining, so the arm would\n"
            "train on a degraded canonical and the only symptom would be a weak F1.\n"
            "`<sil>` is the expected case: dump_tts_canonicals.py strips it, so seeing\n"
            "it here means the canonicals were produced some other way.\n"
            "Pass --allow-unknown to proceed deliberately.",
            file=sys.stderr,
        )
        sys.exit(1)

    write_manifest(out, list(CONDITIONED_COLUMNS), build.rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
