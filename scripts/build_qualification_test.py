"""Build the annotator qualification test packet — `docs/annotator-qualification.md` §§5-7.

Draws N segments from the WikiNews gold set, strips the marks, and emits three
files: the blank test to send to annotators, the withheld gold, and the
instruction sheet.

**Why 20 segments and not §7's literal "70".** §6's targets are *positions*,
not items: ~202 bucket-B and ~1,067 bucket-A. Those were derived in units of an
`Iqra_train` utterance (19.4 marks); a WikiNews segment carries ~76, about 4x.
70 segments would overshoot to ~1,300/~4,000 at ~4x the annotator hours. Signed
off 2026-10-03.

**Gold is `original_gold`, not `max_gold`.** They differ on 3.72% of Arabic
words, and `original_gold` is the Darwish 2017 reference that published machine
DER — including the CATT figure §9's pass marks are calibrated against — was
measured on. `max_gold_alt` is carried through so §8 step 3's
acceptable-alternatives allowance can be applied at scoring time.

The draw is a function of (seed, band, n), so it can be regenerated and
audited rather than described. The source file's sha256 is recorded in the
manifest because the file is not vendored (CC-BY-SA-4.0; share-alike is viral
over derivatives).

Run with:
    uv run python scripts/build_qualification_test.py
    uv run python scripts/build_qualification_test.py --n 30 --seed 20261003
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from arabic_mdd.data.normalize import strip_diacritics
from arabic_mdd.metrics.der import marked_letters

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TSV = Path("/tmp/WikiNewsMax.tsv")
OUT_DIR = REPO_ROOT / "run" / "qualification"

#: §6's eligibility band, in Arabic words. Also what keeps §5's "short
#: everyday prose" register criterion satisfiable.
MIN_WORDS, MAX_WORDS = 8, 40

#: Distinct from the annotation block's 20261001, so the two draws cannot be
#: confused for one another.
DEFAULT_SEED = 20261003

#: §6, in positions (marks), not items.
TARGET_BUCKET_B = 202
TARGET_BUCKET_A = 1067

ARTICLE_MARKER = "#"


@dataclass(frozen=True)
class Word:
    dediac: str
    original_gold: str
    max_gold: str
    max_gold_alt: str

    @property
    def is_arabic(self) -> bool:
        return any("\u0621" <= c <= "\u064a" for c in self.dediac)


@dataclass(frozen=True)
class Segment:
    index: int
    """Position in the source file, so a drawn segment can be found again."""

    words: list[Word]

    @property
    def arabic_words(self) -> list[Word]:
        return [w for w in self.words if w.is_arabic]

    @property
    def gold_text(self) -> str:
        return " ".join(w.original_gold for w in self.words)

    @property
    def blank_text(self) -> str:
        return " ".join(w.dediac for w in self.words)

    def bucket_counts(self) -> tuple[int, int]:
        """(bucket_A, bucket_B) *marked* positions in `original_gold`.

        Bucket B is the mark on a word's final letter — the positional proxy
        for i'rab, the same one `metrics.der` uses. Bucket A is every other
        marked position. Counts are pre-pausal-filter; §8 applies that at
        scoring time, which can only reduce bucket B.
        """
        a = b = 0
        for word in self.arabic_words:
            pairs = marked_letters(word.original_gold)
            for i, (_letter, marks) in enumerate(pairs):
                if not marks:
                    continue
                if i == len(pairs) - 1:
                    b += 1
                else:
                    a += 1
        return a, b


def read_segments(tsv: Path) -> list[Segment]:
    """Segment the word-per-row TSV on blank rows.

    Article-marker rows (`#`) also break a segment: a headline and the body
    that follows it are not one sentence.
    """
    with tsv.open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))[1:]

    segments: list[Segment] = []
    current: list[Word] = []

    def flush() -> None:
        if current and any(w.is_arabic for w in current):
            segments.append(Segment(index=len(segments), words=list(current)))
        current.clear()

    for row in rows:
        cells = (row + ["", "", "", ""])[:4]
        if not cells[0].strip() or cells[0].strip() == ARTICLE_MARKER:
            flush()
            continue
        current.append(
            Word(
                dediac=unicodedata.normalize("NFC", cells[0]),
                original_gold=unicodedata.normalize("NFC", cells[1]),
                max_gold=unicodedata.normalize("NFC", cells[2]),
                max_gold_alt=unicodedata.normalize("NFC", cells[3]),
            )
        )
    flush()
    return segments


def _verify_skeleton(segments: list[Segment]) -> int:
    """`strip_diacritics(original_gold)` must reproduce the file's own `dediac`.

    If it does not, the blank text we hand an annotator is not the text the
    gold was written for, and §8 step 1's skeleton check would fire on our
    own preprocessing rather than on the annotator.
    """
    return sum(
        1
        for segment in segments
        for w in segment.words
        if strip_diacritics(w.original_gold) != w.dediac
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--n", type=int, default=20, help="segments to draw (signed off: 20)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    segments = read_segments(args.tsv)
    mismatches = _verify_skeleton(segments)
    if mismatches:
        raise SystemExit(
            f"{mismatches} words where strip_diacritics(original_gold) != dediac; "
            "the blank text would not match the gold. Refusing to build a packet."
        )

    band = [s for s in segments if MIN_WORDS <= len(s.arabic_words) <= MAX_WORDS]
    print(f"source      : {args.tsv} ({args.tsv.stat().st_size} bytes)")
    print(
        f"segments    : {len(segments)} with Arabic, {len(band)} in the "
        f"{MIN_WORDS}-{MAX_WORDS}-word band"
    )
    print("skeleton    : strip_diacritics(original_gold) == dediac on all words")

    if args.n > len(band):
        raise SystemExit(f"asked for {args.n} segments, band has {len(band)}")

    drawn = sorted(random.Random(args.seed).sample(band, args.n), key=lambda s: s.index)

    bucket_a = bucket_b = 0
    alt_words = 0
    for segment in drawn:
        a, b = segment.bucket_counts()
        bucket_a += a
        bucket_b += b
        alt_words += sum(1 for w in segment.arabic_words if w.max_gold_alt)

    words = sum(len(s.arabic_words) for s in drawn)
    print(f"\ndrawn       : {args.n} segments, seed {args.seed}, {words} Arabic words")
    print(
        f"bucket A    : {bucket_a} marked positions (target ~{TARGET_BUCKET_A}) "
        f"{'MET' if bucket_a >= TARGET_BUCKET_A else 'SHORT'}"
    )
    print(
        f"bucket B    : {bucket_b} marked positions (target ~{TARGET_BUCKET_B}) "
        f"{'MET' if bucket_b >= TARGET_BUCKET_B else 'SHORT'}"
    )
    print(f"alt readings: {alt_words} words carry a max_gold_alt (§8 step 3 allowance)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.tsv.read_bytes()).hexdigest()

    blank = ["# Arabic diacritization — qualification test", ""]
    blank.append("Add the diacritics to each sentence below. Do not change the letters.")
    blank.append("Write your answer on the line beginning `>`. See instructions.md.")
    blank.append("")
    for i, segment in enumerate(drawn, start=1):
        blank.append(f"## Item {i} of {args.n}")
        blank.append("")
        blank.append(segment.blank_text)
        blank.append("")
        # No trailing space: the annotator types after the `>`, and a trailing
        # one makes the committed packet un-regenerable (pre-commit strips it).
        blank.append(">")
        blank.append("")
    (args.out_dir / "blank.txt").write_text("\n".join(blank), encoding="utf-8")

    gold = {
        "source": "CAMeL-Lab/wild_diacritics data/WikiNewsMax.tsv",
        "licence": "CC-BY-SA-4.0; cite Darwish et al. 2017 additionally",
        "source_sha256": digest,
        "gold_column": "original_gold",
        "gold_column_note": (
            "original_gold is the Darwish 2017 reference that published machine DER "
            "(incl. the CATT figure §9's pass marks are calibrated against) was measured "
            "on. max_gold differs on 3.72% of Arabic words corpus-wide."
        ),
        "seed": args.seed,
        "n": args.n,
        "band": [MIN_WORDS, MAX_WORDS],
        "band_size": len(band),
        "bucket_a_positions": bucket_a,
        "bucket_b_positions": bucket_b,
        "arabic_words": words,
        "alt_reading_words": alt_words,
        "items": [
            {
                "item": i,
                "source_segment_index": segment.index,
                "blank": segment.blank_text,
                "gold": segment.gold_text,
                "words": [
                    {
                        "dediac": w.dediac,
                        "original_gold": w.original_gold,
                        "max_gold": w.max_gold,
                        "max_gold_alt": w.max_gold_alt,
                    }
                    for w in segment.words
                ],
            }
            for i, segment in enumerate(drawn, start=1)
        ],
    }
    (args.out_dir / "gold.json").write_text(
        json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    (args.out_dir / "instructions.md").write_text(
        _INSTRUCTIONS.format(n=args.n, items=args.n), encoding="utf-8"
    )

    print(f"\nwrote {args.out_dir.relative_to(REPO_ROOT)}/blank.txt         <- send this")
    print(f"wrote {args.out_dir.relative_to(REPO_ROOT)}/instructions.md  <- send this")
    print(f"wrote {args.out_dir.relative_to(REPO_ROOT)}/gold.json        <- DO NOT send")


_INSTRUCTIONS = """# Qualification test — instructions

Please read this once before starting. It should take two minutes.

## What to do

You will receive `blank.txt` with **{items} short Modern Standard Arabic
sentences** with all diacritics removed. For each one, write the same sentence
back with the diacritics added, on the line beginning with `>`.

## Rules

1. **Work alone.** Do not discuss the items with anyone, including the other
   annotator.
2. **Do not use any automatic diacritization tool**, website, or language
   model. This test measures your judgment, not a tool's.
3. **Do not look up the source article.** These are real news sentences; finding
   the original would defeat the purpose.
4. **Do not change the letters.** Add marks only. Do not fix spelling, do not
   add or remove words, do not change the punctuation.
5. **Attempt every item.** If you are unsure, write your best reading rather
   than leaving it blank — skipping hard items makes the result less useful,
   not more flattering. If you are genuinely torn between two readings, write
   the one you think more likely; there is no penalty for a near miss.
6. **Mark fully.** Include sukun where it belongs, and shadda where it belongs.
   Where a word's final mark is grammatically determined, write it.

## Timing

**Record your start time and your finish time** and send them with your answers.
There is no time limit and speed is not scored. We need the elapsed time only
to size the main annotation job realistically — an honest number helps you,
because it is what stops the main job from being under-budgeted.

If you take a long break, note that too, so we can subtract it.

## What happens to this

Your answers are scored against a specialist-prepared reference, in two
separate categories: word-final marks (which depend on grammatical function)
and everything else. The two are reported separately because they are
different skills, and the thresholds were fixed in writing before the test was
built. Nobody sees the reference before scoring, including the person scoring.

The purpose is to establish suitability by measurement rather than by
credential, and to decide how the main annotation work is divided. Raw
submissions are kept as evidence for the paper's reliability section.

Thank you — please send back the completed file plus your two timestamps.
"""


if __name__ == "__main__":
    main()
