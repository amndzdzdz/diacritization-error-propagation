"""Score a qualification submission — `docs/annotator-qualification.md` §8.

Implements §8 steps 1-5 (step 6, inter-annotator agreement, needs two
submissions and is reported only when both are present).

**Positions are marks, not letters.** §4's table counts 50,188 *marks* in
`Iqra_train` dev, so a letter carrying `<shadda><fatha>` is two positions, not
one. This matters: `build_qualification_test.py` counted marked *letters* when
it sized the draw, which overstates bucket B (it counted the free classes) and
understates bucket A (it merged shadda with its harakah). The realized scored
counts are therefore reported here and are the ones §6's targets should be
checked against.

**The pausal filter, from §4.** Two mark classes are free because `C_gold` is
pausal and the convention strips them mechanically downstream:

  - **tanwin** (U+064B-U+064D) — 3.49% of dev marks;
  - **the final word's final mark** — 3.03%.

Everything else is scored, in two buckets that are *never* combined (§4: a
single score is dominated by bucket A and would pass someone weak at exactly
the positions §3.3a found dominate this arm):

  - **bucket A** — word-internal marks, plus shadda anywhere;
  - **bucket B** — a non-final word's final mark, i.e. the case ending.

Gold is `original_gold` (§5). Where `max_gold_alt` is non-empty the word has a
second defensible reading and either scores as correct (§8 step 3); the count
of words where that allowance was actually invoked is reported, so its effect
on the rate is visible rather than buried.

Run with:
    uv run python scripts/score_qualification.py run/qualification/blank.txt
    uv run python scripts/score_qualification.py a.txt b.txt --names ann1 ann2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from arabic_mdd.data.normalize import skeleton, strip_diacritics
from arabic_mdd.metrics.der import arabic_words, marked_letters

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLD = REPO_ROOT / "run" / "qualification" / "gold.json"

#: §4: tanwin is stripped by the pausal convention, so it is not scored.
TANWIN = frozenset({"\u064b", "\u064c", "\u064d"})
SHADDA = "\u0651"

BUCKET_A = "A"
BUCKET_B = "B"
FREE = "free"

ITEM_RE = re.compile(r"^##\s*Item\s+(\d+)\s+of\s+(\d+)\s*$")


@dataclass
class Tally:
    scored: int = 0
    errors: int = 0
    omissions: int = 0
    substitutions: int = 0
    spurious: int = 0
    """Annotator marked a position the gold leaves unmarked."""

    @property
    def rate(self) -> float:
        return 100.0 * self.errors / self.scored if self.scored else 0.0

    def wilson(self) -> tuple[float, float]:
        """95% Wilson score interval, in percent. §8 step 4."""
        if not self.scored:
            return (0.0, 0.0)
        z = 1.959963984540054
        n, p = self.scored, self.errors / self.scored
        d = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / d
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (100.0 * max(0.0, centre - half), 100.0 * min(1.0, centre + half))


@dataclass
class Result:
    name: str
    buckets: dict[str, Tally] = field(
        default_factory=lambda: {BUCKET_A: Tally(), BUCKET_B: Tally()}
    )
    free_positions: int = 0
    items_attempted: int = 0
    items_total: int = 0
    skeleton_flags: list[int] = field(default_factory=list)
    word_count_flags: list[int] = field(default_factory=list)
    alt_invoked: int = 0
    alt_available: int = 0
    exact_words: int = 0
    total_words: int = 0
    error_examples: list[tuple[int, str, str, str, str]] = field(default_factory=list)
    confusions: Counter[str] = field(default_factory=Counter)


def parse_submission(path: Path) -> dict[int, str]:
    """Pull `> answer` lines out of the packet, keyed by item number."""
    answers: dict[int, str] = {}
    current: int | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = ITEM_RE.match(raw.strip())
        if m:
            current = int(m.group(1))
            continue
        if current is not None and raw.lstrip().startswith(">"):
            text = raw.lstrip()[1:].strip()
            if text:
                answers[current] = unicodedata.normalize("NFC", text)
            current = None
    return answers


def _mark_positions(word: str) -> list[tuple[int, int, str]]:
    """(letter_index, n_letters, mark) for every mark in the word, in order."""
    pairs = marked_letters(word)
    out: list[tuple[int, int, str]] = []
    for i, (_letter, marks) in enumerate(pairs):
        for mark in sorted(marks):
            out.append((i, len(pairs), mark))
    return out


def _bucket(letter_index: int, n_letters: int, mark: str, is_final_word: bool) -> str:
    """§4's classification. Order matters: shadda beats word-final."""
    is_word_final = letter_index == n_letters - 1
    if mark in TANWIN:
        return FREE
    if is_final_word and is_word_final:
        return FREE
    if mark == SHADDA:
        return BUCKET_A
    if is_word_final:
        return BUCKET_B
    return BUCKET_A


def score_word(
    result: Result,
    item: int,
    gold_word: str,
    alt_word: str,
    hyp_word: str,
    is_final_word: bool,
) -> None:
    result.total_words += 1
    if alt_word:
        result.alt_available += 1

    gold_norm = unicodedata.normalize("NFC", gold_word)
    hyp_norm = unicodedata.normalize("NFC", hyp_word)

    if hyp_norm == gold_norm:
        result.exact_words += 1
    elif alt_word and hyp_norm == unicodedata.normalize("NFC", alt_word):
        # §8 step 3: the second reading is correct too.
        result.alt_invoked += 1
        result.exact_words += 1
        hyp_norm = gold_norm

    gold_marks = {(i, m) for i, _n, m in _mark_positions(gold_norm)}
    hyp_marks = {(i, m) for i, _n, m in _mark_positions(hyp_norm)}
    n_letters = len(marked_letters(gold_norm))

    # Compare per (letter, mark-class) so an omission and a substitution are
    # distinguishable rather than both showing up as "differs".
    gold_by_letter: dict[int, set[str]] = {}
    hyp_by_letter: dict[int, set[str]] = {}
    for i, m in gold_marks:
        gold_by_letter.setdefault(i, set()).add(m)
    for i, m in hyp_marks:
        hyp_by_letter.setdefault(i, set()).add(m)

    for i in range(n_letters):
        g = gold_by_letter.get(i, set())
        h = hyp_by_letter.get(i, set())
        for mark in sorted(g | h):
            bucket = _bucket(i, n_letters, mark if mark in g else mark, is_final_word)
            if bucket == FREE:
                result.free_positions += 1
                continue
            tally = result.buckets[bucket]
            in_g, in_h = mark in g, mark in h
            if in_g and in_h:
                tally.scored += 1
                continue
            if in_g and not in_h:
                tally.scored += 1
                # Omission only if the annotator put nothing of that kind
                # there; otherwise the wrong mark is a substitution.
                same_class = (
                    {m for m in h if m != SHADDA}
                    if mark != SHADDA
                    else {m for m in h if m == SHADDA}
                )
                if same_class:
                    tally.substitutions += 1
                    got = "".join(sorted(same_class))
                    result.confusions[f"{mark}->{got}"] += 1
                else:
                    tally.omissions += 1
                    result.confusions[f"{mark}->(none)"] += 1
                tally.errors += 1
                if len(result.error_examples) < 25:
                    result.error_examples.append(
                        (item, strip_diacritics(gold_norm), gold_norm, hyp_norm, bucket)
                    )
            elif in_h and not in_g:
                # Gold leaves it unmarked. Counted, because writing a mark
                # where the reference has none is a real disagreement.
                tally.scored += 1
                tally.errors += 1
                tally.spurious += 1
                result.confusions[f"(none)->{mark}"] += 1


def score(path: Path, gold: dict, name: str) -> Result:
    answers = parse_submission(path)
    result = Result(name=name, items_total=len(gold["items"]))

    for entry in gold["items"]:
        item = entry["item"]
        hyp = answers.get(item)
        if not hyp:
            continue
        result.items_attempted += 1

        if skeleton(hyp) != skeleton(entry["blank"]):
            result.skeleton_flags.append(item)

        gold_words = [
            w for w in entry["words"] if any("\u0621" <= c <= "\u064a" for c in w["dediac"])
        ]
        hyp_words = arabic_words(hyp)
        if len(hyp_words) != len(gold_words):
            result.word_count_flags.append(item)
            continue

        for idx, (gw, hw) in enumerate(zip(gold_words, hyp_words, strict=True)):
            score_word(
                result,
                item,
                gw["original_gold"],
                gw["max_gold_alt"],
                hw,
                is_final_word=idx == len(gold_words) - 1,
            )
    return result


def report(result: Result) -> None:
    print(f"\n{'=' * 68}")
    print(f"  {result.name}")
    print(f"{'=' * 68}")
    print(
        f"items attempted   : {result.items_attempted}/{result.items_total} "
        f"({100.0 * result.items_attempted / result.items_total:.0f}%)  "
        f"[§9 completion bar: >=95%]"
    )
    print(f"skeleton altered  : {len(result.skeleton_flags)} items {result.skeleton_flags or ''}")
    print(
        f"word-count flags  : {len(result.word_count_flags)} items {result.word_count_flags or ''}"
    )
    print(
        f"words exact       : {result.exact_words}/{result.total_words} "
        f"({100.0 * result.exact_words / result.total_words:.1f}%)"
    )
    print(
        f"alt allowance     : invoked on {result.alt_invoked} of "
        f"{result.alt_available} words that offer one (§8 step 3)"
    )
    print(f"free positions    : {result.free_positions} (tanwin + final word's final mark)")

    print(f"\n{'bucket':<10}{'scored':>8}{'errors':>8}{'rate':>9}{'95% Wilson':>18}")
    for key, label in ((BUCKET_A, "A internal"), (BUCKET_B, "B case end")):
        t = result.buckets[key]
        lo, hi = t.wilson()
        print(f"{label:<10}{t.scored:>8}{t.errors:>8}{t.rate:>8.2f}%{f'[{lo:.2f}, {hi:.2f}]':>18}")

    print("\nerror profile (§8 step 5):")
    for key, label in ((BUCKET_A, "A"), (BUCKET_B, "B")):
        t = result.buckets[key]
        print(
            f"  {label}: {t.omissions} omissions, {t.substitutions} substitutions, "
            f"{t.spurious} spurious"
        )

    if result.confusions:
        print("\ntop confusions:")
        for pair, n in result.confusions.most_common(8):
            print(f"  {pair:<24} {n}")

    if result.error_examples:
        print("\nfirst errors (item, bare, gold, submitted, bucket):")
        seen = set()
        for item, bare, g, h, bucket in result.error_examples:
            if (item, bare) in seen:
                continue
            seen.add((item, bare))
            print(f"  [{item:>2}] {bucket}  {bare}   gold={g}   got={h}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submissions", nargs="+", type=Path)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--gold", type=Path, default=GOLD)
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    names = args.names or [p.stem for p in args.submissions]

    results = [score(path, gold, name) for path, name in zip(args.submissions, names, strict=False)]
    for result in results:
        report(result)

    if len(results) < 2:
        print("\n(one submission: §8 step 6 inter-annotator agreement not computed)")

    print(f"\n{'=' * 68}")
    print("  §9 pass marks are pre-registered in docs/annotator-qualification.md")
    print("  and are NOT reproduced here, so the thresholds cannot drift to fit")
    print("  the result. Read them there and apply them to the table above.")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
