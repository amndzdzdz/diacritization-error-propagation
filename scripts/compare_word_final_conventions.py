"""Compare word-final (case-ending) conventions across the two arms -- MSA-arm step 2.

`docs/msa-arm.md` §3.4 calls the prescriptive-vs-pausal question the thing
that decides what `C_gold` even means, and warns that getting it wrong
would *manufacture* the case-ending effect RQ3 predicts. It also asserts
that both arms must use the same convention, and flags that QuranMB's
`Reference_phn` follows Qur'anic recitation convention while the MSA arm's
references come from the organizers' in-house vowelizer. That assertion was
never checked. This checks it.

**What this can and cannot do.** `phoneme_ref` / `Reference_phn` are flat,
whitespace-separated phoneme strings with no word boundaries, so a per-word
alignment of text to phonemes needs the week-4 phonetizer and is out of
scope here. Two alignment-free contrasts are available now and are enough
to settle the question:

* the distribution of **word-final marks in the Arabic text**, per arm; and
* the distribution of the **utterance-final phoneme**, per arm, cross-tabbed
  against the mark on the utterance's final word.

If an arm marks a case ending in the text and realizes it as a vowel in the
phoneme string, that arm is prescriptive. If the mark is there but the
phoneme string ends on the consonant, that arm is pausal. Different answers
per arm is a cross-arm comparability defect that has to be fixed before any
MSA number is generated.

The QuranMB `annotation_phoneme_string` column adds a third contrast that
exists on neither arm's reference: what reciters *actually produced*
utterance-finally, which is the empirical pausal rate `docs/msa-arm.md`
§3.4 proposes reporting as a measured quantity.

Depends on the JSONL cache written by `scripts/inspect_iqra_train_text.py`.

Run with:
    uv run python scripts/inspect_iqra_train_text.py --splits dev   # first
    uv run python scripts/compare_word_final_conventions.py
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import fsspec
import pyarrow.parquet as pq

from arabic_mdd.data.quranmb import QURANMB_DATASET

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "run" / "tmp" / "iqra_train_text"
QURANMB_CACHE = REPO_ROOT / "run" / "tmp" / "quranmb_text.jsonl"
QURANMB_PARQUET = (
    "https://huggingface.co/datasets/safikhan/quran_mbv2_formatted/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
QURANMB_COLUMNS = [
    "id",
    "reference_arabic_string",
    "reference_phoneme_string",
    "annotation_phoneme_string",
]

MARK_NAMES = {
    chr(0x064B): "tanwin fath",
    chr(0x064C): "tanwin damm",
    chr(0x064D): "tanwin kasr",
    chr(0x064E): "fatha",
    chr(0x064F): "damma",
    chr(0x0650): "kasra",
    chr(0x0651): "shadda",
    chr(0x0652): "sukun",
    chr(0x0670): "superscript alef",
}
CASE_MARKS = {chr(c) for c in (0x064B, 0x064C, 0x064D, 0x064E, 0x064F, 0x0650)}

# `sws_arabic.txt` short vowels, long vowels, and their emphatic-context
# variants (uppercase = emphatic/velarized neighbourhood in this scheme).
SHORT_VOWELS = {"a", "i", "u", "A", "I", "U"}
LONG_VOWELS = {"aa", "ii", "uu", "AA", "II", "UU"}
VOWELS = SHORT_VOWELS | LONG_VOWELS


def final_mark(word: str) -> str:
    """Name the word's case-ending mark: the last non-shadda mark it carries.

    Tanwin is conventionally written on the letter *before* a final alef
    ("كِتَابًا"), so keying on "mark attached to the last letter" would miss
    it. Taking the last mark in the word catches both placements. Shadda is
    skipped because it is a gemination mark that co-occurs with a vowel
    rather than being one.
    """
    for ch in reversed(word):
        if ch == chr(0x0651):
            continue
        if ch in MARK_NAMES:
            return MARK_NAMES[ch]
    return "(bare)"


def words_of(text: str) -> list[str]:
    cleaned = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return [w for w in cleaned.split() if any(ch in MARK_NAMES or ch.isalpha() for ch in w)]


def classify_phoneme(token: str) -> str:
    if token in LONG_VOWELS:
        return "long vowel"
    if token in SHORT_VOWELS:
        return "short vowel"
    return "consonant"


def fetch_quranmb() -> list[dict[str, Any]]:
    if QURANMB_CACHE.exists():
        with QURANMB_CACHE.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]
    QURANMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {QURANMB_DATASET} text columns ...")
    with fsspec.filesystem("http").open(QURANMB_PARQUET) as handle:
        table = pq.ParquetFile(handle).read(columns=QURANMB_COLUMNS)
    rows = table.to_pylist()
    with QURANMB_CACHE.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load_iqra(split: str) -> list[dict[str, Any]]:
    path = CACHE_DIR / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"No cache at {path}.\n"
            f"Run: uv run python scripts/inspect_iqra_train_text.py --splits {split}"
        )
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _table(title: str, counts: Counter[str], total: int, top: int = 10) -> None:
    print(f"\n  {title}  (n={total})")
    for name, count in counts.most_common(top):
        bar = "#" * int(40 * count / total) if total else ""
        print(f"    {name:20s} {100 * count / total:5.1f}%  {bar}")


def analyse_arm(name: str, pairs: list[tuple[str, str]]) -> None:
    """`pairs` is (diacritized arabic text, phoneme string) per utterance."""
    print("\n" + "=" * 72)
    print(f"ARM: {name}")
    print("=" * 72)

    all_word_marks: Counter[str] = Counter()
    final_word_marks: Counter[str] = Counter()
    final_phoneme_class: Counter[str] = Counter()
    cross: Counter[tuple[str, str]] = Counter()

    for text, phonemes in pairs:
        words = words_of(text)
        tokens = phonemes.split()
        if not words or not tokens:
            continue
        for word in words:
            all_word_marks[final_mark(word)] += 1
        mark = final_mark(words[-1])
        final_word_marks[mark] += 1
        klass = classify_phoneme(tokens[-1])
        final_phoneme_class[klass] += 1
        cross[(mark, klass)] += 1

    _table("word-final mark, ALL words", all_word_marks, sum(all_word_marks.values()))
    _table(
        "word-final mark, utterance-FINAL word", final_word_marks, sum(final_word_marks.values())
    )
    _table("utterance-final PHONEME class", final_phoneme_class, sum(final_phoneme_class.values()))

    print("\n  is a marked case ending REALIZED in the phoneme string?")
    print("    (utterance-final word's mark  ->  utterance-final phoneme class)")
    marked = [(m, k) for (m, k) in cross if m in {MARK_NAMES[c] for c in CASE_MARKS}]
    total_marked = sum(cross[key] for key in marked)
    if not total_marked:
        print("      no utterance-final case endings found")
        return
    realized = sum(cross[(m, k)] for (m, k) in marked if k != "consonant")
    realized_pct = 100 * realized / total_marked
    print(f"      marked utterances: {total_marked}")
    print(f"      ending in a vowel (prescriptive/realized): {realized_pct:5.1f}%")
    print(f"      ending in a consonant (pausal/dropped):    {100 - realized_pct:5.1f}%")
    for (mark, klass), count in sorted(cross.items(), key=lambda kv: -kv[1])[:8]:
        print(f"        {mark:20s} -> {klass:12s} {count:6d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iqra-split", default="dev")
    args = parser.parse_args()

    iqra = load_iqra(args.iqra_split)
    analyse_arm(
        f"MSA  --  IqraEval/Iqra_train [{args.iqra_split}], vowelizer output",
        [(r.get("tashkeel_sentence") or "", r.get("phoneme_ref") or "") for r in iqra],
    )

    quranmb = fetch_quranmb()
    analyse_arm(
        "QUR'ANIC  --  QuranMB.v2 reference (the scoring key)",
        [
            (r.get("reference_arabic_string") or "", r.get("reference_phoneme_string") or "")
            for r in quranmb
        ],
    )
    analyse_arm(
        "QUR'ANIC  --  QuranMB.v2 human annotation (what reciters ACTUALLY said)",
        [
            (r.get("reference_arabic_string") or "", r.get("annotation_phoneme_string") or "")
            for r in quranmb
        ],
    )

    print("\n" + "=" * 72)
    print("READ THIS AS: if the two reference arms disagree on the last block,")
    print("they use different case-ending conventions and are not comparable")
    print("until one is chosen (docs/msa-arm.md §3.4). The gap between the")
    print("QuranMB reference and annotation blocks is the empirical pausal")
    print("rate -- the quantity §3.4 proposes reporting rather than scoring.")
    print("=" * 72)


if __name__ == "__main__":
    main()
