"""Inspect `IqraEval/Iqra_train`'s text columns -- MSA-arm step 1.

Answers three questions that `docs/msa-arm.md` leaves open, using data
inspection alone: no phonetizer, no annotation, no GPU.

1. **Is there a second phonetizer round-trip reference, on MSA text?**
   The week-4 exit criterion (`docs/weeks/week-04.md`) round-trips the
   phonetizer on QuranMB.v2's 1,642 Arabic strings, which validates it on
   Qur'anic orthography only -- `docs/msa-arm.md` §4 flags that MSA
   transcripts add digits, Latin script and loanwords the round-trip never
   exercises. `Iqra_train` ships a `tashkeel_sentence` column (the
   organizers' in-house vowelizer output) alongside `phoneme_ref`, so
   `phonetize(tashkeel_sentence) == phoneme_ref` may be a second, much
   larger round-trip set that *does* cover MSA orthography.

2. **Risk (a): does MSA text stay inside the 68-token vocab?**
   `docs/msa-arm.md` §4 -- any phoneme the CTC head cannot emit is a
   guaranteed false rejection for a mechanical reason, indistinguishable in
   the metric from the paper's own claim. The organizers already phonetized
   this text into `sws_arabic.txt`, so their own data partially answers
   this before our phonetizer exists.

3. **What does the undiacritized input to the diacritizers actually look
   like?** The MSA arm feeds bare text to CATT/Shakkala/Mishkal/Farasa. If
   `sentence` is already (partly) diacritized, it is not that input, and
   the arm needs an explicit stripping step whose behaviour must be pinned.

Only the text columns are read -- `pyarrow` pushes the column projection
into the parquet reader, so the 12 GB of audio in these files is never
fetched. Rows are cached as JSONL under `run/tmp/` so step 2
(`scripts/compare_word_final_conventions.py`) can reuse them.

Run with:
    uv run python scripts/inspect_iqra_train_text.py --splits dev
    uv run python scripts/inspect_iqra_train_text.py            # both splits
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
import requests

DATASET = "IqraEval/Iqra_train"
PARQUET_INDEX = "https://datasets-server.huggingface.co/parquet?dataset=IqraEval%2FIqra_train"
TEXT_COLUMNS = ["id", "sentence", "tashkeel_sentence", "phoneme_ref", "phoneme_aug"]

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "run" / "tmp" / "iqra_train_text"
VOCAB_PATH = REPO_ROOT / "scripts" / "baseline_reproduction" / "vocab" / "sws_arabic.txt"

# Arabic combining marks: harakat + tanwin + shadda + sukun (U+064B-U+0652),
# hamza/madda marks (U+0653-U+0655), superscript alef (U+0670), and the
# Qur'anic annotation block (U+06D6-U+06ED). Tatweel (U+0640) is a
# letter-stretching glyph, not a diacritic, but is stripped alongside them.
DIACRITICS = (
    {chr(c) for c in range(0x064B, 0x0653)}
    | {chr(c) for c in range(0x0653, 0x0656)}
    | {chr(0x0670)}
    | {chr(c) for c in range(0x06D6, 0x06EE)}
)
TATWEEL = chr(0x0640)
SHADDA = chr(0x0651)

# Word-final short vowels / tanwin -- the case-ending (i'rab) marks.
CASE_ENDING_MARKS = {
    chr(0x064B): "tanwin fath",
    chr(0x064C): "tanwin damm",
    chr(0x064D): "tanwin kasr",
    chr(0x064E): "fatha",
    chr(0x064F): "damma",
    chr(0x0650): "kasra",
    chr(0x0652): "sukun",
}


def is_arabic_letter(ch: str) -> bool:
    """True for Arabic *letters* -- excludes diacritics, digits, punctuation."""
    return ("\u0621" <= ch <= "\u063a") or ("\u0641" <= ch <= "\u064a") or ch in "\u0671\u0675"


def strip_diacritics(text: str) -> str:
    return "".join(ch for ch in text if ch not in DIACRITICS and ch != TATWEEL)


def _depunctuate(text: str) -> str:
    """Drop punctuation and normalize whitespace, for skeleton comparison."""
    return " ".join(
        "".join(ch for ch in text if not unicodedata.category(ch).startswith("P")).split()
    )


def diacritization_rate(text: str) -> float | None:
    """Fraction of Arabic letters carrying at least one diacritic.

    Returns None for text with no Arabic letters at all. Note a fully
    diacritized string does not reach 1.0: long vowels (alef/waw/ya as
    madd) and the definite article's lam are conventionally bare, so
    well-formed fully-vowelized MSA lands around 0.7-0.9.
    """
    letters = 0
    marked = 0
    chars = list(text)
    for i, ch in enumerate(chars):
        if not is_arabic_letter(ch):
            continue
        letters += 1
        if i + 1 < len(chars) and chars[i + 1] in DIACRITICS:
            marked += 1
    return marked / letters if letters else None


def fetch_split(split: str, force: bool = False) -> Path:
    """Download this split's text columns into a JSONL cache; return its path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{split}.jsonl"
    if out_path.exists() and not force:
        print(f"[cache] {split}: reusing {out_path}")
        return out_path

    index = requests.get(PARQUET_INDEX, timeout=60).json()
    files = [f for f in index["parquet_files"] if f["split"] == split]
    if not files:
        raise SystemExit(f"No parquet files listed for split {split!r}")
    files.sort(key=lambda f: f["filename"])

    fs = fsspec.filesystem("http")
    tmp_path = out_path.with_suffix(".jsonl.partial")
    written = 0
    with tmp_path.open("w", encoding="utf-8") as out:
        for n, entry in enumerate(files, 1):
            with fs.open(entry["url"]) as handle:
                table = pq.ParquetFile(handle).read(columns=TEXT_COLUMNS)
            for row in table.to_pylist():
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            print(f"[fetch] {split}: {entry['filename']} ({n}/{len(files)}) -> {written} rows")
    tmp_path.replace(out_path)
    return out_path


def load_rows(split: str) -> list[dict[str, Any]]:
    path = CACHE_DIR / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"No cache for split {split!r}; run without --no-fetch first.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _pct(count: int, total: int) -> str:
    return f"{count:6d}/{total} ({100 * count / total:5.1f}%)" if total else "n/a"


def report_phoneme_inventory(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("1. PHONEME INVENTORY vs the 68-token sws_arabic.txt  [msa-arm.md risk (a)]")
    print("=" * 72)

    vocab = [line.strip() for line in VOCAB_PATH.read_text(encoding="utf-8").splitlines()]
    vocab_set = {token for token in vocab if token}
    print(f"vocab: {len(vocab_set)} tokens from {VOCAB_PATH.relative_to(REPO_ROOT)}")

    for column in ("phoneme_ref", "phoneme_aug"):
        counts: Counter[str] = Counter()
        for row in rows:
            counts.update((row.get(column) or "").split())
        observed = set(counts)
        out_of_vocab = observed - vocab_set
        unused = vocab_set - observed
        total_tokens = sum(counts.values())

        print(f"\n  {column}: {total_tokens} tokens, {len(observed)} distinct")
        if out_of_vocab:
            oov_tokens = sum(counts[t] for t in out_of_vocab)
            print(f"    OUT OF VOCAB: {len(out_of_vocab)} types, {oov_tokens} tokens")
            for token in sorted(out_of_vocab, key=lambda t: -counts[t])[:20]:
                print(f"      {token!r}: {counts[token]}")
        else:
            print("    OUT OF VOCAB: none -- every token is in the 68-token vocab")
        if unused:
            print(f"    unused vocab tokens ({len(unused)}): {sorted(unused)}")


def report_round_trip_candidacy(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("2. ROUND-TRIP CANDIDACY: phonetize(tashkeel_sentence) == phoneme_ref?")
    print("=" * 72)

    total = len(rows)
    missing_tash = sum(1 for r in rows if not (r.get("tashkeel_sentence") or "").strip())
    missing_phon = sum(1 for r in rows if not (r.get("phoneme_ref") or "").strip())
    print(f"rows: {total}")
    print(f"  empty tashkeel_sentence: {_pct(missing_tash, total)}")
    print(f"  empty phoneme_ref:       {_pct(missing_phon, total)}")

    rates = [
        rate
        for r in rows
        if (rate := diacritization_rate(r.get("tashkeel_sentence") or "")) is not None
    ]
    if rates:
        rates.sort()
        print("\n  tashkeel_sentence diacritization rate (marked Arabic letters / Arabic letters)")
        print(f"    mean {sum(rates) / len(rates):.3f}   median {rates[len(rates) // 2]:.3f}")
        print(f"    p05  {rates[len(rates) // 20]:.3f}   p95    {rates[-len(rates) // 20]:.3f}")
        bare = sum(1 for rate in rates if rate < 0.2)
        print(f"    rows under 0.20 (effectively undiacritized): {_pct(bare, len(rates))}")

    # phoneme_ref vs phoneme_aug: if identical everywhere, phoneme_aug is not
    # an independent "actual production" annotation and cannot stand in for A.
    both = [r for r in rows if r.get("phoneme_ref") and r.get("phoneme_aug")]
    identical = sum(1 for r in both if r["phoneme_ref"] == r["phoneme_aug"])
    print(f"\n  phoneme_aug == phoneme_ref: {_pct(identical, len(both))}")
    print("    (if ~100%, phoneme_aug carries no mispronunciation signal -- msa-arm.md §3.3)")


def report_input_text_state(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("3. WHAT IS `sentence`? (the undiacritized input the diacritizers need)")
    print("=" * 72)

    total = len(rows)
    identical = 0
    same_skeleton = 0
    same_depunctuated = 0
    rates_bare: list[float] = []
    for row in rows:
        sentence = row.get("sentence") or ""
        tashkeel = row.get("tashkeel_sentence") or ""
        if sentence == tashkeel:
            identical += 1
        src, out = strip_diacritics(sentence), strip_diacritics(tashkeel)
        if src == out:
            same_skeleton += 1
        if _depunctuate(src) == _depunctuate(out):
            same_depunctuated += 1
        rate = diacritization_rate(sentence)
        if rate is not None:
            rates_bare.append(rate)

    print(f"  sentence == tashkeel_sentence:                 {_pct(identical, total)}")
    print(f"  same consonantal skeleton after stripping:     {_pct(same_skeleton, total)}")
    print(f"  ...and after also removing punctuation:        {_pct(same_depunctuated, total)}")
    print("    (the gap between these two rows is what the vowelizer's own")
    print("     normalization does; the MSA arm must apply the same step)")

    if rates_bare:
        rates_bare.sort()
        buckets = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.01)]
        print("\n  sentence diacritization rate, bucketed:")
        for lo, hi in buckets:
            n = sum(1 for rate in rates_bare if lo <= rate < hi)
            print(f"    [{lo:.2f}, {hi:.2f}): {_pct(n, len(rates_bare))}")
        print("    (a heterogeneous `sentence` column means the MSA arm needs an")
        print("     explicit strip step; bare text cannot be assumed)")


def report_non_arabic(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("4. NON-ARABIC CHARACTERS  [msa-arm.md §4: normalization rules]")
    print("=" * 72)

    offenders: Counter[str] = Counter()
    rows_with: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for row in rows:
        sentence = row.get("sentence") or ""
        seen: set[str] = set()
        for ch in sentence:
            if ch.isspace() or is_arabic_letter(ch) or ch in DIACRITICS or ch == TATWEEL:
                continue
            category = unicodedata.category(ch)
            if category.startswith("P"):
                label = "punctuation"
            elif ch.isdigit():
                label = "digit"
            elif "A" <= ch.upper() <= "Z":
                label = "latin letter"
            else:
                label = f"other ({unicodedata.name(ch, 'UNNAMED')})"
            offenders[label] += 1
            seen.add(label)
            examples.setdefault(label, sentence)
        for label in seen:
            rows_with[label] += 1

    if not offenders:
        print("  none -- `sentence` is pure Arabic script + whitespace")
        return
    total = len(rows)
    for label, count in offenders.most_common(12):
        print(f"  {label:28s} {count:7d} chars, in {_pct(rows_with[label], total)}")
        print(f"      e.g. {examples[label][:70]}")


def report_content_loss(rows: list[dict[str, Any]]) -> None:
    """Does the vowelizer preserve the sentence, or does it drop content?

    Spotted in the dev split: rows whose `sentence` contains Latin script or
    a Qur'anic pause mark come back from the vowelizer with words missing or
    the tail truncated. That matters well beyond tidiness -- `phoneme_ref`
    is the CTC *training target*, so a dropped word is audio the model is
    trained to emit nothing for, and at evaluation it is a guaranteed
    insertion against the reference. Same mechanical-false-rejection
    mechanism as `docs/msa-arm.md` §4, already present in the organizers'
    own data.
    """
    print("\n" + "=" * 72)
    print("5. VOWELIZER CONTENT LOSS: does tashkeel_sentence preserve `sentence`?")
    print("=" * 72)

    total = len(rows)
    truncated: list[dict[str, Any]] = []
    latin_dropped = 0
    latin_rows = 0
    ratios: list[float] = []
    for row in rows:
        sentence = row.get("sentence") or ""
        tashkeel = row.get("tashkeel_sentence") or ""
        src_words = len(strip_diacritics(sentence).split())
        out_words = len(strip_diacritics(tashkeel).split())
        if src_words:
            ratios.append(out_words / src_words)
            if out_words < 0.7 * src_words:
                truncated.append(row)
        has_latin = any("A" <= ch.upper() <= "Z" for ch in sentence)
        if has_latin:
            latin_rows += 1
            if not any("A" <= ch.upper() <= "Z" for ch in tashkeel):
                latin_dropped += 1

    print(f"  rows losing >30% of their words: {_pct(len(truncated), total)}")
    if ratios:
        ratios.sort()
        median = ratios[len(ratios) // 2]
        print(f"  word-count ratio (out/in): median {median:.3f}, min {ratios[0]:.3f}")
    print(f"  rows with Latin script in `sentence`:   {_pct(latin_rows, total)}")
    print(f"    ...where the vowelizer dropped it all: {_pct(latin_dropped, max(latin_rows, 1))}")

    for row in truncated[:5]:
        print(f"\n    SRC : {(row.get('sentence') or '')[:72]}")
        print(f"    TASH: {(row.get('tashkeel_sentence') or '')[:72]}")

    # Superscript alef (U+0670, dagger alef) moves in BOTH directions, which
    # is why the two counts below are reported separately rather than as one
    # "rewrite rate". On the train split the vowelizer strips every U+0670 it
    # is given (120/120) and then re-inserts its own on ~3.8% of rows --
    # mostly Qur'anic/classical forms where its convention wants one
    # ("الرحمن" -> "الرَّحْمَٰن", "موسى" -> "مُوسَىٰ"). So the count that
    # matters for the phonetizer is the OUTPUT one: U+0670 is a live
    # character in `tashkeel_sentence` and must be handled there, regardless
    # of how rare it is in `sentence`.
    dagger = chr(0x0670)
    dagger_src = sum(1 for r in rows if dagger in (r.get("sentence") or ""))
    dagger_out = sum(1 for r in rows if dagger in (r.get("tashkeel_sentence") or ""))
    dagger_kept = sum(
        1
        for r in rows
        if dagger in (r.get("sentence") or "") and dagger in (r.get("tashkeel_sentence") or "")
    )
    dagger_added = sum(
        1
        for r in rows
        if dagger not in (r.get("sentence") or "") and dagger in (r.get("tashkeel_sentence") or "")
    )
    print(f"\n  superscript alef (U+0670) in `sentence`:          {_pct(dagger_src, total)}")
    print(f"  superscript alef (U+0670) in `tashkeel_sentence`: {_pct(dagger_out, total)}")
    print(f"    survived from `sentence`: {dagger_kept}/{dagger_src or 0}")
    print(f"    inserted by the vowelizer where the source had none: {dagger_added}")
    print("    (the phonetizer must handle U+0670 in the OUTPUT, not just the input)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["dev", "train"])
    parser.add_argument("--no-fetch", action="store_true", help="Use the JSONL cache only.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = parser.parse_args()

    for split in args.splits:
        if not args.no_fetch:
            fetch_split(split, force=args.force)
        rows = load_rows(split)
        print("\n\n" + "#" * 72)
        print(f"#  {DATASET}  split={split}  rows={len(rows)}")
        print("#" * 72)
        report_phoneme_inventory(rows)
        report_round_trip_candidacy(rows)
        report_input_text_state(rows)
        report_non_arabic(rows)
        report_content_loss(rows)


if __name__ == "__main__":
    main()
