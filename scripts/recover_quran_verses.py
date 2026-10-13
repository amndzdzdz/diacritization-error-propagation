"""Recover each QuranMB.v2 prompt's verse location — week-05 task 1, route (c).

Produces `run/quranmb_verse_map.json`: for each of the **98 prompt indices**,
the surah:ayah span it reads and the authoritative diacritized text of that
span, so the Qur'anic arm's label path has trustworthy Arabic to strip and
re-diacritize.

**This is route (c), a reproduction and not a validation.** Routes (a) and (b)
are both closed: `ID` carries no verse reference (measured, `insights/
week-05.md`), and the organizers publish no test Arabic — `IqraEval/
test_references` is `ID` + phonemes only, `IqraEval/QuranMB.v2` is `ID` +
audio, and `IqraEval/Iqra_Extra_IS26` is gated (401). So the query here is the
consonant skeleton of `reference_arabic_string`, which the uploader built *from*
`reference_phoneme_string` — the same algorithm class the uploader ran. Task 1
pre-committed to reporting that honestly rather than calling it a validation.
`scripts/verify_verse_map.py` adds the one genuinely independent instrument
available: a human listening to the audio.

**The method, and why this shape.** The prompts are verse *fragments*, not whole
verses — whole-verse phoneme lookup recovers 4–8/98 and the nearest whole verse
sits at edit distance ~20 on length-35 sequences. So the Qur'an is flattened to
one word stream and the prompt's word-skeleton sequence is matched as a
contiguous span. Exact matching only: no fuzzy threshold, nothing to tune.

**The edition is load-bearing and is the single biggest lever here.** Same
algorithm, two Tanzil editions: `simple` locates 98/98, `uthmani` 16/98. Uthmani
writes alef wasla (U+0671), which `normalize.is_arabic_letter` treats as a
letter, so it survives `strip_diacritics` and never matches a bare alef. That
was a written prediction in the week plan; it is an 82-prompt swing.

Not vendored: Tanzil ships a copyright block asking that it not be removed, so
the text stays local and is pinned by sha256 here instead.

Run with:
    uv run python scripts/recover_quran_verses.py --download
    uv run python scripts/recover_quran_verses.py --tanzil run/quran_simple.xml.local
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from arabic_mdd.data.normalize import strip_diacritics, unsupported_characters
from arabic_mdd.data.phonetizer import Phonetizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TANZIL = REPO_ROOT / "run" / "quran_simple.xml.local"
OUT = REPO_ROOT / "run" / "quranmb_verse_map.json"

#: The `simple` edition, NOT `uthmani`. See the module docstring: this choice
#: is worth 82 of the 98 recoveries, and it is also what keeps §9's alef-wasla
#: falsifier from firing. `imlaei-simple` is a 404 on this endpoint.
TANZIL_URL = "https://tanzil.net/pub/download/index.php?quranType=simple&outType=xml&agree=true"
TANZIL_EDITION = "tanzil-simple-xml"

#: `reference_phoneme_string` is prescriptive (full case endings, tanwin as
#: recited) — `docs/msa-arm.md` §3.4 and the Phonetizer's own docstring.
CONVENTION = "prescriptive"

QURANMB_JOIN = "safikhan/quran_mbv2_formatted"

#: Spans the skeleton matcher got wrong, corrected from the audio.
#:
#: The matcher assumes a prompt is a contiguous run of **whole Qur'anic
#: words**. It is not. QuranMB.v2 trims and adds proclitics relative to the
#: Qur'anic text, and one prompt's reference string begins partway through a
#: word. Nothing in the text columns can reveal that — `reference_arabic_string`
#: is derived from `reference_phoneme_string`, so both carry the same boundary.
#: The audio is the only independent witness, which is the verification pass's
#: justification restated as a result.
#:
#: Each entry is a human verdict from `scripts/verify_verse_map.py`, recorded
#: with what was heard. `text: None` means the recovered text is correct and the
#: finding is about the audio instead. These are applied as explicit overrides
#: rather than edited into the matcher so that the matcher stays honest about
#: what it can and cannot do.
#:
#: Separately measured while adjudicating 00087, and relevant to week 7 rather
#: than to recovery: over all 1642 clips, duration ≈ 1.53s + 0.083s × n_phones.
#: Every clip carries ~1.53s of fixed material outside the annotated phone
#: sequence — about 3.3 words at the marginal rate. It is audible as speech at
#: least sometimes (00087). Any MDD system fed whole clips will see it, and the
#: benchmark's labels do not account for it.
AUDIO_OVERRIDES: dict[str, dict[str, str | None]] = {
    "00035": {
        "text": "إِنَّ اللَّهَ غَنِيٌّ عَنِ الْعَالَمِينَ",
        "heard": "an extra initial 'inna' on every take",
        "why": "3:97 reads فَإِنَّ; the benchmark drops the fa, so the skeleton "
        "search could not match the word and started one word late. d 5 -> 3.",
    },
    "00036": {
        "text": "وَإِنَّ اللَّهَ عَلَىٰ كُلِّ شَيْءٍ قَدِيرٌ",
        "heard": "an extra initial 'wa' on every take",
        "why": "the reference phonemes do begin 'w a < i nn a'; the skeleton span "
        "omitted the conjunction. d 3 -> 1.",
    },
    "00069": {
        "text": "لَوْلَا أَن ثَبَّتْنَاكَ لَقَدْ كِدتَّ تَرْكَنُ إِلَيْهِمْ شَيْئًا قَلِيلًا",
        "heard": "the leading 'wa' is in the text but not in the audio",
        "why": "the reference phonemes begin 'l a w l aa' with no wa — the only "
        "override that REMOVES a word. d 3 -> 1.",
    },
    "00088": {
        "text": "يَغْشَى النَّاسَ هَٰذَا عَذَابٌ أَلِيمٌ",
        "heard": "an extra word at the start",
        "why": "the reference begins 'x $ aa nn aa s a' — a truncated يَغْشَى, with "
        "خ for غ. d is unchanged at 3 because that truncation and typo are in the "
        "benchmark, not in the recovery; the audio is what settles the span.",
    },
    "00087": {
        "text": None,
        "heard": "an extra initial عِندَهَا",
        "why": "Deliberately NOT added, on three independent grounds. (1) The "
        "recovered text is already d=0 against the reference; adding عِندَهَا "
        "raises it to 7. (2) All 7 annotation_phoneme_strings are 10-11 phones "
        "beginning 'r i s q A n aa' — the benchmark's own annotators, hearing the "
        "same clips, did not transcribe it either. (3) recovered_text has to stay "
        "in the frame reference_phoneme_string defines, because that is the frame "
        "the annotation labels are aligned to; adding a 7-phone word would shift "
        "every label for this prompt. The audio really does contain it, and that "
        "is a property of the clips, not an error in the text.",
    },
}


@dataclass
class PromptEntry:
    prompt: str
    n_words: int
    n_clips: int
    surah: int | None
    ayah_start: int | None
    ayah_end: int | None
    recovered_text: str
    #: Distinct candidate texts when the skeleton span occurs in more than one
    #: place. Kept rather than silently resolved — see `ambiguous_text` below.
    candidates: list[dict] = field(default_factory=list)
    n_locations: int = 0
    ambiguous_text: bool = False
    phoneme_distance: int | None = None
    phoneme_exact: bool = False
    dataset_agrees: bool = False
    #: Set when `AUDIO_OVERRIDES` replaced the skeleton match. `skeleton_text`
    #: keeps what the matcher produced so the correction stays auditable rather
    #: than disappearing into the output.
    override: dict | None = None
    skeleton_text: str = ""


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Token-level edit distance. Small inputs; no dependency worth adding."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def download_tanzil(dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {TANZIL_EDITION} -> {dest}")
    with urllib.request.urlopen(TANZIL_URL, timeout=120) as r:  # noqa: S310
        dest.write_bytes(r.read())


def load_verses(path: Path) -> list[tuple[int, int, str]]:
    root = ET.parse(path).getroot()
    out: list[tuple[int, int, str]] = []
    for sura in root.iter("sura"):
        s = int(sura.get("index"))  # type: ignore[arg-type]
        for aya in sura.iter("aya"):
            # `bismillah` is a separate attribute and is NOT part of the ayah
            # text; including it would insert four words that no prompt reads.
            out.append((s, int(aya.get("index")), aya.get("text", "")))  # type: ignore[arg-type]
    if not out:
        raise SystemExit(f"{path}: no <aya> elements — wrong file or a 404 page?")
    return out


def load_prompts() -> tuple[dict[str, tuple[str, str]], Counter[str]]:
    """prompt index -> (reference_arabic, reference_phonemes), plus clip counts.

    Keyed on **prompt index (98)**, not on Arabic text (96) — prompts 9/42 share
    an Arabic string but differ in phoneme string (one pausal, one liaison
    word-final), so keying on text merges them and discards the distinction;
    73/91 are a true duplicate. Prompt index is metadata, so unlike the Arabic
    column it is not downstream of the uploader's phonemisation.
    """
    import datasets

    ds = datasets.load_dataset(QURANMB_JOIN, split="train").remove_columns("audio")
    prompts: dict[str, tuple[str, str]] = {}
    clips: Counter[str] = Counter()
    for row in ds:
        key = row["id"].split("_")[1][:5]
        clips[key] += 1
        prompts.setdefault(key, (row["reference_arabic_string"], row["reference_phoneme_string"]))
    return prompts, clips


def recover(verses: list[tuple[int, int, str]], prompts, clips) -> list[PromptEntry]:
    words = [(s, a, w) for s, a, text in verses for w in text.split()]
    skel = [strip_diacritics(w) for _s, _a, w in words]
    phonetizer = Phonetizer(CONVENTION)

    entries: list[PromptEntry] = []
    for key in sorted(prompts):
        arabic, phonemes = prompts[key]
        query = [strip_diacritics(w) for w in arabic.split()]
        n = len(query)
        hits = [i for i in range(len(skel) - n + 1) if skel[i : i + n] == query]

        texts: dict[str, list[int]] = {}
        for i in hits:
            texts.setdefault(" ".join(w for _s, _a, w in words[i : i + n]), []).append(i)

        entry = PromptEntry(
            prompt=key,
            n_words=n,
            n_clips=clips[key],
            surah=None,
            ayah_start=None,
            ayah_end=None,
            recovered_text="",
            n_locations=len(hits),
            ambiguous_text=len(texts) > 1,
        )
        if hits:
            # Order candidates by scriptural order so the listing is stable.
            for text, starts in sorted(texts.items(), key=lambda kv: kv[1][0]):
                i = starts[0]
                entry.candidates.append(
                    {
                        "surah": words[i][0],
                        "ayah_start": words[i][1],
                        "ayah_end": words[i + n - 1][1],
                        "text": text,
                        "n_occurrences": len(starts),
                        "phoneme_distance": _levenshtein(
                            phonetizer.phonetize(text), phonemes.split()
                        ),
                    }
                )
            # Primary = scripturally first. Deliberately NOT the lowest phoneme
            # distance: picking by closeness to the phoneme string is the
            # uploader's own matcher, so it would deepen the circularity the
            # verification pass exists to offset. Ambiguity is flagged for a
            # human instead.
            best = entry.candidates[0]
            entry.surah = best["surah"]
            entry.ayah_start = best["ayah_start"]
            entry.ayah_end = best["ayah_end"]
            entry.recovered_text = best["text"]
            entry.phoneme_distance = best["phoneme_distance"]
            entry.phoneme_exact = best["phoneme_distance"] == 0
            entry.dataset_agrees = unicodedata.normalize(
                "NFC", best["text"]
            ) == unicodedata.normalize("NFC", arabic)

        override = AUDIO_OVERRIDES.get(key)
        if override is not None:
            entry.override = dict(override)
            text = override["text"]
            if text is not None:
                entry.skeleton_text = entry.recovered_text
                entry.recovered_text = text
                entry.n_words = len(text.split())
                entry.phoneme_distance = _levenshtein(phonetizer.phonetize(text), phonemes.split())
                entry.phoneme_exact = entry.phoneme_distance == 0
                entry.dataset_agrees = False
                # Relocate if the corrected span is itself a contiguous Qur'anic
                # run. It usually is not -- that is exactly why the matcher
                # missed it -- in which case the skeleton match's location was
                # still right and is kept.
                q = [strip_diacritics(w) for w in text.split()]
                k = len(q)
                found = [i for i in range(len(skel) - k + 1) if skel[i : i + k] == q]
                entry.override["relocated"] = bool(found)
                if found:
                    i = found[0]
                    entry.surah = words[i][0]
                    entry.ayah_start = words[i][1]
                    entry.ayah_end = words[i + k - 1][1]
                    entry.n_locations = len(found)
                    entry.ambiguous_text = False
                    entry.candidates = []
        entries.append(entry)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tanzil", type=Path, default=DEFAULT_TANZIL)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if args.download or not args.tanzil.exists():
        if not args.download:
            raise SystemExit(f"{args.tanzil} not found; pass --download to fetch it")
        download_tanzil(args.tanzil)

    raw = args.tanzil.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    verses = load_verses(args.tanzil)
    prompts, clips = load_prompts()
    entries = recover(verses, prompts, clips)

    # §9's alef-wasla falsifier is tested on the text we actually chose, rather
    # than assumed from the edition's name. The week plan was explicit that
    # picking an edition to dodge the falsifier obliges us to measure the dodge.
    corpus = " ".join(t for _s, _a, t in verses)
    audit = {
        "characters": len(corpus),
        "alef_wasla_U+0671": corpus.count("\u0671"),
        "dagger_alef_U+0670": corpus.count("\u0670"),
        "quranic_marks_U+06D6_U+06ED": sum(1 for c in corpus if "\u06d6" <= c <= "\u06ed"),
        "unsupported_characters": {k: v for k, v in sorted(unsupported_characters(corpus).items())},
    }

    # Exit criterion 1 asks for this over the RECOVERED TEXT, which is not the
    # same thing. The corpus audit is a superset only while every recovered text
    # is a verbatim corpus span, and `AUDIO_OVERRIDES` writes four spans the
    # Qur'an does not contain contiguously. So audit what actually feeds the
    # label path, separately from what tests the edition.
    recovered = " ".join(e.recovered_text for e in entries)
    recovered_audit = {
        "characters": len(recovered),
        "unsupported_characters": {
            k: v for k, v in sorted(unsupported_characters(recovered).items())
        },
    }

    located = [e for e in entries if e.surah is not None]
    exact = [e for e in located if e.phoneme_exact]
    payload = {
        "decision": "docs/weeks/week-05.md task 1; docs/msa-arm.md 3.3b D1",
        "generated_by": "scripts/recover_quran_verses.py",
        "route": "c",
        "route_note": (
            "Consonant-skeleton span match against an authoritative text. Routes "
            "(a) and (b) are closed: ID carries no verse reference, and the "
            "organizers publish no test Arabic (test_references is ID+phonemes, "
            "QuranMB.v2 is ID+audio, Iqra_Extra_IS26 is gated). The query is the "
            "skeleton of reference_arabic_string, which the uploader derived from "
            "reference_phoneme_string, so this REPRODUCES the uploader's recovery "
            "and does not independently validate it. The independent instrument is "
            "scripts/verify_verse_map.py (human listening)."
        ),
        "edition": {
            "name": TANZIL_EDITION,
            "url": TANZIL_URL,
            "sha256": digest,
            "verses": len(verses),
            "vendored": False,
            "licence_note": (
                "Tanzil ships a copyright block asking it not be removed, so the "
                "text is kept local and pinned by sha256 rather than committed."
            ),
        },
        "convention": CONVENTION,
        "key": "prompt index (98), not Arabic text (96)",
        "phonetize_ceiling": (
            "97/98 by construction: prompts 9 and 42 share one Arabic string but "
            "have different phoneme strings (pausal vs liaison final word), and "
            "phonetize is deterministic, so both cannot match. Subtract this "
            "BEFORE reading the exact-match count, not after."
        ),
        "summary": {
            "prompts": len(entries),
            "located": len(located),
            "unique_location": sum(1 for e in entries if e.n_locations == 1),
            "text_unambiguous": sum(1 for e in entries if not e.ambiguous_text),
            "phoneme_exact": len(exact),
            "dataset_arabic_agrees": sum(1 for e in entries if e.dataset_agrees),
        },
        "character_audit": audit,
        "recovered_text_audit": recovered_audit,
        "prompts": [asdict(e) for e in entries],
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    s = payload["summary"]
    print(f"\ntanzil {TANZIL_EDITION} sha256 {digest[:16]}… — {len(verses)} verses")
    print(f"located            : {s['located']}/{s['prompts']}")
    print(f"unique location    : {s['unique_location']}/{s['prompts']}")
    print(f"text unambiguous   : {s['text_unambiguous']}/{s['prompts']}")
    print(f"phoneme exact      : {s['phoneme_exact']}/{s['prompts']}  (ceiling 97)")
    print(f"dataset text agrees: {s['dataset_arabic_agrees']}/{s['prompts']}")
    print(f"\ncharacter audit (edition) : {audit}")
    print(f"character audit (recovered): {recovered_audit}")
    dist = Counter(e.phoneme_distance for e in located)
    print(f"phoneme distance distribution: {dict(sorted(dist.items(), key=lambda kv: kv[0]))}")
    print(f"\nwrote {args.out}")
    print("NOT verified against audio yet — run scripts/verify_verse_map.py")


if __name__ == "__main__":
    main()
