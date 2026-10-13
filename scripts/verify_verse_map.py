"""Human verification of `run/quranmb_verse_map.json` — the independent half of task 1.

`recover_quran_verses.py` is **route (c)**: it matches the consonant skeleton of
`reference_arabic_string`, which the uploader derived *from*
`reference_phoneme_string`. So it reproduces the uploader's recovery rather than
testing it, and no amount of agreement between those two columns is evidence.

**The audio is the one instrument that is not downstream of that.** It shipped
with the benchmark, it was not generated from the text, and a human can hear
which words are being recited. That is why this pass exists and why it upgrades
route (c) from "a matcher we wrote agrees with a matcher they wrote" to "the
recovered text is what the reciters read."

**What you are judging, and what you are NOT.** QuranMB.v2 is a
*mispronunciation* benchmark: `annotation_phoneme_string` differs from
`reference_phoneme_string` on many clips because the reciters really do deviate,
sometimes deliberately. So the question is **word identity, not pronunciation
accuracy**:

> Is the reciter *attempting* these words, in this order?

A clip where the reciter fumbles a vowel still confirms the recovery. Judging
pronunciation here would reject correct recoveries and would also leak the
benchmark's own labels into a text decision. If one take sounds wrong, press
`t` for another take by a different speaker before concluding anything — there
are ~17 clips per prompt.

**Order is by risk, not by prompt number.** Prompts whose span occurs in more
than one place come first, then those whose text phonetizes away from the
reference, then the self-confirming remainder. So stopping early still buys the
most information per minute.

Resumable: writes after every decision to `run/quranmb_verse_map_verified.json`,
leaving the recovered map untouched.

    uv run python scripts/verify_verse_map.py
    uv run python scripts/verify_verse_map.py --only ambiguous
    uv run python scripts/verify_verse_map.py --review
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import tempfile
import wave
import webbrowser
from collections import Counter
from pathlib import Path

from arabic_mdd.data.normalize import strip_diacritics
from arabic_mdd.data.phonetizer import Phonetizer

REPO_ROOT = Path(__file__).resolve().parent.parent
MAP = REPO_ROOT / "run" / "quranmb_verse_map.json"
OUT = REPO_ROOT / "run" / "quranmb_verse_map_verified.json"
SHEET = REPO_ROOT / "run" / "quranmb_verse_map_sheet.html"
QURANMB_JOIN = "safikhan/quran_mbv2_formatted"

VERDICTS = {
    "y": "confirmed",
    "n": "wrong_verse",
    "b": "boundary_wrong",
    "p": "pick_candidate",
    "u": "unusable_audio",
    "s": "skip",
}

HELP = """
  y  CONFIRMED  — the reciter is attempting exactly these words, in this order
  n  WRONG      — different words entirely; the recovery is wrong
  b  BOUNDARY   — right verse, wrong extent: audio has MORE or FEWER words
                  than the text shown (the fragment's start/end is off).
                  Press 't' first: if only THIS take runs on into the next
                  verse it is that reciter, not the recovered boundary.
  p  PICK       — a different candidate TEXT (only prompts flagged AMBIGUOUS).
                  A bare "N locations" flag needs no action: the text is
                  identical at each, only the citation label differs.
  u  UNUSABLE   — audio truncated/silent/undecidable, no verdict possible
  t  next take  — another clip of the same prompt, usually a different speaker
  r  replay
  w  words      — word-by-word, one script per line
  o  open       — this prompt in the browser, where Arabic actually renders
  s  skip       — defer; returns to the BACK of the queue this session
  ?  this help
  q  save and quit
"""


_PHONETIZER = Phonetizer("prescriptive")

_RESHAPER: tuple[object, object] | None = None


def enable_reshaping() -> bool:
    """Lay the Arabic out ourselves, because the terminal won't.

    A terminal with no bidi pass prints codepoints in logical (memory) order,
    left to right, with every letter in its isolated form — which is
    unreadable even to a fluent reader. `arabic_reshaper` substitutes the
    contextual presentation forms and `get_display` reverses to visual order,
    so what reaches the terminal is already laid out and it only has to draw
    glyphs.

    `delete_harakat: False` is not optional here: the harakat are the entire
    object of study. Ligatures are off because lam-alef as a single glyph
    hides a word boundary this pass sometimes needs to see.
    """
    global _RESHAPER
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        print("  (arabic-reshaper / python-bidi not installed — printing logical order)")
        return False
    _RESHAPER = (
        arabic_reshaper.ArabicReshaper({"delete_harakat": False, "support_ligatures": False}),
        get_display,
    )
    return True


def _ar(text: str) -> str:
    """Arabic, laid out for a terminal that does no bidi of its own."""
    if _RESHAPER is None:
        return text
    reshaper, get_display = _RESHAPER
    return get_display(reshaper.reshape(text))  # type: ignore[operator]


_MARKS = {
    "\u064b": ("fathatan", "-an"),
    "\u064c": ("dammatan", "-un"),
    "\u064d": ("kasratan", "-in"),
    "\u064e": ("fatha", "-a"),
    "\u064f": ("damma", "-u"),
    "\u0650": ("kasra", "-i"),
    "\u0652": ("sukun", "— no vowel"),
}


def _final_mark(text: str) -> str:
    """Name the last word's final diacritic, in words.

    Candidates for an ambiguous prompt differ *only* in this mark, so it is
    the entire decision — and it is also the single character a terminal is
    least able to draw, since a combining mark that gets its own cell instead
    of stacking is exactly what makes diacritized Arabic unreadable here.
    Naming it sidesteps rendering altogether.
    """
    for char in reversed(text.strip()):
        if char in _MARKS:
            name, phon = _MARKS[char]
            return f"{name} ({phon})"
        if char == "\u0651":  # shadda sits before the vowel it doubles
            continue
        if not char.isspace():
            return "no final mark"
    return "no final mark"


def _ascii(text: str) -> str:
    """Halabi phonemes — the one form of the text a terminal cannot mangle.

    Terminals vary wildly on Arabic: many do no bidi reordering at all, fewer
    shape the joined letter forms, and almost none position stacked combining
    harakat correctly — which is precisely the part this pass has to read. So
    every Arabic string below is also printed as phonemes, which are pure
    ASCII and in the same space `phoneme_distance` is measured in.

    Deliberately *not* shown anywhere: the benchmark's own
    `reference_phoneme_string`. Picking a candidate by matching it would
    reproduce the uploader's matcher, which is the circularity this whole pass
    exists to escape. The ear decides; `d` is a flag, not a vote.
    """
    try:
        return " ".join(_PHONETIZER.phonetize(text)) or "(empty)"
    except Exception:  # noqa: BLE001 — a display aid must never block a verdict
        return "(phonetization failed)"


def _play(path: Path) -> None:
    """Best-effort playback. Never fatal — the path is always printed."""
    for player in ("ffplay", "paplay", "aplay", "afplay"):
        exe = shutil.which(player)
        if not exe:
            continue
        cmd = [exe, str(path)]
        if player == "ffplay":
            cmd = [exe, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
        try:
            subprocess.run(cmd, check=False, timeout=180)
            return
        except Exception:  # noqa: BLE001 — playback is a convenience
            continue
    print("  (no audio player found; the wav path is printed above)")


def _duration(path: Path) -> float | None:
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001
        return None


class Clips:
    """Lazy id -> wav bytes, without decoding.

    `datasets` needs `torchcodec` to *decode* audio, which nothing here wants:
    `Audio(decode=False)` hands back the original wav bytes and keeps random
    access, so a 221 MB corpus costs one memory-mapped read per clip.
    """

    def __init__(self, tmpdir: Path) -> None:
        import datasets

        self.tmpdir = tmpdir
        self.ds = datasets.load_dataset(QURANMB_JOIN, split="train").cast_column(
            "audio", datasets.Audio(decode=False)
        )
        self.by_prompt: dict[str, list[int]] = {}
        for i, row_id in enumerate(self.ds["id"]):
            self.by_prompt.setdefault(row_id.split("_")[1][:5], []).append(i)

    def take(self, prompt: str, nth: int) -> tuple[str, Path] | None:
        rows = self.by_prompt.get(prompt, [])
        if not rows:
            return None
        row = self.ds[rows[nth % len(rows)]]
        dest = self.tmpdir / f"{row['id']}.wav"
        if not dest.exists():
            dest.write_bytes(row["audio"]["bytes"])
        return row["id"], dest

    def count(self, prompt: str) -> int:
        return len(self.by_prompt.get(prompt, []))


def risk_order(entries: list[dict]) -> list[dict]:
    """Ambiguous first, then phonetically divergent, then self-confirming.

    Within a tier, descending phoneme distance: the further the recovered text
    sits from the reference phonemes, the more a human ear is worth.
    """

    def tier(e: dict) -> int:
        if e["ambiguous_text"]:
            return 0
        if e["n_locations"] != 1:
            return 1
        if not e["phoneme_exact"]:
            return 2
        return 3

    return sorted(entries, key=lambda e: (tier(e), -(e["phoneme_distance"] or 0), e["prompt"]))


def show(entry: dict, clips: Clips, nth: int) -> tuple[str, Path] | None:
    bare = " ".join(strip_diacritics(w) for w in entry["recovered_text"].split())
    print("\n" + "=" * 72)
    print(f"  prompt {entry['prompt']}   —   SURAH {entry['surah']} : AYAH ", end="")
    if entry["ayah_start"] == entry["ayah_end"]:
        print(f"{entry['ayah_start']}")
    else:
        n_ayahs = entry["ayah_end"] - entry["ayah_start"] + 1
        print(f"{entry['ayah_start']}–{entry['ayah_end']}  (spans {n_ayahs} ayahs)")
    print("=" * 72)
    # Undiacritized and numbered is the primary view, as in `pilot_annotate.py`:
    # a terminal cell grid cannot stack combining harakat, so the diacritized
    # string is the one form that reliably renders wrong. Word identity is what
    # this pass judges and the bare skeleton carries it exactly; the vowels are
    # carried by the phoneme line and by `ends:` on the candidates.
    if entry.get("override"):
        o = entry["override"]
        print(f"\n  ** CORRECTED from your last pass: {o['heard']}")
        was = entry.get("skeleton_text") or ""
        if not was:
            print("  ** text unchanged; the extra audio is outside the prompt by design")
        elif entry["n_words"] > len(was.split()):
            print("  ** that word has been ADDED below — confirm the span is now complete")
        else:
            print("  ** that word has been REMOVED below — confirm it is not recited")
    print(f"\n  recovered text — {entry['n_words']} words, undiacritized:\n")
    for i, word in enumerate(bare.split(), start=1):
        print(f"    {i:>2}. {_ar(word)}")
    print(f"\n  phonemes      :  {_ascii(entry['recovered_text'])}")
    print("  diacritized   :  (below — may render badly; the phonemes above are authoritative)")
    print(f"    {_ar(entry['recovered_text'])}")

    flags = []
    if entry["ambiguous_text"]:
        flags.append(f"AMBIGUOUS — {len(entry['candidates'])} candidate texts")
    if entry["n_locations"] > 1 and not entry["ambiguous_text"]:
        # Not a problem to solve: the recovered text is character-identical at
        # every one of these locations, so only the citation label is a choice
        # and nothing downstream reads it. Saying so prevents the flag being
        # mistaken for the AMBIGUOUS case above, which *is* a decision.
        flags.append(
            f"{entry['n_locations']} locations in the Qur'an — identical text at each, "
            "citation is the first; nothing to decide"
        )
    elif entry["n_locations"] > 1:
        flags.append(f"{entry['n_locations']} locations in the Qur'an")
    if not entry["phoneme_exact"]:
        flags.append(f"phoneme distance {entry['phoneme_distance']} from the reference")
    if flags:
        print(f"  flags         :  {'; '.join(flags)}")

    if entry["ambiguous_text"]:
        print("\n  candidates (primary is [0], chosen as scripturally first):")
        print("  decide by EAR — d comes from the benchmark's own phoneme column.")
        for i, c in enumerate(entry["candidates"]):
            print(
                f"\n    [{i}] first at {c['surah']}:{c['ayah_start']}, "
                f"{c['n_occurrences']}x in the Qur'an, d={c['phoneme_distance']}"
            )
            print(f"        ends: {_final_mark(c['text'])}")
            print(f"        {_ascii(c['text'])}")
            print(f"        {_ar(c['text'])}")

    got = clips.take(entry["prompt"], nth)
    if got is None:
        print("\n  !! no audio for this prompt")
        return None
    clip_id, path = got
    dur = _duration(path)
    n = clips.count(entry["prompt"])
    print(
        f"\n  clip          :  {clip_id}   (take {nth % n + 1} of {n})"
        + (f"   {dur:.1f}s" if dur else "")
    )
    if dur:
        # A gross duration/word-count mismatch is the cheap tell for `b`.
        print(f"  pace          :  {dur / max(entry['n_words'], 1):.2f}s per word")
    print(f"  path          :  {path}")
    return clip_id, path


_SHEET_CSS = """
  body { background:#14161a; color:#d8dee9; font:16px/1.6 system-ui, sans-serif;
         margin:0 auto; max-width:60rem; padding:2rem; }
  section { border-top:1px solid #2b313b; padding:1.5rem 0; }
  section:target { background:#1d2129; }
  h2 { font:600 15px/1 ui-monospace, monospace; color:#88c0d0; margin:0 0 .25rem; }
  .loc { font:13px ui-monospace, monospace; color:#7a8595; margin:0 0 1rem; }
  .ar { direction:rtl; text-align:right; font-size:34px; line-height:2.1;
        font-family:'Amiri','Scheherazade New','Noto Naskh Arabic','Traditional Arabic',serif;
        margin:.4rem 0; }
  .bare { color:#8fa3b8; font-size:26px; }
  .phon { font:13px ui-monospace, monospace; color:#7a8595; word-break:break-all; }
  .cand { border-right:3px solid #4c566a; margin:1rem 0; padding-right:1rem; }
  .cand b { font:600 13px ui-monospace, monospace; color:#ebcb8b; }
  .flag { font:13px ui-monospace, monospace; color:#d08770; }
"""


def make_sheet(entries: list[dict], path: Path) -> Path:
    """Write the reading sheet the terminal cannot be.

    A terminal is a fixed-width cell grid: every glyph gets one cell, so Arabic
    letters cannot join and combining harakat cannot stack on the letter they
    belong to. That is not a bug in any one terminal and no amount of reshaping
    or reordering fixes it — it is what a cell grid *is*. A browser does
    proportional shaping and mark positioning natively.

    So the split is: the terminal plays audio and takes verdicts, this file is
    what you read the Arabic from. Text here is in logical order, unreshaped —
    the browser does its own bidi, and pre-reversing would fight it.
    """
    parts = [
        "<!doctype html><html lang='ar' dir='rtl'><meta charset='utf-8'>",
        "<title>QuranMB verse map — reading sheet</title>",
        f"<style>{_SHEET_CSS}</style>",
        "<body><div dir='ltr'>",
        f"<p class='loc'>{len(entries)} prompts, risk-ordered — same order as the CLI queue. "
        "Jump with your browser's find; the CLI prints the prompt id.</p>",
    ]
    for entry in entries:
        ayah = (
            f"{entry['surah']}:{entry['ayah_start']}"
            if entry["ayah_start"] == entry["ayah_end"]
            else f"{entry['surah']}:{entry['ayah_start']}–{entry['ayah_end']}"
        )
        bare = " ".join(strip_diacritics(w) for w in entry["recovered_text"].split())
        parts.append(f"<section id='p{entry['prompt']}'><h2>prompt {entry['prompt']}</h2>")
        parts.append(
            f"<p class='loc'>surah {ayah} &middot; {entry['n_words']} words &middot; "
            f"{entry['n_clips']} clips</p>"
        )
        parts.append(f"<div class='ar'>{html.escape(entry['recovered_text'])}</div>")
        parts.append(f"<div class='ar bare'>{html.escape(bare)}</div>")
        parts.append(f"<p class='phon'>{html.escape(_ascii(entry['recovered_text']))}</p>")
        flags = []
        if entry["n_locations"] > 1:
            flags.append(f"{entry['n_locations']} locations in the Qur'an")
        if not entry["phoneme_exact"]:
            flags.append(f"phoneme distance {entry['phoneme_distance']}")
        if flags:
            parts.append(f"<p class='flag'>{html.escape('; '.join(flags))}</p>")
        if entry["ambiguous_text"]:
            parts.append("<p class='flag'>AMBIGUOUS — decide by ear, not by d</p>")
            for i, c in enumerate(entry["candidates"]):
                parts.append(
                    f"<div class='cand'><b>[{i}] first at {c['surah']}:{c['ayah_start']}, "
                    f"{c['n_occurrences']}x, d={c['phoneme_distance']}, "
                    f"ends {html.escape(_final_mark(c['text']))}</b>"
                    f"<div class='ar'>{html.escape(c['text'])}</div>"
                    f"<p class='phon'>{html.escape(_ascii(c['text']))}</p></div>"
                )
        parts.append("</section>")
    parts.append("</div></body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def words(entry: dict) -> None:
    """Word-by-word, one script per line — the fallback when shaping is broken.

    A long fragment is where a non-shaping terminal does the most damage, since
    the letters of adjacent words run together visually. One word per block,
    with its own phonemes, is readable even then.
    """
    print()
    for i, word in enumerate(entry["recovered_text"].split(), start=1):
        print(f"    {i}/{entry['n_words']}   {_ascii(word)}")
        print(f"        {_ar(word)}")
        print(f"        {_ar(strip_diacritics(word))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, default=MAP)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--only",
        choices=["all", "ambiguous", "inexact", "unverified"],
        default="unverified",
        help="which prompts to queue (default: everything without a verdict)",
    )
    parser.add_argument("--review", action="store_true", help="print verdicts and exit")
    parser.add_argument(
        "--recheck",
        nargs="+",
        metavar="PROMPT",
        default=[],
        help="re-queue these prompts even though they already have a verdict. The old "
        "verdict is kept under 'superseded' rather than discarded: a prompt whose text "
        "changed was judged against text that no longer exists, and the record should "
        "show that it was re-decided, not that it was always right.",
    )
    parser.add_argument("--sheet", type=Path, default=SHEET)
    parser.add_argument(
        "--sheet-only",
        action="store_true",
        help="write the HTML reading sheet and exit, without starting a session",
    )
    parser.add_argument(
        "--no-reshape",
        action="store_true",
        help="print Arabic in logical order; use this only if your terminal does its own "
        "bidi reordering, since doing it twice reverses the text back",
    )
    args = parser.parse_args()

    if not args.no_reshape:
        enable_reshaping()

    payload = json.loads(args.map.read_text(encoding="utf-8"))
    entries = {e["prompt"]: e for e in payload["prompts"]}

    verdicts: dict[str, dict] = {}
    superseded: dict[str, list[dict]] = {}
    if args.out.exists():
        prior = json.loads(args.out.read_text(encoding="utf-8"))
        verdicts = prior.get("verdicts", {})
        superseded = prior.get("superseded", {})

    for prompt in args.recheck:
        old = verdicts.pop(prompt, None)
        if old is None:
            print(f"  --recheck {prompt}: no existing verdict, queuing it anyway")
            continue
        superseded.setdefault(prompt, []).append(old)
        print(f"  --recheck {prompt}: superseding {old['verdict']!r}")

    if args.review:
        counts = Counter(v["verdict"] for v in verdicts.values())
        print(f"\n{len(verdicts)}/{len(entries)} prompts decided: {dict(counts)}")
        for p in sorted(verdicts):
            v = verdicts[p]
            e = entries[p]
            line = f"  {p}  {v['verdict']:<14} {e['surah']}:{e['ayah_start']}"
            if v.get("chosen_candidate") is not None:
                line += f"  -> candidate [{v['chosen_candidate']}]"
            if v.get("note"):
                line += f"   {v['note']}"
            print(line)
        return

    sheet = make_sheet(risk_order(list(entries.values())), args.sheet)
    if args.sheet_only:
        print(f"wrote {sheet}")
        webbrowser.open(sheet.as_uri())
        return

    queue = risk_order(list(entries.values()))
    if args.only == "ambiguous":
        queue = [e for e in queue if e["ambiguous_text"] or e["n_locations"] > 1]
    elif args.only == "inexact":
        queue = [e for e in queue if not e["phoneme_exact"]]
    elif args.only == "unverified":
        queue = [e for e in queue if e["prompt"] not in verdicts]

    if not queue:
        print("nothing queued — all prompts in this selection already have a verdict")
        return

    print(__doc__)
    print(f"queued: {len(queue)} prompts, risk-ordered")
    print(f"\nreading sheet (open this in a browser and keep it beside the terminal):\n  {sheet}")
    print("  a terminal cell grid cannot join Arabic letters or stack harakat; a browser can.")
    print("  'o' jumps the browser to the current prompt.")
    print(HELP)

    def save() -> None:
        args.out.write_text(
            json.dumps(
                {
                    "verifies": str(args.map.relative_to(REPO_ROOT)),
                    "edition_sha256": payload["edition"]["sha256"],
                    "route": payload["route"],
                    "instrument": "human listening — independent of the text columns",
                    "judged": "word identity, NOT pronunciation accuracy",
                    "n_prompts": len(entries),
                    "verdicts": verdicts,
                    "superseded": superseded,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    with tempfile.TemporaryDirectory(prefix="quranmb_verify_") as td:
        clips = Clips(Path(td))
        pending = list(queue)
        while pending:
            entry = pending.pop(0)
            nth = 0
            got = show(entry, clips, nth)
            if got:
                _play(got[1])
            while True:
                key = input("\n  [y/n/b/p/u/t/r/w/o/s/?/q] > ").strip().lower()
                if key == "?":
                    print(HELP)
                    continue
                if key == "w":
                    words(entry)
                    continue
                if key == "o":
                    webbrowser.open(f"{sheet.as_uri()}#p{entry['prompt']}")
                    continue
                if key == "q":
                    save()
                    print(f"\nsaved {len(verdicts)}/{len(entries)} to {args.out}")
                    return
                if key == "r" and got:
                    _play(got[1])
                    continue
                if key == "t":
                    nth += 1
                    got = show(entry, clips, nth)
                    if got:
                        _play(got[1])
                    continue
                if key == "s":
                    pending.append(entry)
                    break
                if key not in VERDICTS:
                    print("  unrecognised — ? for help")
                    continue

                record = {
                    "verdict": VERDICTS[key],
                    "clip": got[0] if got else None,
                    "takes_heard": nth + 1,
                }
                if key == "p":
                    raw = input(f"  candidate index [0-{len(entry['candidates']) - 1}] > ").strip()
                    if not raw.isdigit() or int(raw) >= len(entry["candidates"]):
                        print("  not a candidate index")
                        continue
                    record["chosen_candidate"] = int(raw)
                    record["verdict"] = "confirmed"
                if key in ("n", "b", "u"):
                    record["note"] = input("  what did you hear? > ").strip()
                else:
                    note = input("  note (optional) > ").strip()
                    if note:
                        record["note"] = note
                verdicts[entry["prompt"]] = record
                save()
                done = len(verdicts)
                print(f"  recorded — {done}/{len(entries)} decided, {len(pending)} left in queue")
                break

    save()
    counts = Counter(v["verdict"] for v in verdicts.values())
    print(f"\nqueue exhausted. {len(verdicts)}/{len(entries)} decided: {dict(counts)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
