"""Blinded, resumable annotation UI for the 50-utterance listening pilot.

Implements `docs/annotation-protocol-pilot.md`. Three things it enforces
that hand-editing `run/msa_pilot_sample.json` would not:

1. **Blinding.** The manifest carries the model's `prediction`, the
   `disagreement` score and the `stratum`. Seeing any of them while judging
   anchors the verdict to the model, which would make the pilot measure
   agreement-with-the-model rather than the speaker's production -- the
   exact confound that already cost us the CV-Ar dev screen. None are ever
   printed here.
2. **Interleaved order.** The manifest is stored stratum-by-stratum: 25
   high-disagreement rows followed by 25 uniform ones. Annotating in file
   order means 25 error-rich utterances in a row, which recalibrates the
   annotator's strictness before the uniform stratum -- the stratum that
   actually carries the base rate. A fixed shuffle interleaves them.
3. **Word-level capture.** Per the protocol's power calculation, a binary
   per-utterance verdict bounds the base rate only at 12%, which is useless
   against QuranMB's 12.41%. Marking words bounds it at 2.1%.

Writes to a SEPARATE file (`run/msa_pilot_annotated.json`), leaving the
sampled manifest pristine. Saves after every utterance, so it is safe to
stop and resume at any point.

    uv run python scripts/pilot_annotate.py
    uv run python scripts/pilot_annotate.py --play          # auto-play audio
    uv run python scripts/pilot_annotate.py --review        # re-read own verdicts
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed so the presentation order is reproducible and stated in the paper.
SHUFFLE_SEED = 20260923

VERDICTS = {"c": "clean", "e": "error", "u": "unusable", "s": "skip"}

TAGS = {
    "sub": "substitution -- different consonant/vowel than written",
    "del": "deletion -- a written phoneme is absent",
    "ins": "insertion -- a phoneme not in the written word",
    "word": "wrong/skipped/added word",
    "dialect": "dialectal realization (th->t/s, q->g, j->zh, ...)",
    "case": "case ending (i'rab) -- NOT an error under our pausal convention",
    "hesit": "hesitation/restart, word ultimately correct",
    "unclear": "still unsure after three listens",
}


def _play(path: Path) -> None:
    """Best-effort playback. Never fatal -- the path is always printed."""
    for player in ("paplay", "aplay", "afplay", "ffplay"):
        exe = shutil.which(player)
        if not exe:
            continue
        cmd = [exe, str(path)]
        if player == "ffplay":
            cmd = [exe, "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]
        try:
            subprocess.run(cmd, check=False, timeout=120)
            return
        except Exception:  # noqa: BLE001 - playback is a convenience, not a requirement
            continue
    print("  (no audio player found -- open the path above manually)")


def _numbered_words(text: str) -> str:
    words = (text or "").split()
    if not words:
        return "  (no text)"
    # Right-to-left script with left-to-right indices: print one per line so
    # the pairing is unambiguous. Terminal bidi rendering of inline numbers
    # is not reliable enough to trust for data entry.
    return "\n".join(f"    {i:>2}. {w}" for i, w in enumerate(words, 1))


def _ask_verdict() -> str:
    while True:
        raw = input("  verdict [c]lean / [e]rror / [u]nusable / [s]kip: ").strip().lower()
        if raw in VERDICTS:
            return VERDICTS[raw]
        if raw in VERDICTS.values():
            return raw
        print("  -> enter c, e, u or s")


def _ask_words(n_words: int) -> list[int]:
    while True:
        raw = input(f"  which words are wrong? (numbers 1-{n_words}, comma-separated): ")
        parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
        try:
            idx = sorted({int(p) for p in parts})
        except ValueError:
            print("  -> numbers only, e.g. '2,5'")
            continue
        if not idx:
            print("  -> an 'error' verdict needs at least one word; use 'c' if it is clean")
            continue
        bad = [i for i in idx if not 1 <= i <= n_words]
        if bad:
            print(f"  -> out of range: {bad}")
            continue
        return idx


def _ask_tags() -> list[str]:
    print("  tags: " + ", ".join(TAGS))
    while True:
        raw = input("  tags (comma-separated, blank = sub): ").strip().lower()
        if not raw:
            return ["sub"]
        tags = [t.strip() for t in raw.replace(" ", ",").split(",") if t.strip()]
        unknown = [t for t in tags if t not in TAGS]
        if unknown:
            print(f"  -> unknown tag(s): {unknown}")
            continue
        return tags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "run" / "msa_pilot_sample.json"))
    parser.add_argument("--output", default=str(REPO_ROOT / "run" / "msa_pilot_annotated.json"))
    parser.add_argument("--audio-dir", default=str(REPO_ROOT / "run" / "pilot_audio"))
    parser.add_argument("--play", action="store_true", help="Auto-play each utterance.")
    parser.add_argument("--review", action="store_true", help="Print own verdicts and exit.")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out_path = Path(args.output)
    done: dict[str, dict] = {}
    if out_path.exists():
        done = {r["id"]: r for r in json.loads(out_path.read_text(encoding="utf-8"))}

    order = list(manifest)
    random.Random(SHUFFLE_SEED).shuffle(order)

    if args.review:
        if not done:
            raise SystemExit(f"Nothing annotated yet at {out_path}.")
        print(f"{len(done)}/{len(manifest)} annotated\n")
        for row in order:
            rec = done.get(row["id"])
            if rec:
                words = rec.get("error_words") or []
                tags = ",".join(rec.get("tags") or []) or "-"
                print(f"  {row['id']:>8s}  {rec['verdict']:<9s} words={words} tags={tags}")
        return

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        print(f"WARNING: no audio directory at {audio_dir}.")
        print("The wavs live on the cluster (run/pilot_audio/). Copy them over first,")
        print("or pass --audio-dir. Continuing: paths will be printed but not playable.\n")

    remaining = [r for r in order if r["id"] not in done]
    print("=" * 72)
    print("PILOT ANNOTATION  --  docs/annotation-protocol-pilot.md")
    print("=" * 72)
    print(f"  {len(done)} done, {len(remaining)} to go, {len(manifest)} total")
    print("  Model prediction, disagreement score and stratum are HIDDEN by design.")
    print("  A dropped utterance-final case ending is NOT an error (pausal convention).")
    print("  Ctrl-C to stop; progress is saved after every utterance.\n")

    try:
        for n, row in enumerate(remaining, 1):
            words = (row.get("sentence") or "").split()
            wav = audio_dir / f"{row['id']}.wav"
            print("-" * 72)
            print(f"[{n}/{len(remaining)}]  id {row['id']}")
            print(f"  audio: {wav}")
            print("\n  undiacritized:")
            print(f"    {row.get('sentence')}")
            print("\n  reference (diacritized), word by word:")
            print(_numbered_words(row.get("tashkeel_sentence") or row.get("sentence") or ""))
            print()
            if args.play and wav.exists():
                _play(wav)

            verdict = _ask_verdict()
            if verdict == "skip":
                print("  skipped -- will reappear next run\n")
                continue

            error_words: list[int] = []
            tags: list[str] = []
            if verdict == "error":
                error_words = _ask_words(len(words))
                tags = _ask_tags()
            note = input("  note (optional, required for 'unusable'): ").strip()
            while verdict == "unusable" and not note:
                note = input("  note REQUIRED for unusable: ").strip()

            done[row["id"]] = {
                "id": row["id"],
                "verdict": verdict,
                "error_words": error_words,
                "tags": tags,
                "note": note,
                "n_words": len(words),
                "n_canonical_phonemes": row.get("n_canonical_phonemes"),
            }
            out_path.write_text(
                json.dumps(list(done.values()), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  saved ({len(done)}/{len(manifest)})\n")
    except KeyboardInterrupt:
        print("\n\ninterrupted -- progress saved.")

    print(f"\n{len(done)}/{len(manifest)} annotated -> {out_path}")
    if len(done) == len(manifest):
        print("\nComplete. Now run:")
        print("  uv run python scripts/pilot_baserate.py")
    else:
        print("Re-run the same command to continue where you left off.")


if __name__ == "__main__":
    main()
