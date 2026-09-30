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
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed so the presentation order is reproducible and stated in the paper.
SHUFFLE_SEED = 20260923

VERDICTS = {"c": "clean", "e": "error", "u": "unusable", "s": "skip", "j": "jump"}

# `skip` and `jump` both defer the current utterance rather than discarding
# it: it returns to the back of the queue within the SAME session. The
# earlier version recomputed the queue once at startup, so a skip only
# reappeared on the next run -- which in practice means the hard utterances
# quietly never get annotated. Non-random missingness in a 25-row stratum is
# not a rounding error; the hard ones are exactly where the errors live, so
# dropping them biases the base rate DOWN.

TAGS = {
    "sub": "substitution -- different consonant/vowel than written",
    "del": "deletion -- a written phoneme is absent",
    "ins": "insertion -- a phoneme not in the written word",
    "word": "wrong/skipped/added word",
    "dialect": "dialectal realization (th->t/s, q->g, j->zh, ...)",
    "case": "i'rab only -- wrong/dropped GRAMMATICAL ending, word itself intact",
    "hesit": "hesitation/restart, word ultimately correct",
    "unclear": "still unsure after three listens",
    "refbad": "the VOWELIZER's diacritics are wrong here, not the speaker",
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


def _numbered_words(plain: str, vowelized: str, *, show_vowelizer: bool = True) -> str:
    """Number the words the SPEAKER READ, i.e. the undiacritized sentence.

    The vowelized form is shown beside each word as a hint, never as the
    thing being numbered. Two reasons, both load-bearing:

    * `tashkeel_sentence` is an automatic vowelizer's output, not gold. Its
      errors must not become the annotator's error count.
    * The two strings do not always align: 3/50 utterances in this pilot
      have different word counts (the vowelizer drops punctuation-only
      tokens, and loses content outright in ~0.8% of the corpus). Numbering
      the vowelized list while validating and scoring against the plain one
      -- which is what an earlier version of this script did -- silently
      mismatches the indices on exactly those rows.

    With `show_vowelizer=False` the hints are withheld entirely: same word
    numbers, no machine output. That is stage 1 of the two-stage display
    (see `_ask_reveal`).
    """
    words = (plain or "").split()
    if not words:
        return "  (no text)"
    if not show_vowelizer:
        return "\n".join(f"    {i:>2}. {w}" for i, w in enumerate(words, 1))
    vow = (vowelized or "").split()
    aligned = len(vow) == len(words)
    lines = []
    for i, w in enumerate(words, 1):
        hint = f"   [{vow[i - 1]}]" if aligned else ""
        lines.append(f"    {i:>2}. {w}{hint}")
    if not aligned and vow:
        lines.append(f"\n    vowelizer (does NOT align word-for-word): {vowelized}")
    return "\n".join(lines)


# Shown on '?'. Deliberately short: the point is to make the three calls that
# actually matter decidable in a few seconds, not to restate the protocol.
HELP = """
  ---------------------------------------------------------------------
  Only ONE question matters:
      Is what you heard a LEGITIMATE READING of the written sentence?
      yes -> c        no -> e (then: which words, then a tag)
      Not 'does it match the brackets'. The brackets are a MACHINE's
      guess and are often wrong; that tool's error rate is this paper's
      subject, not the speaker's.
  ---------------------------------------------------------------------
  Getting the TAG right matters much less than you think. dialect, case,
  hesit and refbad are all EXCLUDED from the headline (lenient) rate and
  reported separately. A mis-tagged word moves between two numbers that
  both get published. It does not corrupt the result.
  ---------------------------------------------------------------------
  The three cases people stall on:

    word itself changed   kataba -> kutiba ("he wrote" -> "it was
                          written")                        ->  e + sub
    only the GRAMMATICAL  al-kitaabu -> al-kitaaba,
    ending, mid-sentence  word intact                      ->  e + case
    ending on the LAST    utterance ends al-kitaab                ->  c
    word of the utterance (pausal form -- correct by our convention)

  Regional accent (th->t/s, q->g, j->zh)                   ->  e + dialect
  Speaker right, brackets wrong                            ->  e + refbad
  Hesitates/restarts but gets there in the end             ->  e + hesit
  Still unsure after three listens                         ->  e + unclear
  ---------------------------------------------------------------------
  Out of scope entirely -- never an error: speed, intonation, stress,
  recording quality, anything you can still identify the word through.
  ---------------------------------------------------------------------
"""


def _ask_reveal(wav: Path | None = None) -> str:
    """Stage 1: undiacritized text only, until the annotator asks for more.

    Implements step 1 of the four-step procedure in
    `docs/annotation-protocol-pilot.md` §6a. The annotator reads the
    unvowelized sentence and settles on their OWN reading before any machine
    output is on screen; `d` then reveals the vowelizer's guess and the
    verdict options together.

    This is an anchoring control, and it is the same class of control as the
    blinding in §5 -- the model's prediction is hidden for exactly this
    reason. The vowelizer's output was never hidden because it has to be
    correctable (`refbad`), but showing it *first* means the annotator's
    reading is formed in its presence. Given that the pilot found the
    vowelizer wrong on 43.6% of usable utterances, anchoring to it is not a
    theoretical worry.

    Returns "display", or a deferral verdict if the annotator postpones
    before revealing.
    """
    prompt = "  [d]isplay vowelizer + verdict options / [r]eplay / [s]kip / [j]ump / [?]help: "
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("d", "display", ""):
            return "display"
        if raw in ("r", "replay"):
            if wav and wav.exists():
                _play(wav)
            else:
                print("  -> no playable audio for this utterance")
            continue
        if raw in ("?", "h", "help"):
            print(HELP)
            continue
        if raw in ("s", "skip", "j", "jump"):
            return VERDICTS[raw[0]]
        if raw in ("c", "e", "u", "clean", "error", "unusable"):
            print("  -> press 'd' first: decide your own reading before seeing the machine's")
            continue
        print("  -> enter d, r, s, j or ?")


def _ask_verdict(wav: Path | None = None) -> str:
    while True:
        prompt = (
            "  verdict [c]lean / [e]rror / [u]nusable / [r]eplay / "
            "[s]kip / [j]ump-random / [?]help: "
        )
        raw = input(prompt).strip().lower()
        if raw in ("r", "replay"):
            if wav and wav.exists():
                _play(wav)
            else:
                print("  -> no playable audio for this utterance")
            continue
        if raw in ("?", "h", "help"):
            print(HELP)
            continue
        if raw in VERDICTS:
            return VERDICTS[raw]
        if raw in VERDICTS.values():
            return raw
        print("  -> enter c, e, u, r, s, j or ?")


def _parse_words(raw: str, n_words: int) -> list[int] | None:
    parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    try:
        idx = sorted({int(p) for p in parts})
    except ValueError:
        print("  -> numbers only, e.g. '2,5'")
        return None
    bad = [i for i in idx if not 1 <= i <= n_words]
    if bad:
        print(f"  -> out of range: {bad}")
        return None
    return idx


def _parse_tags(raw: str) -> list[str] | None:
    if not raw:
        return ["sub"]
    tags = [t.strip() for t in raw.replace(" ", ",").split(",") if t.strip()]
    unknown = [t for t in tags if t not in TAGS]
    if unknown:
        print(f"  -> unknown tag(s): {unknown}  (valid: {', '.join(TAGS)})")
        return None
    return tags


def _group_by_tags(word_tags: dict[int, set[str]]) -> list[dict]:
    """Collapse a per-word tag map into (words, tags) groups for storage."""
    buckets: dict[tuple[str, ...], list[int]] = {}
    for word in sorted(word_tags):
        key = tuple(sorted(word_tags[word]))
        buckets.setdefault(key, []).append(word)
    return [{"words": w, "tags": list(k)} for k, w in buckets.items()]


def _ask_errors(n_words: int) -> list[dict]:
    """Collect a tag set PER WORD, then emit (words, tags) groups.

    Two constraints have to hold at once, and an earlier design satisfied
    only the first.

    **Tags cannot live on the utterance.** Read MSA routinely drops
    mid-sentence i'rab, so one utterance can carry five `case` words beside a
    single genuine `sub`. Asked "are ALL this utterance's tags excludable?"
    the answer is no, and all six words land in the lenient numerator -- a 6x
    overcount that biases toward the >=3% band, i.e. toward the result that
    suits the paper.

    **One word can carry two independent facts.** The speaker mispronounces
    word 4 AND the automatic vowelizer got word 4 wrong: that is `sub` plus
    `refbad`, and both must be recorded. The previous version made groups
    primary and rejected a word that was already assigned, so the annotator
    could only express this by thinking of both tags simultaneously -- and
    hit a dead end otherwise, mid-utterance, with no way back.

    Keying on the word fixes both: re-entering a word UNIONS the new tags
    into it. Groups are derived at the end purely as a storage format, so
    every word appears exactly once and the scorer's per-group exclusion
    stays exact.
    """
    print(f"  tags: {', '.join(TAGS)}")
    print("  Enter words, then their tags. Re-enter a word to ADD tags to it.")
    print("  Blank line when done.")
    word_tags: dict[int, set[str]] = {}
    while True:
        label = "words that are wrong" if not word_tags else "more words (blank = done)"
        raw = input(f"  {label} -- numbers 1-{n_words}: ").strip().lower()
        if raw in ("?", "h", "help"):
            print(HELP)
            for name, desc in TAGS.items():
                print(f"    {name:<8s} {desc}")
            continue
        if not raw:
            if word_tags:
                return _group_by_tags(word_tags)
            print("  -> an 'error' verdict needs at least one word; use 'c' if it is clean")
            continue
        idx = _parse_words(raw, n_words)
        if idx is None:
            continue
        seen = sorted(w for w in idx if w in word_tags)
        if seen:
            have = {w: ",".join(sorted(word_tags[w])) for w in seen}
            print(f"    (already tagged: {have} -- new tags will be added to them)")
        while True:
            traw = input(f"  tags for word(s) {idx} (blank = sub, ? = help): ").strip().lower()
            if traw in ("?", "h", "help"):
                print(HELP)
                for name, desc in TAGS.items():
                    print(f"    {name:<8s} {desc}")
                continue
            tags = _parse_tags(traw)
            if tags is None:
                continue
            break
        for word in idx:
            word_tags.setdefault(word, set()).update(tags)
        print("    -> " + "  ".join(f"{w}={','.join(sorted(word_tags[w]))}" for w in sorted(idx)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "run" / "msa_pilot_sample.json"))
    parser.add_argument("--output", default=str(REPO_ROOT / "run" / "msa_pilot_annotated.json"))
    parser.add_argument("--audio-dir", default=str(REPO_ROOT / "run" / "pilot_audio"))
    parser.add_argument("--play", action="store_true", help="Auto-play each utterance.")
    parser.add_argument("--review", action="store_true", help="Print own verdicts and exit.")
    parser.add_argument(
        "--allow-no-audio",
        action="store_true",
        help="Run without wavs. For interface testing ONLY -- the verdicts are not valid data.",
    )
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
            if not rec:
                continue
            groups = rec.get("errors")
            if groups is None:  # pre-grouping record
                groups = (
                    [{"words": rec.get("error_words") or [], "tags": rec.get("tags") or []}]
                    if rec.get("error_words")
                    else []
                )
            detail = "  ".join(f"{g['words']}={','.join(g['tags'])}" for g in groups) or "-"
            print(f"  {row['id']:>8s}  {rec['verdict']:<9s} {detail}")
        return

    audio_dir = Path(args.audio_dir)
    n_wavs = len(list(audio_dir.glob("*.wav"))) if audio_dir.is_dir() else 0
    if n_wavs == 0 and not args.allow_no_audio:
        raise SystemExit(
            f"No .wav files found in {audio_dir}.\n\n"
            "This is a LISTENING pilot: the verdict is about what the speaker\n"
            "actually produced, which cannot be judged from the text. Annotating\n"
            "without audio would measure your reading of the orthography, not\n"
            "pronunciation, and the base rate would be meaningless.\n\n"
            "The wavs are on the cluster, written by run/cv_dev_baserate.slurm\n"
            "Stage B. Copy them over FROM YOUR LOCAL MACHINE (not from a shell\n"
            "on the cluster -- rsync pulls, so it runs on the destination):\n\n"
            "  rsync -avP dziri@amazonit:diacritization-error-propagation/run/pilot_audio/ \\\n"
            "      run/pilot_audio/\n\n"
            "Trailing slashes matter: they copy the directory's CONTENTS.\n"
            "Verify on the cluster first with:  ls run/pilot_audio/*.wav | wc -l\n\n"
            "(--allow-no-audio exists only for testing the interface.)"
        )
    if n_wavs and n_wavs < len(manifest):
        print(f"WARNING: only {n_wavs} wavs for {len(manifest)} utterances.")
        print("Missing ones will show an unplayable path -- mark those 'skip'.\n")

    pending = [r for r in order if r["id"] not in done]
    jump_rng = random.Random(SHUFFLE_SEED + 1)
    deferrals: Counter[str] = Counter()

    print("=" * 72)
    print("PILOT ANNOTATION  --  docs/annotation-protocol-pilot.md")
    print("=" * 72)
    print(f"  {len(done)} done, {len(pending)} to go, {len(manifest)} total")
    print("  Model prediction, disagreement score and stratum are HIDDEN by design.")
    print(HELP)
    print("  Two stages per utterance: the undiacritized sentence first, so you")
    print("  settle on YOUR reading; 'd' then reveals the vowelizer's guess and")
    print("  the verdict options. Protocol section 6a, step 1.")
    print("  'r' replays. '?' reprints the above at any prompt.")
    print("  's' defers this one to the back of the queue; 'j' defers it AND")
    print("  jumps to a random other one -- both come back before the session")
    print("  ends, so a hard utterance is postponed, never dropped.")
    print("  Ctrl-C to stop -- progress saves after every utterance.\n")

    try:
        consecutive_defers = 0
        while pending:
            row = pending.pop(0)
            words = (row.get("sentence") or "").split()
            wav = audio_dir / f"{row['id']}.wav"
            print("-" * 72)
            n_def = deferrals[row["id"]]
            seen = f"  (returning, deferred {n_def}x)" if n_def else ""
            print(f"[{len(done) + 1}/{len(manifest)}]  id {row['id']}{seen}")
            print(f"  audio: {wav}")
            print("\n  WHAT THE SPEAKER READ (this is the reference):")
            print(f"    {row.get('sentence')}")
            print("\n  word by word:")
            print(
                _numbered_words(
                    row.get("sentence") or "",
                    row.get("tashkeel_sentence") or "",
                    show_vowelizer=False,
                )
            )
            print("\n  Decide YOUR OWN reading of this first. Then 'd'.")
            print()
            if args.play and wav.exists():
                _play(wav)

            verdict = _ask_reveal(wav)
            if verdict == "display":
                print("\n  automatic vowelizer's guess -- MAY BE WRONG, tag 'refbad':")
                print(
                    _numbered_words(
                        row.get("sentence") or "",
                        row.get("tashkeel_sentence") or "",
                    )
                )
                print()
                verdict = _ask_verdict(wav)
            if verdict in ("skip", "jump"):
                deferrals[row["id"]] += 1
                pending.append(row)
                consecutive_defers += 1
                if verdict == "jump" and len(pending) > 1:
                    # Deferred row is at the back; draw the next from the rest
                    # so 'j' cannot hand back the utterance just postponed.
                    i = jump_rng.randrange(len(pending) - 1)
                    pending.insert(0, pending.pop(i))
                    print(f"  deferred {row['id']} -- jumping to a random other one\n")
                else:
                    print(f"  deferred {row['id']} -- it returns later this session\n")
                if consecutive_defers >= len(pending) > 0:
                    print(f"  NOTE: all {len(pending)} remaining have been deferred at least")
                    print("  once without a verdict. Ctrl-C and come back rested rather")
                    print("  than cycling -- and if one is genuinely undecidable, 'e' +")
                    print("  tag 'unclear' is a valid answer that stays visible.\n")
                continue
            consecutive_defers = 0

            errors: list[dict] = []
            if verdict == "error":
                errors = _ask_errors(len(words))
            note = input("  note (optional, required for 'unusable'): ").strip()
            while verdict == "unusable" and not note:
                note = input("  note REQUIRED for unusable: ").strip()

            # `errors` is the authoritative per-word record. `error_words` and
            # `tags` are flattened conveniences for review/tag-counting only --
            # never compute the lenient rate from them (see _ask_errors).
            error_words = sorted({i for g in errors for i in g["words"]})
            tags = sorted({t for g in errors for t in g["tags"]})
            done[row["id"]] = {
                "id": row["id"],
                "verdict": verdict,
                "errors": errors,
                "error_words": error_words,
                "tags": tags,
                "note": note,
                "n_words": len(words),
                "n_canonical_phonemes": row.get("n_canonical_phonemes"),
                # How often this one was postponed before a verdict landed.
                # A cheap difficulty proxy: if the utterances that took three
                # passes are also the ones carrying errors, that is worth
                # saying in the write-up rather than discovering in review.
                "n_deferrals": deferrals[row["id"]],
            }
            out_path.write_text(
                json.dumps(list(done.values()), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  saved ({len(done)}/{len(manifest)})\n")
    except (KeyboardInterrupt, EOFError):
        print("\n\ninterrupted -- progress saved.")

    print(f"\n{len(done)}/{len(manifest)} annotated -> {out_path}")
    left_deferred = [i for i in deferrals if i not in done]
    if left_deferred:
        print(f"\n{len(left_deferred)} deferred without a verdict: {left_deferred}")
        print("These are the ones you found hardest, which makes them the ones")
        print("most likely to contain errors. Leaving them out would bias the")
        print("base rate DOWN. Re-run and finish them, or give them 'e' +")
        print("'unclear' so the ambiguity is recorded instead of missing.")
    if len(done) == len(manifest):
        print("\nComplete. Now run:")
        print("  uv run python scripts/pilot_baserate.py")
    else:
        print("Re-run the same command to continue where you left off.")


if __name__ == "__main__":
    main()
