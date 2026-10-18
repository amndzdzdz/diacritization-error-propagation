"""Build the `catt-eo` drafts the anchored condition shows — protocol §2.

    uv run python scripts/prepare_annotation_drafts.py --write
        -> run/msa_annotation_drafts.json

One draft per row of `run/msa_annotation_block.json`, produced once and
committed, rather than generated inside the annotation tool. Three reasons, in
descending order of how much they would have cost to learn later:

1. **`tool_version` has to be a fact about the artifact.** §6 logs the
   `catt-tashkeel` version that produced `suggested`. Generating drafts inside
   the UI makes that a property of whichever machine the annotator sat at; a
   mid-block environment change would be invisible in the data, which is the
   exact failure `protocol_version` exists to prevent for the document.
2. **The anchored and blank arms must see the same utterances in the same
   state.** §8's Δ is a within-utterance contrast, so the draft for an
   utterance must be identical whenever it is shown. A frozen file guarantees
   that; a model call per session does not.
3. It is **3.2 seconds for all 500 rows** on CPU, so there is nothing to
   optimize. The draft is not the expensive part of this arm; the annotator is.

## What this script checks while it is here, and one real finding

§3's skeleton rule applies to the annotator, but a draft whose own skeleton
differs from the frame's hands them a starting point that **cannot be submitted
as-is** — the ingest check fires through no fault of theirs. So every draft is
checked and the offenders are named in the artifact, where the tool reads them
and warns at that row instead of letting the check fire mysteriously.

Measured: **1 of 500**. Rank 380 (id 64376), the Latin row — `catt-eo` deletes
`Eddie`, both occurrences, exactly as the organizers' vowelizer does (§7 puts
that deletion rate at 100% and calls it "a real `C_auto` error and D5's
subject"). §7 tells the annotator to leave Latin untouched, so on this row they
must put `Eddie` back. That is a one-row instruction, and it is now carried in
the artifact rather than in someone's memory.

The word count is checked too (Arabic words only, so Latin deletions do not
register) and is **0 of 500**.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

from arabic_mdd.data.normalize import DIACRITICS, skeleton
from arabic_mdd.diacritizers import build
from arabic_mdd.metrics.der import arabic_words

REPO_ROOT = Path(__file__).resolve().parent.parent
BLOCK = REPO_ROOT / "run" / "msa_annotation_block.json"
OUT = REPO_ROOT / "run" / "msa_annotation_drafts.json"
PROTOCOL = REPO_ROOT / "docs" / "annotation-protocol-msa.md"

#: §2: "the draft shown in the anchored condition is CATT encoder-only,
#: `catt-tashkeel==1.0.2`, pin checked at runtime." Not the organizers'
#: `tashkeel_sentence` — that column is `C_auto`, the thing D5 measures
#: `C_gold` against, and seeding the reference with the object under test would
#: make D5's number mechanically small.
TOOL = "catt-eo"
PINNED = "1.0.2"


def stacked_marks(text: str) -> int:
    """Letters carrying two short vowels/sukun — impossible, so always a bug.

    This check exists because the splice above had this bug and nothing else
    would have found it: §6's four ingest checks all run on the skeleton or on
    word counts, and both are blind to diacritics by construction. A draft with
    `طَاغُونَُ` in it passes every check the protocol specifies.

    A shadda legitimately precedes a vowel, so only the mutually exclusive
    marks count: two of {fatha, damma, kasra, sukun, the three tanwin} in a row.
    """
    exclusive = set("\u064b\u064c\u064d\u064e\u064f\u0650\u0652")
    return sum(1 for a, b in zip(text, text[1:], strict=False) if a in exclusive and b in exclusive)


def _punctuation(text: str) -> list[str]:
    """Everything that is neither a letter, a space, nor a diacritic, in order."""
    return [c for c in text if not c.isalnum() and not c.isspace() and c not in DIACRITICS]


def restore_punctuation(source: str, draft: str) -> str:
    """Put back the characters `catt-eo` drops but §7 says to leave untouched.

    Measured: the model strips punctuation from **all 299** of the 500 rows that
    have any. Left alone that is not a scoring error — `skeleton()` ignores
    punctuation, so §6's ingest check never fires — but it is two real problems:

    1. §7 tells the annotator punctuation is left untouched. A draft with none
       makes that instruction unfollowable on 299 rows without retyping, which
       is busywork the protocol never asked for.
    2. The anchored arm would then produce punctuation-free `final` strings
       while the blank arm, typing from the source, keeps them. §8's Δ is a
       within-utterance contrast between those two, so the arms would differ
       systematically on something that is not diacritization.

    The splice is exact rather than heuristic because the draft is the source's
    letters with diacritics inserted — `skeleton(draft) == skeleton(source)` for
    499 of 500 rows. So a two-pointer merge over the two strings is
    well-defined: advance together on shared characters, emit the draft's
    diacritics as they come, and emit any *non-diacritic* source character the
    draft skipped. On the one row where the skeleton does differ (rank 380,
    Latin) this cannot align, so it is left exactly as the model produced it and
    stays on the annotator's warning list.

    **A source diacritic the draft skipped is dropped, not restored**, and that
    exclusion is the whole correctness argument. 71 of the 500 rows already
    carry marks (§7's 59 flagged `partially_vowelized`, plus 12 below the
    0.05 flag threshold), and on those rows the model frequently marks a letter
    differently from the source. Carrying the source's mark across as if it were
    a dropped character stacks it on top of the draft's own — `طَاغُونُ` becomes
    `طَاغُونَُ`, two short vowels on one letter, a string neither the source nor
    the model proposed. §6's checks would not catch it, because `skeleton()`
    strips diacritics, so it would have reached the annotator as a draft that
    silently cannot be right. Measured before the exclusion: **8 rows corrupted
    this way.** §7 settles which side wins where they disagree — the source's
    "existing marks have no authority" — so the draft's mark stands and the
    annotator corrects it, which is what they are there for.
    """
    if skeleton(draft) != skeleton(source):
        return draft

    out: list[str] = []
    i = j = 0
    while i < len(source) and j < len(draft):
        if source[i] == draft[j]:
            out.append(source[i])
            i += 1
            j += 1
        elif draft[j] in DIACRITICS:
            out.append(draft[j])
            j += 1
        elif source[i] in DIACRITICS:
            # The model marked this letter differently. §7: the source's marks
            # have no authority, and stacking them would produce a letter with
            # two vowels that no check downstream would reject.
            i += 1
        else:
            # In the source, absent from the draft: the dropped punctuation.
            out.append(source[i])
            i += 1
    out.append(draft[j:])
    out.append("".join(c for c in source[i:] if c not in DIACRITICS))
    return "".join(out)


def protocol_version() -> str | None:
    """The protocol document's git sha — §6's `protocol_version`.

    Blob sha of the file itself, not the repo HEAD: HEAD moves when anything in
    the repo changes, so it would not answer "which version of the protocol was
    this row annotated under", which is the only question the field exists for.
    """
    try:
        out = subprocess.run(
            ["git", "hash-object", str(PROTOCOL)],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=Path, default=BLOCK)
    parser.add_argument("--limit", type=int, default=0, help="first N rows only (smoke run)")
    parser.add_argument("--write", action="store_true", help=f"write {OUT.name}")
    args = parser.parse_args()

    installed = version("catt-tashkeel")
    if installed != PINNED:
        raise SystemExit(
            f"catt-tashkeel is {installed}, protocol §2 pins {PINNED}. "
            "Either the pin moved or the environment did; both need a decision, "
            "not a draft generated under the wrong tool."
        )

    frame = json.loads(args.block.read_text(encoding="utf-8"))
    rows = frame["block"][: args.limit] if args.limit else frame["block"]
    print(f"block       : {args.block} (seed {frame['seed']}, {len(frame['block'])} rows)")
    print(f"drafting    : {len(rows)} rows with {TOOL} {installed}")

    diacritizer = build(TOOL)
    started = time.time()
    raw = {str(row["id"]): diacritizer.diacritize(row["sentence"]) for row in rows}
    seconds = time.time() - started
    drafts = {
        str(row["id"]): restore_punctuation(row["sentence"], raw[str(row["id"])]) for row in rows
    }
    print(f"took        : {seconds:.1f}s ({seconds / max(len(rows), 1):.3f}s/row)")

    skeleton_differs = []
    word_count_differs = []
    punctuation_differs = []
    punctuation_restored = 0
    stacked = []
    for row in rows:
        draft = drafts[str(row["id"])]
        if stacked_marks(draft):
            stacked.append({"rank": row["rank"], "id": str(row["id"])})
        if skeleton(draft) != skeleton(row["sentence"]):
            skeleton_differs.append({"rank": row["rank"], "id": str(row["id"])})
        if len(arabic_words(draft)) != len(arabic_words(row["sentence"])):
            word_count_differs.append({"rank": row["rank"], "id": str(row["id"])})
        if _punctuation(draft) != _punctuation(row["sentence"]):
            punctuation_differs.append({"rank": row["rank"], "id": str(row["id"])})
        if draft != raw[str(row["id"])]:
            punctuation_restored += 1

    print(f"skeleton    : {len(skeleton_differs)} drafts differ from the frame's letters")
    for bad in skeleton_differs:
        print(f"              rank {bad['rank']} id {bad['id']} — annotator must restore them (§7)")
    print(f"word count  : {len(word_count_differs)} drafts differ (Arabic words)")
    print(f"punctuation : restored on {punctuation_restored} rows, {len(punctuation_differs)} left")
    for bad in punctuation_differs:
        print(f"              rank {bad['rank']} id {bad['id']} — could not splice (§7)")
    print(f"stacked     : {len(stacked)} drafts put two vowels on one letter")
    if stacked:
        raise SystemExit(
            f"{len(stacked)} drafts have a letter carrying two mutually exclusive "
            f"marks: {stacked[:5]}. No §6 check would reject this, so it is "
            "fatal here rather than a warning."
        )

    artifact = {
        "purpose": "the catt-eo drafts shown in the anchored condition — "
        "docs/annotation-protocol-msa.md §2",
        "generated_by": "scripts/prepare_annotation_drafts.py",
        "tool": TOOL,
        "tool_version": installed,
        "protocol_version": protocol_version(),
        "block": str(args.block.relative_to(REPO_ROOT)),
        "block_seed": frame["seed"],
        "n_rows": len(drafts),
        "seconds": round(seconds, 3),
        "skeleton_differs_from_frame": skeleton_differs,
        "word_count_differs_from_frame": word_count_differs,
        "punctuation_restored_rows": punctuation_restored,
        "punctuation_differs_from_frame": punctuation_differs,
        "stacked_marks_rows": len(stacked),
        "note": "a row listed in skeleton_differs_from_frame hands the annotator a "
        "starting point that fails §6's ingest check as-is; the tool warns on it.",
        "drafts": drafts,
    }

    if args.write:
        OUT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT}")
    else:
        print(f"(dry run — pass --write to produce {OUT})")


if __name__ == "__main__":
    main()
