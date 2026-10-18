"""The MSA block's annotation tool — writes `docs/annotation-protocol-msa.md` §6.

    uv run python scripts/annotate_msa.py --annotator a1
    uv run python scripts/annotate_msa.py --annotator a1 --status
    uv run python scripts/annotate_msa.py --annotator a1 --validate

The protocol's own "Open, and deliberately not decided here" list opens with
*"The annotation tool. `scripts/pilot_annotate.py` is a listening interface and
cannot be reused. The schema above is fixed; the thing that writes it is not
built. This is the next piece of code the block needs."* This is that thing, and
it is the only piece of machinery between a pre-registered protocol and 500
annotated utterances.

Nothing here is a design decision. The schema is §6, the role assignment is §4,
the conventions are §3 and §7, and the checks are §6's four. All of that lives
in `arabic_mdd.data.annotation`, with tests, because §6 requires the checks to
be "automated at ingest" and ingest happens again in weeks 7–9 when the label
path consumes `C_gold`. This file is the interface and the clock.

## Why the text is typed in an editor and not at a prompt

Two constraints collide. §6 needs **real** per-utterance timings — "D4's
extension decision is explicitly deferred to real timings, and self-reported
durations are not that" — and the task is typing a fully diacritized Arabic
sentence, which at a bare `input()` prompt means no cursor movement, no
correction, and no usable IME behaviour. The qualification test was done by
editing a text file for exactly that reason, and that workflow is the one the
annotator has already passed a test under.

So: the tool opens `$EDITOR` on a small per-row buffer, pre-filled with the
draft (anchored) or the bare sentence (blank), and takes `started_at` before
launching it and `finished_at` when it returns. The timing is wall-clock around
a real editing session, which is the quantity D4 needs, and the typing happens
somewhere a person can actually type. `--inline` falls back to a prompt for
environments with no editor; it is honest about being the worse path.

## The three things this tool must not do

Named in `docs/weeks/week-06.md` task 4 and each one enforced here rather than
trusted:

1. **It must not reorder the block.** The draw is committed (seed 20261001) and
   §4 fixes crossover-first. `assignments()` returns that order; this file
   consumes it front to back and has no skip, no jump and no shuffle — a
   deliberate difference from `pilot_annotate.py`, where deferral existed
   because a *listening* judgment can genuinely stall. Here the escape hatch is
   §5's `unusable`, which is narrow and recorded.
2. **It must not show `catt-eo`'s suggestion on the blank-page ranks.** The
   drafts are loaded once, and on a blank row the renderer is handed `None` —
   `_render` cannot reach the drafts dict at all. §4: "Blank page means blank."
3. **It must not accept a submission that alters the consonantal skeleton
   without flagging it.** A blocking check loops back into the editor with the
   diff shown. §3: "flagged and re-done, not silently scored."

## What it does *not* do, on purpose

No adjudication UI. §5 gives the user — not the annotator — flagged positions
to adjudicate, and §6 keeps `final_before` so the change is auditable; the
record carries `adjudicated: null` and a later, separate pass fills it. Writing
both halves now would mean guessing an adjudication workflow before a single
flag exists, and the flags are what will tell us what it should look like.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arabic_mdd.data.annotation import (
    ANNOTATORS,
    DONE,
    MALFORMED_UPSTREAM,
    UNUSABLE,
    Assignment,
    assignments,
    build_record,
    check_submission,
    elapsed_seconds,
    validate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BLOCK = REPO_ROOT / "run" / "msa_annotation_block.json"
DRAFTS = REPO_ROOT / "run" / "msa_annotation_drafts.json"

#: §7: two block rows carry a shadda on a long-vowel carrier, an orthographic
#: impossibility present in the source. Diacritize as written, leave the
#: cluster, set `note`. Do not repair the skeleton.
MALFORMED_RANKS = (447, 475)

RULES = """
  --------------------------------------------------------------------------
  THE JOB (§1).  Produce C_gold: the CANONICAL diacritization of the WRITTEN
  sentence. What the prompt text calls for -- not what any speaker did.
  There is no audio in this task.

  THE FOUR CONVENTIONS (§3).  C_gold is PAUSAL.
    R1  Write the full, correct case ending (i'rab) on EVERY word except the
        last.                                              <- SCORED
    R2  The LAST word's final mark is free. Write it or not.
    R3  Tanwin is free, anywhere. Write it or not.
    R4  NEVER write a liaison vowel.                       <- SCORED
          min + AL-maa'  ->  keep the sukun:  مِنْ الْمَاءِ
          NOT mina:                           مِنَ الْمَاءِ
        31 positions in the 500 rows. This tool flags them for you.

  THE LETTERS MUST NOT CHANGE (§3). Marks only. Checked automatically.

  LEAVE ALONE (§7): Latin script, digits, punctuation, Qur'anic pause marks.
  COMPLETE AND CORRECT (§7): partially vowelized text. Existing marks have no
  authority.

  THE DRAFT HAS NO AUTHORITY (§2). catt-eo disagrees with the organizers'
  vowelizer on 4.85% of words and is wrong somewhere in there. Typing aid only.
  --------------------------------------------------------------------------
"""

EDITOR_HEADER = """# {header}
#
# Edit the line below this block. Marks only -- the LETTERS must not change.
# Lines starting with '#' are ignored. Save and quit when done.
#
# R1 case ending on every word but the last   R2/R3 last mark + tanwin free
# R4 never write a liaison vowel (keep the sukun before AL-)
{extra}#
# the sentence as written (your letters must match this):
# {sentence}
{body}
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _edit(initial: str, header: str, sentence: str, extra: str = "") -> str | None:
    """Open `$EDITOR` on a buffer and return the edited text, or None if blanked.

    The buffer carries the conventions as comments, because the one moment a
    rule is worth having on screen is while the sentence is being typed.
    """
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if not editor:
        for candidate in ("nano", "vim", "vi", "micro"):
            if shutil.which(candidate):
                editor = candidate
                break
    if not editor:
        return _inline(initial, sentence)

    content = EDITOR_HEADER.format(header=header, sentence=sentence, extra=extra, body=initial)
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", encoding="utf-8", delete=False) as handle:
        handle.write(content)
        path = Path(handle.name)
    try:
        subprocess.run([*editor.split(), str(path)], check=False)
        lines = [
            line.rstrip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("#")
        ]
    finally:
        path.unlink(missing_ok=True)
    text = " ".join(line for line in lines if line.strip()).strip()
    return text or None


def _inline(initial: str, sentence: str) -> str | None:
    """Prompt fallback. Worse, and says so."""
    print("  (no editor found — typing inline; cursor editing will be limited)")
    print(f"  as written: {sentence}")
    if initial:
        print(f"  draft     : {initial}")
    text = input("  C_gold: ").strip()
    return text or None


def _render(assignment: Assignment, suggested: str | None, position: str) -> None:
    """Print one row. `suggested` is None on every blank-condition row.

    The drafts mapping is deliberately not a parameter: the only way this
    function can print a draft is if the caller already decided it was allowed
    to, and the caller's decision is one line, right next to §4's rule.
    """
    print("-" * 74)
    kind = "CROSSOVER" if assignment.crossover else "block"
    where = f"[{position}]  rank {assignment.rank}  id {assignment.id}"
    print(f"{where}  {kind}  {assignment.condition.upper()}")
    print()
    print("  SENTENCE AS WRITTEN:")
    print(f"    {assignment.sentence}")
    if assignment.condition == "blank":
        print()
        print("  BLANK PAGE (§4): no draft, no vowelizer output, no diacritization")
        print("  tool, no LLM, no looking up the source. Your own reading only.")
    elif suggested:
        print()
        print("  catt-eo DRAFT (a typing aid with no authority, §2):")
        print(f"    {suggested}")
    if assignment.rank in MALFORMED_RANKS:
        print()
        print("  §7 MALFORMED UPSTREAM: this row carries a shadda on a long-vowel")
        print("  carrier, which is impossible orthography present in the SOURCE.")
        print("  Diacritize as written, leave the cluster, do NOT repair the letters.")
        print(f"  `note` will be set to {MALFORMED_UPSTREAM!r}.")
    print()


def _ask_indices(prompt: str, n_words: int) -> list[int]:
    """Word indices, 1-based for the human and stored 1-based, as §6 implies.

    The protocol says "[word indices]" without fixing the base; the qualification
    packet and `pilot_annotate.py` both numbered words from 1 for the annotator,
    so 1 is what the annotator sees and 1 is what is stored. Stated here because
    an off-by-one in a flag list is the kind of thing that is discovered during
    adjudication and cannot be repaired afterwards.
    """
    while True:
        raw = input(f"  {prompt} (1-{n_words}, comma-separated, blank = none): ").strip()
        if not raw:
            return []
        parts = [p for p in raw.replace(" ", ",").split(",") if p]
        try:
            idx = sorted({int(p) for p in parts})
        except ValueError:
            print("  -> numbers only, e.g. '2,5'")
            continue
        bad = [i for i in idx if not 1 <= i <= n_words]
        if bad:
            print(f"  -> out of range: {bad}")
            continue
        return idx


def _ask_ambiguous(n_words: int) -> list[dict[str, Any]]:
    """§5.3: the sentence genuinely admits two readings and both are defensible."""
    out: list[dict[str, Any]] = []
    idx = _ask_indices("ambiguous words — two defensible readings (§5.3)", n_words)
    for word in idx:
        alt = input(f"    alternative reading for word {word}: ").strip()
        if alt:
            out.append({"word": word, "alternative": alt})
    return out


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _report(report: Any, assigned: list[Assignment]) -> None:
    n_crossover = sum(1 for a in assigned if a.crossover)
    print(f"  records     : {report.n_records} of {report.n_expected} assigned")
    print(f"  done        : {report.done}   unusable: {report.unusable}")
    print(f"  crossover   : {report.crossover_done} done of {n_crossover} (§4 wants ≥45 usable)")
    if report.crossover_extension_required:
        print("  §4 EXTENSION REQUIRED: fewer than 45 of the 50 crossover rows are")
        print("  usable. Extend with the next rows in draw order and RECORD that it")
        print("  was done — the rule was pre-registered, so applying it is not a")
        print("  judgment call, but hiding that it fired would be.")
    median = report.median_seconds
    if median is not None:
        total = sum(report.seconds) / 3600.0
        print(f"  timings     : median {median:.0f}s/row, {total:.2f}h so far (D4's input)")
    print(f"  flagged     : {report.flagged_positions} positions (§5.1, the user adjudicates)")
    print(f"  ambiguous   : {report.ambiguous_positions} positions (§5.3)")
    print(f"  R4 reported : {report.n_r4_violations} in {len(report.r4_violation_rows)} rows")
    for label, rows in (
        ("BLOCKING (skeleton/word count)", report.blocking_rows),
        ("written twice", [f"{i}/{c}" for i, c in report.duplicates]),
        ("WRONG CONDITION — crossover blinding", report.wrong_condition),
        ("BLINDING BREACH — draft present on a blank row", report.blinding_breach),
        ("not assigned to this annotator", report.unknown_rows),
    ):
        if rows:
            print(f"  {label}: {rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotator", choices=ANNOTATORS, required=True)
    parser.add_argument("--block", type=Path, default=BLOCK)
    parser.add_argument("--drafts", type=Path, default=DRAFTS)
    parser.add_argument("--output", type=Path, default=None, help="default run/msa_c_gold_<a>.json")
    parser.add_argument("--status", action="store_true", help="progress and checks, then exit")
    parser.add_argument("--validate", action="store_true", help="re-run §6's checks, then exit")
    parser.add_argument("--limit", type=int, default=0, help="annotate at most N rows this session")
    args = parser.parse_args()

    frame = json.loads(args.block.read_text(encoding="utf-8"))
    assigned = assignments(frame["block"], args.annotator)
    out_path = args.output or REPO_ROOT / "run" / f"msa_c_gold_{args.annotator}.json"
    records = _load_records(out_path)

    drafts_meta: dict[str, Any] = {}
    drafts: dict[str, str] = {}
    if args.drafts.exists():
        drafts_meta = json.loads(args.drafts.read_text(encoding="utf-8"))
        drafts = drafts_meta.get("drafts", {})
    skeleton_warn = {d["id"] for d in drafts_meta.get("skeleton_differs_from_frame", [])}

    # The blank arm's R4 sites come from the frame's own `tashkeel_sentence`:
    # the site set is a property of the sentence, not of the draft, and the
    # blank annotator never sees either. Only the site definition touches it;
    # nothing from this mapping is ever printed.
    tashkeel = {str(r["id"]): r.get("tashkeel_sentence") or "" for r in frame["block"]}

    if args.status or args.validate:
        print(f"=== {args.annotator} — {out_path.name} ===")
        _report(validate(records, assigned, args.annotator, references=tashkeel), assigned)
        return

    if not drafts:
        raise SystemExit(
            f"{args.drafts} not found or empty. The anchored condition cannot run "
            "without the committed drafts:\n"
            "  uv run python scripts/prepare_annotation_drafts.py --write"
        )

    done_keys = {(str(r["id"]), str(r["condition"])) for r in records}
    pending = [a for a in assigned if (a.id, a.condition) not in done_keys]

    print("=" * 74)
    print(f"MSA BLOCK ANNOTATION  —  docs/annotation-protocol-msa.md  —  {args.annotator}")
    print("=" * 74)
    print(f"  {len(records)} done, {len(pending)} to go, {len(assigned)} assigned")
    print(f"  tool: {drafts_meta.get('tool')} {drafts_meta.get('tool_version')}")
    print("  Order is the committed draw (seed 20261001), crossover first (§4).")
    print("  There is no skip: §5's `unusable` is the only escape hatch and it is")
    print("  narrow — empty text, no Arabic, or text too corrupt to diacritize.")
    print("  Ctrl-C to stop; every row is saved as it is accepted.")
    print(RULES)

    annotated = 0
    try:
        for assignment in pending:
            if args.limit and annotated >= args.limit:
                print(f"  --limit {args.limit} reached.")
                break
            suggested = drafts.get(assignment.id) if assignment.anchored else None
            reference = suggested or tashkeel.get(assignment.id) or ""
            position = f"{len(records) + 1}/{len(assigned)}"
            _render(assignment, suggested, position)

            if assignment.anchored and assignment.id in skeleton_warn:
                print("  NOTE: the draft for this row DROPPED non-Arabic text that §7 says")
                print("  to leave untouched. Put it back, or the skeleton check will fire.")
                print()

            extra = ""
            if assignment.rank in MALFORMED_RANKS:
                extra = "# §7: malformed cluster in the SOURCE — leave it, do not repair.\n"

            started_at = _now()
            initial = suggested or ""
            final: str | None = None
            status = DONE
            unusable_reason: str | None = None
            while True:
                final = _edit(
                    initial,
                    header=f"rank {assignment.rank}  id {assignment.id}  "
                    f"{assignment.condition}  ({args.annotator})",
                    sentence=assignment.sentence,
                    extra=extra,
                )
                if final is None:
                    print("  empty submission.")
                    choice = input("  [r]etry / mark [u]nusable (§5) / [q]uit session: ").strip()
                    if choice.startswith("u"):
                        reason = ""
                        while not reason:
                            reason = input("  reason REQUIRED (§5): ").strip()
                        status, unusable_reason, final = UNUSABLE, reason, ""
                        break
                    if choice.startswith("q"):
                        raise KeyboardInterrupt
                    continue

                checks = check_submission(assignment.sentence, final, reference=reference)
                for problem in checks.problems:
                    print(f"  ! {problem}")
                if not checks.blocking:
                    if checks.r4_sites:
                        words = [i + 1 for i in checks.r4_sites]
                        print(f"  R4 sites in this sentence: words {words}")
                    break
                print("  §3: the row is re-done, not scored as-is.")
                choice = input("  [r]etry / mark [u]nusable (§5): ").strip()
                if choice.startswith("u"):
                    reason = ""
                    while not reason:
                        reason = input("  reason REQUIRED (§5): ").strip()
                    status, unusable_reason = UNUSABLE, reason
                    break
                initial = final

            finished_at = _now()

            flagged: list[int] = []
            ambiguous: list[dict[str, Any]] = []
            note = ""
            if status == DONE:
                n_words = len((final or "").split())
                flagged = _ask_indices(
                    "flag uncertain CASE ENDINGS — still write your best guess (§5.1)", n_words
                )
                ambiguous = _ask_ambiguous(n_words)
                note = input("  note (optional): ").strip()
            if assignment.rank in MALFORMED_RANKS and not note:
                note = MALFORMED_UPSTREAM

            record = build_record(
                assignment,
                annotator=args.annotator,
                final=final or "",
                suggested=suggested,
                flagged_words=flagged,
                ambiguous=ambiguous,
                status=status,
                unusable_reason=unusable_reason,
                note=note,
                started_at=started_at,
                finished_at=finished_at,
                tool_version=drafts_meta.get("tool_version"),
                protocol_version=drafts_meta.get("protocol_version"),
            )
            records.append(record)
            _save(out_path, records)
            annotated += 1
            secs = elapsed_seconds(record)
            took = f" in {secs:.0f}s" if secs is not None else ""
            print(f"  saved {status}{took}  ({len(records)}/{len(assigned)})\n")
    except (KeyboardInterrupt, EOFError):
        print("\n\ninterrupted — every accepted row is already saved.")

    print(f"\n{len(records)}/{len(assigned)} -> {out_path}")
    _report(validate(records, assigned, args.annotator, references=tashkeel), assigned)


if __name__ == "__main__":
    main()
