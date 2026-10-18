"""The MSA block's annotation record: role assignment, the schema, ingest checks.

Implements `docs/annotation-protocol-msa.md` §§3–6. The protocol is
**pre-registered** and this module's job is to obey it, not to improve on it —
every constant below is a quotation, and the test module checks the quotations
rather than the behaviour where the two could drift.

Lives in `src/` rather than in the script that drives it for one reason: §6
requires the four ingest checks to be "automated at ingest, so the checks are
not trusted to anyone's discipline". Ingest happens again in weeks 7–9, when
`C_gold` is consumed by the label-path measurement, and a check implemented
inside an interactive tool is a check that only runs while someone is typing.

## Role assignment is a *committed draw*, not a sampling decision

`run/msa_annotation_block.json` is fixed: 500 rows, seed 20261001, `crossover:
true` on the first 50 in draw order. §4's table assigns conditions by rank:

| draw rank | annotator 1 | annotator 2 |
|---|---|---|
| 0–24 | anchored (sees the `catt-eo` draft) | blank page |
| 25–49 | blank page | anchored |

and §1 splits the 450 non-crossover rows by draw-order parity — annotator 1
takes even ranks, annotator 2 odd — so that neither gets a contiguous and
therefore possibly atypical stretch. The 450 are all **anchored**: §8's
pre-committed readings describe correct-the-machine as the design for the whole
block, with blank-page reserved for the crossover and for the >15 pp
re-annotation branch.

`assignments()` is therefore deterministic and order-bearing. §4 fixes the
order too — the 50 crossover rows come **first**, because doing them last would
measure anchoring in annotators who had already spent hours learning
`catt-eo`'s habits, "which is a different and smaller quantity than the one the
paper will claim".

## `suggested` is null in the blank condition, and that is not a loss

§6's schema line says `suggested` is "null in the blank condition"; its bullet
says the draft is "stored even in the blank condition's *sibling* rows". Both
are true and the word doing the work is *sibling*. Every crossover utterance is
annotated by both annotators, one anchored and one blank, so the blank record's
sibling — the other annotator's record for the same `id` — carries the draft,
and §8's distance metric can be computed for both arms from it. Nothing was
shown on a blank row, so nothing is recorded as shown on it. Writing the draft
into a blank row's record would also put the thing the arm is blind to inside
the blind arm's own data, which is the one file where it has no business being.

## The four ingest checks, and which of them may block a submission

§6: "skeleton unchanged (§3), word count matches the frame, R4 violations
counted and *reported rather than corrected*, and no row written twice in the
same condition by the same annotator."

Two of those are **blocking**. §3 says an utterance whose consonant skeleton
was altered "is flagged and re-done, not silently scored", so `skeleton_ok` and
`word_count_ok` gate acceptance. The other two are **reports**: R4 is evidence
*about the protocol* — §3 measured 31 such positions in the block and told
annotators about them precisely because leaving it uninstructed makes it a bias
rather than noise — and silently fixing a violation destroys the evidence that
the instruction did not take. Duplicates are a cross-row property and cannot be
decided from one submission at all.

## How an R4 site is identified, and why the draft defines it

R4: when a word ending in sukun is followed by a word beginning with `ال`,
careful recitation inserts a helping vowel (`مِنْ الْمَاءِ` read *mina
l-māʾi*) and the annotator must **not** write it.

Detecting a violation needs to know the word *should* carry sukun, which is
morphology and not available from the skeleton. A skeleton-only rule — "any
word before an `ال`-initial word" — finds **387** sites in the block, almost
all of them legitimate R1 case endings (`كتابُ الولدِ` takes damma and is
correct). So the site is defined by a *reference* that already carries marks:
a non-final word whose final letter carries sukun in the reference, followed by
an `ال`-initial word.

Run against the frame's own `tashkeel_sentence` this reproduces §3's figure
exactly — **31 positions in 29 of the 500 rows** — which is why this definition
and not another one is the implemented one. It inherits the reference's errors,
so the output is a *candidate* count for a human to read, never a correction.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

from arabic_mdd.data.normalize import skeleton, strip_diacritics
from arabic_mdd.metrics.der import arabic_words, marked_letters

#: §4's two conditions. `anchored` sees the `catt-eo` draft; `blank` sees
#: nothing — "no draft, no `tashkeel_sentence`, no diacritization tool, no LLM,
#: no looking up the source".
ANCHORED = "anchored"
BLANK = "blank"

#: §1 D3: two annotators, both qualified against gold before annotating.
ANNOTATORS = ("a1", "a2")

#: §5's two statuses. `unusable` is narrower than the pilot's: empty text, no
#: Arabic, or text too corrupt to diacritize. Never "hard", never "ambiguous",
#: never "the draft is bad".
DONE = "done"
UNUSABLE = "unusable"

#: §6's schema, verbatim and in order. The record written to disk has exactly
#: these keys — no more, because a field invented by the tool would not be
#: pre-registered, and no fewer, because five of them exist only to keep a
#: measurement possible.
RECORD_FIELDS: tuple[str, ...] = (
    "id",
    "rank",
    "annotator",
    "condition",
    "suggested",
    "final",
    "flagged_words",
    "ambiguous",
    "status",
    "unusable_reason",
    "note",
    "started_at",
    "finished_at",
    "adjudicated",
    "tool_version",
    "protocol_version",
)

#: §4's power rule, stated rather than discovered: "if more than 5 of the 50
#: are `unusable` (§5), extend the crossover with the next rows in draw order
#: until 45 usable remain, and record that this was done."
CROSSOVER_MIN_USABLE = 45

_SUKUN = "\u0652"
_SHORT_VOWELS = frozenset({"\u064e", "\u064f", "\u0650"})  # fatha, damma, kasra
_DEFINITE_ARTICLE = "\u0627\u0644"  # alef + lam

#: §7's rule for the two rows carrying a shadda on a long-vowel carrier
#: (ranks 447 and 475): diacritize as written, leave the malformed cluster, and
#: set `note` to this. Do not repair the skeleton.
MALFORMED_UPSTREAM = "malformed_upstream"


@dataclass(frozen=True)
class Assignment:
    """One row of work for one annotator, in the order the protocol fixes."""

    id: str
    rank: int
    crossover: bool
    condition: str
    sentence: str
    """The frame's undiacritized `sentence`. The annotator's `final` must keep
    this skeleton (§3) and this Arabic word count (§6)."""

    @property
    def anchored(self) -> bool:
        return self.condition == ANCHORED


def condition_for(rank: int, *, crossover: bool, annotator: str) -> str:
    """§4's table for the crossover, §1's design for the rest.

    The 450 non-crossover rows are anchored: correct-the-machine is the block's
    design (§8), and blank-page is the crossover's instrument plus the branch
    §8 reserves for a >15 pp anchoring effect.
    """
    if annotator not in ANNOTATORS:
        raise ValueError(f"unknown annotator {annotator!r}; expected one of {ANNOTATORS}")
    if not crossover:
        return ANCHORED
    first_half = rank < 25
    if annotator == "a1":
        return ANCHORED if first_half else BLANK
    return BLANK if first_half else ANCHORED


def owns(rank: int, *, crossover: bool, annotator: str) -> bool:
    """Whether this annotator does this row at all.

    All 50 crossover rows are done by **both** (§4). The 450 split 225/225 by
    draw-order parity: annotator 1 even ranks, annotator 2 odd (§1).
    """
    if crossover:
        return True
    return (rank % 2 == 0) if annotator == "a1" else (rank % 2 == 1)


def assignments(block: list[dict[str, Any]], annotator: str) -> list[Assignment]:
    """This annotator's queue, in the only order the protocol permits.

    Crossover rows first, then the rest, each in ascending draw rank. The block
    is never reordered, never filtered for difficulty and never shortened — §1:
    "Do not re-draw, do not substitute a row that looks hard, and do not skip
    one."
    """
    mine = [
        Assignment(
            id=str(row["id"]),
            rank=int(row["rank"]),
            crossover=bool(row.get("crossover")),
            condition=condition_for(
                int(row["rank"]), crossover=bool(row.get("crossover")), annotator=annotator
            ),
            sentence=row["sentence"],
        )
        for row in block
        if owns(int(row["rank"]), crossover=bool(row.get("crossover")), annotator=annotator)
    ]
    return sorted(mine, key=lambda a: (not a.crossover, a.rank))


def r4_sites(reference: str) -> list[int]:
    """Word indices (0-based) where R4 can be violated, per the module docstring.

    A non-final word whose final letter carries sukun **in `reference`**,
    followed by a word whose skeleton begins `ال`. The index is the *preceding*
    word's — the one whose final mark R4 governs.
    """
    words = arabic_words(unicodedata.normalize("NFC", reference))
    sites: list[int] = []
    for i in range(len(words) - 1):
        if not strip_diacritics(words[i + 1]).startswith(_DEFINITE_ARTICLE):
            continue
        pairs = marked_letters(words[i])
        if pairs and _SUKUN in pairs[-1][1]:
            sites.append(i)
    return sites


def r4_violations(reference: str, final: str) -> list[int]:
    """Sites where `final` writes a liaison vowel instead of keeping the sukun.

    A *candidate* list. The site definition inherits `reference`'s errors (see
    the module docstring), and §6 is explicit that these are counted and
    reported, not corrected — a violation is evidence that §3's instruction did
    not take, and repairing it deletes the evidence.

    Word indices are into `final`'s Arabic words, so this is only meaningful
    when the word counts agree; it returns `[]` when they do not, leaving the
    word-count check to say so on its own.
    """
    ref_words = arabic_words(unicodedata.normalize("NFC", reference))
    fin_words = arabic_words(unicodedata.normalize("NFC", final))
    if len(ref_words) != len(fin_words):
        return []
    out: list[int] = []
    for i in r4_sites(reference):
        pairs = marked_letters(fin_words[i])
        if pairs and pairs[-1][1] & _SHORT_VOWELS:
            out.append(i)
    return out


@dataclass(frozen=True)
class SubmissionChecks:
    """§6's per-row ingest checks. Two block, two report."""

    skeleton_ok: bool
    frame_skeleton: str
    final_skeleton: str
    word_count_ok: bool
    n_words_frame: int
    n_words_final: int
    r4_sites: tuple[int, ...]
    r4_violations: tuple[int, ...]

    @property
    def blocking(self) -> bool:
        """§3: a row whose skeleton or word count moved "is flagged and re-done"."""
        return not (self.skeleton_ok and self.word_count_ok)

    @property
    def problems(self) -> list[str]:
        """Human-readable, in the order a person should act on them."""
        out: list[str] = []
        if not self.skeleton_ok:
            out.append(
                "the LETTERS changed — marks only (§3). "
                f"frame: {self.frame_skeleton!r} / yours: {self.final_skeleton!r}"
            )
        if not self.word_count_ok:
            out.append(
                f"word count moved: frame has {self.n_words_frame} Arabic words, "
                f"yours has {self.n_words_final} (§6)"
            )
        if self.r4_violations:
            out.append(
                f"R4 (§3): liaison vowel written at word index(es) "
                f"{list(self.r4_violations)} — keep the sukun. Reported, not corrected."
            )
        return out


def check_submission(
    sentence: str, final: str, *, reference: str | None = None
) -> SubmissionChecks:
    """Run §6's two per-row checks plus the R4 report.

    `reference` is the marked text that defines R4 sites — the `catt-eo` draft
    in the anchored condition, or the frame's `tashkeel_sentence` in the blank
    one, where no draft exists but the site set is a property of the *sentence*
    and not of the annotator. Passing `None` skips the R4 report rather than
    guessing.
    """
    frame_words = arabic_words(sentence)
    final_words = arabic_words(final)
    sites: tuple[int, ...] = ()
    violations: tuple[int, ...] = ()
    if reference is not None:
        sites = tuple(r4_sites(reference))
        violations = tuple(r4_violations(reference, final))
    return SubmissionChecks(
        skeleton_ok=skeleton(final) == skeleton(sentence),
        frame_skeleton=skeleton(sentence),
        final_skeleton=skeleton(final),
        word_count_ok=len(final_words) == len(frame_words),
        n_words_frame=len(frame_words),
        n_words_final=len(final_words),
        r4_sites=sites,
        r4_violations=violations,
    )


def build_record(
    assignment: Assignment,
    *,
    annotator: str,
    final: str,
    suggested: str | None,
    flagged_words: list[int],
    ambiguous: list[dict[str, Any]],
    status: str,
    unusable_reason: str | None,
    note: str | None,
    started_at: str,
    finished_at: str,
    tool_version: str | None,
    protocol_version: str | None,
) -> dict[str, Any]:
    """Assemble §6's record. Exactly `RECORD_FIELDS`, in order.

    `suggested` is forced to `None` in the blank condition regardless of what
    the caller passes. That is not defensive programming for its own sake: the
    one way to spoil the crossover irrecoverably is to let the draft reach the
    blind arm, and this is the single place where a record is constructed.
    """
    if status not in (DONE, UNUSABLE):
        raise ValueError(f"status must be {DONE!r} or {UNUSABLE!r}, got {status!r}")
    if status == UNUSABLE and not (unusable_reason or "").strip():
        raise ValueError("§5: `unusable` requires a free-text reason")
    if annotator not in ANNOTATORS:
        raise ValueError(f"unknown annotator {annotator!r}; expected one of {ANNOTATORS}")
    record = {
        "id": assignment.id,
        "rank": assignment.rank,
        "annotator": annotator,
        "condition": assignment.condition,
        "suggested": suggested if assignment.anchored else None,
        "final": final,
        "flagged_words": sorted(set(flagged_words)),
        "ambiguous": list(ambiguous),
        "status": status,
        "unusable_reason": unusable_reason or None,
        "note": note or None,
        "started_at": started_at,
        "finished_at": finished_at,
        "adjudicated": None,
        "tool_version": tool_version,
        "protocol_version": protocol_version,
    }
    assert tuple(record) == RECORD_FIELDS, "record keys drifted from §6's schema"
    return record


@dataclass
class FileReport:
    """§6's cross-row checks, plus §4's crossover-usability rule."""

    annotator: str
    n_records: int = 0
    n_expected: int = 0
    done: int = 0
    unusable: int = 0
    duplicates: list[tuple[str, str]] = field(default_factory=list)
    """(`id`, `condition`) pairs written twice by this annotator — §6's fourth check."""
    unknown_rows: list[str] = field(default_factory=list)
    """Records whose (`id`, `condition`) is not in this annotator's assignment."""
    wrong_condition: list[str] = field(default_factory=list)
    blinding_breach: list[str] = field(default_factory=list)
    """Blank-condition records carrying a non-null `suggested`.

    `build_record` cannot produce one — it forces `suggested` to None on the
    blank arm. But this is the ingest gate, not the writer's own assertion, and
    a file that reached ingest by any other route still has to be told apart
    from a valid one. A breach here invalidates §8's Δ for that utterance: the
    blank arm was not blank.
    """
    crossover_done: int = 0
    crossover_unusable: int = 0
    blocking_rows: list[str] = field(default_factory=list)
    r4_violation_rows: list[str] = field(default_factory=list)
    n_r4_violations: int = 0
    flagged_positions: int = 0
    ambiguous_positions: int = 0
    seconds: list[float] = field(default_factory=list)

    @property
    def crossover_extension_required(self) -> bool:
        """§4: extend the crossover when fewer than 45 of the 50 are usable.

        True only once all 50 have been attempted — before that the shortfall is
        work not yet done, and reading it as the rule firing would extend the
        crossover on the strength of an unfinished session.
        """
        attempted = self.crossover_done + self.crossover_unusable
        return attempted >= 50 and self.crossover_done < CROSSOVER_MIN_USABLE

    @property
    def median_seconds(self) -> float | None:
        """D4's extension decision is deferred to real timings (§6)."""
        if not self.seconds:
            return None
        ordered = sorted(self.seconds)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])


def elapsed_seconds(record: dict[str, Any]) -> float | None:
    """`finished_at − started_at`, or None if either timestamp is unusable."""
    from datetime import datetime

    try:
        start = datetime.fromisoformat(record["started_at"])
        finish = datetime.fromisoformat(record["finished_at"])
    except (KeyError, TypeError, ValueError):
        return None
    delta = (finish - start).total_seconds()
    return delta if delta >= 0 else None


def validate(
    records: list[dict[str, Any]],
    assigned: list[Assignment],
    annotator: str,
    *,
    references: dict[str, str] | None = None,
) -> FileReport:
    """Re-run every §6 check over a whole annotation file.

    This is the ingest gate, and it is deliberately runnable on a finished file
    rather than only at the keystroke: the same checks must hold when weeks 7–9
    read `C_gold`, long after the tool that wrote it stopped running.
    """
    by_key = {(a.id, a.condition): a for a in assigned}
    report = FileReport(annotator=annotator, n_records=len(records), n_expected=len(assigned))
    seen: set[tuple[str, str]] = set()
    for rec in records:
        key = (str(rec.get("id")), str(rec.get("condition")))
        if key in seen:
            report.duplicates.append(key)
        seen.add(key)
        assignment = by_key.get(key)
        if assignment is None:
            # Either a row this annotator does not own, or the right row under
            # the wrong condition. The second is the serious one: it means the
            # crossover's blinding was broken for that utterance.
            if any(a.id == key[0] for a in assigned):
                report.wrong_condition.append(key[0])
            else:
                report.unknown_rows.append(key[0])
            continue
        if rec.get("status") == UNUSABLE:
            report.unusable += 1
            if assignment.crossover:
                report.crossover_unusable += 1
            continue
        report.done += 1
        if assignment.crossover:
            report.crossover_done += 1
        if not assignment.anchored and rec.get("suggested"):
            report.blinding_breach.append(assignment.id)
        reference = (references or {}).get(assignment.id)
        if assignment.anchored and rec.get("suggested"):
            reference = rec["suggested"]
        checks = check_submission(assignment.sentence, rec.get("final") or "", reference=reference)
        if checks.blocking:
            report.blocking_rows.append(assignment.id)
        if checks.r4_violations:
            report.r4_violation_rows.append(assignment.id)
            report.n_r4_violations += len(checks.r4_violations)
        report.flagged_positions += len(rec.get("flagged_words") or [])
        report.ambiguous_positions += len(rec.get("ambiguous") or [])
        secs = elapsed_seconds(rec)
        if secs is not None:
            report.seconds.append(secs)
    return report
