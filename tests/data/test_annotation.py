"""`docs/annotation-protocol-msa.md` §§3-6, as executable assertions.

The protocol is pre-registered, so these tests check *quotations* rather than
behaviour wherever the two could drift: §4's role table, §6's field list,
§4's crossover rule, §3's 31 R4 positions. A refactor that changes any of
them should fail here and be taken back to the document.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arabic_mdd.data.annotation import (
    ANCHORED,
    BLANK,
    CROSSOVER_MIN_USABLE,
    DONE,
    RECORD_FIELDS,
    UNUSABLE,
    Assignment,
    assignments,
    build_record,
    check_submission,
    condition_for,
    elapsed_seconds,
    owns,
    r4_sites,
    r4_violations,
    validate,
)

BLOCK_PATH = Path(__file__).resolve().parents[2] / "run" / "msa_annotation_block.json"


@pytest.fixture(scope="module")
def block() -> list[dict]:
    return json.loads(BLOCK_PATH.read_text(encoding="utf-8"))["block"]


def _assignment(rank: int, *, crossover: bool = False, condition: str = ANCHORED) -> Assignment:
    return Assignment(
        id=f"id{rank}", rank=rank, crossover=crossover, condition=condition, sentence="كتب الولد"
    )


# --- §4's role table -------------------------------------------------------


@pytest.mark.parametrize(
    ("rank", "annotator", "expected"),
    [
        (0, "a1", ANCHORED),
        (24, "a1", ANCHORED),
        (25, "a1", BLANK),
        (49, "a1", BLANK),
        (0, "a2", BLANK),
        (24, "a2", BLANK),
        (25, "a2", ANCHORED),
        (49, "a2", ANCHORED),
    ],
)
def test_crossover_conditions_follow_the_protocol_table(rank, annotator, expected):
    assert condition_for(rank, crossover=True, annotator=annotator) == expected


def test_non_crossover_rows_are_anchored_for_both_annotators():
    # §8: correct-the-machine is the design for the whole block; blank-page is
    # the crossover's instrument.
    for annotator in ("a1", "a2"):
        assert condition_for(60, crossover=False, annotator=annotator) == ANCHORED


def test_both_annotators_do_every_crossover_row():
    for rank in range(50):
        assert owns(rank, crossover=True, annotator="a1")
        assert owns(rank, crossover=True, annotator="a2")


def test_non_crossover_rows_split_by_draw_order_parity():
    assert owns(50, crossover=False, annotator="a1")
    assert not owns(50, crossover=False, annotator="a2")
    assert owns(51, crossover=False, annotator="a2")
    assert not owns(51, crossover=False, annotator="a1")


def test_unknown_annotator_is_refused():
    with pytest.raises(ValueError, match="unknown annotator"):
        condition_for(0, crossover=True, annotator="a3")


# --- the committed draw, on the real block ---------------------------------


def test_assignments_on_the_committed_block(block):
    a1 = assignments(block, "a1")
    a2 = assignments(block, "a2")
    # 50 crossover (both) + 225 each of the 450.
    assert len(a1) == len(a2) == 275
    assert sum(1 for a in a1 if a.crossover) == 50
    # Every non-crossover row is done exactly once, across the two annotators.
    non_crossover = {a.id for a in a1 if not a.crossover} | {a.id for a in a2 if not a.crossover}
    assert len(non_crossover) == 450
    assert not {a.id for a in a1 if not a.crossover} & {a.id for a in a2 if not a.crossover}


def test_the_block_is_never_reordered(block):
    """§4: crossover first, then the rest, each in ascending draw rank."""
    for annotator in ("a1", "a2"):
        queue = assignments(block, annotator)
        crossover = [a.rank for a in queue if a.crossover]
        rest = [a.rank for a in queue if not a.crossover]
        assert queue[: len(crossover)] == [a for a in queue if a.crossover]
        assert crossover == sorted(crossover)
        assert rest == sorted(rest)
        assert max(crossover) < min(rest)


def test_each_annotator_sees_exactly_25_anchored_crossover_rows(block):
    for annotator in ("a1", "a2"):
        crossover = [a for a in assignments(block, annotator) if a.crossover]
        assert sum(1 for a in crossover if a.condition == ANCHORED) == 25
        assert sum(1 for a in crossover if a.condition == BLANK) == 25


def test_every_crossover_utterance_has_one_anchored_and_one_blank_record(block):
    """What makes §6's "sibling" wording work: the draft is always recoverable."""
    a1 = {a.id: a.condition for a in assignments(block, "a1") if a.crossover}
    a2 = {a.id: a.condition for a in assignments(block, "a2") if a.crossover}
    assert set(a1) == set(a2)
    for row_id in a1:
        assert {a1[row_id], a2[row_id]} == {ANCHORED, BLANK}


# --- §3's R4 --------------------------------------------------------------


def test_r4_site_is_a_sukun_before_the_definite_article():
    # §3's own example: min + al-maa' — the sukun is kept.
    assert r4_sites("مِنْ الْمَاءِ مَرَّتَيْنِ") == [0]


def test_r4_ignores_a_case_ending_before_the_definite_article():
    """The reason the site needs a marked reference and not just the skeleton.

    A genitive construct takes a real vowel before `ال` and is correct R1, not
    an R4 violation. A skeleton-only rule finds 387 such sites in the block.
    """
    assert r4_sites("كِتَابُ الْوَلَدِ") == []


def test_r4_violation_is_the_liaison_vowel_and_nothing_else():
    reference = "مِنْ الْمَاءِ مَرَّتَيْنِ"
    assert r4_violations(reference, "مِنْ الْمَاءِ مَرَّتَيْنِ") == []
    assert r4_violations(reference, "مِنَ الْمَاءِ مَرَّتَيْنِ") == [0]


def test_r4_declines_to_guess_when_the_word_counts_disagree():
    assert r4_violations("مِنْ الْمَاءِ مَرَّتَيْنِ", "مِنَ الْمَاءِ") == []


def test_r4_reproduces_the_protocols_31_positions(block):
    """§3: "31 positions in the 500-row block, 1.49% of non-final words"."""
    total = sum(len(r4_sites(row["tashkeel_sentence"])) for row in block)
    rows = sum(1 for row in block if r4_sites(row["tashkeel_sentence"]))
    assert total == 31
    assert rows == 29


# --- §6's two blocking checks ---------------------------------------------


def test_adding_marks_passes_both_checks():
    checks = check_submission("كتب الولد", "كَتَبَ الْوَلَدُ")
    assert checks.skeleton_ok and checks.word_count_ok
    assert not checks.blocking


def test_changing_a_letter_blocks():
    checks = check_submission("كتب الولد", "كَتَبَ الْوَلَدِين")
    assert not checks.skeleton_ok
    assert checks.blocking
    assert "LETTERS changed" in checks.problems[0]


def test_dropping_a_word_blocks():
    checks = check_submission("كتب الولد", "كَتَبَ")
    assert checks.blocking
    assert checks.n_words_frame == 2
    assert checks.n_words_final == 1


def test_punctuation_and_diacritics_do_not_count_as_letter_changes():
    # §7: punctuation is left as written and is handled downstream; the
    # skeleton key depunctuates, so a stray comma is not a skeleton change.
    assert not check_submission("كتب الولد،", "كَتَبَ الْوَلَدُ").blocking


def test_r4_violation_is_reported_but_does_not_block():
    checks = check_submission("من الماء مرتين", "مِنَ الْمَاءِ مَرَّتَيْنِ", reference="مِنْ الْمَاءِ مَرَّتَيْنِ")
    assert not checks.blocking
    assert checks.r4_violations == (0,)
    assert any("R4" in p for p in checks.problems)


def test_r4_is_skipped_rather_than_guessed_without_a_reference():
    checks = check_submission("من الماء", "مِنَ الْمَاءِ")
    assert checks.r4_sites == ()
    assert checks.r4_violations == ()


# --- §6's schema ----------------------------------------------------------


def test_record_has_exactly_the_protocols_fields_in_order():
    record = build_record(
        _assignment(0, crossover=True),
        annotator="a1",
        final="كَتَبَ الْوَلَدُ",
        suggested="كَتَبَ الْوَلَدَ",
        flagged_words=[2],
        ambiguous=[],
        status=DONE,
        unusable_reason=None,
        note=None,
        started_at="2026-10-14T10:00:00+00:00",
        finished_at="2026-10-14T10:01:40+00:00",
        tool_version="1.0.2",
        protocol_version="abc123",
    )
    assert tuple(record) == RECORD_FIELDS
    assert record["adjudicated"] is None  # filled by a later, separate pass
    assert elapsed_seconds(record) == 100.0


def test_the_draft_can_never_be_written_into_a_blank_record():
    """§4: "Blank page means blank." The record builder is the chokepoint."""
    record = build_record(
        _assignment(25, crossover=True, condition=BLANK),
        annotator="a1",
        final="كَتَبَ الْوَلَدُ",
        suggested="كَتَبَ الْوَلَدَ",  # a caller bug
        flagged_words=[],
        ambiguous=[],
        status=DONE,
        unusable_reason=None,
        note=None,
        started_at="2026-10-14T10:00:00+00:00",
        finished_at="2026-10-14T10:00:30+00:00",
        tool_version="1.0.2",
        protocol_version="abc123",
    )
    assert record["suggested"] is None


def test_unusable_requires_a_reason():
    with pytest.raises(ValueError, match="requires a free-text reason"):
        build_record(
            _assignment(0),
            annotator="a1",
            final="",
            suggested=None,
            flagged_words=[],
            ambiguous=[],
            status=UNUSABLE,
            unusable_reason="  ",
            note=None,
            started_at="2026-10-14T10:00:00+00:00",
            finished_at="2026-10-14T10:00:05+00:00",
            tool_version="1.0.2",
            protocol_version="abc123",
        )


def test_flagged_words_are_deduplicated_and_sorted():
    record = build_record(
        _assignment(0),
        annotator="a1",
        final="كَتَبَ الْوَلَدُ",
        suggested=None,
        flagged_words=[2, 1, 2],
        ambiguous=[],
        status=DONE,
        unusable_reason=None,
        note=None,
        started_at="2026-10-14T10:00:00+00:00",
        finished_at="2026-10-14T10:00:05+00:00",
        tool_version="1.0.2",
        protocol_version="abc123",
    )
    assert record["flagged_words"] == [1, 2]


# --- §6's cross-row checks and §4's crossover rule ------------------------


def _done(assignment: Assignment, final: str, **over) -> dict:
    base = dict(
        annotator="a1",
        final=final,
        suggested=None,
        flagged_words=[],
        ambiguous=[],
        status=DONE,
        unusable_reason=None,
        note=None,
        started_at="2026-10-14T10:00:00+00:00",
        finished_at="2026-10-14T10:00:50+00:00",
        tool_version="1.0.2",
        protocol_version="abc",
    )
    base.update(over)
    return build_record(assignment, **base)


def test_validate_catches_a_row_written_twice_in_the_same_condition():
    assignment = _assignment(0)
    record = _done(assignment, "كَتَبَ الْوَلَدُ")
    report = validate([record, dict(record)], [assignment], "a1")
    assert report.duplicates == [(assignment.id, ANCHORED)]


def test_validate_flags_a_row_recorded_under_the_wrong_condition():
    """The serious one: it means the crossover's blinding broke for that row."""
    assigned = _assignment(25, crossover=True, condition=BLANK)
    record = _done(_assignment(25, crossover=True, condition=ANCHORED), "كَتَبَ الْوَلَدُ")
    report = validate([record], [assigned], "a1")
    assert report.wrong_condition == [assigned.id]
    assert report.done == 0


def test_validate_catches_a_draft_present_on_a_blank_row():
    """A blank record carrying `suggested` means the blank arm was not blank.

    `build_record` cannot produce one, so such a file did not come from the
    tool — which is exactly why the ingest gate checks it rather than trusting
    the writer. §8's Δ is a within-utterance contrast, so a breach here does
    not add noise, it removes the comparison for that utterance.
    """
    assignment = _assignment(25, crossover=True, condition=BLANK)
    record = _done(assignment, "كَتَبَ الْوَلَدُ")
    assert record["suggested"] is None  # the writer refuses
    record["suggested"] = "كَتَبَ الْوَلَدُ"  # tampered with afterwards
    report = validate([record], [assignment], "a1")
    assert report.blinding_breach == [assignment.id]
    assert report.done == 1  # still counted: the row was annotated, just not blind


def test_validate_reports_blocking_rows_after_the_fact():
    assignment = _assignment(0)
    report = validate([_done(assignment, "كَتَبَ الْأَوْلَادُ")], [assignment], "a1")
    assert report.blocking_rows == [assignment.id]


def test_validate_collects_timings_for_d4():
    assignment = _assignment(0)
    report = validate([_done(assignment, "كَتَبَ الْوَلَدُ")], [assignment], "a1")
    assert report.seconds == [50.0]
    assert report.median_seconds == 50.0


def test_crossover_extension_fires_only_once_all_fifty_are_attempted():
    rows = [_assignment(r, crossover=True) for r in range(50)]
    records = [_done(a, "كَتَبَ الْوَلَدُ") for a in rows[:40]]
    partial = validate(records, rows, "a1")
    assert partial.crossover_done == 40
    assert not partial.crossover_extension_required  # 10 rows not yet done

    unusable = [
        build_record(
            a,
            annotator="a1",
            final="",
            suggested=None,
            flagged_words=[],
            ambiguous=[],
            status=UNUSABLE,
            unusable_reason="no Arabic in the sentence",
            note=None,
            started_at="2026-10-14T10:00:00+00:00",
            finished_at="2026-10-14T10:00:05+00:00",
            tool_version="1.0.2",
            protocol_version="abc",
        )
        for a in rows[40:]
    ]
    full = validate(records + unusable, rows, "a1")
    assert full.crossover_done == 40 < CROSSOVER_MIN_USABLE
    assert full.crossover_extension_required
