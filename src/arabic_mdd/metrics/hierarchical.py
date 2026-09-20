"""Hierarchical MDD detection/diagnosis metric (TA/TR/FA/FR + Correct/Error Diagnosis).

Re-derived (not copied verbatim) from the official Iqra'Eval baseline's
reference implementation:
`github.com/Iqra-Eval/interspeech_IqraEval`, `mdd_eval/metric.py`,
`mdd_eval/align_data.py`, `mdd_eval/ins_del_cor_sub_analysis.py`. This is
the metric the week 1-3 baseline-reproduction gate is scored with (see
`docs/weeks/week-02.md` / `week-03.md`), and it also underlies RQ1-RQ5's
downstream MDD-quality comparisons.

Algorithm
---------
For one utterance, given:

- ``canonical``: the canonical/target phoneme sequence (correct
  pronunciation, from the text).
- ``human_annotation``: a human annotator's transcription of what was
  actually pronounced (the ground truth of the (mis)pronunciation).
- ``prediction``: an MDD system's predicted phoneme sequence.

three pairwise global alignments are computed (canonical vs
human_annotation, human_annotation vs prediction, canonical vs
prediction), each classified position-by-position into
Correct/Substitution/Insertion/Deletion (C/S/I/D) against ``<eps>`` gap
tokens.

The canonical-vs-annotation alignment tells us whether the speaker
actually made an error at a given canonical position. The
annotation-vs-prediction alignment (re-synced to the same position via the
shared annotation sequence) tells us whether the system's prediction
agrees with what actually happened. Combining the two classifies every
canonical position into:

- TA (True Acceptance): no real error; system predicted no error.
- FR (False Rejection): no real error; system predicted an error anyway.
- FA (False Acceptance): a real error; system missed it (predicted no
  error, or predicted the canonical phoneme).
- TR (True Rejection): a real error; system predicted an error too, split
  into:

  - Correct Diagnosis (CD): the predicted phoneme matches the annotation
    exactly.
  - Error Diagnosis (ED): the system flagged an error but predicted the
    wrong phoneme.

Deletion errors (a canonical phoneme the speaker dropped entirely) are
scored separately, by re-syncing the canonical-vs-annotation and
canonical-vs-prediction alignments on the canonical sequence, since a
dropped phoneme has no corresponding annotation-vs-prediction position to
anchor on.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

EPS = "<eps>"

_MATCH_AWARD = 1
_MISMATCH_PENALTY = -1
_GAP_PENALTY = -1

_Op = str  # one of "C", "S", "I", "D"


def _match_score(a: str, b: str) -> int:
    if a == b:
        return _MATCH_AWARD
    if a == EPS or b == EPS:
        return _GAP_PENALTY
    return _MISMATCH_PENALTY


def needleman_wunsch(seq1: Sequence[str], seq2: Sequence[str]) -> tuple[list[str], list[str]]:
    """Global alignment of two token sequences with a uniform gap penalty.

    Returns two equal-length lists (``seq1`` and ``seq2``, padded with
    ``<eps>`` gap tokens) representing one optimal alignment under
    match=+1 / mismatch=-1 / gap=-1, matching the reference
    implementation's ``metric.Align``.
    """
    n, m = len(seq1), len(seq2)
    score = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        score[i][0] = _GAP_PENALTY * i
    for j in range(n + 1):
        score[0][j] = _GAP_PENALTY * j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = score[i - 1][j - 1] + _match_score(seq1[j - 1], seq2[i - 1])
            delete = score[i - 1][j] + _GAP_PENALTY
            insert = score[i][j - 1] + _GAP_PENALTY
            score[i][j] = max(match, delete, insert)

    align1: list[str] = []
    align2: list[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        current = score[i][j]
        if current == score[i - 1][j - 1] + _match_score(seq1[j - 1], seq2[i - 1]):
            align1.append(seq1[j - 1])
            align2.append(seq2[i - 1])
            i -= 1
            j -= 1
        elif current == score[i][j - 1] + _GAP_PENALTY:
            align1.append(seq1[j - 1])
            align2.append(EPS)
            j -= 1
        else:
            align1.append(EPS)
            align2.append(seq2[i - 1])
            i -= 1
    while j > 0:
        align1.append(seq1[j - 1])
        align2.append(EPS)
        j -= 1
    while i > 0:
        align1.append(EPS)
        align2.append(seq2[i - 1])
        i -= 1

    align1.reverse()
    align2.reverse()
    return align1, align2


def _classify_ops(aligned_a: Sequence[str], aligned_b: Sequence[str]) -> list[_Op]:
    """Per-position C/S/I/D classification of an alignment pair.

    ``D`` (deletion): ``a`` has a token, ``b`` is a gap (``a``'s token was
    dropped). ``I`` (insertion): ``a`` is a gap, ``b`` has a token (``b``
    added a token not in ``a``). ``S`` (substitution): both present but
    differ. ``C`` (correct): both present and equal.
    """
    ops: list[_Op] = []
    for a, b in zip(aligned_a, aligned_b, strict=True):
        if a != EPS and b == EPS:
            ops.append("D")
        elif a == EPS and b != EPS:
            ops.append("I")
        elif a != b:
            ops.append("S")
        else:
            ops.append("C")
    return ops


def _sync_positions(anchor_a: Sequence[str], anchor_b: Sequence[str]) -> list[tuple[int, int]]:
    """Re-sync two independently-computed alignments of the same underlying
    sequence.

    ``anchor_a`` and ``anchor_b`` are two alignments that both contain the
    same original token sequence (e.g. the annotation sequence, aligned
    once against the canonical sequence and once against the prediction).
    Since alignment preserves original token order, a two-pointer scan
    re-establishes the correspondence between the two eps-padded views.

    Yields ``(index_in_a, index_in_b)`` for each non-eps anchor token, in
    order. Mirrors the reference implementation's resync loop exactly,
    including silently skipping a position if the anchor tokens don't line
    up (which should not happen when both alignments are of the same
    original sequence, but is preserved for fidelity).
    """
    pairs: list[tuple[int, int]] = []
    b_idx = 0
    for a_idx, tok in enumerate(anchor_a):
        if tok == EPS:
            continue
        while b_idx < len(anchor_b) and anchor_b[b_idx] == EPS:
            b_idx += 1
        if b_idx < len(anchor_b) and anchor_b[b_idx] == tok:
            pairs.append((a_idx, b_idx))
            b_idx += 1
    return pairs


def _score_correctness(
    canon_ca: Sequence[str],
    annotation_ca: Sequence[str],
    op_ca: Sequence[_Op],
    annotation_ab: Sequence[str],
    prediction_ab: Sequence[str],
    op_ab: Sequence[_Op],
) -> Counter[str]:
    """Score TA/FR and substitution/insertion TR positions.

    Synced on the annotation sequence, shared between the
    canonical-vs-annotation alignment (``_ca``) and the
    annotation-vs-prediction alignment (``_ab``).
    """
    counts: Counter[str] = Counter()
    for i, flag in _sync_positions(annotation_ca, annotation_ab):
        op = op_ca[i]
        op2 = op_ab[flag]
        if op == "C":
            counts["cor_cor" if op2 == "C" else "cor_nocor"] += 1
        elif op == "S":
            if op2 == "C":
                counts["sub_sub"] += 1
            elif canon_ca[i] != prediction_ab[flag]:
                counts["sub_sub1"] += 1
            else:
                counts["sub_nosub"] += 1
        elif op == "I":
            if op2 == "C":
                counts["ins_ins"] += 1
            elif op2 != "D":
                counts["ins_ins1"] += 1
            else:
                counts["ins_noins"] += 1
    return counts


def _score_deletions(
    canon_ca: Sequence[str],
    op_ca: Sequence[_Op],
    canon_cb: Sequence[str],
    op_cb: Sequence[_Op],
) -> Counter[str]:
    """Score deletion-error TR/FA positions.

    Synced on the canonical sequence, shared between the
    canonical-vs-annotation alignment (``_ca``) and the
    canonical-vs-prediction alignment (``_cb``). Only positions where the
    speaker actually dropped a canonical phoneme (``op_ca == "D"``) are
    scored here; C/S/I positions are handled by ``_score_correctness``.
    """
    counts: Counter[str] = Counter()
    for i, flag in _sync_positions(canon_ca, canon_cb):
        if op_ca[i] != "D":
            continue
        op3 = op_cb[flag]
        if op3 == "D":
            counts["del_del"] += 1
        elif op3 == "C":
            counts["del_nodel"] += 1
        else:
            counts["del_del1"] += 1
    return counts


@dataclass
class MDDCounts:
    """Raw tallies underlying the hierarchical metric, additive across utterances."""

    ta: int = 0
    fr: int = 0
    fa: int = 0
    tr: int = 0
    correct_diagnosis: int = 0
    error_diagnosis: int = 0

    def __add__(self, other: MDDCounts) -> MDDCounts:
        return MDDCounts(
            ta=self.ta + other.ta,
            fr=self.fr + other.fr,
            fa=self.fa + other.fa,
            tr=self.tr + other.tr,
            correct_diagnosis=self.correct_diagnosis + other.correct_diagnosis,
            error_diagnosis=self.error_diagnosis + other.error_diagnosis,
        )


@dataclass(frozen=True)
class MDDMetrics:
    """Derived metrics computed from an (aggregated) `MDDCounts`."""

    counts: MDDCounts
    precision: float
    recall: float
    f1: float
    far: float
    frr: float
    der: float
    detection_accuracy: float


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def evaluate_utterance(
    canonical: Sequence[str],
    human_annotation: Sequence[str],
    prediction: Sequence[str],
) -> MDDCounts:
    """Compute raw TA/TR/FA/FR + CD/ED tallies for one utterance.

    ``canonical``, ``human_annotation``, and ``prediction`` are phoneme
    token sequences (e.g. the 68-phoneme Halabi inventory used by the
    Iqra'Eval baseline). Counts from multiple utterances should be summed
    (``+`` or `aggregate`) before computing corpus-level metrics via
    `compute_metrics`, matching the reference implementation's
    corpus-pooled (not per-utterance-averaged) precision/recall/F1.
    """
    canon_ca, annotation_ca = needleman_wunsch(canonical, human_annotation)
    op_ca = _classify_ops(canon_ca, annotation_ca)

    annotation_ab, prediction_ab = needleman_wunsch(human_annotation, prediction)
    op_ab = _classify_ops(annotation_ab, prediction_ab)

    canon_cb, prediction_cb = needleman_wunsch(canonical, prediction)
    op_cb = _classify_ops(canon_cb, prediction_cb)

    counts = _score_correctness(canon_ca, annotation_ca, op_ca, annotation_ab, prediction_ab, op_ab)
    counts.update(_score_deletions(canon_ca, op_ca, canon_cb, op_cb))

    ta = counts["cor_cor"]
    fr = counts["cor_nocor"]
    fa = counts["sub_nosub"] + counts["ins_noins"] + counts["del_nodel"]
    tr = (
        counts["sub_sub"]
        + counts["sub_sub1"]
        + counts["ins_ins"]
        + counts["ins_ins1"]
        + counts["del_del"]
        + counts["del_del1"]
    )
    correct_diagnosis = counts["sub_sub"] + counts["ins_ins"] + counts["del_del"]
    error_diagnosis = counts["sub_sub1"] + counts["ins_ins1"] + counts["del_del1"]

    return MDDCounts(
        ta=ta,
        fr=fr,
        fa=fa,
        tr=tr,
        correct_diagnosis=correct_diagnosis,
        error_diagnosis=error_diagnosis,
    )


def aggregate(counts: Iterable[MDDCounts]) -> MDDCounts:
    """Sum per-utterance `MDDCounts` into a single corpus-level tally."""
    total = MDDCounts()
    for c in counts:
        total = total + c
    return total


def compute_metrics(counts: MDDCounts) -> MDDMetrics:
    """Derive precision/recall/F1/FAR/FRR/DER/detection accuracy from tallies."""
    ta, fr, fa, tr = counts.ta, counts.fr, counts.fa, counts.tr
    precision = _safe_div(tr, tr + fr)
    recall = _safe_div(tr, tr + fa)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    far = 1 - recall
    frr = _safe_div(fr, fr + ta)
    der = _safe_div(counts.error_diagnosis, counts.error_diagnosis + counts.correct_diagnosis)
    detection_accuracy = _safe_div(ta + tr, tr + ta + fr + fa)
    return MDDMetrics(
        counts=counts,
        precision=precision,
        recall=recall,
        f1=f1,
        far=far,
        frr=frr,
        der=der,
        detection_accuracy=detection_accuracy,
    )


def evaluate(
    canonical: Sequence[str],
    human_annotation: Sequence[str],
    prediction: Sequence[str],
) -> MDDMetrics:
    """Convenience wrapper: score a single utterance end to end.

    For corpus-level scoring (what the week-3 gate needs), prefer calling
    `evaluate_utterance` per utterance, `aggregate` over the results, and
    `compute_metrics` once on the total -- pooling counts before computing
    precision/recall/F1 matches the reference implementation and avoids
    the bias of averaging per-utterance ratios over utterances of very
    different lengths.
    """
    return compute_metrics(evaluate_utterance(canonical, human_annotation, prediction))
