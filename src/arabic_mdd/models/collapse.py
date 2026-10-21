"""Did the text-dependent arm learn to read audio, or to copy the canonical?

**Why this exists.** The training data makes a shortcut available, and the
shortcut would not look like a failure -- it would look like a spectacular
result. The baseline's manifests set the CTC target as follows
(`scripts/baseline_reproduction/download_hugg_data*.py`):

* `Iqra_train` (79 h): target = `phoneme_ref`, which **is** the canonical. For
  these rows a model that ignores the audio and emits the conditioned canonical
  scores a perfect loss.
* `Iqra_TTS` (52 h): target = `phoneme_mis`, the *mispronounced* sequence that
  was actually synthesized, while `phoneme_ref` remains the canonical. For
  these rows copying the canonical is wrong -- which is exactly what makes them
  the informative 40% of training hours.

So the pressure against collapse is real but partial, and the degenerate
solution is locally attractive: copy the canonical except where the audio
objects loudly. That is also, stated charitably, the *correct* inductive bias
for canonical-conditioned MDD. The difference between the right bias and a
collapsed model is one of degree, and degree needs a number.

**Why collapse would be mistaken for a finding.** A collapsed model's output is
a function of the canonical alone. Swap `C_gold` for `C_auto` at inference --
D10's operation, the one the entire 2x2 rests on -- and the prediction changes
wherever the diacritizer erred, so |ΔF1| on the input path would be enormous.
That is the paper's headline number arriving as a pure artifact. D10 already
names the expected 2x2 signature as a bug guard; this module is the guard for
the specific bug that would most flatter us.

**Deliberately a diagnostic, not a mitigation.** The cheap option is one number
read off artifacts the run already writes, not a canonical-corruption scheme
bolted into training. Corrupting canonicals during training would also change
what the arm measures -- a model trained on corrupted `C` is *less* brittle to
corrupted `C`, which shrinks |ΔF1| and biases the headline in the opposite
direction from D10's stated caveat. If the numbers here say the arm collapsed,
that is the moment to weigh such a scheme, with the bias direction recorded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "D12_MIN_INFORMATIVE",
    "D12_THRESHOLD",
    "CollapseReport",
    "D12Verdict",
    "canonical_copy_rate",
    "collapse_report",
    "d12_verdict",
    "token_edit_distance",
]

#: D12's falsifier, pre-registered in `docs/weeks/week-08.md` before any arm was
#: evaluated on QuranMB.v2. A **difference** from the ablated arm rather than an
#: absolute level, because week 7 falsified the absolute form: on the dev split
#: both the conditioned arm and the `use_canonical=False` arm -- which cannot
#: causally prefer a canonical -- reported `canonical_preference = 1.0000`. The
#: statistic has a scale only relative to a model that cannot copy.
D12_THRESHOLD = 0.15
#: Below this many informative utterances the comparison is declared
#: underpowered and the cell is reported with the count rather than a verdict.
#: Week 7's dev run had 3, which is where the absolute threshold "fired" on a
#: model that provably was not copying.
D12_MIN_INFORMATIVE = 200


def token_edit_distance(first: Sequence[str], second: Sequence[str]) -> int:
    """Levenshtein distance over phoneme tokens.

    A local two-row DP rather than a call into `metrics.hierarchical`: that
    module's Needleman-Wunsch is an *alignment* with task-specific gap and
    match awards, built to produce TA/TR/FA/FR. Reusing it here would couple a
    sanity check to the metric it is meant to sanity-check, and a change to the
    metric's scoring would silently move this diagnostic.
    """
    if not first:
        return len(second)
    if not second:
        return len(first)
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, start=1):
        current = [i]
        for j, b in enumerate(second, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (a != b),  # substitution / match
                )
            )
        previous = current
    return previous[-1]


def canonical_copy_rate(
    hypotheses: Sequence[Sequence[str]],
    canonicals: Sequence[Sequence[str]],
) -> float:
    """Fraction of utterances whose prediction is *identical* to the canonical.

    The bluntest collapse signal and the one that needs no alignment. A value
    near 1.0 means the arm is a canonical echo and detects nothing, which in
    the hierarchical metric shows up as TR and FR both near zero -- i.e. as
    "bad recall", which is indistinguishable from an honestly weak detector
    unless this number is also on the table.
    """
    if len(hypotheses) != len(canonicals):
        raise ValueError(f"{len(hypotheses)} hypotheses but {len(canonicals)} canonicals")
    if not hypotheses:
        raise ValueError("no utterances: a copy rate over an empty dev set is not a number")
    identical = sum(list(h) == list(c) for h, c in zip(hypotheses, canonicals))
    return identical / len(hypotheses)


@dataclass(frozen=True)
class CollapseReport:
    """The three numbers that separate a collapsed arm from a working one."""

    utterances: int
    copy_rate: float
    """Fraction of all utterances where prediction == canonical exactly."""
    informative_utterances: int
    """Utterances where canonical != target, i.e. where the two disagree about
    what was said. Only these can distinguish reading from copying."""
    canonical_preference: float
    """Over the informative utterances, the fraction where the prediction is
    strictly closer to the canonical than to the target.

    This is the number to watch. At ~1.0 the arm resolves every disagreement in
    the canonical's favour, which is collapse. At ~0.0 it resolves them in the
    audio's favour, which is the behaviour MDD needs. Chance is ~0.5 only if
    the two are equidistant on average, which they are not, so read it as a
    trend across checkpoints rather than against a fixed threshold.
    """

    def as_dict(self) -> dict[str, float | int]:
        return {
            "utterances": self.utterances,
            "copy_rate": self.copy_rate,
            "informative_utterances": self.informative_utterances,
            "canonical_preference": self.canonical_preference,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float | int]) -> CollapseReport:
        """Rebuild from `as_dict()`, so a stored report can be a D12 argument.

        The ablated arm is evaluated in its own pass and its numbers reach the
        conditioned arm's verdict as JSON. Parsing that back into the dataclass
        keeps `d12_verdict` a function of two reports rather than of one report
        and a dict whose keys are checked nowhere.
        """
        return cls(
            utterances=int(payload["utterances"]),
            copy_rate=float(payload["copy_rate"]),
            informative_utterances=int(payload["informative_utterances"]),
            canonical_preference=float(payload["canonical_preference"]),
        )


def collapse_report(
    hypotheses: Sequence[Sequence[str]],
    canonicals: Sequence[Sequence[str]],
    targets: Sequence[Sequence[str]],
) -> CollapseReport:
    """Compare predictions against both the canonical and the actual target.

    `targets` is the CTC groundtruth -- what the speaker actually said, which
    for `Iqra_TTS` rows differs from the canonical by construction. Utterances
    where canonical and target agree carry no information about collapse and
    are excluded from `canonical_preference`; they are still counted in
    `copy_rate`, because an echo of an agreeing canonical is still an echo.

    Ties (prediction equidistant from both) count as *not* preferring the
    canonical. That is the conservative direction: it makes collapse harder to
    claim, so a high `canonical_preference` is not an artifact of tie-breaking.
    """
    lengths = {len(hypotheses), len(canonicals), len(targets)}
    if len(lengths) != 1:
        raise ValueError(
            f"ragged inputs: {len(hypotheses)} hypotheses, {len(canonicals)} "
            f"canonicals, {len(targets)} targets"
        )

    informative = 0
    prefers_canonical = 0
    # Lengths already checked above, so plain `zip` cannot truncate silently.
    for hypothesis, canonical, target in zip(hypotheses, canonicals, targets):
        if list(canonical) == list(target):
            continue
        informative += 1
        to_canonical = token_edit_distance(hypothesis, canonical)
        to_target = token_edit_distance(hypothesis, target)
        if to_canonical < to_target:
            prefers_canonical += 1

    return CollapseReport(
        utterances=len(hypotheses),
        copy_rate=canonical_copy_rate(hypotheses, canonicals),
        informative_utterances=informative,
        # NaN rather than 0.0 when nothing is informative: a dev set with no
        # canonical/target disagreements cannot measure this, and reporting 0.0
        # would read as "no collapse detected".
        canonical_preference=(prefers_canonical / informative if informative else float("nan")),
    )


@dataclass(frozen=True)
class D12Verdict:
    """Which of D12's three branches the numbers selected."""

    delta: float
    """`canonical_preference(conditioned) - canonical_preference(ablated)`.

    NaN if either arm had no informative utterances, in which case the verdict
    is `underpowered` on the count anyway.
    """
    informative_utterances: int
    """The **smaller** of the two arms' counts. Both arms are evaluated on the
    same manifest so the counts should agree; taking the minimum means a
    disagreement makes the comparison weaker rather than stronger."""
    powered: bool
    fires: bool
    verdict: str
    """One of `underpowered`, `fires`, `does_not_fire`."""

    def as_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "delta": self.delta,
            "informative_utterances": self.informative_utterances,
            "powered": self.powered,
            "fires": self.fires,
            "verdict": self.verdict,
        }


def d12_verdict(conditioned: CollapseReport, ablated: CollapseReport) -> D12Verdict:
    """Apply D12's pre-registered falsifier to a conditioned/ablated pair.

    The rule, in one line: the falsifier fires when the conditioned arm prefers
    the canonical more than the ablated arm does by more than
    `D12_THRESHOLD`, and only when at least `D12_MIN_INFORMATIVE` utterances
    could have shown the difference.

    Order matters and is not symmetric -- `d12_verdict(ablated, conditioned)`
    asks a different and meaningless question -- which is why this is a tested
    function and not a comparison written inline at each call site.

    The power test is checked **before** the threshold, so an underpowered run
    never reports a firing falsifier. That is the direction week 7 needed: a
    delta of +1.0 over 3 informative utterances is noise, and the absolute form
    of this rule did "fire" there on a model that provably cannot copy.
    """
    delta = conditioned.canonical_preference - ablated.canonical_preference
    informative = min(conditioned.informative_utterances, ablated.informative_utterances)
    powered = informative >= D12_MIN_INFORMATIVE
    # NaN > threshold is False, so an undefined delta cannot fire. Stated
    # rather than relied upon: the `powered` gate already excludes it, since a
    # NaN preference means zero informative utterances.
    fires = bool(powered and delta > D12_THRESHOLD)
    return D12Verdict(
        delta=delta,
        informative_utterances=informative,
        powered=powered,
        fires=fires,
        verdict="underpowered" if not powered else "fires" if fires else "does_not_fire",
    )
