"""The RQ5 interaction power estimate — week-06 exit criterion, clause 2.

A number, not a verdict, and then one decision taken against a rule that was
pre-registered before the number existed.

## What RQ5 asks and why it needs its own power estimate

RQ5 asks whether label-path corruption hurts a **text-dependent** (canonical-
conditioned) MDD system *more* than it hurts a prompt-free one. That is an
**interaction**, not a main effect. The main effect is already settled by
somebody else -- CROTTC-IF reports that conditioning on a correct canonical
hurts -- so the only thing left for us to contribute is the interaction, and
interactions cost more power than the main effects they sit on.

The estimand is therefore a difference-in-differences:

    DiD = dF1(text-dependent arm) - dF1(prompt-free arm)

Week 6 has measured the right-hand term (`run/quranic_label_path.json`). The
left-hand term needs the text-dependent arm, which is weeks 7-9. So the honest
thing this script can produce is the **minimum detectable DiD** at 80% power,
alpha = 0.05 two-sided, as an explicit function of the one quantity that cannot
yet be measured.

    MDE = (z_{1-alpha/2} + z_{power}) * SE(DiD) = 2.8016 * SE(DiD)

## The assumption that cannot be measured, carried as an assumption

The text-dependent arm does not exist, so its per-cluster variance is assumed,
not observed. Two parameters carry that:

- `k` -- the ratio of the text-dependent arm's per-cluster variance to the
  prompt-free arm's. Reported at **1x, 1.5x and 2x**, as week 6's clause 2
  requires. `k > 1` is the likely direction: a conditioned model consumes `C`,
  so corrupting `C` moves both its input and its score and the per-prompt
  spread should widen.
- `rho` -- the correlation between the two arms' per-prompt deltas. The two
  arms are scored on the **same 98 prompts against the same corrupted
  canonicals**, so they are paired and positively correlated, and pairing
  shrinks the SE of a difference. `rho = 0` (independence) is therefore the
  conservative choice and is the headline; `rho = 0.5` is reported beside it to
  show how much is being given away.

    SE(DiD) = SE_PF * sqrt(1 + k - 2*rho*sqrt(k))

An MDE quoted without naming `k` and `rho` would be cited later as if it had
been measured. Every figure in the artifact is keyed by both.

## Clustering, and what it costs -- reported as a bound, with its sign

`C` is constant within a prompt. For a `C`-driven effect, 1,642 clips carry
only 98 independent units, and week 6's plan asks for the naive and the
cluster-corrected MDE side by side so the design effect is visible rather than
implied.

The cluster SE is **measured**: `measure_label_path.py` already ran a
10,000-replicate bootstrap over the 98 prompts, and this script re-derives it
from the stored per-prompt counts so the two agree (they do, to ~5%, which is
itself a check that the stored SD is the quantity it claims to be).

The naive SE is **bounded, not measured**, and the direction is stated because
it matters. A true clip-level i.i.d. bootstrap needs per-clip counts; the
artifact stores per-*prompt* counts, because storing 1,642 count rows per
condition would have inflated it tenfold for a diagnostic. What this script
does instead is a **clip-equivalent bootstrap**: 1,642 draws from the clip
multiset (prompt `j` drawn with probability proportional to its `m_j` clips),
each draw contributing that prompt's counts scaled by `1/m_j`. That is the
clip-level bootstrap under the approximation that every clip of a prompt equals
the prompt's mean clip -- which **deletes within-prompt variance**. So:

    SE_naive(estimated)  <=  SE_naive(true)
    design effect        >=  truth

i.e. the naive MDE reported here is *smaller* than a real clip bootstrap would
give, and the design effect is an **upper** bound. Both errors point the same
way as the plan's concern -- the naive figure overstates power -- so quoting it
as the thing not to quote is safe. The second handle is analytic: for a purely
`C`-driven effect the intra-cluster correlation is 1 and the classical design
effect is `1 + (m_bar - 1) * 1 = m_bar = 16.76`, so the SE ratio should be near
`sqrt(16.76) = 4.09`. Two independent routes to the same quantity.

On Kadambi's mixed-effects template (plan section 5): the prompt bootstrap is
the design-effect equivalent of a random intercept for utterance, which is the
grouping factor that matters here. The other grouping factor, **speaker, is not
modelled** -- the artifact has no per-clip counts, so speaker cannot be
recovered from it, and the 18 speakers are *crossed* with the 98 prompts rather
than nested. A prompt-only bootstrap treats speaker variation as within-cluster
noise, which **inflates** the per-prompt variance it measures. So this too errs
toward a larger MDE, and the recommendation below is not resting on an
optimistic variance.

## Two sample sizes, because the paper has two arms

- **Qur'anic**: K = 98 prompt clusters, 1,642 clips. Measured SE.
- **MSA**: K = 500 utterances (D4's seeded draw). Nothing is measured here, so
  the per-cluster spread is taken to be the Qur'anic arm's per-prompt dF1 sd
  and scaled as `s / sqrt(500)`. That is a *generous* transfer in one respect
  and a *conservative* one in another, and both are stated in the artifact: a
  Qur'anic prompt pools ~16.8 clips, so its per-prompt dF1 is an average and
  less variable than a single MSA utterance's would be (optimistic), while 500
  independent utterances beat 98 clusters by `sqrt(500/98) = 2.26` (the real
  gain). The MSA row is a projection and is labelled one.

## The decision rule, and the threshold's provenance

Pre-registered in `docs/weeks/week-06.md` clause 2, from plan section 6: **if
the MDE exceeds the input-path main effect CROTTC-IF reports, RQ5 is cut in
November** rather than discovered to be underpowered in December. RQ5 is the
designated cuttable RQ, so cutting it is a planned outcome.

Our own notes had recorded only the *endpoint* of that main effect
(F1 -> 40.52%), not the baseline it drops from, which left the rule not
evaluable. Resolved from the paper on 14 Oct 2026 and pinned in
`THRESHOLD_PROVENANCE` below. Table 4(c), section 6.4.3: **CROTTC-LLaMA scores
56.87% F1 and the same system with canonical injection scores 40.52%**, on the
same 900-utterance L2-ARCTIC test set. The input-path main effect is therefore
**16.35 F1 points**, and it is a within-system ablation -- one model, one
corpus, one test set, the canonical conditioning switched on and off. That is
the right shape of comparison to lend us a threshold, and it is a closer analogy
to our input-path row than the abstract's 71.77% headline, which is a *different*
system (CROTTC-IF, the implicit-fusion fix) and therefore prices the repair
rather than the damage.

The caveat is still real and is stored next to the number: different system
(LLM-based), different corpus, different language. It is being used as a
*threshold* -- an order-of-magnitude statement about how large an input-path
effect is in this literature -- not as a prediction of our own. The rule was
pre-registered in those terms and is honoured in those terms.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from arabic_mdd.metrics.hierarchical import MDDCounts, aggregate, compute_metrics

ARTIFACT = Path("run/quranic_label_path.json")
OUT = Path("run/rq5_power.json")

# 80% power, alpha = 0.05 two-sided. Stored rather than computed so the
# artifact names the design it was produced under.
ALPHA = 0.05
POWER = 0.80
Z_ALPHA = 1.9599639845400545  # Phi^-1(0.975)
Z_POWER = 0.8416212335729143  # Phi^-1(0.80)
MDE_MULTIPLIER = Z_ALPHA + Z_POWER  # 2.8015852...

VARIANCE_RATIOS = (1.0, 1.5, 2.0)  # k: text-dependent arm vs prompt-free arm
ARM_CORRELATIONS = (0.0, 0.5)  # rho: headline is 0.0, the conservative one

MSA_UTTERANCES = 500  # docs/msa-arm.md D4

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261014  # same seed as the label-path run, deliberately

# The pre-registered threshold, pinned 14 Oct 2026. See the module docstring.
THRESHOLD_PROVENANCE = {
    "quantity": "the input-path main effect: the same system scored with and "
    "without canonical conditioning",
    "source": "arXiv:2604.22133 section 6.4.3, table 4(c); read 14 Oct 2026",
    "non_conditioned_f1": 0.5687,
    "non_conditioned_system": "CROTTC-LLaMA (56.87% F1)",
    "conditioned_f1": 0.4052,
    "conditioned_system": "CROTTC-LLaMA w/ canonical injection (40.52% F1)",
    "dataset": "L2-ARCTIC, 900 test utterances, 6 speakers",
    "main_effect_f1_points": 0.1635,
    "why_this_pair": "a within-system ablation — one model, one corpus, one test "
    "set, the canonical conditioning switched on and off. The abstract's 71.77% "
    "CROTTC-IF headline is a different system (the implicit-fusion fix) and so "
    "prices the repair rather than the damage; it is the wrong shape of "
    "comparison to lend a threshold to our input-path row.",
    "caveat": "different system (LLM-based), different corpus (L2-ARCTIC), "
    "different language (English L2). Used as an order-of-magnitude threshold "
    "for how large an input-path effect is in this literature, not as a "
    "prediction for the Arabic arms. The rule was pre-registered in these terms.",
}
THRESHOLD = 0.5687 - 0.4052  # 0.1635 F1


def _f1(counts: MDDCounts) -> float:
    return compute_metrics(counts).f1


def _counts(raw: dict[str, int]) -> MDDCounts:
    return MDDCounts(**raw)


def per_prompt_delta_f1(
    gold: dict[str, MDDCounts], auto: dict[str, MDDCounts], prompts: list[str]
) -> np.ndarray:
    """dF1 computed independently within each prompt.

    Not the estimator -- the estimator pools counts and calls `compute_metrics`
    once, per week 3's finding that per-utterance averaging biases toward short
    utterances. This is the *variance input*: the per-cluster spread that a
    cluster bootstrap resamples and that the MSA projection rescales.
    """
    return np.array([_f1(auto[p]) - _f1(gold[p]) for p in prompts], dtype=float)


def pooled_delta_f1(
    gold: dict[str, MDDCounts], auto: dict[str, MDDCounts], prompts: list[str]
) -> float:
    return _f1(aggregate(auto[p] for p in prompts)) - _f1(aggregate(gold[p] for p in prompts))


def cluster_bootstrap_se(
    gold: dict[str, MDDCounts],
    auto: dict[str, MDDCounts],
    prompts: list[str],
    replicates: int,
    seed: int,
) -> float:
    """SE of the pooled dF1 under resampling of the 98 prompts, paired draw.

    Re-derivation of what `measure_label_path.py` already stored. Kept here so
    the artifact's `recomputed` field can be compared against its `as_measured`
    field: if they disagree by more than a few percent, one of the two scripts
    is not doing what it says.
    """
    rng = np.random.default_rng(seed)
    k = len(prompts)
    draws = np.empty(replicates, dtype=float)
    for i in range(replicates):
        idx = rng.integers(0, k, size=k)
        chosen = [prompts[j] for j in idx]
        draws[i] = _f1(aggregate(auto[p] for p in chosen)) - _f1(aggregate(gold[p] for p in chosen))
    return float(draws.std(ddof=1))


def clip_equivalent_bootstrap_se(
    gold: dict[str, MDDCounts],
    auto: dict[str, MDDCounts],
    prompts: list[str],
    clips_per_prompt: dict[str, int],
    replicates: int,
    seed: int,
) -> float:
    """A *bound* on the naive clip-level SE. See the module docstring.

    1,642 draws from the clip multiset, each contributing its prompt's counts
    divided by that prompt's clip count. Equals a clip-level i.i.d. bootstrap
    under the approximation that all clips of a prompt are the prompt's mean
    clip, which deletes within-prompt variance -- so the SE returned is a lower
    bound on the true naive SE and the design effect built from it is an upper
    bound.

    Counts become fractional here, so this works on float vectors rather than
    `MDDCounts`; `compute_metrics`' arithmetic is scale-free in the four cells
    it needs, so F1 is unaffected by that.
    """
    rng = np.random.default_rng(seed)
    order = ("ta", "fr", "fa", "tr")
    per_clip_gold = np.array(
        [[getattr(gold[p], f) / clips_per_prompt[p] for f in order] for p in prompts]
    )
    per_clip_auto = np.array(
        [[getattr(auto[p], f) / clips_per_prompt[p] for f in order] for p in prompts]
    )
    weights = np.array([clips_per_prompt[p] for p in prompts], dtype=float)
    weights /= weights.sum()
    n_clips = int(sum(clips_per_prompt[p] for p in prompts))

    def f1_from_vector(v: np.ndarray) -> float:
        _ta, fr, fa, tr = v
        precision = tr / (tr + fr) if tr + fr else float("nan")
        recall = tr / (tr + fa) if tr + fa else float("nan")
        denom = precision + recall
        return 2 * precision * recall / denom if denom else float("nan")

    draws = np.empty(replicates, dtype=float)
    for i in range(replicates):
        idx = rng.choice(len(prompts), size=n_clips, replace=True, p=weights)
        draws[i] = f1_from_vector(per_clip_auto[idx].sum(axis=0)) - f1_from_vector(
            per_clip_gold[idx].sum(axis=0)
        )
    return float(draws.std(ddof=1))


def se_did(se_prompt_free: float, k: float, rho: float) -> float:
    """SE of the difference-in-differences given the assumed TD/PF variance ratio."""
    return se_prompt_free * math.sqrt(1.0 + k - 2.0 * rho * math.sqrt(k))


def mde(se: float) -> float:
    return MDE_MULTIPLIER * se


def mde_grid(se_prompt_free: float) -> dict[str, dict[str, float]]:
    """MDE for every (k, rho) in the sensitivity design, keyed by both."""
    return {
        f"k={k}": {f"rho={rho}": mde(se_did(se_prompt_free, k, rho)) for rho in ARM_CORRELATIONS}
        for k in VARIANCE_RATIOS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--write", action="store_true", help=f"write {OUT.name}")
    args = parser.parse_args()

    if not args.artifact.exists():
        raise SystemExit(
            f"{args.artifact} not found -- run scripts/measure_label_path.py --write first; "
            "this script measures nothing itself, it only re-reads that artifact"
        )

    label_path = json.loads(args.artifact.read_text(encoding="utf-8"))
    gold = {p: _counts(c) for p, c in label_path["c_gold_cell"]["per_prompt_counts"].items()}
    clips_per_prompt = {str(p): int(n) for p, n in label_path["clips_per_prompt"].items()}
    prompts = sorted(gold)
    n_clips = sum(clips_per_prompt[p] for p in prompts)
    mean_cluster = n_clips / len(prompts)
    analytic_design_effect = math.sqrt(1.0 + (mean_cluster - 1.0) * 1.0)

    print("=== RQ5 interaction power — week-06 exit criterion, clause 2 ===")
    print(f"source      : {args.artifact} (git {label_path.get('git_sha', '?')})")
    print("estimand    : DiD = dF1(text-dependent) - dF1(prompt-free)")
    print(f"design      : {POWER:.0%} power, alpha = {ALPHA} two-sided, z-sum {MDE_MULTIPLIER:.4f}")
    print(f"quranic n   : {n_clips} clips, {len(prompts)} clusters, {mean_cluster:.2f}/cluster")
    print(f"MSA n       : {MSA_UTTERANCES} utterances (D4), 1 cluster each")
    print(
        f"threshold   : {THRESHOLD:.4f} F1 — CROTTC-LLaMA "
        f"{THRESHOLD_PROVENANCE['non_conditioned_f1']:.4f} minus the same system with "
        f"canonical injection {THRESHOLD_PROVENANCE['conditioned_f1']:.4f}"
    )
    print("              (different system/corpus/language; a threshold, not a prediction)\n")

    tools: dict[str, Any] = {}
    for tool, result in label_path["results"].items():
        auto = {p: _counts(c) for p, c in result["per_prompt_counts"].items()}
        deltas = per_prompt_delta_f1(gold, auto, prompts)
        sd = float(deltas.std(ddof=1))
        se_sd_over_sqrt_k = sd / math.sqrt(len(prompts))
        se_measured = float(result["delta_vs_gold"]["bootstrap_sd"])
        se_recomputed = cluster_bootstrap_se(gold, auto, prompts, args.bootstrap, args.seed)
        se_naive_bound = clip_equivalent_bootstrap_se(
            gold, auto, prompts, clips_per_prompt, args.bootstrap, args.seed
        )
        se_msa = sd / math.sqrt(MSA_UTTERANCES)

        print(f"--- {tool} (pooled dF1 {pooled_delta_f1(gold, auto, prompts):+.4f}) ---")
        print(
            f"  per-prompt dF1: mean {deltas.mean():+.4f}  sd {sd:.4f}  "
            f"sd/sqrt(98) {se_sd_over_sqrt_k:.5f}"
        )
        print(
            f"  SE_PF cluster : {se_recomputed:.5f} recomputed  "
            f"vs {se_measured:.5f} as measured in the label-path run"
        )
        print(
            f"  SE_PF naive   : {se_naive_bound:.5f} (clip-equivalent, a LOWER bound)  "
            f"-> design effect <= {se_recomputed / se_naive_bound:.2f}x "
            f"(analytic, rho_icc=1: {analytic_design_effect:.2f}x)"
        )
        cluster_grid = mde_grid(se_recomputed)
        naive_grid = mde_grid(se_naive_bound)
        msa_grid = mde_grid(se_msa)
        for k in VARIANCE_RATIOS:
            key = f"k={k}"
            print(
                f"  MDE k={k:<4} : quranic cluster {cluster_grid[key]['rho=0.0']:.4f} "
                f"(rho=0.5 {cluster_grid[key]['rho=0.5']:.4f})   "
                f"naive {naive_grid[key]['rho=0.0']:.4f}   "
                f"MSA n=500 {msa_grid[key]['rho=0.0']:.4f}"
            )
        worst = cluster_grid[f"k={max(VARIANCE_RATIOS)}"]["rho=0.0"]
        realistic = cluster_grid["k=1.5"]["rho=0.5"]
        own_effect = pooled_delta_f1(gold, auto, prompts)
        print(
            f"  headline MDE  : {worst:.4f} (k=2, rho=0, clustered) "
            f"vs threshold {THRESHOLD:.4f} -> {'POWERED' if worst < THRESHOLD else 'UNDERPOWERED'}"
        )
        # The second, harsher question: is the MDE small against *this tool's
        # own* label-path effect? A ratio above 1 means the interaction would
        # have to exceed the main effect it modifies to be visible.
        flag = "" if worst < abs(own_effect) else "   <-- exceeds this tool's whole effect"
        print(
            f"  vs own effect : MDE/|dF1| = {worst / abs(own_effect):.2f} worst corner, "
            f"{realistic / abs(own_effect):.2f} at k=1.5 rho=0.5{flag}\n"
        )

        tools[tool] = {
            "pooled_delta_f1": own_effect,
            "per_prompt_delta_f1": {
                "mean": float(deltas.mean()),
                "sd": sd,
                "n": len(prompts),
                "se_as_sd_over_sqrt_k": se_sd_over_sqrt_k,
            },
            "se_prompt_free": {
                "cluster_recomputed": se_recomputed,
                "cluster_as_measured": se_measured,
                "cluster_agreement_ratio": se_recomputed / se_measured,
                "naive_clip_equivalent_lower_bound": se_naive_bound,
                "msa_projection_n500": se_msa,
            },
            "design_effect": {
                "empirical_upper_bound": se_recomputed / se_naive_bound,
                "analytic_icc1": analytic_design_effect,
                "note": "empirical figure is an upper bound because the naive SE it "
                "divides is a lower bound; see the module docstring",
            },
            "mde": {
                "quranic_cluster_98": cluster_grid,
                "quranic_naive_1642_lower_bound": naive_grid,
                "msa_projection_500": msa_grid,
            },
            "headline_mde": worst,
            "realistic_mde": realistic,
            "powered_at_threshold": bool(worst < THRESHOLD),
            "mde_over_own_label_path_effect": {
                "worst_corner": worst / abs(own_effect),
                "k1.5_rho0.5": realistic / abs(own_effect),
                "reading": "a ratio above 1 means the RQ5 interaction would have to "
                "be larger than the label-path main effect it modifies before this n "
                "could see it",
            },
        }

    # The decision. Pre-registered rule: cut if the MDE exceeds the threshold.
    # Evaluated at the most adversarial corner of the sensitivity design that
    # the plan actually asked for -- k = 2, rho = 0, clustered -- and on the
    # tool with the widest per-prompt spread, so a PASS is not an artefact of
    # having picked the friendly cell.
    headline_tool = max(tools, key=lambda t: tools[t]["headline_mde"])
    headline = tools[headline_tool]["headline_mde"]
    keep = headline < THRESHOLD
    margin = THRESHOLD / headline

    recommendation = {
        "decision": "keep" if keep else "cut",
        "evaluated_at": {
            "tool": headline_tool,
            "reason": "widest per-prompt dF1 spread of the four, so the most "
            "adversarial variance available",
            "variance_ratio_k": max(VARIANCE_RATIOS),
            "arm_correlation_rho": 0.0,
            "clustering": "prompt (98 clusters)",
        },
        "mde": headline,
        "threshold": THRESHOLD,
        "margin_factor": margin,
        "rule": "docs/weeks/week-06.md clause 2, from plan section 6: if the MDE "
        "exceeds the input-path main effect CROTTC-IF reports, RQ5 is cut in "
        "November rather than discovered to be underpowered in December.",
        "reading": (
            f"The worst-corner MDE is {headline:.4f} F1 against a {THRESHOLD:.4f} "
            f"threshold, a factor of {margin:.2f}. The rule is satisfied and RQ5 is "
            f"kept — but {margin:.2f}x is a pass, not comfort, and the margin is "
            "worth stating as such. What makes the pass defensible is that every "
            "approximation in it points toward a *larger* MDE: the design effect is "
            "an upper bound, speaker variance is folded into the per-prompt spread "
            "rather than modelled, rho = 0 discards the pairing the design actually "
            "has, and k = 2 doubles a variance nobody has observed. At the corner "
            "the design more plausibly sits at (k = 1.5, rho = 0.5) the MDE is "
            f"{tools[headline_tool]['realistic_mde']:.4f}, a factor of "
            f"{THRESHOLD / tools[headline_tool]['realistic_mde']:.2f}."
        )
        if keep
        else (
            f"The worst-corner MDE is {headline:.4f} F1 against a {THRESHOLD:.4f} "
            "threshold. RQ5 is cut in November per the pre-registered rule."
        ),
        "condition_on_keeping": (
            "RQ5 is powered against the *threshold*, not against every tool's own "
            "label-path effect. The MDE/|dF1| column shows the split: for shakkala "
            "and mishkal the MDE is a third to a half of the main effect, so an "
            "interaction of plausible relative size is visible; for catt-eo and "
            "catt-ed the worst-corner MDE *exceeds* the tool's entire label-path "
            "effect, so an interaction there would have to be larger than the "
            "effect it modifies. The practical consequence is a scoping decision "
            "rather than a cut: RQ5 is answered on the high-corruption tools, and "
            "the low-DER tools are reported as not having the resolution to say. "
            "Writing that in now is cheaper than discovering it in December, which "
            "is the entire purpose of the clause."
        ),
        "what_this_does_not_say": "That RQ5's interaction is large, or that it "
        "exists. Only that an interaction of the size the input-path literature "
        "deals in would be detectable at this n. A null would be interpretable "
        "rather than uninformative, which is what clause 2 exists to establish.",
    }

    print("=== recommendation ===")
    print(f"  {recommendation['decision'].upper()} RQ5")
    print(f"  {recommendation['reading']}\n")

    # --- D2 ---------------------------------------------------------------
    # docs/msa-arm.md "Still open": whether `A := C_gold` is carried as an
    # assumption with the 2.3%/19.8% base-rate band as a sensitivity row, or
    # whether the residual interaction term is bounded explicitly. Week 6 task
    # 3's rule: "if the MDE is comfortably below the effects of interest, bound
    # the term; if it is not, the sensitivity row is the honest and cheaper
    # option."
    d2 = {
        "question": "is `A := C_gold` an assumption with the 2.3%/19.8% band as a "
        "sensitivity row, or is the residual interaction term bounded?",
        "rule": "docs/weeks/week-06.md task 3: bound it if the MDE is comfortably "
        "below the effects of interest; otherwise the sensitivity row.",
        "decision": "bound the term, and keep the sensitivity row as a cross-check",
        "because": (
            "not because the MDE is comfortable — it is a factor of "
            f"{margin:.2f} below the threshold, which is a pass and not much more. "
            "The decisive argument is the other one. The MSA deviation base rate "
            "spans 2.3%-19.8% across the pilot's four definitions, an 8.6x spread, "
            "and a sensitivity row carrying that spread would be several times "
            f"wider than the {headline:.4f} MDE it is meant to qualify. It would "
            "swallow the interaction whole and the paper would be left unable to "
            "say anything in either direction. A bound is the only option that "
            "leaves RQ5 readable, so the rule's first branch is taken on "
            "informativeness rather than on margin."
        ),
        "how_the_bound_is_obtained": (
            "On the Qur'anic arm both quantities exist: real `A` "
            "(annotation_phoneme_string) and `C_gold`. Re-scoring the four "
            "label-path conditions a second time with `A := C_gold` substituted "
            "gives dF1 under the assumption directly, and the difference against "
            "the measured dF1 *is* the residual term -- one extra scoring "
            "condition per tool on the existing harness, no new instrument and no "
            "new annotation. The MSA arm then inherits that bound scaled by the "
            "ratio of its deviation base rate to the Qur'anic arm's."
        ),
        "scheduled": "week 10-11, with the MSA label-path column it qualifies; not "
        "week 6, which has no use for it yet",
        "first_order_argument_retained": "the measurement is a delta (same `P`, "
        "same `A`, only `C` swapped), so genuine mispronunciations enter both "
        "terms and cancel to first order. The bound prices the second order.",
    }
    print("=== D2 ===")
    print(f"  {d2['decision']}")
    print(f"  because: {d2['because']}\n")

    artifact = {
        "purpose": "minimum detectable difference-in-differences for RQ5, and the "
        "keep/cut recommendation against the pre-registered rule",
        "generated_by": "scripts/power_rq5.py",
        "source_artifact": str(args.artifact),
        "source_git_sha": label_path.get("git_sha"),
        "design": {
            "estimand": "DiD = dF1(text-dependent arm) - dF1(prompt-free arm)",
            "power": POWER,
            "alpha": ALPHA,
            "sided": 2,
            "mde_multiplier": MDE_MULTIPLIER,
            "se_did_formula": "SE_PF * sqrt(1 + k - 2*rho*sqrt(k))",
            "variance_ratios_k": list(VARIANCE_RATIOS),
            "arm_correlations_rho": list(ARM_CORRELATIONS),
            "headline_corner": "k=2, rho=0, prompt-clustered — the most adversarial "
            "cell of the sensitivity design",
            "measured_term": "prompt-free arm, run/quranic_label_path.json",
            "assumed_term": "text-dependent arm, weeks 7-9; its variance is `k`",
        },
        "samples": {
            "quranic": {
                "clips": n_clips,
                "clusters": len(prompts),
                "cluster_unit": "prompt",
                "mean_clips_per_cluster": mean_cluster,
            },
            "msa": {
                "utterances": MSA_UTTERANCES,
                "clusters": MSA_UTTERANCES,
                "cluster_unit": "utterance",
                "status": "projection — per-utterance spread assumed equal to the "
                "Qur'anic per-prompt dF1 sd, which averages ~16.8 clips and is "
                "therefore the optimistic side of that transfer",
            },
        },
        "clustering": {
            "cluster_se": "measured, 10,000-replicate paired bootstrap over prompts",
            "naive_se": "bounded below by a clip-equivalent bootstrap; a true "
            "clip-level bootstrap needs per-clip counts the label-path artifact "
            "does not store",
            "design_effect_direction": "reported figure is an upper bound",
            "analytic_cross_check": analytic_design_effect,
            "speaker": "18 speakers, crossed with the 98 prompts, NOT modelled — "
            "speaker variation is folded into the per-prompt spread, which inflates "
            "it, so the MDE is conservative in this respect too",
            "kadambi_template": "the prompt bootstrap is the design-effect "
            "equivalent of a random intercept for utterance; the speaker intercept "
            "is the part not reproduced here",
        },
        "threshold": THRESHOLD,
        "threshold_provenance": THRESHOLD_PROVENANCE,
        "bootstrap": {"replicates": args.bootstrap, "seed": args.seed},
        "tools": tools,
        "recommendation": recommendation,
        "d2": d2,
    }

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT}")
    else:
        print(f"(dry run — pass --write to produce {OUT})")


if __name__ == "__main__":
    main()
