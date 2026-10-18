"""The label-path column on the Qur'anic arm — week-06 exit criterion, clause 1.

The paper's primary number for this week. `A` (what the reciter said) and `P`
(what the model predicted) are held **fixed**; only `C` (the canonical the
scoring uses) is swapped. So every difference reported here is attributable to
the diacritizer that produced `C`, and nothing else moved.

## What the 2x2 is, and why only one column exists

Plan section 1's design is input-path x label-path. The prompt-free baseline
never consumes `C`, so for it the input-path rows collapse and **only the
scoring column exists**. What this script delivers is the top-right cell
measured against the top-left: *label-path corruption alone, with the model
unimpaired*. The input-path row needs the text-dependent arm and is weeks 7-9.
A reader who sees "2x2" in week 6 and finds one column should find that stated
rather than inferred.

## Two legs, because `C_gold` is a choice and the tempting one is wrong

D9 (`docs/msa-arm.md` section 3.3b, 14 Oct 2026) settled this. Recapped because
it is the whole basis of the number below:

- **Leg A, the delta.** `C_gold = phonetize(recovered_text)` and
  `C_auto = phonetize(diacritize(strip_diacritics(recovered_text)))`. Same
  phonetizer, same text, same convention on both sides, so the only thing
  differing between the cells is the diacritizer. **This is the measurement.**
- **Leg B, the anchor.** Score once with `C = reference_phoneme_string`, the
  benchmark's own canonical, and confirm the published F1 = 0.4415 reproduces.

Using leg B's canonical as `C_gold` would have made the gold cell reproduce the
published F1 by construction -- and would have made every delta the *sum* of
two differences: the diacritizer's error, which is wanted, and our phonetizer's
disagreement with the uploader's, which is 44 of 98 prompts of confound
(tajwid: madd lengthening and nun assimilation, which a recitation reference
implements and a general-purpose MSA phonetiser does not). The write-up would
have billed all of it to diacritization. So leg B is reported as a separate
third number and the difference between the legs is the price of owning the
pipeline, named in the limitations rather than hidden in dF1.

## Pooling, and what it forces on the bootstrap

`compute_metrics` is applied **once to the summed counts**, matching the
reference implementation. Week 3 already found that per-utterance averaging
biases toward short utterances. That constrains the CI: the bootstrap resamples
*clusters*, re-pools their counts and recomputes, rather than resampling
per-utterance F1 values.

Clusters are the **98 prompts**, not the 1,642 clips. `C` is constant within a
prompt, so for a `C`-driven effect the effective n is far nearer 98 than 1,642;
a clip-level bootstrap would treat 1,642 draws as independent and report a CI
narrower than the design supports.

## The attribution diagnostic, and why it is not the obvious one

Week 6's plan asks what fraction of the FR/FA delta falls at positions where
`C_auto` and `C_gold` disagree. Read against `metrics/hierarchical.py` that
question is close to tautological: `A` and `P` are fixed, so the A-vs-P
alignment is *identical* across conditions, and `FR` is exactly the positions
where `op_ca == "C"` and `op_ab != "C"`. Only `op_ca` can move when `C` is
swapped, so the delta arises *only* where the two canonicals relate to `A`
differently -- i.e. the answer is ~100% by construction, and reporting it would
be a fake confirmation.

What the plan is actually worried about -- three-way Needleman-Wunsch
realignment manufacturing the effect rather than label corruption causing it --
is testable instead by a bound. Per clip, let `d` be the number of positions
where `C_gold` and `C_auto` disagree. If realignment is well-behaved, damage is
bounded by the corruption that caused it, so `|delta(FR+FA)| <= d`. Clips
exceeding that are realignment *amplifying* corruption. The reported number is
the damage-per-corrupted-position ratio plus the count of clips over the bound,
which unlike the tautology can come out wrong.

Deliberately one diagnostic. Plan section 4.2's corruption-attributable FR rate
is a weeks-10-12 deliverable with its own framing; the instrument is not built
early.

## What is *not* measured here, stated so it is not read as measured

Prediction 3 concerns the **case-ending share of label damage**. At phoneme
level that needs word boundaries inside the phoneme sequence, and per-word
phonetization diverges from whole-utterance phonetization -- section 4.1.1's
own example has one text come out 6 tokens spaced and 8 merged, because
vowel-length and glide rules apply across the join. So a word-aligned
phonetizer would be required, and that is out of proportion to one caveat.

Reported instead: the **text-level** case-ending share, as the CE-vs-noCE DER
ratio per tool, which the existing variants already give for free. The artifact
records `case_ending_share.phoneme_level: null` with this reason, so the gap is
visible at the point of use.

Run with:
    uv run python scripts/measure_label_path.py --limit 10          # smoke
    uv run python scripts/measure_label_path.py --tools catt-eo     # one tool
    uv run python scripts/measure_label_path.py --write
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from arabic_mdd.data.normalize import normalize, strip_diacritics, unsupported_characters
from arabic_mdd.data.phonemes import parse_phoneme_sequence
from arabic_mdd.data.phonetizer import Phonetizer
from arabic_mdd.data.quranmb import MDDGroundTruth, load_ground_truth
from arabic_mdd.diacritizers import DIACRITIZERS
from arabic_mdd.diacritizers.chunking import diacritize_long
from arabic_mdd.metrics import der_published
from arabic_mdd.metrics.der import (
    CATT_WITH_CASE_ENDINGS,
    CATT_WITHOUT_CASE_ENDINGS,
    MARKED_ONLY_WITH_CASE_ENDINGS,
    MARKED_ONLY_WITHOUT_CASE_ENDINGS,
    score_corpus,
)
from arabic_mdd.metrics.hierarchical import (
    MDDCounts,
    aggregate,
    compute_metrics,
    evaluate_utterance,
    needleman_wunsch,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP = REPO_ROOT / "run" / "quranmb_verse_map.json"
DEFAULT_PREDICTIONS = REPO_ROOT / "run" / "quranmb_predictions_official_ckpt.json"
OUT = REPO_ROOT / "run" / "quranic_label_path.json"

VARIANTS = (
    CATT_WITH_CASE_ENDINGS,
    CATT_WITHOUT_CASE_ENDINGS,
    MARKED_ONLY_WITH_CASE_ENDINGS,
    MARKED_ONLY_WITHOUT_CASE_ENDINGS,
)

#: Week 3's gate, reproduced to +-0.0001 with the organizers' own checkpoint.
#: Leg B must land here or the harness has drifted and no delta below is
#: trustworthy.
ANCHOR_F1 = 0.4415
ANCHOR_TOLERANCE = 0.001

#: Pre-registered in `docs/weeks/week-06.md` task 2, prediction 2: |dF1| is
#: monotone in CE DER. Three unrelated instruments already agree on this
#: ordering, so a fourth disagreeing is evidence about the metric, not the tools.
EXPECTED_ORDER = ("catt-eo", "catt-ed", "shakkala", "mishkal")

#: The gate's clause-1 threshold names the worst tool specifically: the CI must
#: exclude zero for at least this one.
WORST_TOOL = "mishkal"

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261014


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _f1(counts: MDDCounts) -> float:
    return compute_metrics(counts).f1


def _counts_dict(counts: MDDCounts) -> dict[str, int]:
    return {
        "ta": counts.ta,
        "fr": counts.fr,
        "fa": counts.fa,
        "tr": counts.tr,
        "correct_diagnosis": counts.correct_diagnosis,
        "error_diagnosis": counts.error_diagnosis,
    }


def _metrics_dict(counts: MDDCounts) -> dict[str, object]:
    m = compute_metrics(counts)
    return {
        "counts": _counts_dict(counts),
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "far": m.far,
        "frr": m.frr,
        "diagnostic_error_rate": m.der,
        "detection_accuracy": m.detection_accuracy,
    }


def load_prompt_text(verse_map: Path) -> dict[str, str]:
    """The 98 prompts' gold Qur'anic text, keyed by prompt index.

    `recovered_text` is the audio-verified recovery from week 5 (98/98 located,
    every span confirmed by ear). Week 6's plan calls this field
    `recovered_verse`; the artifact's key is `recovered_text`.
    """
    payload = json.loads(verse_map.read_text(encoding="utf-8"))
    return {p["prompt"]: p["recovered_text"] for p in payload["prompts"]}


def build_canonicals(
    prompt_text: dict[str, str],
    tools: list[str],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, dict], dict[str, str]]:
    """Phonemize `C_gold` and one `C_auto` per tool, for every prompt.

    Returns `(canonicals, tool_meta, normalized_text)` where `canonicals` maps
    condition name -> prompt index -> phoneme sequence.
    """
    phonetizer = Phonetizer("prescriptive")

    prompts = sorted(prompt_text)
    normalized = {}
    for key in prompts:
        result = normalize(prompt_text[key])
        if result.flags:
            # The verse map already reports `unsupported_characters` empty over
            # all 4,442 recovered characters, so a flag here means the text or
            # the contract moved since. Either way the delta would be measured
            # on something other than what was verified.
            raise SystemExit(
                f"prompt {key} is flagged {sorted(result.flags)} by the "
                f"normalization contract; week 5 verified these 98 prompts as "
                f"clean, so this is drift and not a new corpus"
            )
        normalized[key] = result.text

    canonicals: dict[str, dict[str, list[str]]] = {
        "gold": {k: phonetizer.phonetize(normalized[k]) for k in prompts}
    }
    empty = [k for k, seq in canonicals["gold"].items() if not seq]
    if empty:
        raise SystemExit(f"C_gold is empty for prompts {empty}; refusing to score against it")

    # One input per prompt, shared by every tool: the consonantal skeleton.
    # `strip_diacritics` also removes U+0670, consistent with D9's deletion rule.
    inputs = [strip_diacritics(normalized[k]) for k in prompts]
    residual = {k: v for text in inputs for k, v in unsupported_characters(text).items()}
    if residual:
        raise SystemExit(f"stripped input carries characters the phonetiser cannot map: {residual}")

    tool_meta: dict[str, dict] = {}
    for name in tools:
        tool = DIACRITIZERS[name]()
        print(f"=== {name} ===", flush=True)
        started = time.monotonic()
        batch = diacritize_long(tool, inputs)
        elapsed = time.monotonic() - started
        print(
            f"  diacritized {len(batch.texts)} prompts in {elapsed:.1f}s "
            f"({elapsed / max(len(batch.texts), 1):.3f} s/prompt)",
            flush=True,
        )

        auto_text = dict(zip(prompts, batch.texts, strict=True))
        canonicals[name] = {k: phonetizer.phonetize(auto_text[k]) for k in prompts}
        blank = [k for k, seq in canonicals[name].items() if not seq]
        if blank:
            # A tool returning nothing scores as total corruption and would look
            # like a huge effect. That is a tool failure, not a finding.
            raise SystemExit(
                f"{name} produced an empty canonical for prompts {blank}; a blank "
                f"C_auto would be scored as total label corruption and inflate dF1"
            )

        # Both DER conventions, per tool, so the dF1-vs-DER relation in
        # prediction 2 can be read off this one artifact. Reference is the gold
        # Qur'anic text, so these are accuracy rates and not agreement rates.
        pairs = [(normalized[k], auto_text[k]) for k in prompts]
        pub_ce = der_published.score_corpus(pairs, case_endings=True)
        pub_noce = der_published.score_corpus(pairs, case_endings=False)
        ours = {v.name: score_corpus(pairs, v) for v in VARIANTS}
        tool_meta[name] = {
            "provenance": tool.provenance(),
            "max_input_chars": tool.max_input_chars,
            "chunking": batch.report(),
            "diacritize_seconds": elapsed,
            "der": {
                "reference_is_gold": True,
                "published_convention": {
                    "ce_der": pub_ce.der,
                    "ce_wer": pub_ce.wer,
                    "noce_der": pub_noce.der,
                    "noce_wer": pub_noce.wer,
                },
                "our_convention": {
                    name_: {"der": score.der, "wer": score.wer} for name_, score in ours.items()
                },
            },
            # Prediction 3, at text level only -- see the module docstring for
            # why the phoneme-level share is not measured here.
            "case_ending_share": {
                "text_level_ce_minus_noce_der": pub_ce.der - pub_noce.der,
                "text_level_ratio": (pub_ce.der / pub_noce.der) if pub_noce.der else None,
                "phoneme_level": None,
                "phoneme_level_note": (
                    "not measured: needs word boundaries inside the phoneme "
                    "sequence, and per-word phonetization diverges from "
                    "whole-utterance phonetization (section 4.1.1). Deferred "
                    "rather than approximated."
                ),
            },
        }
        print(
            f"  CE DER {pub_ce.der:7.3f} (published)  "
            f"noCE DER {pub_noce.der:7.3f}  "
            f"chunked {batch.chunked_items}/{len(batch.texts)}",
            flush=True,
        )

    return canonicals, tool_meta, normalized


def scored_clips(
    prompts: Iterable[str],
    ground_truth: dict[str, MDDGroundTruth],
    predictions: dict[str, list[str]],
) -> list[str]:
    """The clip ids leg A actually scores: joined on id, and in a kept prompt.

    Leg B must be restricted to this same set before it is differenced against
    leg A, or `--limit` silently compares 1,642 clips against a handful.
    """
    keep = set(prompts)
    return sorted(
        clip_id
        for clip_id, truth in ground_truth.items()
        if clip_id in predictions and truth.prompt_index in keep
    )


def score_condition(
    canonical: dict[str, list[str]],
    ground_truth: dict[str, MDDGroundTruth],
    predictions: dict[str, list[str]],
    clips: Iterable[str],
) -> tuple[dict[str, MDDCounts], dict[str, int]]:
    """Per-prompt pooled counts for one choice of `C`, plus clips per prompt.

    Returns prompt index -> counts summed over that prompt's clips. Keeping the
    per-prompt level (rather than one corpus total) is what makes the cluster
    bootstrap and week 6 task 3's variance estimate possible without rescoring.
    """
    per_prompt: dict[str, list[MDDCounts]] = defaultdict(list)
    for clip_id in clips:
        truth = ground_truth[clip_id]
        prompt = truth.prompt_index
        per_prompt[prompt].append(
            evaluate_utterance(canonical[prompt], truth.human_annotation, predictions[clip_id])
        )
    return (
        {prompt: aggregate(counts) for prompt, counts in per_prompt.items()},
        {prompt: len(counts) for prompt, counts in per_prompt.items()},
    )


def bootstrap_delta(
    auto: dict[str, MDDCounts],
    gold: dict[str, MDDCounts],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Cluster bootstrap of dF1 = F1(C_auto) - F1(C_gold), resampling prompts.

    Both conditions are resampled with the **same** prompt draw, because they
    share `A` and `P` and differ only in `C` -- the delta is paired, and
    resampling the two independently would add variance the design does not
    have.
    """
    prompts = sorted(set(auto) & set(gold))
    point = _f1(aggregate(auto[p] for p in prompts)) - _f1(aggregate(gold[p] for p in prompts))

    rng = random.Random(seed)
    deltas: list[float] = []
    n = len(prompts)
    for _ in range(replicates):
        draw = [prompts[rng.randrange(n)] for _ in range(n)]
        auto_f1 = _f1(aggregate(auto[p] for p in draw))
        gold_f1 = _f1(aggregate(gold[p] for p in draw))
        deltas.append(auto_f1 - gold_f1)

    deltas.sort()
    lo = deltas[int(0.025 * replicates)]
    hi = deltas[min(int(0.975 * replicates), replicates - 1)]
    return {
        "delta_f1": point,
        "ci95": [lo, hi],
        "excludes_zero": (lo > 0.0) or (hi < 0.0),
        "bootstrap_sd": statistics.stdev(deltas) if replicates > 1 else None,
        "replicates": replicates,
        "seed": seed,
        "clusters": n,
        "cluster_unit": "prompt",
    }


def attribution(
    gold_canonical: dict[str, list[str]],
    auto_canonical: dict[str, list[str]],
    auto: dict[str, MDDCounts],
    gold: dict[str, MDDCounts],
    clips_per_prompt: dict[str, int],
) -> dict[str, object]:
    """Is the damage bounded by the corruption that caused it?

    Per prompt: `d` = positions where `C_gold` and `C_auto` disagree under their
    own alignment, **times the prompt's clip count**, because one corrupted
    canonical is re-used by every clip reading that prompt and can therefore do
    damage once per clip. Without that scaling the ratio is inflated by ~16.8x
    (the mean clips per prompt) and a well-behaved run looks like amplification.

    If realignment merely transmits corruption, `|delta(FR+FA)| <= d`. A ratio
    above 1 is realignment amplifying corruption; a prompt with damage but zero
    corrupted sites is a pure alignment artifact, which is the stronger check
    because it cannot be explained by scaling at all.
    """
    prompts = sorted(set(auto) & set(gold))
    total_damage = 0
    total_sites = 0
    per_prompt_sites: dict[str, int] = {}
    over_bound: list[str] = []
    artifact_prompts: list[str] = []
    for prompt in prompts:
        aligned_gold, aligned_auto = needleman_wunsch(
            gold_canonical[prompt], auto_canonical[prompt]
        )
        sites = sum(1 for g, a in zip(aligned_gold, aligned_auto, strict=True) if g != a)
        per_prompt_sites[prompt] = sites

        clips = clips_per_prompt.get(prompt, 0)
        damage = (auto[prompt].fr + auto[prompt].fa) - (gold[prompt].fr + gold[prompt].fa)
        total_damage += abs(damage)
        total_sites += sites * clips
        if sites == 0 and damage != 0:
            artifact_prompts.append(prompt)
        elif abs(damage) > sites * clips:
            over_bound.append(prompt)

    return {
        "corrupted_sites": total_sites,
        "corrupted_sites_unit": "canonical disagreement positions x clips reading that prompt",
        "abs_fr_fa_delta": total_damage,
        "damage_per_corrupted_site": (total_damage / total_sites) if total_sites else None,
        "prompts_over_bound": over_bound,
        "prompts_damaged_with_zero_corruption": artifact_prompts,
        "per_prompt_corrupted_sites": per_prompt_sites,
        "note": (
            "A ratio <= 1 means label corruption is transmitted, not amplified, "
            "by three-way Needleman-Wunsch. A nonzero delta on a prompt whose "
            "C_auto equals C_gold would be a pure alignment artifact. Diagnostic "
            "for the gate only -- not plan section 4.2's "
            "corruption-attributable FR rate, which is a weeks-10-12 deliverable."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verse-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--tools", nargs="*", default=list(DIACRITIZERS))
    parser.add_argument("--limit", type=int, default=0, help="first N prompts only (smoke run)")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--write", action="store_true", help=f"write {OUT.name}")
    args = parser.parse_args()

    for path in (args.verse_map, args.predictions):
        if not path.exists():
            raise SystemExit(f"{path} not found")

    unknown = [t for t in args.tools if t not in DIACRITIZERS]
    if unknown:
        raise SystemExit(f"unknown tools {unknown}; registered: {sorted(DIACRITIZERS)}")

    prompt_text = load_prompt_text(args.verse_map)
    if args.limit:
        prompt_text = {k: prompt_text[k] for k in sorted(prompt_text)[: args.limit]}

    raw_predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = {k: parse_phoneme_sequence(v) for k, v in raw_predictions.items()}
    ground_truth = load_ground_truth()

    print("=== the label-path column — week-06 exit criterion, clause 1 ===")
    print(f"prompts     : {len(prompt_text)} (C varies per prompt)")
    print(f"clips       : {len(predictions)} predictions, {len(ground_truth)} ground truth")
    print("fixed       : A = annotation_phoneme_string, P = official checkpoint")
    print("varying     : C only")
    print("C_gold rule : D9 — phonetize(recovered_text), U+0670 deleted")
    print(f"order       : pre-declared {' < '.join(EXPECTED_ORDER)}\n")

    canonicals, tool_meta, _normalized = build_canonicals(prompt_text, args.tools)

    # Leg B: the anchor. Scored against the benchmark's own canonical, which is
    # per clip rather than per prompt, so it does not go through `canonicals`.
    # Two numbers, because they answer different questions: over **all** clips
    # it is the reproduction of the published condition, and over the clips leg
    # A scores it is the quantity that can be differenced against leg A.
    def _anchor(clip_ids: Iterable[str]) -> MDDCounts:
        return aggregate(
            evaluate_utterance(
                ground_truth[c].canonical, ground_truth[c].human_annotation, predictions[c]
            )
            for c in clip_ids
        )

    clips = scored_clips(prompt_text, ground_truth, predictions)
    all_clips = sorted(c for c in ground_truth if c in predictions)
    anchor_counts = _anchor(all_clips)
    anchor_f1 = _f1(anchor_counts)
    anchor_ok = abs(anchor_f1 - ANCHOR_F1) <= ANCHOR_TOLERANCE
    subset_counts = anchor_counts if len(clips) == len(all_clips) else _anchor(clips)
    subset_f1 = _f1(subset_counts)

    print("=== leg B: the anchor (C = reference_phoneme_string) ===")
    print(
        f"  F1 {anchor_f1:.4f} over all {len(all_clips)} clips "
        f"vs published {ANCHOR_F1:.4f} -> {'OK' if anchor_ok else 'DRIFT'}"
    )
    if len(clips) != len(all_clips):
        print(f"  F1 {subset_f1:.4f} over the {len(clips)} clips leg A scores (--limit active)")
    if not anchor_ok:
        print(
            "  WARNING: the harness no longer reproduces the published condition, "
            "so no delta below should be reported until that is explained.",
            flush=True,
        )

    gold_per_prompt, clips_per_prompt = score_condition(
        canonicals["gold"], ground_truth, predictions, clips
    )
    gold_total = aggregate(gold_per_prompt.values())
    print("\n=== leg A: C_gold (our phonetizer on gold Qur'anic text) ===")
    print(f"  F1 {_f1(gold_total):.4f}   {_counts_dict(gold_total)}")
    print(
        f"  leg B minus leg A = {subset_f1 - _f1(gold_total):+.4f} "
        f"(the phonetizer's own price, same {len(clips)} clips both sides)"
    )

    results: dict[str, dict] = {}
    print("\n=== leg A: per tool ===")
    for name in args.tools:
        per_prompt, _ = score_condition(canonicals[name], ground_truth, predictions, clips)
        total = aggregate(per_prompt.values())
        boot = bootstrap_delta(per_prompt, gold_per_prompt, args.bootstrap, args.seed)
        results[name] = {
            **tool_meta[name],
            "metrics": _metrics_dict(total),
            "delta_vs_gold": {
                "f1": boot["delta_f1"],
                "ci95": boot["ci95"],
                "excludes_zero": boot["excludes_zero"],
                "bootstrap_sd": boot["bootstrap_sd"],
                "ta": total.ta - gold_total.ta,
                "fr": total.fr - gold_total.fr,
                "fa": total.fa - gold_total.fa,
                "tr": total.tr - gold_total.tr,
            },
            "bootstrap": {k: boot[k] for k in ("replicates", "seed", "clusters", "cluster_unit")},
            "attribution": attribution(
                canonicals["gold"],
                canonicals[name],
                per_prompt,
                gold_per_prompt,
                clips_per_prompt,
            ),
            "per_prompt_counts": {p: _counts_dict(c) for p, c in sorted(per_prompt.items())},
        }
        ci = boot["ci95"]
        print(
            f"  {name:<10} F1 {_f1(total):.4f}  dF1 {boot['delta_f1']:+.4f}  "
            f"95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
            f"{'excludes 0' if boot['excludes_zero'] else 'includes 0'}",
            flush=True,
        )

    # --- the gate, read out rather than left to the reader ---
    print("\n=== clause 1, against the criteria fixed before the run ===")
    worst = results.get(WORST_TOOL)
    worst_ok = bool(worst and worst["delta_vs_gold"]["excludes_zero"])
    print(
        f"  CI excludes zero for {WORST_TOOL}: "
        f"{'YES' if worst_ok else 'NO' if worst else 'NOT RUN'}"
    )

    ranked = [(n, abs(results[n]["delta_vs_gold"]["f1"])) for n in EXPECTED_ORDER if n in results]
    monotone = [n for n, _ in ranked] == [n for n, _ in sorted(ranked, key=lambda r: r[1])]
    for name, magnitude in ranked:
        print(f"    {name:<10} |dF1| {magnitude:.4f}")
    print(f"  monotone in the pre-declared DER order: {'YES' if monotone else 'NO'}")
    if not monotone:
        print(
            "    An inversion is the most interesting possible failure here: three "
            "unrelated instruments already agree on that ordering, so this is "
            "first evidence for RQ3's claim that DER is miscalibrated."
        )

    passed = worst_ok and monotone and len(args.tools) == len(EXPECTED_ORDER)
    print(f"  CLAUSE 1: {'PASS' if passed else 'NOT MET'}")
    if not passed and not args.limit:
        print(
            "    If all four CIs include zero this is a null on the paper's spine, "
            "and plan section 5's pre-committed position binds: the contribution "
            "becomes the protocol, the metric and the bias correction. It does not "
            "become a different paper. Distinguish it from a power failure with "
            "week 6 task 3's MDE before writing anything."
        )

    signs = {n: results[n]["delta_vs_gold"]["f1"] for n in results}
    negative = all(v < 0 for v in signs.values())
    print(
        f"\n  prediction 1 (dF1 negative, FA-driven): "
        f"{'holds' if negative else 'FALSIFIED'} — signs "
        f"{ {n: round(v, 4) for n, v in signs.items()} }"
    )
    for name in results:
        delta = results[name]["delta_vs_gold"]
        movers = {k: delta[k] for k in ("ta", "fr", "fa", "tr")}
        dominant = max(movers, key=lambda k: abs(movers[k]))
        print(f"    {name:<10} dominant moving cell: {dominant} ({movers[dominant]:+d}) {movers}")

    payload = {
        "purpose": "docs/weeks/week-06.md exit criterion, clause 1 — the label-path column",
        "generated_by": "scripts/measure_label_path.py",
        "git_sha": _git_sha(),
        "design": {
            "fixed": {
                "A": "QuranMB.v2 annotation_phoneme_string",
                "P": str(args.predictions.relative_to(REPO_ROOT)),
                "P_note": "official organizers' checkpoint, reproduces published F1 to +-0.0001",
            },
            "varying": "C only",
            "join": "prompt_index, from id.split('_')[1][:5]",
            "pooling": "counts summed, compute_metrics applied once (never averaged ratios)",
            "cell": (
                "top-right against top-left of plan section 1's 2x2: label-path "
                "corruption alone, model unimpaired. The input-path row needs the "
                "text-dependent arm and is weeks 7-9, so only the scoring column "
                "exists here."
            ),
        },
        "c_gold": {
            "decision": "D9, docs/msa-arm.md section 3.3b, 14 Oct 2026",
            "rule": "phonetize(normalize(recovered_text)), U+0670 deleted",
            "phonetizer": Phonetizer("prescriptive").provenance(),
            "dagger_alef_rule": "delete (54/98 phoneme_exact, vs 48/98 for U+0670 -> alef)",
            "source": str(args.verse_map.relative_to(REPO_ROOT)),
        },
        "leg_b_anchor": {
            "c": "QuranMB.v2 reference_phoneme_string (the benchmark's own canonical)",
            "f1": anchor_f1,
            "f1_clips": len(all_clips),
            "published_f1": ANCHOR_F1,
            "reproduces": anchor_ok,
            "metrics": _metrics_dict(anchor_counts),
            "f1_on_scored_clips": subset_f1,
            "leg_b_minus_leg_a_f1": subset_f1 - _f1(gold_total),
            "note": (
                "The gap is our phonetizer's disagreement with the uploader's "
                "(44/98 prompts, tajwid conventions), reported separately so it "
                "is not inside any dF1. Belongs in the limitations."
            ),
        },
        "c_gold_cell": {
            "metrics": _metrics_dict(gold_total),
            "per_prompt_counts": {p: _counts_dict(c) for p, c in sorted(gold_per_prompt.items())},
        },
        "prompts": len(prompt_text),
        "clips_scored": len(clips),
        "clips_per_prompt": clips_per_prompt,
        "limited": bool(args.limit),
        "expected_order": list(EXPECTED_ORDER),
        "predictions_registered": {
            "1": "dF1 is negative and the mechanism is FA inflation",
            "2": "|dF1| is monotone in CE DER: catt-eo < catt-ed < shakkala < mishkal",
            "3": (
                "the case-ending share of label damage is high on this arm — a "
                "caveat, not a finding; Qur'anic recitation pronounces case "
                "endings, so plan section 1's MSA pausal argument does not "
                "transfer. Measured at text level only; see results.*."
                "case_ending_share.phoneme_level"
            ),
        },
        "clause_1": {
            "worst_tool": WORST_TOOL,
            "worst_tool_ci_excludes_zero": worst_ok,
            "monotone_in_der": monotone,
            "passed": passed,
            "criteria": (
                "CI excludes zero for at least the worst diacritizer, and the "
                "per-tool point estimates are monotone in DER. Two-sided: plan "
                "section 1 says nobody knows the sign."
            ),
        },
        "results": results,
    }

    if args.write:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")
    else:
        print(f"\n(dry run — pass --write to produce {OUT.name})")


if __name__ == "__main__":
    main()
