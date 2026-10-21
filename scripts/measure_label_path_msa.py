"""The label-path column on the MSA arm — week-08 task 8, started and stopped on purpose.

The Qur'anic arm's version is `scripts/measure_label_path.py`. This is not a
fork of it: the two arms differ in the three things that matter, so sharing a
driver would mean a driver made of branches.

| | Qur'anic arm | MSA arm |
|---|---|---|
| `C_gold` | `phonetize(recovered_text)`, D9 | **human annotation**, not yet closed |
| `A` | real, `annotation_phoneme_string` | **does not exist** (D1 dropped it) |
| convention | prescriptive, as recited | **pausal** (protocol section 3) |
| cluster | 98 prompts, ~16.8 clips each | one utterance = one clip = one `C` |

## Why this stops where it stops

Week-08 task 8 is scoped as *start*: "build to the point where closing
annotation is the only remaining input, then stop." That boundary is real and
it is not the annotation of `A` -- it is `C_gold`. Everything on the `C_auto`
side needs no human at all, costs the whole diacritizer runtime, and owns every
refusal path, so it runs now and is cached. When the 500 rows close, the
remaining step is one invocation with `--c-gold`.

So this script has two phases and will take the first one today:

1. **`C_auto`, five conditions, 500 rows.** The four evaluated tools on
   `strip_diacritics(normalize(sentence))`, plus the organizers'
   `tashkeel_sentence` -- which is not a fifth tool but **the naive-deployment
   condition itself**: it is the column a group would score against if they
   evaluated on MSA without thinking about any of this, and plan section 1's
   bottom-right cell is defined by it.
2. **`C_gold`, and the scoring.** Blocked on annotation. Implemented, refuses
   cleanly, and names what is missing.

## `A := C_gold`, which is D2 and not a shortcut taken here

D1 (`docs/msa-arm.md:640`) decided the MSA arm is label-path only and does not
annotate `A`. D2 (`:946`) then decided that `A := C_gold` -- assuming the
speaker said what the reference says -- is carried as a **bounded** assumption
rather than as a sensitivity row, because the pilot's deviation band spans
2.3%-19.8% and a row that wide would swallow the 0.0907 MDE whole. The bound
itself is weeks 10-11 and is obtained on the *Qur'anic* arm, where both `A` and
`C_gold` exist. Nothing here anticipates it.

What this file must do is apply the substitution where the precedent already
applies it: `evaluate_utterance(C, C_gold, P)`. That is the same call shape as
`scripts/tts_control_baserate.py`, where `A == C` is an identity rather than an
assumption, and the quantity it yields is the same one -- a false-rejection
rate against a reference with no real positives in it.

**Which forces the headline to be ΔFRR and not ΔF1, and that is worth saying
before the number exists.** Under `A := C_gold` the gold cell has `C == A` at
every position, so `TR` and `FA` are **zero by construction** and its F1 is 0
via `_safe_div`, not a measurement. `ΔF1` against that is not the bias in a
naive evaluation; it is an artifact of the substitution. The MSA arm's
quantities are:

* gold cell `FRR` -- pure model error on real MSA speech, the week-4 CV-Ar dev
  figure (0.0708) re-measured on this block;
* auto cell `TR` -- apparent detections **manufactured** by reference
  corruption, which is contribution 4, the one D1 explicitly preserved;
* `ΔFRR` and `Δprecision` between them.

F1 is still written to the artifact, flagged degenerate, so a later reader
cannot pick it up without seeing why it is not the number.

    uv run python scripts/measure_label_path_msa.py                  # phase 1
    uv run python scripts/measure_label_path_msa.py --tools catt-eo  # phase 1, one tool
    uv run python scripts/measure_label_path_msa.py --c-gold run/msa_c_gold_a1.json \
        --c-gold run/msa_c_gold_a2.json --write                      # phase 2, once closed
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from arabic_mdd.data.normalize import normalize, strip_diacritics, unsupported_characters
from arabic_mdd.data.phonemes import parse_phoneme_sequence
from arabic_mdd.data.phonetizer import Phonetizer
from arabic_mdd.diacritizers import DIACRITIZERS
from arabic_mdd.diacritizers.chunking import diacritize_long
from arabic_mdd.metrics.hierarchical import (
    MDDCounts,
    aggregate,
    compute_metrics,
    evaluate_utterance,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BLOCK = REPO_ROOT / "run" / "msa_annotation_block.json"
PREDICTIONS = REPO_ROOT / "run" / "cv_dev_predictions.json"
C_AUTO_CACHE = REPO_ROOT / "run" / "msa_c_auto.json"
OUT = REPO_ROOT / "run" / "msa_label_path.json"

#: The organizers' own vowelized column, carried in the frame. Named as a
#: condition rather than as a tool because that is what it is: plan section 1's
#: bottom-right cell is "what a group would report if they evaluated on MSA with
#: an automatic diacritizer and did not think about any of this", and this is
#: that diacritizer. `docs/annotation-protocol-msa.md` section 2 is explicit
#: that `tashkeel_sentence` is `C_auto` and must never seed `C_gold`.
VOWELIZER = "organizers-vowelizer"

#: Week-4's CV-Ar dev false-rejection rate over all 2,588 dev utterances
#: (`insights/week-04.md`), and the Qur'anic arm's, for scale. The gold cell's
#: FRR on this 500-row subset should land near the first.
CV_DEV_FRR = 0.0708
QURANMB_FRR = 0.1241

BOOTSTRAP_REPLICATES = 10_000
#: Same seed as the Qur'anic arm, deliberately. The two arms resample different
#: objects, so sharing it buys nothing statistically -- it just means neither
#: artifact carries a seed the other does not explain.
BOOTSTRAP_SEED = 20261014


def load_block(path: Path) -> list[dict]:
    """The committed 500-row frame. Not re-drawn, not filtered, not reordered."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    block = payload["block"]
    if payload.get("seed") != 20261001 or payload.get("block_size") != 500:
        raise SystemExit(
            f"{path.name} reports seed {payload.get('seed')} and block_size "
            f"{payload.get('block_size')}; the frame is committed as seed 20261001 / 500 rows "
            "and a different draw is a different experiment"
        )
    return block


def build_c_auto(
    rows: list[dict],
    tools: list[str],
) -> tuple[dict[str, dict[str, str]], dict[str, dict]]:
    """Diacritize each row's bare `sentence` with every tool.

    Returns `(texts, meta)` where `texts` maps condition -> id -> diacritized
    Arabic. Phonetization happens later, so the cache stays debuggable as text:
    a wrong phoneme sequence is unreadable, a wrong diacritization is not.

    The input is one per row and shared by every tool, which is the same
    discipline the Qur'anic arm uses -- a per-tool input would let a tool be
    scored on text no other tool saw.
    """
    ids = [r["id"] for r in rows]
    inputs = [strip_diacritics(normalize(r["sentence"]).text) for r in rows]
    residual = {k: v for text in inputs for k, v in unsupported_characters(text).items()}
    if residual:
        raise SystemExit(f"stripped input carries characters the phonetiser cannot map: {residual}")

    texts: dict[str, dict[str, str]] = {
        VOWELIZER: {r["id"]: normalize(r["tashkeel_sentence"]).text for r in rows}
    }
    meta: dict[str, dict] = {
        VOWELIZER: {
            "provenance": "IqraEval/Iqra_train dev `tashkeel_sentence`, as published",
            "note": (
                "not run here — carried in the frame. This is the naive-deployment "
                "condition, and docs/msa-arm.md D5 measures C_gold against it."
            ),
        }
    }

    for name in tools:
        tool = DIACRITIZERS[name]()
        print(f"=== {name} ===", flush=True)
        started = time.monotonic()
        batch = diacritize_long(tool, inputs)
        elapsed = time.monotonic() - started
        print(
            f"  diacritized {len(batch.texts)} rows in {elapsed:.1f}s "
            f"({elapsed / max(len(batch.texts), 1):.3f} s/row), "
            f"chunked {batch.chunked_items}",
            flush=True,
        )
        texts[name] = dict(zip(ids, batch.texts, strict=True))
        meta[name] = {
            "provenance": tool.provenance(),
            "max_input_chars": tool.max_input_chars,
            "chunking": batch.report(),
            "diacritize_seconds": elapsed,
        }
    return texts, meta


def phonetize_conditions(
    texts: dict[str, dict[str, str]],
    ids: list[str],
) -> dict[str, dict[str, list[str]]]:
    """Pausal phonetization of every condition.

    **Pausal on both sides of the delta**, per `docs/annotation-protocol-msa.md`
    section 3: R2 frees the utterance-final mark and R3 frees tanwin, and the
    annotators were instructed accordingly. Phonetizing `C_auto` under the
    prescriptive convention instead would bill the convention difference to the
    diacritizer -- the same mistake D9 caught on the Qur'anic arm between legs A
    and B, where it was 44 of 98 prompts of confound.
    """
    phonetizer = Phonetizer("pausal")
    out: dict[str, dict[str, list[str]]] = {}
    for condition, by_id in texts.items():
        sequences = {k: phonetizer.phonetize(by_id[k]) for k in ids}
        blank = [k for k, seq in sequences.items() if not seq]
        if blank:
            # A blank canonical scores as total corruption and would look like a
            # huge effect. That is a tool failure, not a finding.
            raise SystemExit(
                f"{condition} phonetizes to nothing for rows {blank}; a blank C "
                "would be scored as total label corruption and inflate the delta"
            )
        out[condition] = sequences
    return out


def load_c_gold(paths: list[Path], ids: set[str]) -> dict[str, str]:
    """The annotators' `final` per id, ingest-gated, or a refusal naming the gap.

    `arabic_mdd.data.annotation.validate` is the gate and it is not re-run here
    -- `scripts/annotate_msa.py --validate` owns that, and duplicating it would
    give two answers to one question. What this does is the join: every row of
    the frame must have exactly one usable `final`, and the 50 crossover rows
    have two, so one of them has to be chosen rather than silently last-wins.
    """
    finals: dict[str, list[tuple[str, str]]] = {}
    for path in paths:
        if not path.exists():
            raise SystemExit(f"{path} not found; run scripts/annotate_msa.py first")
        for record in json.loads(path.read_text(encoding="utf-8")):
            if record.get("status") != "done":
                continue
            adjudicated = record.get("adjudicated")
            final = (adjudicated or {}).get("final") or record.get("final") or ""
            if final.strip():
                finals.setdefault(str(record["id"]), []).append((path.name, final))

    missing = sorted(ids - set(finals))
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(ids)} frame rows have no usable `final` yet "
            f"(first few: {missing[:5]}). C_gold is the only remaining input to this "
            "measurement; the C_auto half is already cached. Finish the block, then "
            "re-run with the same --c-gold arguments."
        )

    disagreeing = sorted(key for key, entries in finals.items() if len({f for _, f in entries}) > 1)
    if disagreeing:
        raise SystemExit(
            f"{len(disagreeing)} rows carry two different `final` values across annotators "
            f"(first few: {disagreeing[:5]}). These are the crossover rows and the "
            "disagreement is section 8's Δ, not noise to average away. Which C_gold is "
            "released — pre- or post-adjudication — is an open item in "
            "docs/annotation-protocol-msa.md; adjudicate them before scoring rather "
            "than letting file order decide."
        )
    return {key: entries[0][1] for key, entries in finals.items()}


def score(
    canonical: dict[str, list[str]],
    gold: dict[str, list[str]],
    predictions: dict[str, list[str]],
    ids: list[str],
) -> dict[str, MDDCounts]:
    """Per-utterance counts for one choice of `C`, with `A := C_gold` (D2).

    Per-utterance rather than one total because the bootstrap resamples
    utterances and must re-pool, not resample ratios.
    """
    return {key: evaluate_utterance(canonical[key], gold[key], predictions[key]) for key in ids}


def bootstrap_frr(
    auto: dict[str, MDDCounts],
    gold: dict[str, MDDCounts],
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Paired utterance bootstrap of ΔFRR = FRR(C_auto) − FRR(C_gold).

    The cluster unit is the **utterance**, and here that is not a weakening of
    the Qur'anic arm's prompt clustering -- it is the same rule. `C` is constant
    within a cluster on both arms; on QuranMB.v2 one prompt carries ~16.8 clips,
    while the MSA block is 500 distinct sentences with one clip each, so the
    cluster and the clip coincide.

    FRR rather than F1 because under `A := C_gold` the gold cell has no
    positives, so its F1 is 0 by construction -- see the module docstring.
    """
    keys = sorted(set(auto) & set(gold))

    def frr(counts: dict[str, MDDCounts], draw: list[str]) -> float:
        return compute_metrics(aggregate(counts[k] for k in draw)).frr

    point = frr(auto, keys) - frr(gold, keys)
    rng = random.Random(seed)
    n = len(keys)
    deltas: list[float] = []
    for _ in range(replicates):
        # One draw used for both conditions: same P, same A, only C differs, so
        # the delta is paired and independent draws would add variance the
        # design does not have.
        draw = [keys[rng.randrange(n)] for _ in range(n)]
        deltas.append(frr(auto, draw) - frr(gold, draw))
    deltas.sort()
    lo = deltas[int(0.025 * replicates)]
    hi = deltas[min(int(0.975 * replicates), replicates - 1)]
    return {
        "delta_frr": point,
        "ci95": [lo, hi],
        "excludes_zero": (lo > 0.0) or (hi < 0.0),
        "bootstrap_sd": statistics.stdev(deltas) if replicates > 1 else None,
        "replicates": replicates,
        "seed": seed,
        "clusters": n,
        "cluster_unit": "utterance",
    }


def _metrics_dict(counts: MDDCounts) -> dict[str, object]:
    m = compute_metrics(counts)
    return {
        "counts": {
            "ta": counts.ta,
            "fr": counts.fr,
            "fa": counts.fa,
            "tr": counts.tr,
            "correct_diagnosis": counts.correct_diagnosis,
            "error_diagnosis": counts.error_diagnosis,
        },
        "frr": m.frr,
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "f1_note": (
            "degenerate under A := C_gold — the gold cell has no positives, so F1 "
            "is 0 by construction and ΔF1 is an artifact of the substitution, not "
            "the bias in a naive evaluation. Read FRR and TR."
        ),
        "detection_accuracy": m.detection_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", type=Path, default=BLOCK)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--tools", nargs="*", default=list(DIACRITIZERS))
    parser.add_argument(
        "--c-gold",
        type=Path,
        action="append",
        default=[],
        help="an annotator's msa_c_gold_*.json. Phase 2; omit it to build C_auto only.",
    )
    parser.add_argument("--cache", type=Path, default=C_AUTO_CACHE)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="re-run the diacritizers even if the cache is valid",
    )
    parser.add_argument("--limit", type=int, default=0, help="first N frame rows (smoke run)")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    unknown = [t for t in args.tools if t not in DIACRITIZERS]
    if unknown:
        raise SystemExit(f"unknown tools {unknown}; registered: {sorted(DIACRITIZERS)}")

    rows = load_block(args.block)
    if args.limit:
        rows = rows[: args.limit]

    # The 3 Latin rows cannot be scored fairly: Latin letters are valid
    # Buckwalter symbols, so the phonetizer invents phonemes for them
    # (`data/normalize.py` opening note). Excluding beats cleaning, and the
    # count is reported rather than quietly applied.
    unfair = [r["id"] for r in rows if not normalize(r["sentence"]).fair]
    rows = [r for r in rows if r["id"] not in set(unfair)]
    ids = [r["id"] for r in rows]

    raw_predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    no_prediction = [key for key in ids if key not in raw_predictions]
    if no_prediction:
        raise SystemExit(
            f"{len(no_prediction)} frame rows have no prediction in "
            f"{args.predictions.name} (first few: {no_prediction[:5]})"
        )
    predictions = {key: parse_phoneme_sequence(raw_predictions[key]) for key in ids}

    print("=== the label-path column — MSA arm (week-08 task 8) ===")
    print(f"frame       : {len(ids)} rows scored, {len(unfair)} excluded unfair {unfair}")
    print(f"P           : {args.predictions.name}")
    print("A           : := C_gold (D2) — no annotation of A on this arm (D1)")
    print("convention  : pausal, both sides (protocol section 3, R1-R4)")
    print("cluster     : utterance (one clip, one sentence, one C)\n")

    cached = (
        json.loads(args.cache.read_text(encoding="utf-8"))
        if args.cache.exists() and not args.refresh_cache
        else None
    )
    conditions = [VOWELIZER, *args.tools]
    reusable = bool(
        cached
        and set(conditions) <= set(cached["texts"])
        and all(set(ids) <= set(cached["texts"][c]) for c in conditions)
    )
    if reusable:
        print(f"reusing {args.cache.name} (diacritizers not re-run)\n")
        texts = {c: {k: cached["texts"][c][k] for k in ids} for c in conditions}
        meta = cached["meta"]
    else:
        texts, meta = build_c_auto(rows, args.tools)
        payload = {
            "purpose": "C_auto for the MSA label path — the half that needs no annotation",
            "generated_by": "scripts/measure_label_path_msa.py",
            "input": "strip_diacritics(normalize(sentence)) from run/msa_annotation_block.json",
            "rows": len(ids),
            "conditions": conditions,
            "meta": meta,
            "texts": texts,
        }
        args.cache.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.cache}")

    canonicals = phonetize_conditions(texts, ids)
    print("\n=== C_auto, phonetized (pausal) ===")
    for condition in conditions:
        lengths = [len(canonicals[condition][k]) for k in ids]
        print(f"  {condition:<22} {sum(lengths):6d} phonemes, mean {statistics.mean(lengths):.1f}")

    if not args.c_gold:
        print(
            "\n=== phase 2 not run: C_gold is the only remaining input ===\n"
            "  The 500 rows are annotated by scripts/annotate_msa.py, which writes\n"
            "  run/msa_c_gold_{a1,a2}.json. Validate with --validate, adjudicate the\n"
            "  50 crossover rows, then re-run this with --c-gold for each file.\n"
            "  Nothing else is missing: P joins 1:1, C_auto is cached above, and the\n"
            "  scoring path below it is implemented.\n"
            "  Note before the numbers exist: under A := C_gold the gold cell has no\n"
            "  positives, so this arm's quantities are ΔFRR and manufactured TR, not\n"
            "  ΔF1. See the module docstring."
        )
        return

    gold_text = load_c_gold(args.c_gold, set(ids))
    phonetizer = Phonetizer("pausal")
    gold = {k: phonetizer.phonetize(normalize(gold_text[k]).text) for k in ids}
    blank = [k for k, seq in gold.items() if not seq]
    if blank:
        raise SystemExit(f"C_gold phonetizes to nothing for rows {blank}")

    gold_counts = score(gold, gold, predictions, ids)
    gold_total = aggregate(gold_counts.values())
    gold_metrics = compute_metrics(gold_total)
    print("\n=== gold cell: C = C_gold = A — pure model error on real MSA speech ===")
    print(f"  FRR {gold_metrics.frr:.4f}   (CV-Ar dev {CV_DEV_FRR:.4f}, QuranMB {QURANMB_FRR:.4f})")
    print(f"  TR {gold_total.tr} and FA {gold_total.fa} are 0 by construction under A := C_gold")

    results: dict[str, dict] = {}
    print("\n=== per condition ===")
    for condition in conditions:
        counts = score(canonicals[condition], gold, predictions, ids)
        total = aggregate(counts.values())
        boot = bootstrap_frr(counts, gold_counts, args.bootstrap, args.seed)
        results[condition] = {
            **meta.get(condition, {}),
            "metrics": _metrics_dict(total),
            "delta_vs_gold": {
                "frr": boot["delta_frr"],
                "ci95": boot["ci95"],
                "excludes_zero": boot["excludes_zero"],
                "bootstrap_sd": boot["bootstrap_sd"],
                "manufactured_tr": total.tr - gold_total.tr,
                "manufactured_fa": total.fa - gold_total.fa,
                "ta": total.ta - gold_total.ta,
                "fr": total.fr - gold_total.fr,
            },
            "bootstrap": {k: boot[k] for k in ("replicates", "seed", "clusters", "cluster_unit")},
            "per_utterance_counts": {k: counts[k].__dict__ for k in ids},
        }
        ci = boot["ci95"]
        print(
            f"  {condition:<22} FRR {compute_metrics(total).frr:.4f}  "
            f"dFRR {boot['delta_frr']:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
            f"manufactured TR {total.tr - gold_total.tr:+d}",
            flush=True,
        )

    payload = {
        "purpose": "docs/weeks/week-08.md task 8 — the MSA arm's label-path column",
        "generated_by": "scripts/measure_label_path_msa.py",
        "design": {
            "fixed": {"A": ":= C_gold (D2)", "P": str(args.predictions)},
            "varying": "C only",
            "convention": "pausal on both sides (docs/annotation-protocol-msa.md section 3)",
            "cluster_unit": "utterance",
            "a_substitution": (
                "D1 dropped annotation of A on this arm; D2 carries A := C_gold as a "
                "bounded assumption, bounded on the Qur'anic arm in weeks 10-11. The "
                "gold cell therefore has no positives and ΔF1 is not this arm's "
                "quantity — ΔFRR and manufactured TR are."
            ),
            "naive_deployment_cell": (
                f"{VOWELIZER} is plan section 1's bottom-right condition: the column a "
                "group would score against with no thought given to any of this."
            ),
        },
        "rows_scored": len(ids),
        "rows_excluded_unfair": unfair,
        "gold_cell": {"metrics": _metrics_dict(gold_total)},
        "cv_dev_frr": CV_DEV_FRR,
        "quranmb_frr": QURANMB_FRR,
        "c_gold_sources": [str(p) for p in args.c_gold],
        "results": results,
    }
    if args.write:
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.output}")
    else:
        print(f"\n(dry run — pass --write to produce {args.output.name})")


if __name__ == "__main__":
    main()
