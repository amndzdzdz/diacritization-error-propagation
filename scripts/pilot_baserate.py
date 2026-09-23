"""Score the completed listening pilot into a reportable base rate.

Reads `run/msa_pilot_annotated.json` (produced by `pilot_annotate.py`) and
answers `docs/msa-arm.md` §3.3's open question: does Common Voice Arabic
read speech contain enough genuine mispronunciations to support the MSA
arm's RQ1?

Two things here that hand computation reliably gets wrong.

**The 50 are not a random sample.** 25 were drawn from the 100
highest-disagreement utterances, specifically to surface errors if any
exist; the other 25 are uniform. Pooling all 50 overstates the base rate
badly. Only the `uniform_random` stratum estimates it. The enriched stratum
is reported separately, where it answers a different question -- when model
and reference disagree, who is wrong -- which feeds §3.4.1 rather than §3.3.

**The point estimate alone is not reportable at n=25.** Every rate carries
a Clopper-Pearson 95% interval, computed from the stdlib so this needs no
new dependency.

Strict vs lenient is the protocol's central distinction: `dialect`, `case`
and `hesit` words are excluded from the lenient rate, because counting
regional accent as mispronunciation would push the rate toward 100% and
make it meaningless. Both are reported; the pre-committed decision bands in
the protocol are read against the **lenient word-level** rate.

    uv run python scripts/pilot_baserate.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Tags that do not count as mispronunciation under the lenient definition.
# See docs/annotation-protocol-pilot.md §4.
LENIENT_EXCLUDED = {"dialect", "case", "hesit"}

QURANMB_FR_RATE = 0.1241


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI by bisection on the binomial CDF.

    Avoids a scipy dependency for ~20 lines of bisection. Exact rather than
    normal-approximate matters here: at n=25 with k=0 or 1 the normal
    approximation is simply wrong.
    """
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(200):
            mid = (a + b) / 2
            if 1 - _binom_cdf(k - 1, n, mid) > alpha / 2:
                b = mid
            else:
                a = mid
        lo = a
    hi = 1.0
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(200):
            mid = (a + b) / 2
            if _binom_cdf(k, n, mid) < alpha / 2:
                b = mid
            else:
                a = mid
        hi = b
    return (lo, hi)


def _fmt(k: int, n: int) -> str:
    if n == 0:
        return f"{'n/a':>10s}  (no data)"
    lo, hi = _clopper_pearson(k, n)
    return f"{k:>4d}/{n:<5d} {k / n:>7.2%}   95% CI [{lo:.2%}, {hi:.2%}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(REPO_ROOT / "run" / "msa_pilot_sample.json"))
    parser.add_argument("--annotated", default=str(REPO_ROOT / "run" / "msa_pilot_annotated.json"))
    args = parser.parse_args()

    manifest = {r["id"]: r for r in json.loads(Path(args.manifest).read_text(encoding="utf-8"))}
    ann_path = Path(args.annotated)
    if not ann_path.exists():
        raise SystemExit(
            f"No annotations at {ann_path}.\nRun: uv run python scripts/pilot_annotate.py"
        )
    annotations = json.loads(ann_path.read_text(encoding="utf-8"))

    print(f"{len(annotations)}/{len(manifest)} utterances annotated")
    if len(annotations) < len(manifest):
        print("  WARNING: incomplete. Rates below are provisional.\n")

    by_stratum: dict[str, list[dict]] = {"uniform_random": [], "high_disagreement": []}
    unusable: list[dict] = []
    all_tags: Counter[str] = Counter()

    for rec in annotations:
        row = manifest.get(rec["id"])
        if row is None:
            continue
        all_tags.update(rec.get("tags") or [])
        if rec["verdict"] == "unusable":
            unusable.append(rec)
            continue
        by_stratum.setdefault(row["stratum"], []).append(rec)

    if unusable:
        print(f"  excluded as unusable: {len(unusable)} ({[r['id'] for r in unusable]})")
    if all_tags:
        print(f"  tag counts: {dict(all_tags.most_common())}")

    for stratum in ("uniform_random", "high_disagreement"):
        recs = by_stratum.get(stratum) or []
        if not recs:
            continue
        header = {
            "uniform_random": "UNIFORM STRATUM  --  this is the base rate",
            "high_disagreement": "ENRICHED STRATUM  --  NOT the base rate (diagnostic only)",
        }[stratum]
        print("\n" + "=" * 72)
        print(header)
        print("=" * 72)

        n_utt = len(recs)
        n_words = sum(r.get("n_words") or 0 for r in recs)

        strict_utt = sum(1 for r in recs if r["verdict"] == "error")
        strict_words = sum(len(r.get("error_words") or []) for r in recs)

        # Lenient: a word is excluded if the utterance's tags are entirely
        # within the excluded set. Tags are recorded per utterance, not per
        # word, so this is exact when an utterance has one error type and
        # conservative (counts the word) when it mixes types -- which is the
        # safe direction for a base-rate claim.
        lenient_utt = 0
        lenient_words = 0
        for r in recs:
            if r["verdict"] != "error":
                continue
            tags = set(r.get("tags") or [])
            if tags and tags <= LENIENT_EXCLUDED:
                continue
            lenient_utt += 1
            lenient_words += len(r.get("error_words") or [])

        print(f"\n  utterances: {n_utt}    words: {n_words}")
        print("\n  STRICT (every marked word counts)")
        print(f"    utterance-level  {_fmt(strict_utt, n_utt)}")
        print(f"    word-level       {_fmt(strict_words, n_words)}")
        print("\n  LENIENT (dialect/case/hesitation excluded)")
        print(f"    utterance-level  {_fmt(lenient_utt, n_utt)}")
        print(f"    word-level       {_fmt(lenient_words, n_words)}")

        if stratum == "uniform_random":
            rate = lenient_words / n_words if n_words else 0.0
            lo, hi = _clopper_pearson(lenient_words, n_words) if n_words else (0.0, 1.0)
            print("\n" + "-" * 72)
            print("  VERDICT (pre-committed bands, lenient word-level)")
            print("-" * 72)
            print(f"  base rate = {rate:.2%}   95% CI [{lo:.2%}, {hi:.2%}]")
            print(f"  QuranMB FR rate for scale: {QURANMB_FR_RATE:.2%}")
            if rate >= 0.03:
                print("\n  >= 3%: A REAL BASE RATE EXISTS.")
                print("  §3.3 option 1 proceeds -- annotate 300-500 utterances and")
                print("  measure RQ1 on the MSA arm directly.")
            elif rate >= 0.01:
                print("\n  1-3%: MARGINAL.")
                print("  Option 1 is viable only with a larger annotation set. Cost it")
                print("  explicitly before committing; consider enriching the pool")
                print("  rather than sampling uniformly.")
            else:
                print("\n  < 1%: NEAR-ZERO BASE RATE -- §9's falsification condition fires.")
                print("  The MSA arm falls back to option 2 (label path only); the")
                print("  headline F1-bias number comes from the Qur'anic arm's C_auto")
                print("  condition, and Iqra_TTS's augmented rows supply injected")
                print("  positives (option 3). The paper survives; the framing shifts.")
            if hi >= 0.03 > rate:
                print("\n  CAUTION: the CI still reaches 3%. The point estimate picks a")
                print("  band but the data does not exclude the one above it. Say so.")

    print("\n  Record the outcome in insights/week-04.md before acting on it.")


if __name__ == "__main__":
    main()
