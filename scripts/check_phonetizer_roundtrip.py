"""Week 4 exit criterion: does the phonetizer reproduce each arm's released reference?

Plan §7 / `docs/weeks/week-04.md` task 3. The criterion is
`phonetize(reference_arabic) == reference_phonemes` at **>=99% exact sequence
match, reported separately per arm**, with every mismatch explained.

The two arms are reported separately because they use different conventions
and, as it turns out, have very different reference quality:

* **MSA arm** (`IqraEval/Iqra_train`): `sentence` -> `phoneme_ref`, pausal.
* **Qur'anic arm** (QuranMB.v2): `reference_arabic_string` ->
  `reference_phoneme_string`, prescriptive.

QuranMB is deduplicated before scoring. Its 1,642 rows contain far fewer
distinct sentences, so a per-row percentage would be a weighted average over
repeats rather than a measure of the phonetizer.

Run with: uv run python scripts/check_phonetizer_roundtrip.py
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Sequence

from arabic_mdd.data.iqra_train import load_iqra_train
from arabic_mdd.data.phonetizer import SWS_ARABIC_INVENTORY, Convention, Phonetizer
from arabic_mdd.data.quranmb import load_ground_truth

MAX_MISMATCHES_SHOWN = 15


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(
                previous[j - 1]
                if token_a == token_b
                else 1 + min(previous[j - 1], previous[j], current[j - 1])
            )
        previous = current
    return previous[-1]


def _first_difference(predicted: Sequence[str], reference: Sequence[str]) -> str:
    """Where the two sequences diverge, with a little context on both sides."""
    for i, (p, r) in enumerate(zip(predicted, reference, strict=False)):
        if p != r:
            lo, hi = max(0, i - 3), i + 4
            return f"at {i}: got {' '.join(predicted[lo:hi])!r} want {' '.join(reference[lo:hi])!r}"
    return f"prefix matches; length {len(predicted)} vs {len(reference)}"


def _report(
    name: str,
    convention: Convention,
    pairs: list[tuple[str, str, list[str]]],
    note: str = "",
) -> None:
    """`pairs` is (id, arabic, reference_phonemes)."""
    phonetizer = Phonetizer(convention=convention)

    exact = 0
    total_reference_tokens = 0
    total_distance = 0
    mismatches: list[tuple[str, str, list[str], list[str]]] = []
    out_of_inventory: Counter[str] = Counter()

    for example_id, arabic, reference in pairs:
        predicted = phonetizer.phonetize(arabic)
        out_of_inventory.update(set(predicted) - SWS_ARABIC_INVENTORY)
        total_reference_tokens += len(reference)
        if predicted == reference:
            exact += 1
        else:
            total_distance += _edit_distance(predicted, reference)
            mismatches.append((example_id, arabic, predicted, reference))

    rate = exact / len(pairs) if pairs else 0.0
    ter = total_distance / total_reference_tokens if total_reference_tokens else 0.0

    print(f"=== {name} ({convention}) ===")
    if note:
        print(f"  {note}")
    print(f"  exact sequence match : {exact}/{len(pairs)} = {rate:.2%}")
    print(f"  token error rate     : {ter:.4f}")
    print(f"  exit criterion (>=99%): {'PASS' if rate >= 0.99 else 'FAIL'}")
    if out_of_inventory:
        print(f"  OUT-OF-INVENTORY tokens produced: {dict(out_of_inventory)}")
    print()

    if mismatches:
        print(f"  mismatches (showing {min(len(mismatches), MAX_MISMATCHES_SHOWN)}):")
        for example_id, arabic, predicted, reference in mismatches[:MAX_MISMATCHES_SHOWN]:
            print(f"    [{example_id}] {arabic}")
            print(f"      got  : {' '.join(predicted)}")
            print(f"      want : {' '.join(reference)}")
            print(f"      {_first_difference(predicted, reference)}")
        print()


def _check_msa_arm() -> None:
    # `tashkeel_sentence`, not `sentence`: the undiacritized column cannot be
    # phonetized at all, and it is not what `phoneme_ref` was derived from.
    examples = load_iqra_train(split="dev")
    pairs = [(e.id, e.tashkeel_sentence, e.canonical) for e in examples]
    _report("MSA arm -- Iqra_train dev", "pausal", pairs)


def _check_quranic_arm() -> None:
    ground_truth = load_ground_truth()

    # Deduplicate on the Arabic string. Report how far the corpus is from
    # 1,642 independent sentences, because the exit criterion's statistical
    # weight depends on it.
    unique: dict[str, tuple[str, list[str]]] = {}
    conflicting = 0
    for item in ground_truth.values():
        key = unicodedata.normalize("NFC", item.reference_arabic)
        if key in unique:
            if unique[key][1] != item.canonical:
                conflicting += 1
            continue
        unique[key] = (item.id, item.canonical)

    pairs = [(example_id, arabic, canonical) for arabic, (example_id, canonical) in unique.items()]
    note = (
        f"{len(ground_truth)} rows collapse to {len(pairs)} distinct Arabic strings "
        f"({conflicting} rows disagree with an earlier row's phonemes for the same text)"
    )
    _report("Qur'anic arm -- QuranMB.v2", "prescriptive", pairs, note=note)


def main() -> None:
    _check_msa_arm()
    _check_quranic_arm()


if __name__ == "__main__":
    main()
