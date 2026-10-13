"""DER/WER in the *published* convention — the calibration instrument.

`metrics.der` is this project's own definition and is argued for there. This
module is a different thing: it is the convention the published numbers were
computed in, implemented so that our numbers can be put next to the
literature's without an asterisk.

**It reproduces CATT's Table 5 exactly.** On the authors' own WikiNews
benchmark files, all sixteen published figures come back to three decimals —
EO 5.425 / 22.132 / 3.105 / 12.679, ED 5.963 / 20.060 / 3.631 / 11.310,
Shakkala 9.978 / 37.241 / 6.593 / 24.988, Mishkal 15.246 / 54.187 / 8.558 /
27.337. `scripts/calibrate_der.py` is that check, and
`run/wikinews_der.json` records it.

The reproduction is exact rather than approximate because the label sequence
is built by **CATT's own tokenizer** (`catt_tashkeel.TashkeelTokenizer`,
version-pinned), not by a re-reading of the paper. Only the edit distance is
ours, and `compute_der` upstream is `xer.wer` over the same sequence, i.e. a
Levenshtein distance over labels.

## How this convention differs from `metrics.der`

Four differences, every one of which makes the published number *lower* than
ours on the same text and the same output:

1. **Spaces are scored positions.** Each word gap contributes one `<NT>`
   label, and each utterance two more for `<BOS>`/`<EOS>`. They can never be
   wrong, so they dilute the rate. `metrics.der` scores Arabic letters only.
2. **"Without case endings" masks the position instead of removing it.** The
   letter before a space is relabelled `<NT>` in reference *and* hypothesis,
   so it always agrees but stays in the denominator — which is therefore
   identical to the with-case-endings denominator. `metrics.der` drops the
   position, giving a smaller denominator and a higher rate.
3. **Dagger alef (U+0670) is deleted from both sides** before labelling
   (`split_tashkeel_from_text` strips the Buckwalter backtick), so the
   convention does not hold a tool responsible for it at all. `metrics.der`
   scores it, because it spells a long vowel the reference writes.
4. **Non-Arabic characters become word separators.** `clean_text` replaces
   anything outside its Arabic range with a space, so Latin script, digits and
   punctuation turn into additional always-correct `<NT>` positions.

And one difference that does not change these numbers but matters elsewhere:
the distance is an **edit distance**, so a tool that deletes or merges words
is scored against a cheap alignment rather than refused. `metrics.der` refuses
(`skeleton_mismatch`) precisely because week 4 caught Mishkal merging words.
Both are reported; neither replaces the other.

Also note the convention's own edge case, mirrored here rather than fixed: it
masks the letter *before a space*, so the final word of each line keeps its
case ending even with `case_endings=False`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from typing import Any

#: CATT's Table 5 (arXiv 2407.03236v3), WikiNews, as published. The order is
#: (DER, WER) with case endings, then (DER, WER) without.
CATT_TABLE_5_WIKINEWS: dict[str, tuple[float, float, float, float]] = {
    "catt-eo": (5.425, 22.132, 3.105, 12.679),
    "catt-ed": (5.963, 20.060, 3.631, 11.310),
    "shakkala": (9.978, 37.241, 6.593, 24.988),
    "mishkal": (15.246, 54.187, 8.558, 27.337),
}


@dataclass(frozen=True)
class PublishedScore:
    """Edit distances and reference lengths, pooled across a corpus."""

    der_distance: int
    der_length: int
    wer_distance: int
    wer_length: int

    @property
    def der(self) -> float:
        return 100.0 * self.der_distance / self.der_length if self.der_length else 0.0

    @property
    def wer(self) -> float:
        return 100.0 * self.wer_distance / self.wer_length if self.wer_length else 0.0

    def __add__(self, other: PublishedScore) -> PublishedScore:
        return PublishedScore(
            der_distance=self.der_distance + other.der_distance,
            der_length=self.der_length + other.der_length,
            wer_distance=self.wer_distance + other.wer_distance,
            wer_length=self.wer_length + other.wer_length,
        )


@lru_cache(maxsize=1)
def _tokenizer() -> Any:
    """CATT's tokenizer, which defines the convention. Constructed once."""
    from catt_tashkeel.tokenizer import TashkeelTokenizer

    return TashkeelTokenizer()


def convention() -> dict[str, str]:
    """Provenance of the *definition*, to be recorded next to any rate."""
    return {
        "name": "catt/fadel",
        "source": "abjadai/catt compute_der.py + xer.py; docs/novelty-check.md Table 5",
        "tokenizer": "catt_tashkeel.TashkeelTokenizer",
        "tokenizer_version": version("catt-tashkeel"),
        "note": (
            "spaces and <BOS>/<EOS> are scored positions; without-case-endings masks "
            "the pre-space letter rather than dropping it; dagger alef U+0670 is "
            "deleted before scoring; distance is an edit distance over labels"
        ),
    }


def label_sequence(text: str, *, case_endings: bool) -> list[str]:
    """The per-position diacritic label sequence the published DER runs over."""
    _letters, tashkeel = _tokenizer().get_tashkeel_with_case_ending(
        text.strip(), case_ending=case_endings
    )
    return list(tashkeel)


def word_sequence(text: str, *, case_endings: bool) -> list[str]:
    """The per-word label sequence the published WER runs over."""
    letters, tashkeel = _tokenizer().get_tashkeel_with_case_ending(
        text.strip(), case_ending=case_endings
    )
    words: list[str] = []
    current: list[str] = []
    for letter, tag in zip(letters, tashkeel, strict=True):
        if letter == " ":
            words.append("|".join(current))
            current = []
        else:
            current.append(f"{letter}{tag}")
    words.append("|".join(current))
    return words


def score_utterance(reference: str, hypothesis: str, *, case_endings: bool) -> PublishedScore:
    """DER and WER distances for one reference/hypothesis pair."""
    ref_labels = label_sequence(reference, case_endings=case_endings)
    hyp_labels = label_sequence(hypothesis, case_endings=case_endings)
    ref_words = word_sequence(reference, case_endings=case_endings)
    hyp_words = word_sequence(hypothesis, case_endings=case_endings)
    return PublishedScore(
        der_distance=_levenshtein(ref_labels, hyp_labels),
        der_length=len(ref_labels),
        wer_distance=_levenshtein(ref_words, hyp_words),
        wer_length=len(ref_words),
    )


def score_corpus(pairs: Iterable[tuple[str, str]], *, case_endings: bool) -> PublishedScore:
    """Pool over a corpus. Micro-averaged, as the published figures are."""
    total = PublishedScore(0, 0, 0, 0)
    for reference, hypothesis in pairs:
        total = total + score_utterance(reference, hypothesis, case_endings=case_endings)
    return total


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Token-level edit distance — what `xer.wer` computes via `kaldialign`."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
