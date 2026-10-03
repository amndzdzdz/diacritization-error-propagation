"""Diacritized Arabic → phoneme sequence in the 68-token `sws_arabic.txt` inventory.

This is the project-side wrapper around the vendored Halabi phonetiser
(`arabic_mdd.data._halabi_phonetiser`). The vendored module holds the
grapheme-to-phoneme rules verbatim and is never edited; everything that is a
*project* decision lives here:

* **Input normalization.** The corpora and Halabi disagree on how a shadda'd
  consonant is spelled, and Halabi has no rule for the Qur'anic dagger alef.
* **The prescriptive/pausal convention split.** The two arms of this project
  use different conventions for utterance-final vowels and for tanwin, and the
  phonetizer has to reproduce whichever one its arm's reference used. See
  `convention` below.
* **The collapse onto 68 tokens.** Halabi emits stress-marked vowels (`i0`,
  `i1`, `uu0`, ...) and `sil` boundary tokens; the inventory the Iqra'Eval CTC
  head is trained against has neither.

Each of these is justified by measurement in `insights/week-04.md`, not by
argument — two plausible-sounding rules (mapping the dagger alef to a full
alef; deleting the alef that carries tanwin-fath) made the round-trip *worse*
and were dropped.

Validated by `scripts/check_phonetizer_roundtrip.py` at **99.85%** (2584/2588,
0 out-of-inventory tokens) against `Iqra_train` dev. The script also reports
the Qur'anic arm, but that number is **not** a validation: QuranMB.v2's Arabic
text was recovered by phonemising the Qur'an and matching against its own
phoneme string, so the round-trip there returns the uploader's recovery
residual rather than anything about this module. Known gap: alef wasla
(U+0671) occurs in 0 MSA dev rows, so its handling is asserted, not measured.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from arabic_mdd.data import _halabi_phonetiser

Convention = Literal["prescriptive", "pausal"]

CONVENTIONS: tuple[Convention, ...] = ("prescriptive", "pausal")

# The 68-token inventory the Iqra'Eval baseline's CTC head is trained against
# (`scripts/baseline_reproduction/vocab/sws_arabic.txt`). It decomposes as 28
# consonants x 2 (plain + geminate, the geminate written doubled) + 12 vowels:
# `a aa A AA i ii I II u uu U UU`, where the capitals are the emphatic
# allophones that follow `D S T Z g x q`. `tests/data/test_phonetizer.py`
# asserts this literal still equals that file.
SWS_ARABIC_INVENTORY: frozenset[str] = frozenset(
    """
    < << b bb t tt ^ ^^ j jj H HH x xx d dd * ** r rr z zz s ss $ $$ S SS
    D DD T TT Z ZZ E EE g gg f ff q qq k kk l ll m mm n nn h hh w ww y yy
    a aa A AA i ii I II u uu U UU
    """.split()  # noqa: SIM905 - the grouping into plain/geminate pairs is the documentation
)

# --- Arabic diacritics, by code point -------------------------------------
FATHATAN = "\u064b"
DAMMATAN = "\u064c"
KASRATAN = "\u064d"
FATHA = "\u064e"
DAMMA = "\u064f"
KASRA = "\u0650"
SHADDA = "\u0651"
SUKUN = "\u0652"
DAGGER_ALEF = "\u0670"
ALEF = "\u0627"
ALEF_MAQSURA = "\u0649"

TANWIN = FATHATAN + DAMMATAN + KASRATAN
HARAKAT = TANWIN + FATHA + DAMMA + KASRA

# A shadda'd, voweled consonant can be spelled either way round, and Halabi's
# rules only accept <letter><shadda><harakah>. Left unswapped, every geminate
# silently degeminates (`$$` -> `$`), which is the single largest source of
# round-trip failures.
#
# Note which order is which, because an earlier version of this comment had it
# backwards and called Halabi's order "canonical". It is the opposite: shadda
# is combining class 33 and the harakat are 30, so Unicode canonical order is
# <letter><harakah><shadda>, and this substitution deliberately produces the
# *non*-canonical order Halabi wants. Consequence: NFC/NFKC must never run
# after this point — it would reorder the pair back and degeminate the corpus.
# `arabic_mdd.data.normalize` is therefore specified to run strictly before
# phonetization. Both orders occur in `Iqra_train` (`sentence`: 17,484 rows
# shadda-first, 296 harakah-first; `tashkeel_sentence`: uniformly shadda-first),
# so this rule has to be tolerant rather than assume one spelling.
_SHADDA_AFTER_HARAKAH = re.compile(f"([{HARAKAT}])({SHADDA})")

# A shadda written on a long-vowel letter. `إِلاَّ` and `عَلَىٍّ` put the shadda on
# the alef / alef maqsura rather than on the consonant it belongs to. Neither
# letter is a geminable consonant, so Halabi doubles the long vowel and emits
# `aaaa` -- the only out-of-inventory token the MSA round-trip produces.
# Restricted to these two letters deliberately: waw and yeh are also vowel
# carriers but are real consonants too, and `أَيَّ` must keep its gemination.
_SHADDA_ON_LONG_VOWEL = re.compile(f"([{ALEF}{ALEF_MAQSURA}]){SHADDA}")

# Pausal only: the utterance-final short vowel / sukun is not pronounced.
_FINAL_DIACRITIC = re.compile(f"[{HARAKAT}{SUKUN}]+$")

# Pausal only: tanwin is dropped. Deliberately only the *mark* -- the alef
# that carries tanwin-fath is a real long vowel in the reference, and deleting
# it along with the mark costs ~500 `aa` tokens on the MSA arm. Since this
# substitution is written to leave every letter alone, that falls out for free
# and needs no accompanying "keep the alef" rule.
_ANY_TANWIN = re.compile(f"[{TANWIN}]")

# Halabi marks vowels for stress (`i0` unstressed, `i1` stressed); the 68-token
# inventory does not distinguish them.
_STRESS_MARK = re.compile(r"[01]$")

_SILENCE = "sil"

_PROVENANCE = {
    "phonetiser": "nawarhalabi/Arabic-Phonetiser",
    "commit": "e75e06bdb8069153251adc08fb81f5bdd31ab953",
    "paper": "Halabi & Wald, LREC 2016, https://aclanthology.org/L16-1116/",
    "licence": "CC BY-NC 4.0 (non-commercial)",
    "vendored_at": "src/arabic_mdd/data/_halabi_phonetiser.py",
}


class Phonetizer:
    """Turn diacritized Arabic text into a 68-token phoneme sequence.

    `convention` selects how utterance-final vowels and tanwin are realized,
    and must match the convention of the reference being compared against:

    `"prescriptive"`
        Full case endings and tanwin, as recited. QuranMB.v2's reference
        phoneme strings use this.

    `"pausal"`
        The utterance-final diacritic is dropped, and tanwin is dropped
        everywhere (keeping the alef that carried tanwin-fath as a long
        vowel). `IqraEval/Iqra_train`'s `phoneme_ref` uses this, and
        `insights/week-04.md` fixes the MSA arm's `C_gold` as pausal.

    The instance holds no state beyond `convention`, so it is cheap to build
    and safe to share.
    """

    def __init__(self, convention: Convention = "prescriptive") -> None:
        if convention not in CONVENTIONS:
            raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")
        self.convention = convention

    def phonetize(self, text: str) -> list[str]:
        """Phonetize one utterance. Returns `[]` for blank input."""
        normalized = self.normalize(text)
        if not normalized:
            return []
        # `phonetise` splits on newlines and returns one string per line; the
        # normalizer has already flattened whitespace, so there is exactly one.
        (pronunciation,) = _halabi_phonetiser.phonetise(normalized)[1]
        return _collapse(pronunciation)

    def phonetize_batch(self, texts: list[str]) -> list[list[str]]:
        """Phonetize each utterance independently, preserving order."""
        return [self.phonetize(text) for text in texts]

    def normalize(self, text: str) -> str:
        """Apply this convention's input normalization. Exposed for diagnosis."""
        # Flatten whitespace first: `phonetise` treats a newline as an utterance
        # boundary, so an embedded one would silently split a single input into
        # two pronunciations and desynchronize a whole batch.
        text = " ".join(text.split())
        text = _SHADDA_AFTER_HARAKAH.sub(r"\2\1", text)
        text = _SHADDA_ON_LONG_VOWEL.sub(r"\1", text)
        # The Qur'anic dagger alef has no Halabi rule. It sits on a letter that
        # already carries its own harakah, so dropping it leaves the short vowel
        # the reference expects; mapping it to a full alef would add a spurious
        # `aa`. Validated at 177/177 on the MSA dev rows containing U+0670.
        # (An earlier comment here justified the rule by a 55.05% -> 49.39%
        # comparison on QuranMB. That metric is circular -- QuranMB's Arabic
        # text was recovered *from* its phoneme string -- so it justified
        # nothing. The rule survives on the MSA evidence instead.)
        text = text.replace(DAGGER_ALEF, "")
        if self.convention == "pausal":
            text = _ANY_TANWIN.sub("", text)
            text = _FINAL_DIACRITIC.sub("", text)
        return text.strip()

    def provenance(self) -> dict[str, str]:
        """Where the phonetisation rules come from, for writing up results."""
        return {**_PROVENANCE, "convention": self.convention}


def _collapse(pronunciation: str) -> list[str]:
    """Map Halabi's output onto the 68-token inventory."""
    return [_STRESS_MARK.sub("", token) for token in pronunciation.split() if token != _SILENCE]


def load_inventory(path: Path) -> frozenset[str]:
    """Read a `sws_arabic.txt`-style one-token-per-line vocabulary."""
    return frozenset(path.read_text(encoding="utf-8").split())
