"""The pinned normalization contract — `docs/msa-arm.md` §4.1.

`base.Diacritizer` deliberately refuses to normalize its input, so that this
step stays a single visible stage shared by every tool and by both arms. This
module is that stage.

It exists because the MSA arm's text is not the clean Arabic the plan assumed.
Common Voice transcripts carry punctuation (41.6% of rows), Persian-script
look-alikes, Unicode presentation forms, and a little Latin script — none of
which Qur'anic text contains, so the week-4 round-trip said nothing about
them.

**The failure mode is deletion, not overflow.** Risk (a) (§4) was written
expecting the phonetizer to emit phonemes outside the 68-token inventory. It
cannot: the vendored Halabi phonetiser passes unknown characters through its
Buckwalter transliteration unchanged, they then match no phoneme rule, and
**nothing is emitted for them**. So an unhandled character does not announce
itself as an out-of-vocabulary token — it silently shortens the canonical
sequence, and at evaluation the speaker's audio for it becomes an insertion
against the reference, i.e. a mechanical false rejection. That is the same
damage risk (a) predicted, arriving through a channel no inventory diff can
see. `SUPPORTED` is therefore derived from the vendored module's own map
rather than hand-listed, so it cannot drift away from what the phonetizer
really handles.

Latin script is worse than silently dropped: its letters are *valid
Buckwalter symbols*, so `Memorial` is transliterated as though it were Arabic
and yields `m r i a l` — five invented phonemes. Hence the `LATIN` flag,
which marks a row for exclusion from evaluation rather than merely cleaning it.

**Ordering constraint.** Unicode normalization must run here, *before*
phonetization, and never after it. Canonical order for a shadda'd, voweled
consonant is `<harakah><shadda>` (combining classes 30 then 33), but Halabi's
rules want `<shadda><harakah>`, which `phonetizer.normalize` produces. NFC
applied afterwards would reorder it back and silently degeminate every
geminate in the corpus.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from arabic_mdd.data import _halabi_phonetiser

#: Characters the vendored phonetiser has a transliteration for — 36 letters
#: plus 8 combining marks. Derived, not copied: if the vendored module is ever
#: re-pinned, this follows it.
SUPPORTED: frozenset[str] = frozenset(_halabi_phonetiser.buckwalter)

#: Handled by `phonetizer.normalize` rather than by the Buckwalter map (it is
#: deleted there, deliberately and with measured justification), so it must not
#: be reported as unsupported here.
DAGGER_ALEF = "\u0670"

TATWEEL = "\u0640"

#: Persian/Urdu orthographic variants of letters MSA writes differently. These
#: are the *same* letter, so mapping them is a spelling normalization, not a
#: phonetic claim. Measured on `Iqra_train` train: farsi yeh 143 occurrences,
#: keheh 44, heh doachashmee 2.
SCRIPT_VARIANTS: dict[str, str] = {
    "\u06cc": "\u064a",  # farsi yeh       -> yeh
    "\u06a9": "\u0643",  # keheh           -> kaf
    "\u06be": "\u0647",  # heh doachashmee -> heh
}

# Letters that exist in the Arabic block but encode phonemes MSA does not have
# (`چ` /t͡ʃ/, `ڨ` /g/ or /v/). Any mapping would invent a pronunciation, so they
# are dropped and the row is flagged. 4 occurrences in 71,391 rows.
_NO_MSA_EQUIVALENT = frozenset("\u0686\u06a8")

_LATIN = re.compile(r"[A-Za-z\u00c0-\u024f]")
_WHITESPACE = re.compile(r"\s+")

# Arabic Presentation Forms-A and -B: display ligatures and positional variants
# that NFKC folds back to base letters. Detected explicitly rather than by
# "NFKC changed something", because NFKC *also* canonically reorders
# `<shadda><harakah>` into `<harakah><shadda>`, which it does on ~24.5% of
# `sentence` rows and which is not a presentation form at all.
_PRESENTATION_FORMS = re.compile(r"[\ufb50-\ufdff\ufe70-\ufeff]")

# --- flags ----------------------------------------------------------------
LATIN = "latin"
DIGITS = "digits"
NO_MSA_EQUIVALENT = "no_msa_equivalent"
UNSUPPORTED = "unsupported_character"
PUNCTUATION = "punctuation"
SCRIPT_VARIANT = "script_variant"
PRESENTATION_FORM = "presentation_form"
NON_PHONEMIC_MARK = "non_phonemic_mark"

#: Flags that mean the row cannot be evaluated fairly, because the audio
#: contains speech the reference has no phonemes for. Cleaning the text does
#: not fix that — only excluding the row does.
UNFAIR: frozenset[str] = frozenset({LATIN, DIGITS, NO_MSA_EQUIVALENT})


@dataclass(frozen=True)
class Normalized:
    """Normalized text plus what had to be done to it."""

    text: str
    flags: frozenset[str]
    dropped: tuple[str, ...]
    """Characters removed because nothing downstream could pronounce them."""

    @property
    def fair(self) -> bool:
        """False when the row should be excluded from evaluation, not just cleaned."""
        return not (self.flags & UNFAIR)


def normalize(text: str) -> Normalized:
    """Apply the §4.1 contract. Safe on both the reference and diacritizer paths.

    Deliberately does **not** strip diacritics: that is the separate `strip`
    step of `phonetize(diacritizer(strip(text)))` and applies to one path only.
    Running this on both paths identically is what makes `C_auto` and `C_gold`
    comparable.
    """
    flags: set[str] = set()
    dropped: list[str] = []

    if _LATIN.search(text):
        flags.add(LATIN)
    if any(ch.isdigit() for ch in text):
        flags.add(DIGITS)

    if _PRESENTATION_FORMS.search(text):
        flags.add(PRESENTATION_FORM)

    # Fold presentation forms (`ﻻ` -> `لا`) before anything inspects letters.
    # NFKC also normalizes the ellipsis to three full stops, which the
    # punctuation rule below then handles.
    folded = unicodedata.normalize("NFKC", text)

    out: list[str] = []
    for ch in folded:
        if ch in SUPPORTED or ch == DAGGER_ALEF:
            out.append(ch)
            continue
        if ch.isspace():
            out.append(" ")
            continue
        if ch in SCRIPT_VARIANTS:
            flags.add(SCRIPT_VARIANT)
            out.append(SCRIPT_VARIANTS[ch])
            continue
        # Tatweel is a letter-stretching glyph; an unsupported nonspacing mark
        # is a Qur'anic recitation annotation (U+06D6 "small high sad-lam-alef",
        # U+06DA "small high jeem", ...). Both carry no phonemic content, so
        # dropping them is correct rather than lossy — and validated: 82 of them
        # sit in `tashkeel_sentence`, which round-tripped at 99.85% with the
        # phonetizer discarding them. Classified by Unicode category so the
        # whole annotation block is covered without hardcoding a range.
        if ch == TATWEEL or unicodedata.category(ch) == "Mn":
            flags.add(NON_PHONEMIC_MARK)
            dropped.append(ch)
            continue
        if ch in _NO_MSA_EQUIVALENT:
            flags.add(NO_MSA_EQUIVALENT)
            dropped.append(ch)
            continue
        if unicodedata.category(ch).startswith("P"):
            # Mapped to a space, not deleted. The vowelizer that produced the
            # reference removes punctuation (§4.1), but deleting it here would
            # merge the words on either side: `نعم—نعم` phonetizes as a single
            # run, so the two words' phonemes are concatenated with no boundary.
            flags.add(PUNCTUATION)
            out.append(" ")
            continue
        flags.add(UNSUPPORTED)
        dropped.append(ch)

    return Normalized(
        text=_WHITESPACE.sub(" ", "".join(out)).strip(),
        flags=frozenset(flags),
        dropped=tuple(dropped),
    )


def unsupported_characters(text: str) -> dict[str, int]:
    """Characters that would be silently dropped downstream, with counts.

    The diagnostic behind the §4 inventory diff. Run on a *diacritizer's
    output* it answers the question the phoneme-level diff cannot: a tool that
    emits a character the phonetizer has no rule for costs phonemes without
    ever producing an out-of-vocabulary token.
    """
    counts: dict[str, int] = {}
    for ch in text:
        if ch in SUPPORTED or ch == DAGGER_ALEF or ch.isspace():
            continue
        counts[ch] = counts.get(ch, 0) + 1
    return counts
