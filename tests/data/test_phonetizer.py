"""Offline tests for the phonetizer wrapper.

These exercise the wrapper's own decisions — normalization, the convention
split, the collapse onto 68 tokens — against the *real* vendored phonetiser.
Nothing here is faked, and nothing here touches the network.

What they cannot prove is agreement with the corpora's released phoneme
references; that is `scripts/check_phonetizer_roundtrip.py`, which needs the
datasets.
"""

import unicodedata
from pathlib import Path

import pytest

from arabic_mdd.data.phonetizer import (
    CONVENTIONS,
    SWS_ARABIC_INVENTORY,
    Phonetizer,
    load_inventory,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VOCAB_PATH = REPO_ROOT / "scripts/baseline_reproduction/vocab/sws_arabic.txt"

BASMALA = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
SENTENCE = "ذَهَبَ الْوَلَدُ إِلَى الْمَدْرَسَةِ"


@pytest.fixture(params=CONVENTIONS)
def phonetizer(request: pytest.FixtureRequest) -> Phonetizer:
    return Phonetizer(convention=request.param)


# --- the inventory ---------------------------------------------------------


def test_inventory_literal_matches_the_tracked_vocab_file() -> None:
    """The in-source frozenset is a copy; this is what keeps it honest."""
    assert load_inventory(VOCAB_PATH) == SWS_ARABIC_INVENTORY


def test_inventory_has_68_tokens() -> None:
    assert len(SWS_ARABIC_INVENTORY) == 68


@pytest.mark.parametrize("text", [BASMALA, SENTENCE, "قُلْ هُوَ اللَّهُ أَحَدٌ"])
def test_output_stays_inside_the_inventory(phonetizer: Phonetizer, text: str) -> None:
    out_of_inventory = set(phonetizer.phonetize(text)) - SWS_ARABIC_INVENTORY
    assert not out_of_inventory


# --- normalization ---------------------------------------------------------


def test_shadda_written_after_the_harakah_still_geminates() -> None:
    """The corpora spell it <letter><harakah><shadda>; Halabi wants the reverse.

    Without the swap this degeminates silently -- the output is still a valid
    phoneme sequence, just the wrong one, which is why it needs a test.
    """
    corpus_order = "الرَّحِيمِ"  # kasra/fatha before shadda, as stored
    assert "rr" in Phonetizer().phonetize(corpus_order)


def test_dagger_alef_is_dropped_not_lengthened() -> None:
    """Mapping U+0670 to a full alef measured *worse* on QuranMB (55.0 -> 49.4%)."""
    assert "\u0670" not in Phonetizer().normalize("الرَّحْمَٰنِ")
    # The dagger alef sits on a meem that already carries a fatha, so dropping
    # it leaves a short `a` -- not the `aa` a full alef would have produced.
    with_dagger = Phonetizer().phonetize("الرَّحْمَٰنِ")
    assert with_dagger == Phonetizer().phonetize("الرَّحْمَنِ")
    assert "aa" not in with_dagger


def test_shadda_on_a_long_vowel_letter_is_dropped() -> None:
    """`إِلاَّ` / `عَلَىٍّ` put the shadda on the alef instead of the consonant.

    Alef and alef maqsura are not geminable, so Halabi doubles the long vowel
    and emits `aaaa` -- the only out-of-inventory token the MSA round-trip
    produced (6/2588 dev rows). See `insights/week-04.md`.
    """
    for text in ("إِلاَّ", "عَلَىٍّ", "دُجَىٌّ"):
        produced = Phonetizer("pausal").phonetize(text)
        assert not set(produced) - SWS_ARABIC_INVENTORY, produced


def test_gemination_on_a_consonant_vowel_carrier_survives() -> None:
    """Waw and yeh carry vowels too, so the rule must not reach them.

    `أَيَّ` is a real geminate; widening the rule to every vowel carrier would
    silently degeminate it.
    """
    assert "yy" in Phonetizer("prescriptive").phonetize("أَيَّ")


def test_embedded_newlines_do_not_split_the_utterance() -> None:
    """`phonetise` treats a newline as an utterance boundary.

    Unguarded, one stray newline returns two pronunciations and desynchronizes
    every later row of a batch against its reference.
    """
    joined = Phonetizer().phonetize("بَيْتٌ\nبَيْتٌ")
    assert joined == Phonetizer().phonetize("بَيْتٌ بَيْتٌ")


def test_blank_input_gives_an_empty_sequence(phonetizer: Phonetizer) -> None:
    assert phonetizer.phonetize("") == []
    assert phonetizer.phonetize("   \n  ") == []


# --- the convention split --------------------------------------------------


def test_pausal_drops_the_utterance_final_vowel() -> None:
    text = "ذَهَبَ الْوَلَدُ"
    prescriptive = Phonetizer("prescriptive").phonetize(text)
    pausal = Phonetizer("pausal").phonetize(text)
    assert prescriptive[-1] == "u"
    assert pausal == prescriptive[:-1]


def test_pausal_also_silences_a_final_teh_marbuta() -> None:
    """A consequence of the convention, not a separate rule -- and a big one.

    Once the case ending is gone, Halabi's own rules stop realizing the teh
    marbuta, so `...s a t i` becomes `...s a`: pausal removes *two* tokens
    here, not one. Every teh-marbuta-final utterance in the MSA arm depends on
    this matching `phoneme_ref`.
    """
    prescriptive = Phonetizer("prescriptive").phonetize(SENTENCE)
    pausal = Phonetizer("pausal").phonetize(SENTENCE)
    assert prescriptive[-2:] == ["t", "i"]
    assert pausal == prescriptive[:-2]


def test_pausal_drops_tanwin_everywhere_not_only_finally() -> None:
    """Iqra's `phoneme_ref` has no tanwin at all, including mid-utterance."""
    text = "بَيْتٌ كَبِيرٌ"
    assert "un" not in "".join(Phonetizer("pausal").phonetize(text))
    assert "n" not in Phonetizer("pausal").phonetize(text)
    assert "n" in Phonetizer("prescriptive").phonetize(text)


def test_pausal_keeps_the_alef_carrying_tanwin_fath() -> None:
    """Deleting it with the tanwin costs ~500 `aa` tokens on the MSA arm."""
    assert "aa" in Phonetizer("pausal").phonetize("شُكْرًا")


def test_prescriptive_keeps_case_endings_and_tanwin() -> None:
    assert Phonetizer("prescriptive").phonetize(BASMALA)[-1] == "i"
    assert "n" in Phonetizer("prescriptive").phonetize("بَيْتٌ")


def test_unknown_convention_is_rejected() -> None:
    with pytest.raises(ValueError, match="convention"):
        Phonetizer(convention="pausal-ish")  # type: ignore[arg-type]


# --- the collapse onto 68 tokens -------------------------------------------


def test_stress_marks_are_stripped(phonetizer: Phonetizer) -> None:
    """Halabi emits `i0`/`i1`/`uu0`; the inventory has no stressed vowels."""
    assert not any(token[-1].isdigit() for token in phonetizer.phonetize(BASMALA))


def test_silence_tokens_are_dropped(phonetizer: Phonetizer) -> None:
    assert "sil" not in phonetizer.phonetize(BASMALA)


# --- the batch interface ---------------------------------------------------


def test_batch_matches_one_at_a_time(phonetizer: Phonetizer) -> None:
    texts = [BASMALA, SENTENCE, "بَيْتٌ"]
    assert phonetizer.phonetize_batch(texts) == [phonetizer.phonetize(t) for t in texts]


def test_empty_batch(phonetizer: Phonetizer) -> None:
    assert phonetizer.phonetize_batch([]) == []


# --- provenance ------------------------------------------------------------


def test_provenance_records_the_pinned_commit_and_the_convention() -> None:
    provenance = Phonetizer("pausal").provenance()
    assert provenance["commit"] == "e75e06bdb8069153251adc08fb81f5bdd31ab953"
    assert provenance["convention"] == "pausal"
    assert "NC" in provenance["licence"]


def test_both_shadda_orders_phonetize_identically() -> None:
    """Unicode-canonical order is <harakah><shadda>; Halabi wants the reverse.

    `Iqra_train`'s `sentence` column carries both (17,484 rows shadda-first,
    296 harakah-first), so the rule must accept either. The comment on
    `_SHADDA_AFTER_HARAKAH` once called Halabi's order "canonical", which is
    backwards and would invite an NFC pass that silently degeminates.
    """
    phonetizer = Phonetizer(convention="pausal")
    shadda_first = "\u0625\u0650\u0646\u0651\u064e"  # إِنَّ, <shadda><harakah>
    harakah_first = "\u0625\u0650\u0646\u064e\u0651"  # إِنَّ, <harakah><shadda>

    assert unicodedata.normalize("NFC", harakah_first) == harakah_first
    assert phonetizer.phonetize(shadda_first) == phonetizer.phonetize(harakah_first)
    assert "nn" in phonetizer.phonetize(harakah_first)
