"""The published convention's four differences from `metrics.der`, each pinned.

The exact reproduction of CATT's Table 5 is `scripts/calibrate_der.py`, which
needs the authors' benchmark files and so cannot run here. These tests pin the
properties that reproduction depends on, so a tokenizer upgrade that changes
the convention fails loudly rather than silently moving a calibrated number.
"""

from arabic_mdd.metrics import der, der_published

WORD = "كَتَبَ"
TWO_WORDS = "كَتَبَ الْوَلَدُ"


def test_spaces_are_scored_positions() -> None:
    """Difference 1: each word gap is an always-correct <NT> in the denominator."""
    one = der_published.label_sequence(WORD, case_endings=True)
    two = der_published.label_sequence(TWO_WORDS, case_endings=True)
    # <BOS> + 3 letters + <EOS>
    assert len(one) == 5
    # the space contributes a position of its own
    assert len(two) == len(one) + len("الولد") + 1
    assert "<NT>" in two


def test_bos_and_eos_are_scored_positions() -> None:
    assert der_published.label_sequence(WORD, case_endings=True)[0] == "<BOS>"
    assert der_published.label_sequence(WORD, case_endings=True)[-1] == "<EOS>"


def test_without_case_endings_masks_rather_than_drops() -> None:
    """Difference 2: the denominator is identical, unlike `metrics.der`'s."""
    with_ce = der_published.label_sequence(TWO_WORDS, case_endings=True)
    without = der_published.label_sequence(TWO_WORDS, case_endings=False)
    assert len(without) == len(with_ce)

    # The masked position is the letter *before a space* — so the final word of
    # the line keeps its case ending. Mirrored from upstream, not fixed.
    assert with_ce[3] != "<NT>"
    assert without[3] == "<NT>"
    assert without[-2] == with_ce[-2] != "<NT>"


def test_a_case_ending_error_is_free_without_case_endings() -> None:
    reference = "كَتَبَ الْوَلَدُ أَمْسِ"
    hypothesis = "كَتَبُ الْوَلَدُ أَمْسِ"
    with_ce = der_published.score_utterance(reference, hypothesis, case_endings=True)
    without = der_published.score_utterance(reference, hypothesis, case_endings=False)
    assert with_ce.der_distance == 1
    assert without.der_distance == 0


def test_dagger_alef_is_not_scored() -> None:
    """Difference 3: the convention deletes U+0670; `metrics.der` scores it."""
    reference = "الرَّحْمَٰنِ"
    hypothesis = "الرَّحْمَنِ"  # the same, minus the dagger alef
    assert der_published.score_utterance(reference, hypothesis, case_endings=True).der_distance == 0
    assert der.score_utterance(reference, hypothesis, der.CATT_WITH_CASE_ENDINGS).der > 0


def test_non_arabic_characters_become_separators() -> None:
    """Difference 4: punctuation and Latin turn into always-correct positions."""
    plain = der_published.label_sequence("كَتَبَ الْوَلَدُ", case_endings=True)
    punctuated = der_published.label_sequence("كَتَبَ، الْوَلَدُ", case_endings=True)
    assert len(punctuated) == len(plain)
    latin = der_published.label_sequence("كَتَبَ (Reuters)", case_endings=True)
    assert latin == der_published.label_sequence(WORD, case_endings=True)


def test_distance_is_an_edit_distance_not_a_refusal() -> None:
    """A dropped word costs its length here; `metrics.der` refuses to align it."""
    reference = "كَتَبَ الْوَلَدُ"
    hypothesis = "كَتَبَ"
    score = der_published.score_utterance(reference, hypothesis, case_endings=True)
    assert 0 < score.der_distance < score.der_length
    assert der.score_utterance(reference, hypothesis, der.CATT_WITH_CASE_ENDINGS).skeleton_mismatch


def test_corpus_scores_are_micro_averaged() -> None:
    pairs = [("كَتَبَ", "كَتَبُ"), ("الْوَلَدُ", "الْوَلَدُ")]
    pooled = der_published.score_corpus(pairs, case_endings=True)
    first = der_published.score_utterance(*pairs[0], case_endings=True)
    second = der_published.score_utterance(*pairs[1], case_endings=True)
    assert pooled.der_distance == first.der_distance + second.der_distance
    assert pooled.der_length == first.der_length + second.der_length
    assert pooled.der == 100.0 * pooled.der_distance / pooled.der_length


def test_published_table_is_carried_verbatim() -> None:
    assert der_published.CATT_TABLE_5_WIKINEWS["catt-eo"] == (5.425, 22.132, 3.105, 12.679)
    assert der_published.CATT_TABLE_5_WIKINEWS["catt-ed"] == (5.963, 20.060, 3.631, 11.310)


def test_convention_records_the_tokenizer_version() -> None:
    """The reproduction is exact only for the pinned tokenizer, so it is recorded."""
    assert der_published.convention()["tokenizer_version"]
