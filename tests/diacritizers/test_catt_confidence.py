"""Check the token-probability extraction against CATT's own decode.

The confidence path re-walks `catt_tashkeel`'s private decode, so the thing
worth testing is not "does it return numbers" but "does it return the *same*
decision the package would have made, at the same position". That is the
reconstruction identity asserted throughout: joining `letter + diacritic`
over the returned positions must reproduce `do_tashkeel_batch` exactly.

To assert that offline, the real `CATTEncoderOnly` / `CATTEncoderDecoder` are
instantiated via `__new__` with the real `TashkeelTokenizer` and fake ONNX
sessions. Everything the alignment actually depends on — preprocessing,
tokenization, padding, the <BOS>/<EOS> trimming, the space mask, the decode
loop — is therefore the package's real code; only the neural net is fake.

The fake scores are a deterministic function of the input letter id, which is
what lets EO and ED be compared to each other: given identical scores the two
architectures must reach identical decisions, despite decoding by completely
different routes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from arabic_mdd.diacritizers import catt
from arabic_mdd.diacritizers.catt import CattDiacritizer

TEXTS = ["ذهب الولد", "ثم عاد المعلم من المدرسة", "بيت"]

#: Size of `TashkeelTokenizer.tashkeel_list` (15 tags + <PAD>/<BOS>/<EOS>).
VOCAB_SIZE = 18

#: Tag indices below this are the specials, which a healthy model never picks.
FIRST_REAL_TAG = 3


def _scores(letter_ids: Any) -> np.ndarray:
    """Fake decoder scores: a deterministic, peaked function of the letter.

    The peak height varies with the letter so that confidences differ across
    positions — a constant would let a broken implementation pass by reading
    the wrong position.
    """
    letter_ids = np.asarray(letter_ids)
    logits = np.zeros((*letter_ids.shape, VOCAB_SIZE), dtype=np.float32)
    chosen = FIRST_REAL_TAG + (letter_ids % (VOCAB_SIZE - FIRST_REAL_TAG))
    peak = 2.0 + (letter_ids % 5)
    np.put_along_axis(logits, chosen[..., None], peak[..., None], axis=-1)
    return logits


def _peak_probability(letter_id: int) -> float:
    """The confidence `_scores` implies for a letter, computed independently."""
    peak = 2.0 + (letter_id % 5)
    return float(np.exp(peak) / (np.exp(peak) + (VOCAB_SIZE - 1)))


class _FakeSession:
    """Stands in for an `onnxruntime.InferenceSession`."""

    def __init__(self, respond: Any) -> None:
        self._respond = respond

    def run(self, _output_names: Any, inputs: dict[str, Any]) -> list[Any]:
        return [self._respond(inputs)]


def _build(variant: str) -> Any:
    from catt_tashkeel.models import CATTEncoderDecoder, CATTEncoderOnly
    from catt_tashkeel.tokenizer import TashkeelTokenizer

    cls = CATTEncoderOnly if variant == "eo" else CATTEncoderDecoder
    model = cls.__new__(cls)
    model.tokenizer = TashkeelTokenizer()
    model.auto_preprocess = True
    model.src_pad_idx = model.tokenizer.letters_map["<PAD>"]
    # The encoder is the identity, so `enc_src` is literally the letter ids and
    # the fake decoder can score them without modelling an embedding.
    model.encoder_session = _FakeSession(lambda inputs: inputs["src"])

    if variant == "eo":
        model.decoder_session = _FakeSession(lambda inputs: _scores(inputs["enc_src"]))
    else:
        model.trg_pad_idx = model.tokenizer.tashkeel_map["<PAD>"]
        model.decoder_session = _FakeSession(_ed_scores)
    return model


def _ed_scores(inputs: dict[str, Any]) -> np.ndarray:
    """Score an ED decode step so that it agrees with the EO scores.

    Row `t` of the decoder output predicts target position `t + 1`, which the
    autoregressive loop pairs with source position `t + 1`.
    """
    source, target = inputs["enc_src"], inputs["trg"]
    return _scores(source[:, 1 : 1 + target.shape[1]])


def _diacritizer(variant: str) -> CattDiacritizer:
    return CattDiacritizer(variant=variant, model=_build(variant))


def _reconstruct(positions: list[catt.DiacriticConfidence]) -> str:
    return "".join(position.letter + position.diacritic for position in positions)


@pytest.mark.parametrize("variant", ["eo", "ed"])
def test_reconstruction_matches_the_packages_own_decode(variant: str) -> None:
    diacritizer = _diacritizer(variant)

    expected = diacritizer.model.do_tashkeel_batch(TEXTS, batch_size=16, verbose=False)
    results = diacritizer.diacritize_batch_with_confidence(TEXTS)

    assert [_reconstruct(positions) for positions in results] == expected


@pytest.mark.parametrize("variant", ["eo", "ed"])
def test_letters_are_the_undiacritized_input_in_order(variant: str) -> None:
    results = _diacritizer(variant).diacritize_batch_with_confidence(TEXTS)

    assert ["".join(position.letter for position in row) for row in results] == TEXTS


def test_the_two_architectures_agree_given_the_same_scores() -> None:
    encoder_only = _diacritizer("eo").diacritize_batch_with_confidence(TEXTS)
    encoder_decoder = _diacritizer("ed").diacritize_batch_with_confidence(TEXTS)

    assert encoder_only == encoder_decoder


@pytest.mark.parametrize("variant", ["eo", "ed"])
def test_confidence_is_the_max_softmax_probability(variant: str) -> None:
    diacritizer = _diacritizer(variant)
    letters = diacritizer.model.tokenizer.letters_map

    positions = diacritizer.diacritize_with_confidence("ذهب الولد")

    expected = [_peak_probability(letters[letter]) for letter in "*hb Alwld"]
    assert [position.confidence for position in positions] == pytest.approx(expected)


@pytest.mark.parametrize("variant", ["eo", "ed"])
def test_spaces_are_flagged_forced_and_carry_the_overridden_decision(variant: str) -> None:
    """The space mask discards the model's choice, so RQ4 must not threshold there."""
    positions = _diacritizer(variant).diacritize_with_confidence("ذهب الولد")

    forced = [position for position in positions if position.forced]
    assert [position.letter for position in forced] == [" "]
    # The fake scores put a real diacritic on the space; the mask removed it,
    # while the confidence still describes the prediction that was thrown away.
    assert forced[0].diacritic == ""
    assert forced[0].confidence > 0.5

    assert all(position.letter == " " for position in positions if position.forced)


@pytest.mark.parametrize("variant", ["eo", "ed"])
def test_batching_does_not_change_the_result(variant: str) -> None:
    """Padding differs with batch composition; the per-text answer must not."""
    batched = CattDiacritizer(variant=variant, model=_build(variant), batch_size=2)
    one_at_a_time = CattDiacritizer(variant=variant, model=_build(variant), batch_size=1)

    assert batched.diacritize_batch_with_confidence(TEXTS) == (
        one_at_a_time.diacritize_batch_with_confidence(TEXTS)
    )


@pytest.mark.parametrize("variant", ["eo", "ed"])
def test_empty_batch_needs_no_model(variant: str) -> None:
    assert CattDiacritizer(variant=variant).diacritize_batch_with_confidence([]) == []


def test_confidences_are_probabilities() -> None:
    results = _diacritizer("ed").diacritize_batch_with_confidence(TEXTS)

    assert all(0.0 < position.confidence <= 1.0 for row in results for position in row)


def test_unsupported_catt_version_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The extraction rides on private internals, so drift must not pass silently."""
    monkeypatch.setattr(catt, "version", lambda _package: "2.0.0")

    with pytest.raises(RuntimeError, match="2.0.0"):
        _diacritizer("eo").diacritize_with_confidence("بيت")


def test_the_pinned_version_is_the_installed_one() -> None:
    assert catt.version("catt-tashkeel") == catt.SUPPORTED_CATT_VERSION
