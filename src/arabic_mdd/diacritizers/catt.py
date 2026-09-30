"""CATT (github.com/abjadai/catt), the strong modern system in the tool set.

Both published variants are wrapped: encoder-only (`eo`) and encoder-decoder
(`ed`). ED is the more accurate one per CATT's own numbers, and RQ4's
token-probability work has to be confirmed on it rather than assumed from EO,
so neither variant may be dropped here.

The `catt_tashkeel` PyPI package is ONNX-only and downloads its model
archives on first construction (EO 72 MB, ED 86 MB), so construction is
deferred until the first call — importing this module must stay free.

## Token-level confidence (week 4 task 2, RQ4)

RQ4 — selective scoring by diacritizer abstention — needs a per-position
confidence for every diacritic CATT assigns. The package does not expose one:
`do_tashkeel(_batch)` returns decoded text with the softmax/argmax step
already applied and discarded.

Week 1 (`scripts/check_catt_probabilities.py`, `insights/week-01-03.md`)
established that the logits are reachable through the package's private
methods, on EO. That probe is superseded by
`CattDiacritizer.diacritize_with_confidence` here, which covers **both**
variants and whose alignment is tested rather than eyeballed.

The two architectures need genuinely different extraction paths, which is why
confirming ED separately was a task in its own right:

* **EO** decodes in one shot. `_run_decoder` returns logits for every
  position at once, so the probabilities are one softmax away.
* **ED** decodes autoregressively, one token per loop iteration, and keeps
  only `preds[:, -1, :]` each time. There is no point at which the whole
  logit tensor exists, so the loop itself has to be re-walked, accumulating
  the per-step distribution.

Both paths are aligned so that position *k* describes input letter *k*, and
both are checked against the package's own `do_tashkeel_batch` output — the
reconstruction identity in `tests/diacritizers/test_catt_confidence.py`.

Because this rides on underscore-prefixed methods that carry no compatibility
promise — the risk week 1 flagged when it deferred the production interface
to this block — the package version is pinned in `pyproject.toml` and checked
at runtime. An upstream refactor should fail loudly rather than quietly
return misaligned confidences.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import numpy as np

from arabic_mdd.diacritizers.base import Diacritizer

VARIANTS = ("eo", "ed")

#: The confidence path uses `catt_tashkeel`'s private internals, so it is
#: written against one exact release rather than a floor.
SUPPORTED_CATT_VERSION = "1.0.2"

#: Tokenizer tags that carry no diacritic of their own.
_EMPTY_TAGS = frozenset({"<PAD>", "<BOS>", "<EOS>"})


@dataclass(frozen=True)
class DiacriticConfidence:
    """CATT's decision at one input position, with the confidence behind it."""

    letter: str
    """The Arabic letter (or space) this position covers."""

    diacritic: str
    """The Arabic diacritic(s) CATT assigned here; empty where it assigned none."""

    confidence: float
    """Max softmax probability over CATT's diacritic vocabulary at this position."""

    forced: bool
    """True where CATT overrode the model rather than predicting.

    CATT hard-sets every space position to "no diacritic" via
    `_apply_space_mask`, so `confidence` there describes a prediction that was
    discarded. RQ4 must not threshold on these.
    """


class CattDiacritizer(Diacritizer):
    """CATT, in its encoder-only or encoder-decoder variant."""

    def __init__(self, variant: str = "eo", model: Any = None, batch_size: int = 16) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown CATT variant {variant!r}; expected one of {VARIANTS}")
        self.variant = variant
        self.batch_size = batch_size
        self._model = model

    @property
    def name(self) -> str:
        return f"catt-{self.variant}"

    def provenance(self) -> dict[str, str]:
        return {
            "tool": "catt",
            "variant": self.variant,
            "package": "catt-tashkeel",
            "version": version("catt-tashkeel"),
        }

    @property
    def model(self) -> Any:
        """The underlying `catt_tashkeel` model, constructed on first use."""
        if self._model is None:
            import catt_tashkeel

            cls = (
                catt_tashkeel.CATTEncoderOnly
                if self.variant == "eo"
                else catt_tashkeel.CATTEncoderDecoder
            )
            self._model = cls()
        return self._model

    def diacritize_batch(self, texts: Sequence[str]) -> list[str]:
        if not texts:
            return []
        outputs = self.model.do_tashkeel_batch(
            list(texts), batch_size=self.batch_size, verbose=False
        )
        return [out.strip() for out in outputs]

    def diacritize_with_confidence(self, text: str) -> list[DiacriticConfidence]:
        """Diacritize one text, reporting CATT's confidence at every position."""
        return self.diacritize_batch_with_confidence([text])[0]

    def diacritize_batch_with_confidence(
        self, texts: Sequence[str]
    ) -> list[list[DiacriticConfidence]]:
        """Diacritize a batch, reporting CATT's confidence at every position.

        Position *k* of the result describes input letter *k*, so joining
        `letter + diacritic` over the list reproduces `diacritize` exactly.
        """
        if not texts:
            return []
        _require_supported_catt_version()

        model = self.model
        prepared = [model._preprocess_text(text) for text in texts]

        results: list[list[DiacriticConfidence]] = []
        for chunk in model._get_batches(prepared, self.batch_size):
            input_ids = model._prepare_batch_input(chunk)
            tags, confidences = self._decode_with_probabilities(input_ids)
            results.extend(_assemble(model.tokenizer, input_ids, tags, confidences))
        return results

    def _decode_with_probabilities(self, input_ids: Any) -> tuple[Any, Any]:
        """Re-walk the variant's decode, keeping the distribution it discards.

        Returns `(tags, confidences)`, both shaped `(batch, width)` and aligned
        to `input_ids[:, 1:1 + width]` — that is, offset by one to skip `<BOS>`.
        """
        if self.variant == "eo":
            return self._decode_encoder_only(input_ids)
        return self._decode_encoder_decoder(input_ids)

    def _decode_encoder_only(self, input_ids: Any) -> tuple[Any, Any]:
        model = self.model
        # Mirrors `CATTEncoderOnly._process_batch`, which drops <BOS>/<EOS>.
        trimmed = input_ids[:, 1:-1]
        logits = model._run_decoder(model._run_encoder(trimmed))

        confidences = _softmax(logits).max(axis=-1)
        tags = model._apply_space_mask(np.argmax(logits, axis=-1), trimmed)
        return tags, confidences

    def _decode_encoder_decoder(self, input_ids: Any) -> tuple[Any, Any]:
        model = self.model
        tokenizer = model.tokenizer
        # Mirrors `CATTEncoderDecoder._process_batch`. Unlike EO there is no
        # single logit tensor to softmax: the loop is the only place each
        # position's distribution exists.
        enc_src = model._run_encoder(input_ids)
        target_ids = np.full((len(input_ids), 1), tokenizer.tashkeel_map["<BOS>"], dtype=np.int64)

        step_confidences = []
        for _ in range(input_ids.shape[1] - 1):
            step_logits = model._run_decoder(target_ids, enc_src, input_ids)[:, -1, :]
            step_confidences.append(_softmax(step_logits).max(axis=-1))

            next_tokens = np.expand_dims(np.argmax(step_logits, axis=1), 1)
            target_ids = np.concatenate([target_ids, next_tokens], axis=1)
            target_ids = model._apply_space_mask(target_ids, input_ids[:, : target_ids.shape[1]])

        # Drop the seeded <BOS> column so position k describes input letter k.
        return target_ids[:, 1:], np.stack(step_confidences, axis=1)


def _require_supported_catt_version() -> None:
    installed = version("catt-tashkeel")
    if installed != SUPPORTED_CATT_VERSION:
        raise RuntimeError(
            f"CATT confidence extraction is written against catt-tashkeel "
            f"{SUPPORTED_CATT_VERSION}'s private internals (`_run_encoder`, "
            f"`_run_decoder`, `_apply_space_mask`, `_prepare_batch_input`), which "
            f"carry no compatibility promise; {installed} is installed. Re-derive "
            f"and re-test the alignment before lifting this pin."
        )


def _softmax(logits: Any) -> Any:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def _assemble(
    tokenizer: Any, input_ids: Any, tags: Any, confidences: Any
) -> Iterator[list[DiacriticConfidence]]:
    """Pair each real input letter with its predicted tag and confidence."""
    from catt_tashkeel import bw2ar

    space_id = tokenizer.letters_map[" "]
    width = tags.shape[1]

    for row in range(len(input_ids)):
        positions = []
        for index, letter_id in enumerate(input_ids[row][1 : 1 + width]):
            letter = tokenizer.letters[letter_id]
            if letter in _EMPTY_TAGS:  # <EOS>, or padding past the end
                break
            tag = tokenizer.tashkeel_list[tags[row][index]]
            positions.append(
                DiacriticConfidence(
                    letter=bw2ar.transliterate_word(letter, "bw2ar"),
                    diacritic=bw2ar.transliterate_word(_tag_to_buckwalter(tokenizer, tag), "bw2ar"),
                    confidence=float(confidences[row][index]),
                    forced=bool(letter_id == space_id),
                )
            )
        yield positions


def _tag_to_buckwalter(tokenizer: Any, tag: str) -> str:
    """Render a tokenizer tag the way `combine_tashkeel_with_text` would."""
    if tag in tokenizer.tags:  # composite shadda tags, e.g. <SF> -> ~a
        return tokenizer.tags[tag]
    if tag == tokenizer.no_tashkeel_tag or tag in _EMPTY_TAGS:
        return ""
    return tag
