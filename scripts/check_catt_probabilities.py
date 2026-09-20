"""Week 1 feasibility check: does CATT expose token-level diacritic confidence?

Ad hoc verification script for docs/weeks/week-01.md task 4 / plan §7 item 4.
RQ4 (selective scoring by diacritizer abstention) depends entirely on CATT
exposing per-position confidence for its predicted diacritics. The public
`catt_tashkeel` package (CATTEncoderOnly / CATTEncoderDecoder) only exposes
`do_tashkeel(_batch)`, which returns decoded text with the softmax/argmax
step already applied and discarded.

Finding (see insights/week-01.md for the full write-up): the ONNX decoder
session inside the package *does* return raw per-class logits before argmax
-- `BaseONNXTashkeel._process_batch` computes `preds = self._run_decoder(...)`
and only then applies `np.argmax(preds, axis=-1)`. Calling the same
non-public methods (`_prepare_batch_input`, `_run_encoder`, `_run_decoder`)
directly, and applying softmax ourselves instead of argmax, recovers a
per-character-position confidence score. This works today but relies on
underscore-prefixed (non-public) methods, so it is fragile to upstream
package changes -- flagged as a risk for the Week 4-6 diacritizer pipeline,
which should vendor or pin this logic rather than re-derive it.

This script is a one-off feasibility probe, not a shipped component: it is
deliberately kept out of src/arabic_mdd and untested, per the project's
scope-discipline convention (see CLAUDE.md).

Run with: uv run python scripts/check_catt_probabilities.py
"""

import numpy as np
from catt_tashkeel import CATTEncoderOnly

SAMPLE_TEXT = (
    "وقالت مجلة نيوزويك الأمريكية التحديث الجديد ل إنستجرام "
    "يمكن أن يساهم في إيقاف وكشف الحسابات المزيفة"
)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def main() -> None:
    model = CATTEncoderOnly()

    baseline = model.do_tashkeel(SAMPLE_TEXT, verbose=False)
    print("do_tashkeel() baseline output:")
    print(baseline)
    print()

    # Replicate BaseONNXTashkeel._process_batch, but keep the raw logits
    # instead of discarding them after argmax.
    preprocessed = model._preprocess_text(SAMPLE_TEXT)
    batch_input_ids = model._prepare_batch_input([preprocessed])
    trimmed_input_ids = batch_input_ids[:, 1:-1]  # drop BOS/EOS, as EO does

    enc_src = model._run_encoder(trimmed_input_ids)
    logits = model._run_decoder(enc_src)  # (batch, seq_len, num_tashkeel_classes)
    probs = softmax(logits, axis=-1)

    predictions = np.argmax(logits, axis=-1)
    predictions = model._apply_space_mask(predictions, trimmed_input_ids)
    confidences = np.max(probs, axis=-1)

    reconstructed = model.tokenizer.decode(batch_input_ids, predictions)[0]
    print("Reconstructed from raw logits (should match baseline):")
    print(reconstructed)
    print(f"Matches do_tashkeel() output: {reconstructed == baseline}")
    print()

    letters = [model.tokenizer.letters[i] for i in trimmed_input_ids[0]]
    tashkeel_preds = [model.tokenizer.tashkeel_list[i] for i in predictions[0]]
    print(f"{'char':<6}{'diacritic':<12}confidence")
    for char, diac, conf in zip(letters, tashkeel_preds, confidences[0], strict=True):
        print(f"{char:<6}{diac:<12}{conf:.4f}")

    print()
    print(f"Mean confidence: {confidences[0].mean():.4f}")
    print(f"Min confidence:  {confidences[0].min():.4f}")


if __name__ == "__main__":
    main()
