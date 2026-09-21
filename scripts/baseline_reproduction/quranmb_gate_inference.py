"""Run a trained S3PRL checkpoint over QuranMB.v2 audio, for the week-3 gate.

Project-authored (not vendored) -- reuses `S3PRLModel` from the vendored
`s3prl_inference.py` in this same directory, but drives it over
`safikhan/quran_mbv2_formatted` (see `src/arabic_mdd/data/quranmb.py`)
instead of a local wav directory, and writes clean `{id: predicted_string}`
JSON instead of a pandas CSV with stringified Python lists in a cell.

This is Stage A of a two-stage gate check and must run inside the S3PRL
Python-3.8 venv (needs `s3prl`/`torch`/`torchaudio`, and a GPU for a
reasonable runtime) -- see run/quranmb_gate_check.slurm, which runs this
stage and then Stage B automatically. `arabic_mdd` (this project's own
Python-3.12 package, which does the actual gate scoring) is deliberately
NOT imported here: it lives in a separate uv env this venv doesn't have.

Standalone usage (from inside run/s3prl_checkout/s3prl, S3PRL venv active):
    python /path/to/quranmb_gate_inference.py \
        --ckpt hubert_base_per/dev-best.ckpt \
        --dict_path downstream/ctc/cv_vocab/sws_arabic.txt \
        --output_json quranmb_predictions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3prl_inference import S3PRLModel  # noqa: E402


def _extract_prediction(raw_output: object) -> str:
    """`S3PRLModel.__call__` returns `dummy_records["hypothesis"]` -- a list
    with one decoded phoneme string per sample in the batch. Batch size is
    always 1 here (one wav at a time), so this unwraps to that one string.
    """
    if isinstance(raw_output, (list, tuple)):
        if not raw_output:
            return ""
        return _extract_prediction(raw_output[0])
    return str(raw_output)


def run(ckpt: str, dict_path: str, output_json: str) -> None:
    import datasets

    model = S3PRLModel(ckpt, dict_path)
    # Same public, join-already-done dataset as
    # `arabic_mdd.data.quranmb.load_ground_truth`/`load_audio`.
    ds = datasets.load_dataset("safikhan/quran_mbv2_formatted", split="train")

    predictions: dict[str, str] = {}
    tmp_wav = Path("_quranmb_gate_tmp.wav")
    for idx, row in enumerate(ds):
        example_id = str(row["id"])
        audio = row["audio"]
        # Same decode-then-torchaudio.save round trip as
        # scripts/baseline_reproduction/download_hugg_data.py, which is
        # known to work in this venv for the same HF Audio feature type.
        wav_tensor = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
        torchaudio.save(str(tmp_wav), wav_tensor, audio["sampling_rate"])

        raw_output = model(str(tmp_wav))
        predictions[example_id] = _extract_prediction(raw_output)

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(ds)}...")

    tmp_wav.unlink(missing_ok=True)

    Path(output_json).write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    print(f"Wrote {len(predictions)} predictions to {output_json}")
    # Sanity check: print a couple of examples so a human can eyeball that
    # the predicted strings look like whitespace-separated phonemes, not
    # something unexpected (e.g. raw token ids).
    for example_id, predicted in list(predictions.items())[:3]:
        print(f"  {example_id!r}: {predicted!r}")

    model.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to trained checkpoint.")
    parser.add_argument("--dict_path", required=True, help="Path to vocab/dict file.")
    parser.add_argument(
        "--output_json", default="quranmb_predictions.json", help="Where to write predictions."
    )
    args = parser.parse_args()
    run(args.ckpt, args.dict_path, args.output_json)
