"""Run a trained S3PRL checkpoint over `IqraEval/Iqra_train` dev audio.

Stage A of the MSA-arm base-rate probe (`docs/msa-arm.md` §3.3). Structure
is deliberately identical to `quranmb_gate_inference.py` -- same
`S3PRLModel`, same `{id: predicted_phoneme_string}` JSON output -- only the
dataset differs, so that any difference in the resulting numbers is
attributable to the data rather than to the inference path.

Why this exists: the MSA arm needs to know how often Common Voice Arabic
speakers actually deviate from the prompt. If they essentially never do,
`A == C_gold` everywhere, the detection metric has no positives, and the
arm as designed cannot produce the headline number. Annotating 300-500
utterances to discover that would cost weeks. Instead, run the *already
validated* baseline over the dev audio and compare its output against
`phoneme_ref`: positions where they disagree are model errors UNION real
mispronunciations. Stage B (`scripts/cv_dev_baserate.py`) contrasts that
disagreement rate against the model's known error profile from QuranMB and
draws the stratified sample for the human listening pilot.

Note this split is *in-domain* for the checkpoint -- `Iqra_train` train is
what it was fine-tuned on, and dev is what `dev-best.ckpt` was selected on
(`docs/msa-arm.md` §5.2). That biases the disagreement rate DOWNWARD, which
is the conservative direction here: it makes the probe more likely to
report "no room for real mispronunciations" than less.

Must run inside the S3PRL Python-3.8 venv, with a GPU. See
run/cv_dev_baserate.slurm, which chains this with Stage B.

Standalone usage (from inside run/s3prl_checkout/s3prl, S3PRL venv active):
    python /path/to/cv_dev_inference.py \
        --ckpt mhubert147_per/dev-best.ckpt \
        --dict_path downstream/ctc/cv_vocab/sws_arabic.txt \
        --output_json cv_dev_predictions.json
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

DATASET = "IqraEval/Iqra_train"


def _extract_prediction(raw_output: object) -> str:
    """Unwrap `S3PRLModel.__call__`'s one-element batch of hypotheses."""
    if isinstance(raw_output, (list, tuple)):
        if not raw_output:
            return ""
        return _extract_prediction(raw_output[0])
    return str(raw_output)


def run(
    ckpt: str,
    dict_path: str,
    output_json: str,
    split: str,
    limit: int | None,
    save_audio_dir: str | None,
) -> None:
    import datasets

    model = S3PRLModel(ckpt, dict_path)
    ds = datasets.load_dataset(DATASET, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    # The step-4 listening pilot needs wav files, and this is the only stage
    # that can produce them: Stage B runs in the uv env, where decoding an
    # HF Audio feature needs `torchcodec`, which is not installed (and
    # should not be added just for this). Here we are already decoding every
    # utterance anyway, so keeping a copy is nearly free -- the dev split is
    # ~77 MB. Stage B then just selects the sampled ids out of this
    # directory.
    audio_dir = Path(save_audio_dir) if save_audio_dir else None
    if audio_dir:
        audio_dir.mkdir(parents=True, exist_ok=True)

    predictions: dict[str, str] = {}
    tmp_wav = Path("_cv_dev_tmp.wav")
    for idx, row in enumerate(ds):
        example_id = str(row["id"])
        audio = row["audio"]
        wav_tensor = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
        torchaudio.save(str(tmp_wav), wav_tensor, audio["sampling_rate"])

        predictions[example_id] = _extract_prediction(model(str(tmp_wav)))

        if audio_dir:
            torchaudio.save(str(audio_dir / f"{example_id}.wav"), wav_tensor, audio["sampling_rate"])

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(ds)}...", flush=True)

    tmp_wav.unlink(missing_ok=True)
    if audio_dir:
        print(f"Saved per-utterance wavs to {audio_dir}")

    Path(output_json).write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    print(f"Wrote {len(predictions)} predictions to {output_json}")
    for example_id, predicted in list(predictions.items())[:3]:
        print(f"  {example_id!r}: {predicted!r}")

    model.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path or URL to trained checkpoint.")
    parser.add_argument("--dict_path", required=True, help="Path or URL to vocab/dict file.")
    parser.add_argument("--output_json", default="cv_dev_predictions.json")
    parser.add_argument("--split", default="dev", help="Iqra_train split to run over.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N rows.")
    parser.add_argument(
        "--save_audio_dir",
        default=None,
        help="Also write each utterance's wav here, for the step-4 listening pilot.",
    )
    args = parser.parse_args()
    run(
        args.ckpt,
        args.dict_path,
        args.output_json,
        args.split,
        args.limit,
        args.save_audio_dir,
    )
