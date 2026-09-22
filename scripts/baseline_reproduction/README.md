# Baseline reproduction (mHuBERT + BiLSTM + CTC, via S3PRL)

Vendored, lightly-annotated copy of the organizers' own reproduction recipe
from `github.com/Iqra-Eval/interspeech_IqraEval` (commit on `main`, fetched
2026-09-20). Everything in this directory is what the organizers used to
produce the published F1 = 44.14% number on `QuranMB.v2` — kept here
largely verbatim (not reimplemented) per the week 2 decision documented in
[docs/weeks/week-02.md](../../docs/weeks/week-02.md): a from-scratch
reimplementation risks an ambiguous gate result if it fails to reproduce,
whereas running the organizers' own code removes that risk.

**Not fully verbatim — two schema-drift fixes.** `download_hugg_data.py`
and `download_hugg_data_tts.py`, as fetched from the upstream repo, read
column names (`ID`/`phoneme`/`phoneme_aug`) and default `--path` values
from different, older datasets than `IqraEval/Iqra_train`/`IqraEval/Iqra_TTS`
— the ones the organizers' own README instructs running them against. As
vendored, they would raise `KeyError` on the real, current dataset schemas
(confirmed via `datasets-server.huggingface.co`: `id`/`phoneme_ref` for
Iqra_train, `phoneme_mis` for Iqra_TTS — see the `NOTE` comment at the top
of each file). Patched to match; the download/save logic itself is
untouched. `generate_len_for_bucket_sdaia.py` also gained a `--splits`
flag so it can run non-interactively (the original blocks on an
`input()` prompt, which hangs forever inside a batch job) — the
interactive path still works unchanged for manual use.

## Fastest path: fully automated

```bash
sbatch run/train_baseline.slurm
```

See [run/train_baseline.slurm](../../run/train_baseline.slurm) and
[run/prepare_baseline.sh](../../run/prepare_baseline.sh) — one `sbatch`
call handles S3PRL environment/install, dataset download, data prep,
config wiring, training, and evaluation, and is safe to resubmit if it's
interrupted partway through. The manual steps below are the same
pipeline, kept for reference/debugging if something in the automated path
needs inspecting.

**S3PRL requires Python 3.8, not this project's own Python 3.12/`uv`
stack.** Do not `uv add` anything from this directory. It runs in its own
plain `venv` (stdlib, no conda and no uv — neither is assumed to be
installed or installable on the cluster), built from whatever `python3.8`
interpreter is already available there. `run/prepare_baseline.sh` looks
for `python3.8` on PATH, then a `python3` reporting version 3.8.x, then a
few common install paths; if none of those find it, set `PYTHON38_BIN`
yourself (e.g. after `module load python/3.8`) before submitting. This
sandbox has no GPU and no cluster access — this is why these are
instructions for you to run, not a script I can run myself.

## Contents

```
config/sws.yaml                  Official training config (verbatim)
vocab/sws_arabic.txt             Official 68-phoneme vocab (verbatim)
download_hugg_data.py            Pull IqraEval/Iqra_train -> wav + transcript files
download_hugg_data_tts.py        Pull IqraEval/Iqra_TTS -> wav + transcript files
generate_len_for_bucket_sdaia.py S3PRL bucketing metadata (audio length sorting)
csv_to_tsv_with_transcripts.py   Bucketing CSV -> S3PRL-ready TSV (path, sentence)
get_units.py                     Regenerate vocab from transcripts (sanity check only)
s3prl_inference.py               Run a trained/pretrained checkpoint over a wav directory
inspect_s3prl_ckpt.py            Print the Args/Config a checkpoint was trained with
mdd_eval/                        Official TA/TR/FA/FR scorer (verbatim)
```

`mdd_eval/` was fetched from the live leaderboard HF Space
(`huggingface.co/spaces/IqraEval/Leaderboard/raw/main/mdd_eval/`), which
carries it even though `github.com/Iqra-Eval/interspeech_IqraEval` is the
canonical citation. It needs only pandas/numpy/stdlib, so unlike everything
else in this directory it runs in this project's own env — see
`scripts/compare_metric_to_official.py`, which uses it to cross-check
`src/arabic_mdd/metrics/hierarchical.py`.

## Step 1 — Python 3.8 venv + S3PRL environment

```bash
python3.8 -m venv s3prl_venv   # or whatever your cluster's python3.8 binary is called
source s3prl_venv/bin/activate
git clone https://github.com/s3prl/s3prl.git
cd s3prl
pip install -e ".[all]"
pip install datasets  # for the download_hugg_data*.py scripts below
```

## Step 2 — pull the training data

Same HF dataset IDs this project's own `src/arabic_mdd/data/iqra_train.py`
loaders use (`IqraEval/Iqra_train`, `IqraEval/Iqra_TTS`), but here
materialized to wav+transcript files on disk, since S3PRL expects files,
not an in-memory `datasets.Dataset`.

```bash
python download_hugg_data.py --path "IqraEval/Iqra_train" --split "train" --output_dir "./sws_data/CV-Ar"
python download_hugg_data.py --path "IqraEval/Iqra_train" --split "dev"   --output_dir "./sws_data/CV-Ar"
python download_hugg_data_tts.py --path "IqraEval/Iqra_TTS" --split "train" --output_dir "./data/TTS" --dev_name "Amer"
```

## Step 3 — bucketing + TSV prep (inside your `s3prl` checkout)

```bash
# a) copy generate_len_for_bucket_sdaia.py into s3prl/s3prl/preprocess/, then:
python generate_len_for_bucket_sdaia.py -i path_to_downloaded_data/sws_data/CV-Ar -o ../data/CV-Ar/
# -> s3prl/s3prl/data/CV-Ar/len_for_bucket/{train,dev}.csv

# b) copy csv_to_tsv_with_transcripts.py into s3prl/s3prl/, then:
python csv_to_tsv_with_transcripts.py \
  --csv_path s3prl/s3prl/data/CV-Ar/len_for_bucket/train.csv \
  --transcript_root "./sws_data/CV-Ar/" \
  --output_path "./sws_data/CV-Ar/train/train.tsv"
python csv_to_tsv_with_transcripts.py \
  --csv_path s3prl/s3prl/data/CV-Ar/len_for_bucket/dev.csv \
  --transcript_root "./sws_data/CV-Ar/" \
  --output_path "./sws_data/CV-Ar/dev/dev.tsv"
```

## Step 4 — downstream task setup

```bash
cp vocab/sws_arabic.txt s3prl/s3prl/downstream/ctc/cv_vocab/sws_arabic.txt
cp config/sws.yaml       s3prl/s3prl/downstream/ctc/cv_config/sws.yaml
```

Edit the copied `sws.yaml`'s `downstream_expert.corpus.path` and
`.train`/`.dev`/`.test` to point at the `train.tsv`/`dev.tsv` produced in
step 3 (paths in the vendored copy are placeholders,
`path_to_tsv_file/...`).

**Config vs. paper discrepancy — resolved.** The paper's prose describes
training for 12.5k updates; the shipped config sets `runner.total_steps:
200000`. Week 2 chose to treat the config as source of truth, and reading
the organizers' published checkpoint confirms that was right: it records
`Step = 200000` with `total_steps: 200000`. The prose's 12.5k does not
describe this baseline. Still worth flagging in the writeup, since anyone
budgeting a reproduction from the paper alone will under-provision by ~16x.

## Step 5 — train and evaluate

```bash
exp_dir='mhubert147_per'
python3 run_downstream.py -m train -c downstream/ctc/cv_config/sws.yaml -p ${exp_dir} \
    -u hf_hubert_custom -k utter-project/mHuBERT-147 -d ctc
python3 run_downstream.py -m evaluate -e ${exp_dir}/dev-best.ckpt
```

**⚠ The organizers' own README says `-u hubert_base` here.** That is English
HuBERT Base (LibriSpeech); training with it scored F1 = 0.4058 against the
published 0.4414. The identifier above is the real one, read out of the
organizers' published checkpoint with `inspect_s3prl_ckpt.py`: frozen
(`upstream_trainable: False`) with SUPERB weighted layer-sum over
`hidden_states` (13 layer weights → base-size, 12 transformer layers).
`run/train_baseline.slurm` uses this. See
[insights/week-03.md](../../insights/week-03.md).

This step is a multi-hour GPU job — run it on your cluster, not in this
sandbox.

### Alternative: skip training, score the organizers' trained checkpoint

The organizers publish their *trained* baseline checkpoint (public,
ungated), so the gate can be checked with inference only — no training run,
and no dependence on getting the upstream identifier right:

```bash
sbatch run/quranmb_gate_check.slurm
```

This is `run/quranmb_gate_check.slurm`'s **default** — it passes the
checkpoint URL straight through to `S3PRLModel`, which downloads it via
`download_if_needed`, and writes to
`run/quranmb_predictions_official_ckpt.json`. To score a locally trained
checkpoint instead, set `EXP_DIR` or `CKPT`; the output then defaults to
`run/quranmb_predictions.json`, so the two never overwrite each other.

## Step 6 — score against the gate (back in this project, uv/Python 3.12)

Once you have predictions (from either step 5's `evaluate` or the
`s3prl_inference.py` CSV), get them into `{id: predicted_phoneme_string}`
form and score them with this project's own metric implementation — no
need to touch the organizers' `mdd_eval/` scripts, since
`src/arabic_mdd/metrics/hierarchical.py` is a from-scratch, unit-tested
port of the same TA/TR/FA/FR/CD/ED algorithm:

```python
from arabic_mdd.data.quranmb import load_ground_truth, score_predictions
from arabic_mdd.metrics.hierarchical import compute_metrics

ground_truth = load_ground_truth()
predictions = {...}  # id -> predicted phoneme string, from your S3PRL run
counts = score_predictions(predictions, ground_truth)
metrics = compute_metrics(counts)
print(metrics.f1)  # compare against the gate: 0.4414 +/- 0.02
```

See [docs/weeks/week-03.md](../../docs/weeks/week-03.md) for the full
gate procedure and pass/fail handling.
