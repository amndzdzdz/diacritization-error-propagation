# Week 3 — Baseline reproduction gate

Plan: [docs/weeks/week-03.md](../docs/weeks/week-03.md). Update this file as
each task below produces a result — do not wait until the end of the week.

## Logistics log

Track non-experimental tasks here as they close. One line each, dated.

- [x] Official `mdd_eval/` scorer vendored, 22 Sep 2026,
      `scripts/baseline_reproduction/mdd_eval/` — fetched verbatim from the
      live leaderboard HF Space
      (`huggingface.co/spaces/IqraEval/Leaderboard/raw/main/mdd_eval/`), which
      carries it even though the GitHub repo is the canonical citation.
      Needs only pandas/numpy/stdlib, so it runs in this project's own env
      rather than the S3PRL Python-3.8 venv. Covered by the existing ruff
      exclusion for `scripts/baseline_reproduction`.
- [x] `run/quranmb_gate_check.slurm` now accepts an `http(s)` `CKPT`, 22 Sep
      2026 — `S3PRLModel` already downloaded URLs via `download_if_needed`,
      but the wrapper's `[ ! -f "$CKPT" ]` guard rejected them first. This is
      what lets the organizers' own trained checkpoint be scored without a
      training run. The organizers' checkpoint is now the script's *default*,
      because a bare `sbatch` previously re-ran the already-failed
      `hubert_base` checkpoint and overwrote the predictions this week's
      result rests on.
- [x] Fixed a latent path bug in the same script, 22 Sep 2026: Stage A runs
      with cwd = the s3prl checkout while Stage B runs from the repo root, and
      `quranmb_gate_inference.py` writes to `Path(output_json)` — so a
      *relative* `OUTPUT_JSON` landed inside the s3prl checkout and Stage B
      then couldn't find it. `OUTPUT_JSON` is now resolved against the repo
      root before Stage A. Never triggered, because the only run so far used
      the absolute default.
- [x] Pre-commit no longer rewrites vendored files, 22 Sep 2026 — ruff's
      exclusion lives in `pyproject.toml`, which the `pre-commit-hooks` repo's
      `trailing-whitespace`/`end-of-file-fixer` never read, so they silently
      edited `mdd_eval/` and broke the byte-for-byte guarantee. Both now carry
      `exclude: ^scripts/baseline_reproduction/`.
- [x] `scripts/baseline_reproduction/inspect_s3prl_ckpt.py` added, 22 Sep 2026
      — prints the `Args`/`Config` embedded in any S3PRL downstream
      checkpoint. Needs only torch + stdlib (deliberately does not import the
      vendored `s3prl_inference`, which pulls in `s3prl`/`torchaudio` at module
      load), so it runs in either environment. `run/inspect_ckpt.slurm` wraps
      it for the cluster.
- [x] `run/smoke_train.slurm` added, 22 Sep 2026 — 20-step throwaway training
      run with a job-id-keyed expdir, to prove a new upstream loads before
      committing 48 GPU-hours. Passed before the mHuBERT-147 retrain.
- [x] `transformers` now installed explicitly by `run/prepare_baseline.sh`,
      22 Sep 2026 — `hf_hubert_custom`'s expert imports it at module load, so
      its absence would have killed the 200k-step job at upstream
      construction. Previously relied on s3prl's `[all]` extra pulling it in
      transitively.
- [x] HF cache redirected off NFS in `run/train_baseline.slurm`, 22 Sep 2026 —
      neither it nor `prepare_baseline.sh` set `HF_HOME`/`TMPDIR` at all, so
      mHuBERT-147 (~2.4 GB) plus the Iqra downloads went to the quota-limited
      `$HOME`. Now defaults to node-local scratch, overridable. Deliberately
      *not* trapped for cleanup, unlike the gate-check job: a 48 h job that
      gets resubmitted shouldn't re-download everything.
- [x] `run/evaluate_baseline.slurm` default `EXP_DIR` moved to
      `mhubert147_per`, 22 Sep 2026 — it still pointed at `hubert_base_per`
      and would have silently evaluated the superseded English-HuBERT run.
      Same class of bug as the gate-check `OUTPUT_JSON` default.
- [x] `scripts/verify_quranmb_labels.py` made disk-cheap, 22 Sep 2026 — it
      was downloading the gated dataset's *audio* to read two text columns
      (`remove_columns` after `load_dataset` is too late, and the streaming
      formatter decodes audio anyway). Now streams with `select_columns`,
      which pushes the projection into the parquet reader: nothing cached, no
      `torchcodec` needed.
- [ ] Licence-terms confirmation for `QuranMB.v2`/`Iqra_Extra_IS26` — still
      open, carried from week 2.
- [ ] Wall-clock cost of one XLS-R-300m fine-tune, and the week-7 work-split
      decision — still open, carried from week 1–2.

## Experiment: Baseline gate scoring

### Hypothesis

The trained mHuBERT+BiLSTM+CTC baseline, scored with this project's
hierarchical metric against `QuranMB.v2`, reproduces the organizers'
published F1 = 0.4414 ± 0.02, validating the whole evaluation pipeline
before any downstream RQ work begins.

### Methodology

Cluster job `run/quranmb_gate_check.slurm` ran the trained checkpoint over
QuranMB.v2 audio (Stage A, S3PRL venv) producing
`run/quranmb_predictions.json`, then scored it with
`scripts/quranmb_gate_score.py` (Stage B, this project's env) via
`arabic_mdd.data.quranmb.score_predictions` +
`arabic_mdd.metrics.hierarchical.compute_metrics`.

### Configuration

- Checkpoint: `hubert_base_per/dev-best.ckpt`, trained per
  `run/train_baseline.slurm` — **upstream `-u hubert_base`**.
- Config: `scripts/baseline_reproduction/config/sws.yaml` verbatim
  (`total_steps: 200000`, `eval_step: 5000`, Adam lr 1e-4, batch 16,
  2×1024 BiLSTM, SpecAugment on).
- Ground truth: `safikhan/quran_mbv2_formatted`, 1642 utterances, all 1642
  matched a prediction id.

### Results

**F1 = 0.4058 — gate FAIL** (required [0.4214, 0.4614]).

| | Published baseline | This run |
|---|---|---|
| F1 | 0.4414 | **0.4058** |
| Precision | 0.3093 | 0.2734 |
| Recall | 0.7707 | 0.7869 |
| FR rate | 0.1237 | 0.1503 |
| Correct diagnosis | 0.6120 | 0.5894 |

Raw counts: TA=42324, FR=7489, FA=763, TR=2818, CD=1661, ED=1157.

### Observations

Recall is *higher* than published; precision accounts for the entire
deficit. The gap is ~1,300 excess false rejections (7,489 against the
~6,160 implied by the published 0.1237 FR rate on the same TA+FR
denominator). Recall-intact/precision-down/FR-up is the signature of a
weaker acoustic phoneme recogniser — spurious phoneme errors landing on
correctly-pronounced positions — not of a scoring bug or mislabelled data,
both of which would move recall too.

Having the published *precision and recall* separately, not just F1, is
what made this diagnosable; the gate as written in
[docs/weeks/week-03.md](../docs/weeks/week-03.md) only records F1.

The most likely cause is an upstream mismatch. `run/train_baseline.slurm`
trains with `-u hubert_base` — English HuBERT Base, LibriSpeech — where the
published baseline uses frozen **mHuBERT-147** (94M params, 147 languages).
`scripts/baseline_reproduction/README.md` had already flagged this as an
unresolved stand-in ("if a distinct mHuBERT upstream identifier is required,
check the S3PRL upstream registry before running"); it was never resolved
before the run.

The `total_steps: 200000` vs "12.5k updates" discrepancy flagged in week 2 is
**not currently implicated, but also not excluded**: `run/logs/` came back
empty, so there is no record of how far training actually ran or which step
`dev-best` came from. Worth noting the "12.5k updates" prose is from the
Interspeech benchmark paper (arXiv 2506.07722); whether it describes the
shared-task baseline specifically is unconfirmed.

Two subsets worth recording, since the split is tempting to misread: the
738 `match_type: fuzzy` rows score F1 0.3647 while the 904 `exact` rows
score 0.4459 — i.e. the `exact` subset alone would clear the gate. That is a
biased subset and must not be reported as a pass. See the label-provenance
experiment below for whether the fuzzy rows are mislabelled or merely harder.

### Conclusion

**Gate failed at F1 = 0.4058.** Per plan §"Contingency if the gate fails",
week 4–6 work does not begin. Debugging proceeds in the mandated order:
metric first (below), then data/config, then treat as a genuine finding.

## Experiment: Metric port cross-check against the official scorer

### Hypothesis

`src/arabic_mdd/metrics/hierarchical.py` was re-derived from the official
algorithm rather than copied, and week 2 validated it only on twelve
hand-constructed cases. If the port diverges from the organizers' own
`mdd_eval/` on real data, the gate failure is a scoring artefact rather than
a model result. This is step (1) of the mandated debugging order.

### Methodology

Vendored the official scorer verbatim and ran both implementations over
identical predictions and identical ground truth via
`scripts/compare_metric_to_official.py`. The official path is two-stage,
mirroring how the leaderboard drives it: `align_data.evaluate_from_dfs`
writes three alignment detail files, then
`ins_del_cor_sub_analysis.analyze_alignment` re-parses them into metrics.

### Configuration

- `scripts/baseline_reproduction/mdd_eval/{metric,align_data,ins_del_cor_sub_analysis}.py`
- Predictions `run/quranmb_predictions.json`, 1642 utterances, ground truth
  from `safikhan/quran_mbv2_formatted` for both scorers.

### Results

**Exact agreement on all eight metrics to four decimal places** (F1,
precision, recall, TA rate, FRR, FAR, DER, DetAcc); largest absolute delta
0.0000.

### Observations

Agreement was not guaranteed even for a correct port: the official
`metric.py::Align` is a hand-rolled Needleman–Wunsch whose tie-breaking
among equal-scoring paths is implementation-specific and non-unique, so
small divergences in the raw tallies would have been tolerable. Getting zero
difference across 1642 utterances is a stronger result than the experiment
was designed to require.

Also confirmed by reading the official source: F1 is **detection-only**
(`TR/(TR+FR)`, `TR/(TR+FA)`, corpus-pooled), diagnosis is reported separately
as CD/ED/DER, and there is **no sequence preprocessing at all** — no
`sil`/`sp`/`<unk>` stripping, because none of those tokens exist in the
68-phoneme vocab or in the data.

### Conclusion

**Metric ruled out as the cause of the gate failure.** Debugging step (1)
closed. The week-2 conclusion that the port is trustworthy now holds on real
data, not just hand-constructed cases. Proceed to step (2), data/config.

## Experiment: Organizers' trained checkpoint through our pipeline

### Hypothesis

If the gate failure is our *training run* rather than our evaluation
pipeline, then scoring the organizers' own trained checkpoint through the
identical inference + scoring path should reproduce F1 = 0.4414. If it
does not, the fault is somewhere in the pipeline or the data and the
failure is a genuine reproduction gap.

### Methodology

`run/quranmb_gate_check.slurm` with its new default `CKPT`, the organizers'
public/ungated `IqraEval/Iqra_mhubert_base/mhubert.ckpt`, downloaded by
`S3PRLModel` at job start. Identical Stage A inference
(`quranmb_gate_inference.py`) and identical Stage B scoring
(`scripts/quranmb_gate_score.py`) as the failed run — only the checkpoint
differs.

### Configuration

- Checkpoint: `https://huggingface.co/IqraEval/Iqra_mhubert_base/resolve/main/mhubert.ckpt`
- Vocab: `downstream/ctc/cv_vocab/sws_arabic.txt` (68 phonemes, verbatim).
- Ground truth: `safikhan/quran_mbv2_formatted`, 1642 utterances, all 1642
  matched a prediction id.
- Predictions: `run/quranmb_predictions_official_ckpt.json`.

### Results

**F1 = 0.4415 — gate PASS** (required [0.4214, 0.4614]).

| | Published | Official ckpt | Our `hubert_base` run |
|---|---|---|---|
| F1 | 0.4414 | **0.4415** | 0.4058 |
| Precision | 0.3093 | 0.3088 | 0.2734 |
| Recall | 0.7707 | 0.7744 | 0.7869 |
| FR rate | 0.1237 | 0.1246 | 0.1503 |
| Correct diagnosis | 0.6120 | 0.6101 | 0.5894 |

Raw counts: TA=43606, FR=6207, FA=808, TR=2773, CD=1692, ED=1081.

### Observations

Agreement with the published numbers is to within 0.0001 on F1 and ≤0.004
on every component rate — far tighter than the ±0.02 the gate allows.

The earlier failure analysis is confirmed quantitatively, not just
directionally. Both runs score the same 53,394 phoneme positions and the
same TA+FR denominator of 49,813, so the counts are directly comparable:
the official checkpoint makes **1,282 fewer false rejections** (6,207 vs
7,489). Week 3's prediction from the published rates alone was "~1,300
excess false rejections". Precision moves from 0.2734 to 0.3088 while
recall barely moves (0.7869 → 0.7744, slightly *down*) — exactly the
recall-neutral/precision-up signature expected from swapping in a stronger
acoustic phoneme recogniser.

This also settles the data question without needing the gated labels.
Our third-party `safikhan/quran_mbv2_formatted` join reproduces the
published numbers to 4 decimal places when paired with the organizers' own
model. Labels that had drifted from the audio could not do that. The 738
`match_type: fuzzy` rows are therefore *harder audio*, not mislabelled —
which is what the per-utterance edit distances already suggested, now
confirmed independently.

The `total_steps: 200000` vs "12.5k updates" discrepancy is likewise no
longer load-bearing for the gate, since no training was involved here. It
remains an open question for our *own* training runs.

### Conclusion

**Gate PASSED at F1 = 0.4415.** The evaluation pipeline — metric, data
path, alignment, inference wiring — is validated end to end, and the week
1–3 block's binary gate is met. The 0.4058 shortfall is isolated to our own
training run, whose `-u hubert_base` upstream (English HuBERT Base,
LibriSpeech) stands in for the baseline's frozen mHuBERT-147. Week 4–6 work
is unblocked.

## Finding: the organizers' checkpoint states its own training recipe

S3PRL stores the training `Args` and resolved `Config` inside every
downstream checkpoint. Reading the published `mhubert.ckpt`
(`scripts/baseline_reproduction/inspect_s3prl_ckpt.py`, 22 Sep 2026) settles
three things that had been inferred from prose or left open:

- **Upstream:** `upstream = 'hf_hubert_custom'`, `upstream_ckpt =
  'utter-project/mHuBERT-147'`, `upstream_trainable = False`,
  `upstream_feature_selection = 'hidden_states'`. The featurizer holds 13
  weights — CNN output plus 12 transformer layers — confirming base-size
  mHuBERT under a SUPERB weighted layer-sum. `run/train_baseline.slurm`
  updated accordingly; `-u hubert_base` came from the organizers' *README*,
  not from what they ran.
- **Steps:** `Step = 200000` with `runner.total_steps: 200000`, `Epoch = 1`.
  The benchmark paper's "12.5k updates" does not describe this baseline.
  Week 2's decision to treat the config as source of truth over the prose was
  correct, and a retrain is the expensive 200k-step job, not a cheap one.
- **Our vocab is the right one.** The CTC output layer is 71 units, derivable
  from the optimizer state (`2048 × 71` weight, `71` bias) — our vendored
  68-phoneme `sws_arabic.txt` plus three special tokens. Their config names a
  different file (`sws_tts.txt`), but decoding this checkpoint against
  `sws_arabic.txt` reproduced the published F1 to four decimal places, so the
  two are functionally identical.

Two differences from the vendored `config/sws.yaml` worth carrying forward.
Their corpus is `mixed/{train,dev}.tsv` — real Iqra_train **mixed with**
Iqra_TTS, which matches what `run/prepare_baseline.sh` builds. And `Args`
carries an `override` string setting `config.runner.baseline=superb`,
`freeze_layers=False`, `layer_drop=False`, `upstream_finetune=False`,
`load_feature_extractor_weights=False`, `downstream_pretrained=None` — keys
stock S3PRL's runner does not define, so they trained on a **modified S3PRL
fork**. Every one of those values is the stock default behaviour, so a stock
run should be equivalent, but that is an assumption, not a verified fact.

## Experiment: Full retrain with the correct upstream

### Hypothesis

If the only defect in our training run was the upstream, then retraining
with `hf_hubert_custom` / `utter-project/mHuBERT-147` — everything else
unchanged — should land within run-to-run variance of 0.4414. This is not
a gate attempt (the gate is already passed); it is validation of a
*training recipe* we will need on the MSA arm, where no published number
exists to catch a broken one.

### Methodology

`run/train_baseline.slurm` end to end on the cluster: `prepare_baseline.sh`
data prep, then 200k steps, then `run/quranmb_gate_check.slurm` with
`EXP_DIR=mhubert147_per`. Stock upstream S3PRL, not the organizers' fork.

### Configuration

- Upstream: `-u hf_hubert_custom -k utter-project/mHuBERT-147`, frozen.
- Config: vendored `config/sws.yaml`, 200k steps, Adam 1e-4, batch 16,
  2×1024 BiLSTM + CTC, SpecAugment on.
- Vocab: `sws_arabic.txt` (68 phonemes).
- Predictions: `run/quranmb_predictions.json` (local-checkpoint default).

### Results

**F1 = 0.4406 — gate PASS.**

| | Published | Official ckpt | **Our retrain** | Our `hubert_base` |
|---|---|---|---|---|
| F1 | 0.4414 | 0.4415 | **0.4406** | 0.4058 |
| Precision | 0.3093 | 0.3088 | 0.3086 | 0.2734 |
| Recall | 0.7707 | 0.7744 | 0.7702 | 0.7869 |
| FR rate | 0.1237 | 0.1246 | 0.1241 | 0.1503 |
| Correct diagnosis | 0.6120 | 0.6101 | 0.6167 | 0.5894 |

Raw counts: TA=43633, FR=6180, FA=823, TR=2758, CD=1701, ED=1057. Same
53,394 scored positions and same 49,813 TA+FR denominator as both earlier
runs, so the counts are directly comparable.

### Observations

0.0008 below published on F1, and on the two components it is *closer* to
published than the organizers' own checkpoint was (precision 0.3086 vs
their 0.3088 against 0.3093; recall 0.7702 vs their 0.7744 against 0.7707).

Against the official checkpoint the run differs by 27 positions out of
49,813: 27 fewer false rejections, 15 fewer true rejections, 15 more false
acceptances. Marginally more conservative, but the gap is far below the
±0.01–0.02 run-to-run variance anticipated before the run — tighter than
a frozen upstream with a freshly initialised BiLSTM/CTC head had any right
to be.

This also resolves the open assumption from the checkpoint inspection. The
organizers trained on a **modified** S3PRL fork whose `override` string sets
`config.runner.baseline=superb`, `freeze_layers`, `layer_drop`,
`upstream_finetune`, `load_feature_extractor_weights`,
`downstream_pretrained` — keys stock S3PRL does not define. Reproducing to
0.0008 on stock S3PRL confirms those flags were all no-ops at their chosen
values, as suspected but not verified.

### Conclusion

**The training recipe is independently validated**, not just the evaluation
pipeline. Three independent paths — the organizers' published number, their
checkpoint scored through our pipeline, and our own from-scratch training —
agree within 0.001 F1. Risk (c) below is materially reduced: the upstream,
config, data prep and scoring are all now known-good as a unit, which is the
reference the MSA arm cannot provide for itself.

## Pending

- **Pin `utter-project/mHuBERT-147` to a revision.** The organizers'
  `upstream_revision` is `None`, i.e. whatever the Hub served them; our run
  took whatever the Hub served us. Pin a commit sha before week 4+ runs so
  the validated recipe stays validated.
- **Label provenance** (`scripts/verify_quranmb_labels.py`, written, not yet
  run): now **optional**. Its purpose was to rule out bad labels as the cause
  of the gate failure; the official checkpoint's 0.4415 on these same labels
  does that more decisively. Worth running once for the writeup if `HF_TOKEN`
  access is convenient, not a blocker.
- Licence-terms confirmation for `QuranMB.v2`/`Iqra_Extra_IS26`.
- Wall-clock cost of one XLS-R-300m fine-tune, and the week-7 work-split
  decision.

## Carried into week 4+: what this gate does **not** cover

The gate validates the pipeline for the Qur'anic arm — QuranMB.v2, the
68-phoneme vocab, the organizers' data path. The MSA arm (Iqra_train dev /
Common Voice Ar, plan §Data) changes exactly the pieces the gate held fixed,
and has no published number to check against. Two of these could produce a
false positive for the paper's central claim, so they are recorded here
rather than discovered later.

**Phoneme-inventory mismatch could manufacture the headline result.**
`sws_arabic.txt` is a 68-phoneme Qur'anic inventory. The MSA arm runs
undiacritized text through CATT/Shakkala/Mishkal/Farasa and then a
phonetizer to build canonical `C`. Any phoneme that path emits which the CTC
head cannot produce becomes a **guaranteed false rejection for a purely
mechanical reason** — and that is indistinguishable, in the metric, from
"automatic diacritization corrupts MDD", which is the thing the paper
claims. Diff the phonetizer's output inventory against the 68-token vocab
before generating any MSA numbers; if they differ, decide explicitly whether
to extend the vocab (breaking cross-arm comparability) or to map/drop
(losing coverage) and state the choice in the writeup.

**No reference point makes a weak upstream unfalsifiable.** The `-u
hubert_base` error was catchable here only because 0.4414 existed to
contradict 0.4058. On the MSA arm a silently wrong or weak upstream reads as
"MSA is harder" — a confound sitting directly underneath the Qur'anic-vs-MSA
comparison that is the paper's argument. Mitigations: pin the upstream
identifier and revision for every arm, keep the upstream identical across
arms so the comparison carries no upstream difference, and record
`Args`/`Config` from each trained checkpoint (`inspect_s3prl_ckpt.py`) rather
than trusting the submission script.

Three further items from the same review, lower risk but worth holding:
cross-arm comparability generally (vocab, upstream, training budget should be
pinned across arms, and any deliberate difference stated); separating domain
shift from reference corruption, which the plan's 2×2 handles only if truth
is genuinely held fixed in the bottom-left cell; and the fact that the MSA
arm's `C_gold` must be created by annotation, already flagged in the plan as
the schedule's main risk.
