"""Forked from s3prl `downstream/ctc/data.py` + `corpus/common_voice.py`.

Materialized into the pinned s3prl checkout as `downstream/ctc_cond/data.py`
by `run/prepare_baseline.sh`. Kept in this repo so it is version-controlled and
diffable against the original, and kept in `scripts/baseline_reproduction/`
because that directory is ruff-excluded as vendored third-party code -- which
is the correct home for glue and the wrong home for judgment. Every decision
lives in `src/arabic_mdd/models/`, under test.

**The only substantive change: rows are triples, not pairs.** The upstream
dataset yields `(audio_path, target_tokens)`; this one yields
`(audio_path, target_tokens, canonical_tokens)` read from the manifest's third
column, and `collect_audio_batch` carries the canonical through the
length-sorting so it stays row-aligned with the audio. The runner's
`for (wavs, *others)` unpacking then delivers it to the expert as a positional
argument with no change to `runner.py`.

**Why the canonical cannot be derived from the target.** For `Iqra_train` rows
the two coincide -- the baseline's CTC target *is* `phoneme_ref`. For
`Iqra_TTS` rows they do not: the target is `phoneme_mis`, the mispronounced
sequence that was actually synthesized, while the canonical stays
`phoneme_ref`. Those 52 h are the only training signal that distinguishes
reading the audio from echoing the canonical, so collapsing the two columns
would delete exactly the rows that matter. See
`src/arabic_mdd/models/collapse.py`.

Everything else -- bucketing, the oversized-batch halving, descending-length
sorting, the DistributedSampler branch -- is upstream's, unchanged, because it
is part of the training regime the comparison holds fixed.
"""

import csv
from functools import partial
from os.path import join

import torch
import torchaudio
from torch.distributed import is_initialized
from torch.utils.data import DataLoader, DistributedSampler, Dataset

EVAL_BATCH_SIZE = 1


def collect_audio_batch(batch, split, half_batch_size_wav_len=300000):
    """Collects a batch of (audio_path, target_tokens, canonical_tokens)."""

    def audio_reader(filepath):
        wav, sample_rate = torchaudio.load(filepath)
        return wav.reshape(-1)

    # Bucketed batch should be [[(file1,txt1,can1),(file2,txt2,can2),...]]
    if type(batch[0]) is not tuple:
        batch = batch[0]

    # Make sure that batch size is reasonable
    first_len = audio_reader(str(batch[0][0])).size(0)
    if split == "train":
        if first_len > half_batch_size_wav_len and len(batch) > 1:
            batch = batch[: len(batch) // 2]

    # Read batch
    file, audio_feat, audio_len, text, canonical = [], [], [], [], []
    with torch.no_grad():
        for b in batch:
            file.append(str(b[0]).split("/")[-1].split(".")[0])
            feat = audio_reader(str(b[0])).numpy()
            audio_feat.append(feat)
            audio_len.append(len(feat))
            text.append(torch.LongTensor(b[1]).numpy())
            canonical.append(torch.LongTensor(b[2]).numpy())

    # Descending audio length within each batch. The canonical joins the sort
    # so it stays paired with its own utterance; sorting the other four and
    # leaving it alone would silently condition every utterance on a different
    # utterance's canonical, which trains and converges and is wrong.
    audio_len, file, audio_feat, text, canonical = zip(
        *[
            (feat_len, f_name, feat, txt, can)
            for feat_len, f_name, feat, txt, can in sorted(
                zip(audio_len, file, audio_feat, text, canonical),
                reverse=True,
                key=lambda x: x[0],
            )
        ]
    )

    return audio_feat, text, canonical, file


class ConditionedCommonVoiceDataset(Dataset):
    """`CommonVoiceDataset`, reading a third `canonical` column.

    Upstream reads `row[0]`/`row[1]` and ignores the rest, so pointing it at a
    three-column manifest would have run happily and conditioned on nothing.
    The header check here is what makes that failure loud.
    """

    def __init__(self, split, tokenizer, bucket_size, path, **kwargs):
        self.path = path
        self.bucket_size = bucket_size

        file_list, text, canonical = [], [], []
        for s in split:
            with open(s, "r") as fp:
                rows = csv.reader(fp, delimiter="\t")
                for i, row in enumerate(rows):
                    if i == 0:
                        if [c.strip() for c in row[:3]] != ["path", "sentence", "canonical"]:
                            raise ValueError(
                                f"{s}: header is {row!r}, expected "
                                "['path', 'sentence', 'canonical']. The conditioned arm "
                                "needs the canonical column; build it with "
                                "scripts/build_canonical_manifest.py."
                            )
                        continue
                    if len(row) < 3 or not row[2].strip():
                        raise ValueError(
                            f"{s}: row {i} has no canonical ({row!r}). An empty canonical "
                            "becomes an all-masked attention row, which returns NaN "
                            "instead of raising."
                        )
                    file_list.append(join(path, row[0]))
                    text.append(tokenizer.encode(row[1]))
                    # No eos on the canonical: it is an attention input, not a
                    # CTC target, so an end symbol would just be one more key
                    # for every query to attend over.
                    canonical.append(
                        [tokenizer.vocab_to_idx(tok) for tok in row[2].strip().split()]
                    )
                print(f"Found {len(file_list)} samples.")

        self.file_list, self.text, self.canonical = file_list, text, canonical

    def __getitem__(self, index):
        if self.bucket_size > 1:
            index = min(len(self.file_list) - self.bucket_size, index)
            stop = index + self.bucket_size
            return [
                (f_path, txt, can)
                for f_path, txt, can in zip(
                    self.file_list[index:stop],
                    self.text[index:stop],
                    self.canonical[index:stop],
                )
            ]
        return self.file_list[index], self.text[index], self.canonical[index]

    def __len__(self):
        return len(self.file_list)


def create_dataset(split, tokenizer, name, bucketing, batch_size, **kwargs):
    """Only `common_voice` is reachable here; the arm has one corpus."""
    if name.lower() not in {"common_voice", "sbcsae"}:
        raise NotImplementedError(
            f"corpus {name!r}: the conditioned downstream only supports the "
            "Iqra manifests, which ride the common_voice loader."
        )
    if split == "train":
        kwargs["ratio"] = 1.0
        kwargs["offset"] = 0
        loader_bs = 1 if bucketing else batch_size
        bucket_size = batch_size if bucketing else 1
        dataset = ConditionedCommonVoiceDataset(
            kwargs["train"], tokenizer, bucket_size, **kwargs
        )
    else:
        loader_bs = EVAL_BATCH_SIZE
        dataset = ConditionedCommonVoiceDataset(kwargs[split], tokenizer, 1, **kwargs)
    return dataset, loader_bs


def load_dataset(split, tokenizer, corpus):
    num_workers = corpus.pop("num_workers", 12)
    dataset, loader_bs = create_dataset(split, tokenizer, num_workers=num_workers, **corpus)
    collate_fn = partial(collect_audio_batch, split=split)
    if split == "train":
        sampler = DistributedSampler(dataset) if is_initialized() else None
        dataloader = DataLoader(
            dataset,
            batch_size=loader_bs,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=loader_bs,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )
    return dataloader
