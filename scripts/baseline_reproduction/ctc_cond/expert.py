"""Forked from s3prl `downstream/ctc/expert.py`.

Materialized into the pinned s3prl checkout as `downstream/ctc_cond/expert.py`
by `run/prepare_baseline.sh`. This file is **glue and nothing else**: it wires
s3prl's runner to `arabic_mdd.models`, where the architecture, the loss, the
vocabulary conventions and the collapse diagnostic all live under test. If a
decision ever wants to be made in this file, that is the signal it belongs in
`src/` instead.

**Why fork rather than reimplement the training loop.** The arms have to be
compared against the baseline and against task 6's XLS-R run, and those use
s3prl's optimizer, scheduler, specaug, gradient clipping, bucketing,
checkpointing and step accounting. Reproducing all of that in our own loop
would make every difference in the results ambiguous between "the conditioning
did it" and "our loop differs somewhere". So the regime is s3prl's, unchanged,
and only the model and the extra input column are ours.

**Why the model is imported rather than copied.** `MddModel` is 51 tests deep
in `src/arabic_mdd/models/`, and a copy here would be untested surface in a
ruff-excluded directory. The import is what makes the two arms one
implementation. It is also the reason `src/arabic_mdd/models/` is held to
Python 3.8 syntax -- see that package's `mdd.py` docstring.

**The three substantive differences from upstream.**

1. `self.model` is `MddModel`, built from `MddConfig`, and the projector lives
   inside it. Upstream keeps `self.projector` on the expert and passes already
   projected features to the head; here the projector is part of the model so
   that the conditioning block sits between the two, which is where the arms
   diverge.
2. `forward` takes a `canonicals` positional argument. `runner.py`'s
   `for (wavs, *others)` / `self.downstream.model(split, features, *others, ...)`
   delivers the 4-tuple from `ctc_cond/data.py`'s collate function with no
   change to `runner.py` itself.
3. Evaluation writes a third file, `{split}-canon.ark`, next to upstream's
   `{split}-hyp.ark`/`{split}-ref.ark`. That is the whole input to
   `arabic_mdd.models.collapse.collapse_report`, which answers the question
   this arm cannot be trusted without: did it read the audio, or echo the
   canonical? Writing it costs one file handle and makes the diagnostic
   computable offline, after the 40-hour run, from artifacts the run already
   had in memory.

`text.py` and `metric.py` are *not* forked -- they are imported from the
sibling `..ctc` package, so the tokenizer and the WER that gate checkpoint
selection are byte-identical to the baseline's.
"""

import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from arabic_mdd.models.mdd import MddConfig, MddModel

from ..ctc.metric import *  # noqa: F401,F403  -- `eval(metric)` needs these by name
from ..ctc.text import load_text_encoder
from .data import load_dataset


class DownstreamExpert(nn.Module):
    """s3prl's CTC expert, running `MddModel` and carrying a canonical column."""

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.expdir = expdir
        self.upstream_dim = upstream_dim
        self.corpus = downstream_expert["corpus"]

        # Upstream's tokenizer, unforked: it fixes pad==blank==0, eos==1,
        # unk==2 and the unsorted file order of the 68 phonemes. Our
        # `arabic_mdd.models.vocab` mirrors those conventions for use in the
        # sandbox, but on the cluster the real one is authoritative.
        self.tokenizer = load_text_encoder(**downstream_expert["text"])

        modelrc = downstream_expert["model"]
        rnnrc = modelrc["RNNs"]

        # `upstream_rate` is the upstream's output stride in samples (320 for
        # every wav2vec-family model here). s3prl's head derives a time
        # downsampling factor from `total_rate / upstream_rate`; our port
        # implements only the factor-1 case and `MddConfig` raises otherwise, so
        # a config that would have downsampled fails here instead of silently
        # training a different architecture.
        total_rate = rnnrc["total_rate"]
        sample_rate = 1 if total_rate == -1 else round(total_rate / upstream_rate)

        config = MddConfig(
            vocab_size=self.tokenizer.vocab_size,
            upstream_dim=upstream_dim,
            project_dim=modelrc["project_dim"],
            rnn_dims=tuple(rnnrc["dim"]),
            rnn_dropout=tuple(rnnrc["dropout"]),
            rnn_layer_norm=tuple(rnnrc["layer_norm"]),
            rnn_proj=tuple(rnnrc["proj"]),
            rnn_module=rnnrc["module"],
            bidirectional=rnnrc["bidirection"],
            sample_rate=sample_rate,
            zero_infinity=modelrc["zero_infinity"],
            # The arm switch, and the only line of this config that differs
            # between the two runs being compared.
            use_canonical=modelrc["use_canonical"],
            conditioner_heads=modelrc.get("conditioner_heads", 8),
            conditioner_dropout=modelrc.get("conditioner_dropout", 0.0),
            padding_idx=self.tokenizer.pad_idx,
        )
        self.model = MddModel(config)
        self.use_canonical = config.use_canonical

        # Stated at startup so the log says which arm it is. A conditioned run
        # misconfigured as prompt-free would otherwise be indistinguishable
        # from the ablation until the results disagreed.
        total = sum(p.numel() for p in self.model.parameters())
        conditioning = sum(p.numel() for p in self.model.conditioning_parameters)
        print(
            "[ctc_cond] use_canonical={}  params={:,} (conditioner {:,}, "
            "constructed in both arms for RNG parity)".format(
                self.use_canonical, total, conditioning
            )
        )

        self.objective = nn.CTCLoss(
            blank=self.tokenizer.pad_idx,
            zero_infinity=modelrc["zero_infinity"],
        )
        self.save_best_on = downstream_expert.get("save_best_on", "dev")
        self.metrics = downstream_expert["metric"]
        self.metric_higher_better = downstream_expert["metric_higher_better"]
        self.register_buffer(
            "best_score", torch.ones(1) * (0 if self.metric_higher_better else 1 << 31)
        )

    def _get_task_name(self):
        return "ctc_cond-{}".format(self.corpus["name"].lower())

    # Interface
    def get_dataloader(self, split):
        return load_dataset(split, self.tokenizer, self.corpus)

    # Interface
    def forward(self, split, features, labels, canonicals, filenames, records, **kwargs):
        device = features[0].device

        labels = [torch.LongTensor(label) for label in labels]
        features_len = torch.IntTensor([len(feat) for feat in features])
        labels_len = torch.IntTensor([len(label) for label in labels])
        features = pad_sequence(features, batch_first=True)
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.tokenizer.pad_idx,
        ).to(device=device)

        # Normalized to plain `int` once, up front, and used for both the
        # tensors and the decode below. `data.py` hands these over as numpy
        # arrays, and `tokenizer.idx_to_vocab` is a bare `self._vocab_list[idx]`
        # -- a numpy int indexes a list, a 0-dim tensor raises. Upstream dodges
        # the same edge by calling `.tolist()` before decoding the labels.
        canonical_rows = [[int(i) for i in canonical] for canonical in canonicals]

        # The canonical is padded with the same index, which is also the CTC
        # blank -- harmless because it never reaches the loss, and masked out of
        # the attention by `canonical_lens`.
        canonical_ids = None
        canonical_lens = None
        if self.use_canonical:
            canonical_lens = torch.LongTensor([len(c) for c in canonical_rows]).to(device=device)
            canonical_ids = pad_sequence(
                [torch.LongTensor(c) for c in canonical_rows],
                batch_first=True,
                padding_value=self.tokenizer.pad_idx,
            ).to(device=device)

        logits, log_probs_len = self.model(features, features_len, canonical_ids, canonical_lens)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        loss = self.objective(
            log_probs.transpose(0, 1),  # (N, T, C) -> (T, N, C)
            labels,
            log_probs_len,
            labels_len,
        )
        records["loss"].append(loss.item())

        pred_tokens = log_probs.argmax(dim=-1)
        filtered_tokens = []
        for pred_token in pred_tokens:
            pred_token = pred_token.unique_consecutive()
            filtered_token = [
                token
                for token in pred_token.tolist()
                if token != self.tokenizer.pad_idx and token != self.tokenizer.eos_idx
            ]
            filtered_tokens.append(filtered_token)
        hypothesis = [self.tokenizer.decode(h) for h in filtered_tokens]
        groundtruth = [self.tokenizer.decode(g.tolist()) for g in labels]

        records["hypothesis"] += hypothesis
        records["groundtruth"] += groundtruth
        records["filename"] += filenames
        # Decoded through the same tokenizer as the other two, so the three
        # .ark files are comparable token-for-token offline. Recorded in both
        # arms: the prompt-free arm's copy rate against a canonical it never
        # saw is the null distribution the conditioned arm's number is read
        # against, and it costs nothing to collect.
        records["canonical"] += [self.tokenizer.decode(c) for c in canonical_rows]

        return loss

    # interface
    def log_records(self, split, records, logger, global_step, **kwargs):
        loss = torch.FloatTensor(records["loss"]).mean().item()
        results = {"loss": loss}

        for metric in self.metrics:
            results[metric] = eval(metric)(
                hypothesis=records["hypothesis"],
                groundtruth=records["groundtruth"],
            )

        save_names = []
        for key, value in results.items():
            print(f"{split} {key}: {value}")

            logger.add_scalar(
                f"{self._get_task_name()}/{split}-{key}", value, global_step=global_step
            )
            if key == self.metrics[0]:
                save_criterion = (
                    value > self.best_score
                    if self.metric_higher_better
                    else value < self.best_score
                )
                if split in self.save_best_on and save_criterion:
                    self.best_score = torch.ones(1) * value
                    save_names.append(f"{split}-best.ckpt")

        if "test" in split or "dev" in split:
            # Upstream's two files plus the canonical. All three are written in
            # the same `zip`, so a row index means the same utterance in each --
            # which is what lets `collapse_report` be computed offline without
            # re-deriving an alignment between files.
            hyp_ark = open(os.path.join(self.expdir, f"{split}-hyp.ark"), "w")
            ref_ark = open(os.path.join(self.expdir, f"{split}-ref.ark"), "w")
            canon_ark = open(os.path.join(self.expdir, f"{split}-canon.ark"), "w")
            for filename, hyp, ref, canon in zip(
                records["filename"],
                records["hypothesis"],
                records["groundtruth"],
                records["canonical"],
            ):
                hyp_ark.write(f"{filename} {hyp}\n")
                ref_ark.write(f"{filename} {ref}\n")
                canon_ark.write(f"{filename} {canon}\n")
            hyp_ark.close()
            ref_ark.close()
            canon_ark.close()

        return save_names
