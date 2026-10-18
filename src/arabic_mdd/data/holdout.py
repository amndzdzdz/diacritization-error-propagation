"""Drop contaminated utterances from an S3PRL training manifest.

`run/training_holdout.json` (built by `scripts/build_training_holdout.py`) names
the `Iqra_train` rows that share a consonantal skeleton with one of the two
evaluation pools — the 500 hand-annotated MSA rows (`docs/msa-arm.md` §5.2.1)
and the QuranMB.v2 prompts (`docs/weeks/week-07.md` task 5). This module is the
other half: it removes those rows from the TSV that S3PRL's CTC downstream
actually reads, which is the only place the holdout has any effect.

**Two keys, because an id-only filter has a hole.** `run/prepare_baseline.sh`
step 4 merges Iqra_TTS into the same train directory, and Iqra_TTS is
*synthesized from `Iqra_train`'s phoneme refs*. Its filenames are
`audio_<speaker>_<idx>.wav` rather than `audio_<5-digit-id>.wav`, so no id
filter can reach it: an id-only holdout would drop a held-out utterance and
leave a synthesized reading of the same text in training, silently defeating
the whole exercise. The second key is the transcript string itself, which for
this manifest *is* the phoneme sequence (`csv_to_tsv_with_transcripts.py`
inlines `transcript_<id>.txt`, whose contents are `phoneme_ref`) — so it
compares like with like, without needing the Arabic text the TSV does not
carry. Two utterances with identical phoneme sequences are the same text, so a
phoneme match is never a false positive.

Known residual, flagged rather than fixed: Iqra_TTS's *augmented* rows carry
`phoneme_mis` (a deliberate mispronunciation) instead of `phoneme_ref`, so a
corrupted reading of held-out text will not match either key. Iqra_TTS is not
cached locally, so the size of that residue is unmeasured.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CANONICAL_COLUMN",
    "COLUMNS",
    "CONDITIONED_COLUMNS",
    "Holdout",
    "ManifestFilter",
    "filter_rows",
    "load_holdout",
    "normalize_transcript",
    "read_manifest",
    "utterance_id",
    "write_manifest",
]

# `download_hugg_data.py` writes `audio_{id}.wav`; the bucketing step prefixes
# the split-relative directory, so a manifest path looks like
# `train/wav/audio_12345.wav`. Non-greedy anchoring on the basename keeps the
# TTS form (`audio_Amer_17.wav`) parseable too — it simply never matches an
# `Iqra_train` id, which is the point of the second key.
_AUDIO_ID = re.compile(r"audio_(.+)\.wav$")

#: The manifest's two columns, in order. Written back verbatim so a filtered
#: manifest stays a drop-in replacement for the unfiltered one.
COLUMNS = ("path", "sentence")

# The text-dependent arm's manifest adds a third column holding the canonical
# phoneme sequence (`phoneme_ref`), which for `Iqra_TTS` rows differs from
# `sentence` — there `sentence` is `phoneme_mis`, what was actually said. The
# baseline's own two-column manifest stays valid and unchanged, so the week-3
# reproduction reads the same file it always did.
CANONICAL_COLUMN = "canonical"
CONDITIONED_COLUMNS = (*COLUMNS, CANONICAL_COLUMN)


def utterance_id(path: str) -> str | None:
    """The `Iqra_train` utterance id encoded in a manifest `path`, if there is one.

    Returns `None` when the basename is not of the form `audio_<id>.wav`, which
    is a manifest this filter cannot key by id at all — the caller should say so
    out loud rather than silently filtering nothing.
    """
    match = _AUDIO_ID.search(path.strip())
    return match.group(1) if match else None


def normalize_transcript(text: str) -> str:
    """Collapse whitespace in a transcript so the phoneme key survives a reformat.

    Deliberately *not* `normalize.skeleton`: this column holds a phoneme
    sequence, not Arabic script, and the skeleton's diacritic stripping would
    conflate phonemes that differ only in a vowel — exactly the distinction the
    task is about. Whitespace is the only thing that can vary benignly between
    the Arrow column and the inlined transcript file.
    """
    return " ".join(text.split())


@dataclass(frozen=True)
class Holdout:
    """The two key sets, as read from `run/training_holdout.json`."""

    ids: frozenset[str]
    phoneme_refs: frozenset[str]

    @property
    def is_empty(self) -> bool:
        return not self.ids and not self.phoneme_refs


@dataclass
class ManifestFilter:
    """What `filter_rows` kept, and an audit trail of what it did not."""

    # Full rows, not `(path, sentence)` pairs. The text-dependent arm's
    # manifest carries a third `canonical` column, and truncating to two here
    # would strip it during filtering — producing a manifest that trains the
    # conditioned arm with no canonical to condition on.
    kept: list[list[str]] = field(default_factory=list)
    dropped_by_id: list[list[str]] = field(default_factory=list)
    dropped_by_phoneme_ref: list[list[str]] = field(default_factory=list)
    unparseable_paths: list[str] = field(default_factory=list)

    @property
    def rows_in(self) -> int:
        return len(self.kept) + len(self.dropped_by_id) + len(self.dropped_by_phoneme_ref)

    def report(self) -> dict[str, object]:
        """A JSON-serializable summary, for committing next to the run."""
        return {
            "rows_in": self.rows_in,
            "rows_out": len(self.kept),
            "dropped_total": len(self.dropped_by_id) + len(self.dropped_by_phoneme_ref),
            "dropped_by_id": len(self.dropped_by_id),
            "dropped_by_phoneme_ref": len(self.dropped_by_phoneme_ref),
            "dropped_by_phoneme_ref_note": (
                "rows whose id was not held out but whose phoneme sequence was — "
                "chiefly the merged Iqra_TTS syntheses of held-out text"
            ),
            "unparseable_paths": len(self.unparseable_paths),
            "dropped_paths_by_id": [row[0] for row in self.dropped_by_id],
            "dropped_paths_by_phoneme_ref": [row[0] for row in self.dropped_by_phoneme_ref],
        }


def filter_rows(
    rows: Iterable[Sequence[str]], holdout: Holdout, *, require_keys: bool = True
) -> ManifestFilter:
    """Drop every `(path, sentence)` row matching either holdout key.

    `require_keys` guards the mistake this whole module exists to prevent: a
    filter wired in with an empty key set runs happily, writes a manifest
    identical to its input, and produces a contaminated model that *looks*
    held out. An empty `holdout` is therefore an error by default.
    """
    if require_keys and holdout.is_empty:
        raise ValueError(
            "holdout has neither ids nor phoneme refs — refusing to 'filter' nothing, "
            "which would produce a contaminated manifest that looks held out"
        )
    result = ManifestFilter()
    for row in rows:
        full_row = list(row)
        path, sentence = full_row[0], full_row[1]
        row_id = utterance_id(path)
        if row_id is None:
            result.unparseable_paths.append(path)
        # Keyed on `path` and `sentence` only, whatever else the row carries.
        # The canonical column is deliberately *not* a third key: it duplicates
        # `sentence` for every Iqra_train row, so keying on it would drop the
        # same rows twice and corrupt the audit counts.
        if row_id is not None and row_id in holdout.ids:
            result.dropped_by_id.append(full_row)
        elif normalize_transcript(sentence) in holdout.phoneme_refs:
            result.dropped_by_phoneme_ref.append(full_row)
        else:
            result.kept.append(full_row)
    return result


def load_holdout(payload: Mapping[str, object], *, split: str) -> Holdout:
    """Read one split's keys out of a parsed `training_holdout.json`.

    The phoneme-ref key set is shared across splits by construction: it exists
    to catch synthesized re-readings, and `Iqra_TTS` is merged into *train*
    while the text it was synthesized from may have been held out of either
    split. Keying train on only the train ids' phonemes would leave a synthesis
    of a held-out *dev* sentence in training.
    """
    key = f"holdout_{split}_ids"
    if key not in payload:
        raise KeyError(f"{key} missing — is this a training_holdout.json?")
    ids = payload[key]
    refs = payload.get("holdout_phoneme_refs", [])
    if not isinstance(ids, list) or not isinstance(refs, list):
        raise TypeError(f"{key} and holdout_phoneme_refs must both be lists")
    return Holdout(
        ids=frozenset(str(i) for i in ids),
        phoneme_refs=frozenset(normalize_transcript(str(r)) for r in refs),
    )


def read_manifest(path: Path) -> tuple[list[str], list[list[str]]]:
    """`(header, rows)` from an S3PRL CTC TSV.

    Accepts the baseline's `(path, sentence)` and the conditioned arm's
    `(path, sentence, canonical)`. A third column under any *other* name is
    rejected rather than passed through: s3prl's `CommonVoiceDataset` reads
    `row[0]` and `row[1]` and ignores the rest, so a mislabelled third column
    would be carried silently all the way to a conditioned run that trains on
    the wrong field.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} is empty") from None
        if tuple(header) not in (COLUMNS, CONDITIONED_COLUMNS):
            raise ValueError(
                f"{path} header is {header!r}, expected {list(COLUMNS)} or "
                f"{list(CONDITIONED_COLUMNS)}"
            )
        width = len(header)
        rows = []
        for lineno, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != width:
                raise ValueError(
                    f"{path}:{lineno}: {len(row)} fields but the header declares {width}. "
                    "A short row here means a missing canonical, which the conditioned "
                    "arm would read as an empty attention row."
                )
            rows.append(row)
        return header, rows


def write_manifest(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """Write a TSV back, refusing to emit an empty one.

    An empty manifest is how a key mismatch presents itself: S3PRL would start,
    find nothing to train on, and fail somewhere deep in the dataloader hours
    later. Cheaper to refuse here.
    """
    if not rows:
        raise ValueError(
            f"refusing to write an empty manifest to {path} — every row matched the "
            "holdout, which means the keys do not line up with this TSV"
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
