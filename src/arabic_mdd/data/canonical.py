"""Add the canonical column that the text-dependent arm conditions on.

The baseline's manifest is `(path, sentence)`, where `sentence` is the CTC
target. The conditioned arm needs a third column, `canonical` -- the phoneme
sequence the speaker was *supposed* to produce. This module decides, row by
row, where that third value comes from, and it is a decision rather than a
lookup because **the two halves of the training set answer it differently**:

* **`Iqra_train` rows** (`audio_<digits>.wav`, 79 h). `download_hugg_data.py`
  writes `transcript = data["phoneme_ref"]`, so the inlined `sentence` **is**
  the canonical. No extra file is needed and none is read.
* **`Iqra_TTS` rows** (`audio_<speaker>_<idx>.wav`, 52 h, merged into the same
  train directory by `prepare_baseline.sh` step 4).
  `download_hugg_data_tts.py` writes `transcript = data["phoneme_mis"]` -- the
  *mispronounced* sequence that was actually synthesized. Here `sentence` is
  emphatically not the canonical, and `phoneme_ref` is not on disk at all until
  `scripts/baseline_reproduction/dump_tts_canonicals.py` puts it there.

**Why this distinction is the whole point.** Those 52 h are the only rows in
131 h where reading the audio and echoing the canonical give different answers
-- the only training signal that punishes collapse (see
`arabic_mdd.models.collapse`). A build that quietly set `canonical = sentence`
everywhere would produce a manifest that trains, converges, and scores well,
having deleted exactly the rows that make the text-dependent arm mean anything.
So a TTS row with no canonical file is a hard error here, never a fallback:
`missing_canonical` rows are reported and refused rather than filled in.

**What this module does not do.** It does not fetch anything. `read_canonical`
is injected, so the classification logic is testable without 131 h of audio or
a network, and the filesystem lives in `scripts/build_canonical_manifest.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from arabic_mdd.data.holdout import CONDITIONED_COLUMNS, normalize_transcript, utterance_id

__all__ = [
    "CONDITIONED_COLUMNS",
    "CanonicalBuild",
    "build_canonical_rows",
    "canonical_relative_path",
    "count_unknown_tokens",
    "is_iqra_train_path",
]


def is_iqra_train_path(path: str) -> bool:
    """True for the `audio_<digits>.wav` form, i.e. a row whose `sentence` is the canonical.

    Keyed off `holdout.utterance_id` so there is exactly one regex in this repo
    that knows the manifest's filename convention. The `isdigit` test is what
    separates the two halves: `Iqra_train` ids are numeric
    (`audio_12345.wav`), `Iqra_TTS` names are `audio_<speaker>_<idx>.wav`.
    """
    row_id = utterance_id(path)
    return row_id is not None and row_id.isdigit()


def canonical_relative_path(path: str) -> str:
    """`train/wav/audio_X.wav` -> `train/canonicals/canonical_X.txt`.

    The same substitution `csv_to_tsv_with_transcripts.py` performs to find a
    transcript, against a parallel directory. Mirroring its filename-derived
    join (rather than inventing an index-keyed side table) is deliberate: the
    pairing is then a property of the filename and cannot drift out of
    alignment with the audio.
    """
    row_id = utterance_id(path)
    if row_id is None:
        raise ValueError(
            f"{path!r} is not of the form audio_<id>.wav, so no canonical filename can "
            "be derived from it. A manifest row this filter cannot key is a row the "
            "conditioned arm would train on with no canonical."
        )
    parent = PurePosixPath(path).parent.parent  # strip the `wav/` directory
    return str(parent / "canonicals" / f"canonical_{row_id}.txt")


@dataclass
class CanonicalBuild:
    """Three-column rows, plus the audit trail of where column three came from."""

    rows: list[list[str]] = field(default_factory=list)
    from_sentence: int = 0
    """`Iqra_train` rows, where `sentence` is `phoneme_ref` by construction."""
    from_file: int = 0
    """`Iqra_TTS` rows, where the canonical had to be read from disk."""
    passed_through: int = 0
    """Rows that already carried a canonical, counted apart from the two
    derivations so a rerun's report cannot be mistaken for a fresh build's."""
    missing_canonical: list[str] = field(default_factory=list)
    """TTS paths with no canonical on disk. Fatal; see the module docstring."""

    def report(self) -> dict[str, object]:
        return {
            "rows": len(self.rows),
            "canonical_from_sentence": self.from_sentence,
            "canonical_from_file": self.from_file,
            "canonical_passed_through": self.passed_through,
            "canonical_from_sentence_note": (
                "Iqra_train rows, whose inlined transcript is phoneme_ref -- the "
                "canonical and the CTC target coincide"
            ),
            "canonical_from_file_note": (
                "Iqra_TTS rows, whose transcript is phoneme_mis -- these are the only "
                "rows where reading the audio and echoing the canonical disagree"
            ),
            "missing_canonical": len(self.missing_canonical),
            "missing_canonical_paths": self.missing_canonical[:20],
        }


def build_canonical_rows(
    rows: Iterable[Sequence[str]],
    read_canonical: Callable[[str], str | None],
) -> CanonicalBuild:
    """Turn `(path, sentence)` rows into `(path, sentence, canonical)` rows.

    `read_canonical` takes the *manifest-relative* canonical path returned by
    `canonical_relative_path` and returns its contents, or `None` if there is no
    such file. Returning `None` for a `Iqra_train` row is harmless -- it is
    never called for one -- and fatal for a TTS row.

    Already-three-column input is passed through unchanged, so the step is
    idempotent and a rerun of `prepare_baseline.sh` does not re-derive it.
    """
    build = CanonicalBuild()
    for row in rows:
        full_row = list(row)
        path, sentence = full_row[0], full_row[1]
        if len(full_row) >= 3 and full_row[2].strip():
            build.rows.append(full_row[:3])
            build.passed_through += 1
            continue
        if is_iqra_train_path(path):
            build.rows.append([path, sentence, normalize_transcript(sentence)])
            build.from_sentence += 1
            continue
        text = read_canonical(canonical_relative_path(path))
        if text is None or not text.strip():
            build.missing_canonical.append(path)
            continue
        build.rows.append([path, sentence, normalize_transcript(text)])
        build.from_file += 1
    return build


def count_unknown_tokens(
    rows: Iterable[Sequence[str]], vocabulary: frozenset[str]
) -> tuple[int, list[str]]:
    """`(count, sorted examples)` of canonical tokens outside `vocabulary`.

    Worth a pass because the tokenizer maps an unknown token to `<unk>`
    *silently* (s3prl's `vocab_to_idx`, mirrored by
    `arabic_mdd.models.vocab.to_idx`). A canonical peppered with `<unk>` is a
    degraded conditioning signal that shows up only as a disappointing F1, and
    the cheapest place to catch it is here, while the file is already open.

    The known live case is `<sil>`: `download_hugg_data_tts.py` strips it from
    the transcript it writes but nothing strips it from `phoneme_ref`, and it is
    not one of the 68 tokens in `vocab/sws_arabic.txt`.
    """
    unknown: dict[str, int] = {}
    for row in rows:
        if len(row) < 3:
            continue
        for token in row[2].split():
            if token not in vocabulary:
                unknown[token] = unknown.get(token, 0) + 1
    return sum(unknown.values()), sorted(unknown)
