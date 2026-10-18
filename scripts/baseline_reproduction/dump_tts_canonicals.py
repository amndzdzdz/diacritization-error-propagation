"""Write `Iqra_TTS`'s `phoneme_ref` to disk as `canonicals/canonical_<name>.txt`.

New file, not a patch of a vendored one. `download_hugg_data_tts.py` writes
only `phoneme_mis` (the mispronunciation that was actually synthesized, which is
the correct CTC target and what the baseline trains on). The text-dependent arm
additionally needs `phoneme_ref` -- the sequence the speaker was *supposed* to
produce -- and nothing in the organizers' pipeline puts it anywhere.

Patching the downloader to write both in one loop would be structurally safer,
and would also be useless here: `prepare_baseline.sh` skips the download when
the audio already exists, so on a cluster that has already fetched 131 h the
patched loop would never run. This script is the backfill, designed to be safe
to run against an already-populated data directory.

**The alignment problem, and how it is checked rather than assumed.** TTS
filenames are `audio_{speaker}_{idx}.wav` where `idx` is the position in
`enumerate(load_dataset(..., streaming=True))` -- a stream position, not a
dataset field. Re-deriving it means re-streaming in the same order, which is
deterministic for a fixed dataset revision but is exactly the kind of
assumption that fails quietly: a reordered stream would pair every canonical
with the wrong utterance, and the arm would train, converge, and be wrong.

So the order is verified per row, not trusted. For each row this script reads
the `transcript_{speaker}_{idx}.txt` already on disk and requires it to equal
this row's `phoneme_mis` (after the same `<sil>` removal the downloader
applies). A reordered stream fails on the first row instead of on no row at
all. `--max-mismatch` does **not** make a mismatch survivable -- any mismatch
at all is fatal. It only sets how many are collected before the scan stops, so
that "one row is off" can be told apart from "the whole stream is shifted".

**`<sil>` is stripped**, matching what the downloader does to the transcript it
writes. It is not one of the 68 tokens in `vocab/sws_arabic.txt`, so leaving it
in would put a silent `<unk>` at every silence boundary of every canonical.

**The audio column is dropped before iterating.** With `streaming=True` this
prunes it from the parquet read, so 52 h of audio is never fetched or decoded
to recover two text columns.

    python scripts/baseline_reproduction/dump_tts_canonicals.py \
        --output_dir run/data/TTS

Then re-merge into the combined train split, which `prepare_baseline.sh` step 4
does, or by hand:

    cp -n run/data/TTS/train/canonicals/*.txt run/data/CV-Ar/train/canonicals/
"""

import argparse
import os
import sys

from datasets import load_dataset


def strip_sil(phonemes):
    """The downloader's own normalization, applied to `phoneme_ref` as well."""
    return " ".join(x for x in phonemes.split() if x != "<sil>")


def main(args):
    dataset = load_dataset(args.path, split=args.split, streaming=True)
    # Never fetch the audio: the two text columns are all this needs, and with
    # streaming the pruned column is not read from the parquet at all.
    if "audio" in (dataset.column_names or []):
        dataset = dataset.remove_columns(["audio"])

    written = 0
    skipped_no_transcript = 0
    mismatches = []

    for idx, data in enumerate(dataset):
        speaker = data["speaker"].replace(" ", "_")
        # The downloader splits one speaker off as its own directory; mirror
        # that so the canonical lands next to the transcript it belongs to.
        split_dir = args.dev_name if speaker == args.dev_name else args.split
        base = os.path.join(args.output_dir, split_dir)
        transcript_path = os.path.join(base, "transcripts", f"transcript_{speaker}_{idx}.txt")

        if not os.path.isfile(transcript_path):
            # The row was never downloaded (a partial fetch, or a different
            # --dev_name). Not an alignment failure, so counted separately.
            skipped_no_transcript += 1
            continue

        with open(transcript_path, "r", encoding="utf-8") as handle:
            on_disk = handle.read().replace("\n", "").strip()
        expected = strip_sil(data["phoneme_mis"])
        if on_disk != expected:
            mismatches.append((transcript_path, on_disk, expected))
            if len(mismatches) > args.max_mismatch:
                break
            continue

        canonical_dir = os.path.join(base, "canonicals")
        os.makedirs(canonical_dir, exist_ok=True)
        canonical_path = os.path.join(canonical_dir, f"canonical_{speaker}_{idx}.txt")
        with open(canonical_path, "w", encoding="utf-8") as handle:
            handle.write(strip_sil(data["phoneme_ref"]))
        written += 1

        if idx % 500 == 0:
            print("Processed %d rows (%d written)..." % (idx, written))

    print("wrote %d canonicals under %s" % (written, args.output_dir))
    if skipped_no_transcript:
        print(
            "skipped %d rows with no transcript on disk (not downloaded, or a "
            "different --dev_name)" % skipped_no_transcript
        )

    if mismatches:
        print("", file=sys.stderr)
        print(
            "ERROR: %d row(s) disagreed with the transcript already on disk, so the "
            "stream order no longer matches the one the download used." % len(mismatches),
            file=sys.stderr,
        )
        path, on_disk, expected = mismatches[0]
        print("  %s" % path, file=sys.stderr)
        print("  on disk:  %s" % on_disk[:90], file=sys.stderr)
        print("  streamed: %s" % expected[:90], file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "Do NOT pass --max-mismatch to get past this. The filename index is a\n"
            "stream position, so a reordered stream pairs every canonical with the\n"
            "wrong utterance -- which trains, converges, and is wrong. Re-run\n"
            "download_hugg_data_tts.py so the audio, the transcript and the canonical\n"
            "all come from one pass.",
            file=sys.stderr,
        )
        sys.exit(1)

    if written == 0:
        print(
            "ERROR: no canonicals written. Nothing downstream will say so -- "
            "build_canonical_manifest.py will simply refuse every TTS row.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dump Iqra_TTS phoneme_ref as canonical files.")
    parser.add_argument("--path", type=str, default="IqraEval/Iqra_TTS")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="the same --output_dir download_hugg_data_tts.py was given",
    )
    parser.add_argument(
        "--dev_name",
        type=str,
        default="Amer",
        help="must match the --dev_name used at download time, or nothing lines up",
    )
    parser.add_argument(
        "--max-mismatch",
        type=int,
        default=0,
        help=(
            "how many transcript disagreements to collect before stopping the scan. "
            "Any disagreement is fatal either way; this only distinguishes one bad "
            "row from a wholly shifted stream."
        ),
    )
    main(parser.parse_args())
