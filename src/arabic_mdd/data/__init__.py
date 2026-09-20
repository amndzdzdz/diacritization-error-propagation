"""Dataset loading and preparation (QuranMB.v2, Iqra_train, Common Voice Arabic)."""

from arabic_mdd.data.iqra_train import TrainingExample, load_iqra_train, load_iqra_tts
from arabic_mdd.data.phonemes import parse_phoneme_sequence
from arabic_mdd.data.quranmb import (
    MDDGroundTruth,
    load_audio,
    load_ground_truth,
    score_predictions,
)

__all__ = [
    "MDDGroundTruth",
    "TrainingExample",
    "load_audio",
    "load_ground_truth",
    "load_iqra_train",
    "load_iqra_tts",
    "parse_phoneme_sequence",
    "score_predictions",
]
