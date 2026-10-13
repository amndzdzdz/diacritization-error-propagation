from collections.abc import Sequence

import pytest

from arabic_mdd.diacritizers.base import Diacritizer
from arabic_mdd.diacritizers.chunking import diacritize_long, split_on_words


class FakeTool(Diacritizer):
    """Records what it was handed, and refuses over-long input like the real ones."""

    def __init__(self, limit: int | None) -> None:
        self.max_input_chars = limit
        self.calls: list[list[str]] = []

    @property
    def name(self) -> str:
        return "fake"

    def provenance(self) -> dict[str, str]:
        return {"tool": "fake"}

    def diacritize_batch(self, texts: Sequence[str]) -> list[str]:
        for text in texts:
            if self.max_input_chars is not None and len(text) > self.max_input_chars:
                raise RuntimeError(f"input of {len(text)} exceeds {self.max_input_chars}")
        self.calls.append(list(texts))
        return [text.upper() for text in texts]


def test_short_input_is_passed_through_untouched() -> None:
    """Byte-identical to calling the tool directly — including the whitespace."""
    tool = FakeTool(limit=20)
    batch = diacritize_long(tool, ["a  b\tc"])
    assert tool.calls == [["a  b\tc"]]
    assert batch.texts == ["A  B\tC"]
    assert batch.seams == [0]
    assert batch.report() == {"chunked_items": 0, "seams": 0}


def test_over_long_input_is_split_rejoined_and_counted() -> None:
    tool = FakeTool(limit=5)
    batch = diacritize_long(tool, ["ab cd ef gh"])
    assert tool.calls == [["ab cd", "ef gh"]]
    assert batch.texts == ["AB CD EF GH"]
    assert batch.seams == [1]
    assert batch.report() == {"chunked_items": 1, "seams": 1}


def test_all_chunks_go_to_the_tool_in_one_call() -> None:
    """Otherwise a batching tool loses its batching on exactly the long inputs."""
    tool = FakeTool(limit=5)
    diacritize_long(tool, ["ab cd ef", "xy", "gh ij kl"])
    assert tool.calls == [["ab cd", "ef", "xy", "gh ij", "kl"]]


def test_outputs_stay_aligned_to_inputs() -> None:
    tool = FakeTool(limit=5)
    batch = diacritize_long(tool, ["ab cd ef", "xy", "gh ij kl"])
    assert batch.texts == ["AB CD EF", "XY", "GH IJ KL"]
    assert batch.seams == [1, 0, 1]
    assert batch.chunked_items == 2


def test_an_unbounded_tool_is_never_split() -> None:
    tool = FakeTool(limit=None)
    long = " ".join(["word"] * 500)
    batch = diacritize_long(tool, [long])
    assert tool.calls == [[long]]
    assert batch.seams == [0]


def test_empty_batch() -> None:
    assert diacritize_long(FakeTool(limit=5), []).texts == []


def test_split_packs_greedily_up_to_the_limit() -> None:
    assert split_on_words("aa bb cc", 5) == ["aa bb", "cc"]
    assert split_on_words("aa bb cc", 8) == ["aa bb cc"]
    assert all(len(c) <= 5 for c in split_on_words("aa bb cc dd ee", 5))


def test_a_word_longer_than_the_limit_is_an_error() -> None:
    """Splitting inside a word changes what the tool is asked to diacritize."""
    with pytest.raises(ValueError, match="exceeds the 3-character limit"):
        split_on_words("ab cdefgh", 3)
