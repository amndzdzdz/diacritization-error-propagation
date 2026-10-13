"""The committed 500-row draw must survive refactors of the sampling frame.

`docs/weeks/week-05.md` task 3 promotes `strip_diacritics` / `skeleton` out of
`msa_pool` into `normalize`, and those two functions are exactly what
`build_pool` dedups on and what `draw_block`'s eligible set is therefore a
function of. A one-line move that changed either would silently redraw the
block, which would stop it being the block §3.3b D4 decided and invalidate any
annotation already done against it — "the worst available outcome for a
one-line move", in the plan's words.

This test rebuilds the pool and the draw from the same cached parquet the
script read and compares every field of all 500 rows, in order, against
`run/msa_annotation_block.json`. It is a regression test, not a unit test:
it asserts that today's code reproduces a committed artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arabic_mdd.data.msa_pool import build_pool, draw_block

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "run" / "iqra_dev_text.parquet"
COMMITTED = REPO_ROOT / "run" / "msa_annotation_block.json"


@pytest.fixture(scope="module")
def committed() -> dict:
    if not COMMITTED.exists():  # pragma: no cover - artifact is committed
        pytest.skip(f"{COMMITTED.name} not present")
    return json.loads(COMMITTED.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def redrawn(committed: dict) -> list:
    pq = pytest.importorskip("pyarrow.parquet")
    if not CACHE.exists():  # pragma: no cover - artifact is committed
        pytest.skip(f"{CACHE.name} not present")
    rows = pq.read_table(CACHE, columns=["id", "sentence", "tashkeel_sentence"]).to_pylist()
    pool = build_pool((r["id"], r["sentence"], r["tashkeel_sentence"]) for r in rows)
    block = draw_block(pool, committed["block_size"], seed=committed["seed"])
    return [pool, block]


def test_pool_frame_is_unchanged(committed: dict, redrawn: list) -> None:
    pool, _ = redrawn
    assert pool.total == committed["pool_total"]
    assert len(pool.eligible) == committed["pool_eligible"]
    assert pool.exclusion_counts() == committed["exclusion_counts"]
    assert pool.flag_counts() == committed["flag_counts"]


def test_draw_is_identical_row_for_row(committed: dict, redrawn: list) -> None:
    _, block = redrawn
    assert len(block) == len(committed["block"])

    rebuilt = [
        {
            "id": c.id,
            "rank": rank,
            "crossover": rank < committed["crossover_size"],
            "sentence": c.sentence,
            "tashkeel_sentence": c.tashkeel_sentence,
            "diacritization_rate": c.rate,
            "flags": sorted(c.flags),
        }
        for rank, c in enumerate(block)
    ]
    assert rebuilt == committed["block"]


def test_crossover_sub_block_is_the_first_fifty(committed: dict) -> None:
    """§10: the crossover assignment cannot be reconstructed after the fact, so
    the sub-block's identity is pinned by rank here as well as in the file."""
    crossover = [r for r in committed["block"] if r["crossover"]]
    assert len(crossover) == committed["crossover_size"]
    assert [r["rank"] for r in crossover] == list(range(committed["crossover_size"]))
