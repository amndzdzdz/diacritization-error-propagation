"""Shakkala (pypi.org/project/shakkala), the mid-strength point of the tool set.

Plan §3 names Shakkala specifically: CATT's Table 5 spans 5.4%-19.6% CE DER
across CATT / Shakkala / Mishkal, and that spread is what makes the
degradation a *curve* rather than a before/after pair. Dropping it would
narrow the x-axis of the paper's main RQ1 figure.

**It cannot run in this project's environment.** Shakkala 1.7 pins
`tensorflow==2.9.3`, whose newest wheel is cp310, against this project's
Python 3.12 — the resolver rejects it outright rather than degrading. So it
runs out-of-process under its own interpreter, which is the same pattern
`scripts/baseline_reproduction/` already uses for S3PRL's Python 3.8
environment. The alternatives were dropping Shakkala (narrows the curve) or
converting its Keras weights to ONNX (needs a TensorFlow environment
anyway).

The model loads on every call, which costs seconds, so callers should pass
whole batches rather than looping over `diacritize`.

Shakkala asserts on inputs longer than 315 characters rather than truncating
them. No utterance in QuranMB.v2 (max 118) or `Iqra_train` (max 191) comes
close, so that assertion is left to fire rather than pre-empted by chunking.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from arabic_mdd.diacritizers.base import Diacritizer

SHAKKALA_VERSION = "1.7"
WORKER_PYTHON = "3.10"
WORKER_PATH = Path(__file__).with_name("_shakkala_worker.py")


def default_command() -> tuple[str, ...]:
    """`uv`-managed ephemeral 3.10 environment holding the pinned Shakkala."""
    return (
        "uv",
        "run",
        "--no-project",
        "--python",
        WORKER_PYTHON,
        "--with",
        f"shakkala=={SHAKKALA_VERSION}",
        "python",
        str(WORKER_PATH),
    )


class ShakkalaDiacritizer(Diacritizer):
    """Shakkala, driven through a subprocess running its own interpreter."""

    def __init__(self, command: Sequence[str] | None = None, timeout: float = 3600) -> None:
        self.command = tuple(command) if command is not None else default_command()
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "shakkala"

    def provenance(self) -> dict[str, str]:
        return {
            "tool": "shakkala",
            "package": "shakkala",
            "version": SHAKKALA_VERSION,
            "worker_python": WORKER_PYTHON,
        }

    def diacritize_batch(self, texts: Sequence[str]) -> list[str]:
        if not texts:
            return []
        payload = self._run_worker(list(texts))

        outputs = payload["texts"]
        if len(outputs) != len(texts):
            raise RuntimeError(
                f"Shakkala worker returned {len(outputs)} outputs for {len(texts)} inputs"
            )
        if payload["version"] != SHAKKALA_VERSION:
            raise RuntimeError(
                f"Shakkala worker ran version {payload['version']!r}, "
                f"pinned to {SHAKKALA_VERSION!r}"
            )
        return [output.strip() for output in outputs]

    def _run_worker(self, texts: list[str]) -> dict:
        with tempfile.TemporaryDirectory() as workdir:
            output_path = Path(workdir) / "output.json"
            process = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [*self.command, str(output_path)],
                input=json.dumps({"texts": texts}),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
                check=False,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"Shakkala worker exited {process.returncode}:\n{process.stderr}"
                )
            if not output_path.exists():
                raise RuntimeError(f"Shakkala worker wrote no output file:\n{process.stderr}")
            return json.loads(output_path.read_text(encoding="utf-8"))
