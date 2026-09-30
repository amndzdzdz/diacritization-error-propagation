"""Exercise the real worker script, with a stub standing in for Shakkala itself.

The worker's own environment (TensorFlow 2.9 / Python 3.10) must not be
built by the test suite, but the script is still reachable offline: point it
at a fake `shakkala` package and run it under this interpreter. That covers
the protocol and, more importantly, the module-shadowing trap — the worker
lives next to the project's own `shakkala.py`, so without `sys.path` surgery
its `import shakkala` picks up the wrapper and fails.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from arabic_mdd.diacritizers.shakkala import WORKER_PATH

FAKE_SHAKKALA = """
class Shakkala:
    def __init__(self, version=3):
        self.version = version

    def get_model(self):
        return self, None

    def predict(self, prepared, verbose=0):
        return [prepared]

    def prepare_input(self, text):
        return text

    def logits_to_text(self, logits):
        return logits

    def get_final_text(self, text, harakat):
        return f"{text}-diacritized"
"""

METADATA = "Metadata-Version: 2.1\nName: shakkala\nVersion: 1.7\n"


def _fake_package_dir(tmp_path: Path) -> Path:
    stub = tmp_path / "stub"
    (stub / "shakkala-1.7.dist-info").mkdir(parents=True)
    (stub / "shakkala.py").write_text(FAKE_SHAKKALA, encoding="utf-8")
    (stub / "shakkala-1.7.dist-info" / "METADATA").write_text(METADATA, encoding="utf-8")
    return stub


def test_worker_imports_the_package_not_the_wrapper_and_honours_the_protocol(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "output.json"

    process = subprocess.run(
        [sys.executable, str(WORKER_PATH), str(output_path)],
        input=json.dumps({"texts": ["ذهب", "عاد"]}),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_fake_package_dir(tmp_path))},
        check=False,
    )

    assert process.returncode == 0, process.stderr
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "texts": ["ذهب-diacritized", "عاد-diacritized"],
        "version": "1.7",
    }
