import json
import sys

import pytest

from arabic_mdd.diacritizers.shakkala import (
    SHAKKALA_VERSION,
    WORKER_PATH,
    ShakkalaDiacritizer,
    default_command,
)


def fake_worker(payload: dict, exit_code: int = 0, stderr: str = "") -> tuple[str, ...]:
    """A stand-in worker obeying the same protocol, runnable in this interpreter.

    The real worker needs a TensorFlow 2.9 / Python 3.10 environment, which
    the test suite must not build. What is testable offline — and what the
    wrapper is actually responsible for — is the protocol: stdin in, JSON
    file out, failures surfaced.
    """
    literal = json.dumps(json.dumps(payload, ensure_ascii=False))
    script = (
        "import sys\n"
        "sys.stdin.read()\n"
        f"sys.stderr.write({json.dumps(stderr)})\n"
        f"open(sys.argv[1], 'w', encoding='utf-8').write({literal})\n"
        f"sys.exit({exit_code})\n"
    )
    return (sys.executable, "-c", script)


def test_outputs_are_returned_in_order_and_stripped() -> None:
    worker = fake_worker({"texts": [" ذَهَبَ ", "عَادٌ\n"], "version": SHAKKALA_VERSION})

    assert ShakkalaDiacritizer(command=worker).diacritize_batch(["ذهب", "عاد"]) == [
        "ذَهَبَ",
        "عَادٌ",
    ]


def test_diacritize_delegates_to_the_batch_path() -> None:
    worker = fake_worker({"texts": ["ذَهَبَ"], "version": SHAKKALA_VERSION})

    assert ShakkalaDiacritizer(command=worker).diacritize("ذهب") == "ذَهَبَ"


def test_empty_batch_does_not_spawn_a_subprocess() -> None:
    shakkala = ShakkalaDiacritizer(command=("definitely-not-an-executable",))

    assert shakkala.diacritize_batch([]) == []


def test_worker_failure_surfaces_stderr() -> None:
    worker = fake_worker({"texts": []}, exit_code=1, stderr="ModuleNotFoundError: shakkala")
    shakkala = ShakkalaDiacritizer(command=worker)

    with pytest.raises(RuntimeError, match="ModuleNotFoundError: shakkala"):
        shakkala.diacritize_batch(["ذهب"])


def test_output_count_mismatch_is_an_error() -> None:
    """Silent misalignment would corrupt every downstream utterance-level join."""
    worker = fake_worker({"texts": ["ذَهَبَ"], "version": SHAKKALA_VERSION})
    shakkala = ShakkalaDiacritizer(command=worker)

    with pytest.raises(RuntimeError, match="returned 1 outputs for 2 inputs"):
        shakkala.diacritize_batch(["ذهب", "عاد"])


def test_version_drift_in_the_worker_environment_is_an_error() -> None:
    """The pin is load-bearing (`docs/msa-arm.md` §5.1), not decorative."""
    worker = fake_worker({"texts": ["ذَهَبَ"], "version": "1.6"})
    shakkala = ShakkalaDiacritizer(command=worker)

    with pytest.raises(RuntimeError, match="ran version '1.6'"):
        shakkala.diacritize_batch(["ذهب"])


def test_default_command_pins_the_interpreter_and_the_package() -> None:
    command = default_command()

    assert f"shakkala=={SHAKKALA_VERSION}" in command
    assert "3.10" in command
    assert command[-1] == str(WORKER_PATH)


def test_worker_script_exists() -> None:
    assert WORKER_PATH.is_file()


def test_name_and_provenance() -> None:
    shakkala = ShakkalaDiacritizer(command=fake_worker({}))

    assert shakkala.name == "shakkala"
    assert shakkala.provenance()["version"] == SHAKKALA_VERSION
