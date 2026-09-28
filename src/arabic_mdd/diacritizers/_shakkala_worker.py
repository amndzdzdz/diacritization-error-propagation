"""Out-of-process Shakkala worker. Runs under a *different* interpreter.

Invoked by `arabic_mdd.diacritizers.shakkala.ShakkalaDiacritizer`, which
cannot import Shakkala in-process: Shakkala 1.7 pins `tensorflow==2.9.3`,
whose newest wheel is cp310, against this project's Python 3.12. See that
module's docstring for the full argument.

Consequently this file must not import anything from `arabic_mdd` — the
interpreter running it does not have the project installed.

Protocol: `{"texts": [...]}` as JSON on stdin, `{"texts": [...],
"version": "..."}` as JSON written to the path given in `argv[1]`. The
result goes to a file rather than stdout because Shakkala and TensorFlow
both print to stdout on model load.
"""

import json
import sys

# This file sits next to the project's own `shakkala.py`, and Python puts a
# script's own directory first on `sys.path`, so an unqualified
# `import shakkala` below would find the wrapper instead of the package.
sys.path.pop(0)

from importlib.metadata import version  # noqa: E402

MODEL_VERSION = 3


def main() -> None:
    output_path = sys.argv[1]
    texts = json.load(sys.stdin)["texts"]

    from shakkala import Shakkala

    shakkala = Shakkala(version=MODEL_VERSION)
    model, _graph = shakkala.get_model()

    outputs = []
    for text in texts:
        logits = model.predict(shakkala.prepare_input(text), verbose=0)[0]
        outputs.append(shakkala.get_final_text(text, shakkala.logits_to_text(logits)))

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"texts": outputs, "version": version("shakkala")}, handle)


if __name__ == "__main__":
    main()
