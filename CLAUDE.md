# CLAUDE.md

## Project

Reference-aware evaluation of how automatic diacritization corrupts Arabic
mispronunciation detection and diagnosis (MDD). Target venue: Interspeech
2027. Full plan available in: [docs/mdd-paper-project-plan-v2.md](docs/mdd-paper-project-plan-v2.md).
Literature/novelty review: [docs/novelty-check.md](docs/novelty-check.md).

## Tech stack

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency, environment, and script
  management — never use `pip`/`venv` directly
- `ruff` for linting and formatting
- `pytest` (strict mode) for tests
- `pre-commit` for fast pre-commit checks
- Model/data dependencies (PyTorch, HF `transformers`/`datasets`, etc.) are
  added to `pyproject.toml` as the corresponding weeks' code is implemented —
  not pre-installed speculatively

## Structure

```
docs/                  Project plan, novelty check, per-week implementation plans (docs/weeks/week-NN.md)
insights/              One file per week (week-NN.md): experiment log, filled in as work happens
scripts/               Ad hoc, one-off verification scripts — not covered by CI/tests, not
                       imported by src/. Promote a script's logic into src/arabic_mdd/ (with
                       tests) once it becomes a real pipeline component a later week needs.
src/arabic_mdd/
  data/                Dataset loading (QuranMB.v2, Iqra_train, Common Voice Arabic)
  diacritizers/         Wrappers around diacritization tools under test (CATT, Shakkala, Mishkal, Farasa)
  metrics/             DER/WER and the hierarchical TA/TR/FA/FR MDD metric
  models/              MDD model architectures (prompt-free and canonical-conditioned)
tests/                 Mirrors src/ layout
```

Week numbers are zero-padded two digits (`week-01.md`, ... `week-21.md`)
matching the plan's 21-week timeline. Add new subpackages under
`src/arabic_mdd/` only when a week's work needs them — don't pre-build
future weeks' modules.

## Setup

```bash
uv sync --all-groups     # installs runtime + dev dependencies, creates .venv
uv run pre-commit install
```

## Running the project

There is no single entry point yet; each week's work is run via its own
script or `uv run pytest`/`uv run python -m ...` as it's added. Document the
exact invocation in that week's `docs/weeks/week-NN.md` when it exists.

## Development commands

```bash
uv run ruff check .              # lint
uv run ruff format .             # format
uv run pytest                    # run tests
uv run pre-commit run --all-files  # run all pre-commit hooks manually
```

CI (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`,
and `pytest` on every push/PR to `main`. Pre-commit runs `ruff` (check +
format) and basic hygiene checks (trailing whitespace, EOF, yaml/toml
validity, large files) on every commit.

## Workflow expectations

- **Tests**: add or update tests under `tests/` for every feature or change
  in behavior. No feature is done without a passing test.
- **Insights**: after every experiment, update the matching
  `insights/week-NN.md` for the current week. Use this structure for each
  experiment entry:

  ```
  ### Experiment: <name>
  #### Hypothesis
  #### Methodology
  #### Configuration
  #### Results
  #### Observations
  #### Conclusion
  ```

  Do this immediately after the experiment, not retroactively at the end of
  the week.
- **Week plans**: before starting a new week's work, check for
  `docs/weeks/week-NN.md`. If it doesn't exist yet, write it first (concrete
  tasks derived from the project plan) before writing code.
- **Scope discipline**: implement only the current week's work. Don't
  pre-build future weeks' pipelines, metrics, or models "while you're in
  there."
- **Before committing**: run `uv run ruff check .`, `uv run ruff format .`,
  and `uv run pytest`. Pre-commit will also run automatically on `git commit`.
