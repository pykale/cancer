# Contributing to KaleCancer

Thanks for helping. KaleCancer follows PyKale's
[contributing guidelines](https://github.com/pykale/pykale/blob/main/.github/CONTRIBUTING.md) and
[Code of Conduct](https://github.com/pykale/pykale/blob/main/.github/CODE_OF_CONDUCT.md). This page covers what is
specific to this repository.

## Examples

Before contributing it is good to understand the libraries use case. See the [quickstart](docs/quickstart.md) for examples to help familiarize yourself with the library.

## Set up

Use Python 3.11 or 3.12 with [uv](https://docs.astral.sh/uv/), which installs the exact versions in `uv.lock`:

```bash
uv sync --all-extras
uv run pre-commit install
```

Or with pip, which is more fragile because it resolves versions from the bounds in `pyproject.toml`. The dev tools
are a dependency group, which needs pip 25.1 or later:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tabular,imaging,pathology,interpret]" --group dev
pre-commit install
```

With pip, drop the `uv run` prefix from the commands below.

## Before opening a pull request

Run the same checks as CI:

```bash
uv run pre-commit run --all-files
uv run pytest --cov=kalecancer
```

### Tests

Tests live in a flat `tests/` directory with one file per area, such as `test_loaddata.py`, `test_model_fusion.py`
and `test_pipeline.py`. They build synthetic HDF5 cohorts with `make_cohort` from `tests/conftest.py` and never access
the network. Add tests for new code to the matching file.

## Adding a component

Each kind of component has a guide in `docs/extending/`:

| To add | Guide | Where the code goes |
| --- | --- | --- |
| a new modality | [modalities.md](docs/extending/modalities.md) | `kalecancer/loaddata/modalities.py` |
| an encoder | [encoders.md](docs/extending/encoders.md) | `kalecancer/model/encode/` |
| a fusion method | [fusion.md](docs/extending/fusion.md) | `kalecancer/model/fusion.py` |
| a new endpoint: a target and its head | [endpoints.md](docs/extending/endpoints.md) | `kalecancer/loaddata/targets.py`, `kalecancer/model/heads.py` |

Every component also follows the [Conventions](#conventions) below. Read [docs/architecture.md](docs/architecture.md)
to see how the parts fit together, and [docs/adr/](docs/adr/) for the reasoning behind the current design.

A change to the architecture gets an entry in [docs/adr/](docs/adr/) recording the options considered and the choice
made.

## Conventions

- **The library supplies mechanisms, not experiments.** Nothing in `kalecancer/` may name a dataset, an endpoint
  column, a file pattern or a published split. Those are the dataset's decisions, and belong in `examples/`.
- **There is one estimator, `Pipeline`.** Don't add a pipeline or trainer for a particular modality, fusion method
  or endpoint.
- **Reuse before writing.** Use scikit-learn, Lightning, TorchSurv or PyKale where they already cover what you need.
- **scikit-survival is GPL-licensed.** Use it only as a reference implementation in tests, and never import it from
  `kalecancer/`. If you think importing it would be hugely beneficial, open an issue to discuss it first.
- **`tabicl` is pinned to an exact version**, because `TabICLEncoder` relies on its private internals. Bump it only if
  `tests/test_tabicl.py` passes with the pretrained checkpoint cached, so that the comparison with upstream runs
  rather than being skipped.
- **Imports:** export public classes from their subpackage's `__init__.py`. Users import from
  `kalecancer.<subpackage>`, not from the package root.
- **Configure through constructor arguments**, not globals or environment variables
- **Modules create their weights in `__init__`.** `Pipeline.fit` trains a copy of the model as built and never resets
  its weights, so don't rely on `reset_parameters()` being called, and don't add a `random_state` argument for the
  weights as the user seeds them before building the model using `torch.manual_seed()`.
- **Seed only your own random choices.** Training randomness (shuffling, dropout, sampling) is seeded by
  `Pipeline(random_state=...)`. If a component makes a random choice of its own, such as assigning folds in `fit`,
  give it a `random_state` argument, as `TabICLEncoder` does.
- **Style:** Google-style docstrings, type hints, and a line length of 120. ruff-format decides formatting, so don't
  format by hand. Comments explain why, not what.
