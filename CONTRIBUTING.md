# Contributing to KaleCancer

Thanks for helping. KaleCancer follows PyKale's [contributing guidelines](https://github.com/pykale/pykale/blob/main/.github/CONTRIBUTING.md) and [Code of Conduct](https://github.com/pykale/pykale/blob/main/.github/CODE_OF_CONDUCT.md). This page covers what is specific to this repository.

## Examples

Before contributing it is good to understand the libraries use case. See the [quickstart](docs/quickstart.md) for examples to help familiarize yourself with the library.

## Set up

Use Python 3.11 or 3.12 with [uv](https://docs.astral.sh/uv/), which installs the exact versions in `uv.lock`:

```bash
uv sync --all-extras
uv run pre-commit install
```

Or with pip, which is more fragile because it resolves versions from the bounds in `pyproject.toml`. The dev tools are a dependency group, which needs pip 25.1 or later:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tabular,imaging,pathology,interpret,multiomics]" --group dev
pre-commit install
```

With pip, drop the `uv run` prefix from the commands below.

The `multiomics` extra also needs two compiled PyTorch Geometric packages that cannot be declared; see [Code migrated from PyKale](#code-migrated-from-pykale).

## Before opening a pull request

Run the same checks as CI:

```bash
uv run pre-commit run --all-files
uv run pytest --cov=kalecancer
```

### Tests

Tests mirror the package layout, as in PyKale: code in `kalecancer/model/fusion.py` is tested in `tests/model/test_fusion.py`, and an example in `tests/examples/`. They build synthetic HDF5 cohorts with `make_cohort` from `tests/conftest.py` and never access the network, except the MOGONET tests described in [Code migrated from PyKale](#code-migrated-from-pykale). Add tests for new code to the matching file.

## Releasing

Releases follow PyKale's process, with the steps maintainers did by hand for PyKale 0.2.2 automated:

1. Bump `__version__` in `kalecancer/__init__.py`, e.g. to `0.2.0` or `0.2.0rc1`, and open a pull request from a branch of this repository titled `Release 0.2.0`.
2. `changelog.yml` collects the pull requests merged since the last tag, grouped by label, and commits the section to `.github/CHANGELOG.md` on the branch. Within each group the entries are sorted by pull request number, titles lose any `feat:`-style prefix and start with a capital letter, and earlier release pull requests are left out. It also posts the section as a comment and runs CI on the new commit.
3. Review the section and edit it on the branch if needed. A rerun never overwrites an existing section.
4. Approve the pull request. `release-on-approval.yml` waits for the checks, merges, creates the `v0.2.0` GitHub release with the changelog section as its notes, builds the package with `build-dist.yml` and publishes it: alpha, beta and release-candidate versions go to Test PyPI, the rest to PyPI.

Only approvals from owners, members and collaborators, of the pull request's latest commit, start a release. If a check fails, fix it, push, and approve again.

Publishing uses PyPI trusted publishing, so no PyPI token is stored. pypi.org and test.pypi.org each trust two workflows of `pykale/cancer`: `release-on-approval.yml` and `release.yml`, which publishes a release created by hand. The publishers use the environments `pypi` and `testpypi`. The repository needs the `new feature`, `bug`, `refactor`, `enhancement`, `dependencies`, `tests` and `documentation` labels that `.github/changelog-config.json` groups by. New issues and pull requests are added to the [KaleCancer project](https://github.com/orgs/pykale/projects/14), which needs the `ADD_TO_PROJECT_PAT` secret.

## Code migrated from PyKale

MOGONET, the multiomics classifier, was moved from PyKale in its source format. This round is a migration, not an enhancement: its outputs, public APIs and contracts are those of PyKale 0.2.2, whose MOGONET code is unchanged since 0.2.1. To pass this repository's checks, the import paths, docstrings and type hints changed, and two misuse cases now behave differently: `MultiomicsTrainer` skips its per-step loss logging when no logger is attached instead of failing, and the dataset raises a `RuntimeError` if an adjacency matrix is built before its similarity threshold is set.

| KaleCancer | Moved from PyKale |
| --- | --- |
| `kalecancer/loaddata/multiomics_datasets.py` | `kale.loaddata.multiomics_datasets` |
| `kalecancer/model/encode/mogonet.py` | `kale.embed.model_lib.mogonet` |
| `LinearClassifier` in `kalecancer/model/heads.py` | `kale.predict.decode` |
| `kalecancer/pipeline/multiomics_trainer.py` | `kale.pipeline.multiomics_trainer` |
| `kalecancer/interpret/model_weights.py` | `select_top_features_by_masking` in `kale.interpret.model_weights` |
| `examples/multiomics_mogonet/` | `examples/multiomics_mogonet/` |

`VCDN`, `calculate_distance`, `load_graph_data`, `ToTensor` and `ToOneHotEncoding` stay in PyKale and are imported from `kale`.

This code differs from the rest of KaleCancer, and the differences are deliberate until a later enhancement round:

- **It is built on PyTorch, not on scikit-learn and NumPy.** Each omics type is a PyTorch Geometric graph of similar patients, and training runs in `MultiomicsTrainer`, a PyTorch Lightning module, rather than through the scikit-learn-style `Pipeline` over NumPy arrays and pandas frames. `MultiomicsTrainer` is therefore an exception to the single-estimator convention below, and these modules are not exported from their subpackage `__init__.py`, so importing `kalecancer` does not require PyTorch Geometric.
- **Its dependencies are the `multiomics` extra:** `torch_geometric`, and `yacs` for the example. Like PyKale, KaleCancer cannot declare `torch_scatter` and `torch_sparse`, because they are built for each PyTorch and CUDA version and published only on the PyG wheel index. Install them for the PyTorch you have, as CI does:

  ```bash
  TORCH=$(python -c "import torch; print(torch.__version__)")
  python -m pip install --only-binary :all: torch_scatter torch_sparse -f "https://data.pyg.org/whl/torch-${TORCH}.html"
  ```

  In a uv environment, use `uv pip install` in place of `python -m pip install`.

- **Its tests keep PyKale's format.** `tests/model/encode/test_mogonet.py`, `tests/loaddata/test_multiomics_datasets.py` and `tests/pipeline/test_multiomics_trainer.py` download ROSMAP and TCGA-BRCA from [pykale/data](https://github.com/pykale/data/tree/main/multiomics) into `tests/test_data/`, which git ignores, so they need network access. `tests/interpret/test_model_weights.py` uses synthetic data.
- **The example keeps PyKale's layout.** Run it from its folder: `python main.py --cfg configs/MOGONET_BRCA_quick_test.yaml`.

Don't change this code's behaviour as part of the migration. Adapting it to the conventions below is separate work, recorded in [docs/adr/](docs/adr/) when it happens.

## Adding a component

Each kind of component has a guide in `docs/extending/`:

| To add | Guide | Where the code goes |
| --- | --- | --- |
| a new modality | [modalities.md](docs/extending/modalities.md) | `kalecancer/loaddata/modalities.py` |
| an encoder | [encoders.md](docs/extending/encoders.md) | `kalecancer/model/encode/` |
| a fusion method | [fusion.md](docs/extending/fusion.md) | `kalecancer/model/fusion.py` |
| a new endpoint: a target and its head | [endpoints.md](docs/extending/endpoints.md) | `kalecancer/loaddata/targets.py`, `kalecancer/model/heads.py` |

Every component also follows the [Conventions](#conventions) below. Read [docs/architecture.md](docs/architecture.md) to see how the parts fit together, and [docs/adr/](docs/adr/) for the reasoning behind the current design.

A change to the architecture gets an entry in [docs/adr/](docs/adr/) recording the options considered and the choice made.

## Conventions

- **The library supplies mechanisms, not experiments.** Nothing in `kalecancer/` may name a dataset, an endpoint column, a file pattern or a published split. Those are the dataset's decisions, and belong in `examples/`.
- **There is one estimator, `Pipeline`.** Don't add a pipeline or trainer for a particular modality, fusion method or endpoint.
- **Reuse before writing.** Use scikit-learn, Lightning, TorchSurv or PyKale where they already cover what you need.
- **scikit-survival is GPL-licensed.** Use it only as a reference implementation in tests, and never import it from `kalecancer/`. If you think importing it would be hugely beneficial, open an issue to discuss it first.
- **`tabicl` is pinned to an exact version**, because `TabICLEncoder` relies on its private internals. Bump it only if `tests/model/encode/test_tabicl.py` passes with the pretrained checkpoint cached, so that the comparison with upstream runs rather than being skipped.
- **Imports:** export public classes from their subpackage's `__init__.py`. Users import from `kalecancer.<subpackage>`, not from the package root.
- **Configure through constructor arguments**, not globals or environment variables
- **Modules create their weights in `__init__`.** `Pipeline.fit` trains a copy of the model as built and never resets its weights, so don't rely on `reset_parameters()` being called, and don't add a `random_state` argument for the weights as the user seeds them before building the model using `torch.manual_seed()`.
- **Seed only your own random choices.** Training randomness (shuffling, dropout, sampling) is seeded by `Pipeline(random_state=...)`. If a component makes a random choice of its own, such as assigning folds in `fit`, give it a `random_state` argument, as `TabICLEncoder` does.
- **Style:** Google-style docstrings, type hints, and a line length of 120. ruff-format decides formatting, so don't format by hand. Comments explain why, not what.
