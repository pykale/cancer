# AGENTS.md

Instructions for coding agents working in this repository.

## Setup and commands

Python 3.11 or 3.12, managed with [uv](https://docs.astral.sh/uv/) against the committed `uv.lock`.

| Task | Command |
| --- | --- |
| Install, with dev tools | `uv sync --all-extras` |
| Test | `uv run pytest` |
| Test with coverage | `uv run pytest --cov=kalecancer --cov-report=term-missing` |
| Format, lint, type-check | `uv run pre-commit run --all-files` |
| Smallest end-to-end run | `uv run python examples/synthetic/synthetic_survival.py` |

## Layout

```text
kalecancer/
├── loaddata/   MultimodalDataset, modalities (Tabular, PatchFeatures), targets, train_test_split
├── prepdata/   TableTransform: sklearn transforms for table modalities
├── model/      encode/ (encoders, EncoderList), fusion methods, heads, and the models that wire them together
├── pipeline/   Pipeline (fit, predict, evaluate, encode, run_encoders), Lightning training, YAML configs
├── evaluate/   metrics on prediction frames, cross_validate
└── interpret/  attention export
```

There is **one** estimator, `Pipeline`. It takes a model, transforms for each modality, and training settings, and it
works the same for every model and target. A model is a `Unimodal`, `EarlyFusion`, `IntermediateFusion` or
`LateFusion`, built from a list of encoders for each modality, a fusion method, and a head. A new combination of
modalities is a different dictionary, never a new class.

## Read before changing code

These documents hold the rules the code relies on. Read the relevant one before you start, not after a test fails:

- **Any change to `kalecancer/`:** read the Conventions in [CONTRIBUTING.md](CONTRIBUTING.md).
- **Adding or changing a modality, encoder, fusion method, target or head:** also read the guide for that component
  in [docs/extending/](docs/extending/), listed in CONTRIBUTING.md. The guides hold the contracts that `fit` and the
  dry run enforce.
- **Behaviour users rely on** (patient ids, splitting, leakage guards, target semantics): see
  [docs/architecture.md](docs/architecture.md). Don't change it without recording the decision in
  [docs/adr/](docs/adr/).
- **Tests and examples:** see CONTRIBUTING.md. Keep `examples/synthetic/synthetic_survival.py` learning, because its
  test requires a Harrell's C above 0.7.

## Environment

The dev container has no GPU, so train full cohorts on a GPU machine. The HANCOCK examples need a local copy of the
data in `data/hancock/`, which git ignores and which may not be present.
