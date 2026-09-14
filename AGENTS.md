# AGENTS.md

Instructions for coding agents working in this repository. Human contributors should
read [README.md](README.md) and [CONTRIBUTING](README.md#contributing).

## Setup

Python 3.10–3.12 is required. Install PyTorch first if a specific CUDA build is
needed, then:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"     # Windows: .venv/Scripts/pip
```

Verify the install:

```bash
.venv/bin/python -c "import kalecancer, torch, kale; print(kalecancer.__version__)"
```

## Commands

| Task | Command |
| --- | --- |
| Run tests | `.venv/bin/pytest` |
| Run tests with coverage | `.venv/bin/pytest --cov=kalecancer --cov-report=term-missing` |
| Format | `.venv/bin/ruff format .` |
| Lint | `.venv/bin/ruff check .` |
| Fetch and run on HANCOCK | `.venv/bin/python -m examples.hancock_wsi_survival.main --cfg examples/hancock_wsi_survival/configs/hancock_primary_tumour_quick.yaml DATASET.SOURCE hancock DATASET.PATIENTS 50` |

Both `pre-commit run --all-files` (ruff, ruff-format, mypy) and `pytest` must pass
before a change is complete. These are the same checks CI runs.

## Data

Two sources, selected by `DATASET.SOURCE`:

- `local` reads an existing copy from `DATASET.FEATURE_ROOT` and
  `DATASET.CLINICAL_PATH`.
- `hancock` fetches the published dataset (CC BY 4.0) over HTTP range requests,
  transferring only the selected patients out of a 9.65 GB archive and caching them.
  Requires no credentials. Its fetcher lives in `examples/hancock/dataset.py`
  and is passed to `run(..., fetch=...)` by that example. Keep it small with
  `DATASET.PATIENTS N`.

Tests need neither: they build synthetic HDF5 cohorts in a temporary directory, and
must never reach the network. `tests/conftest.py` shows the pattern, and
`tests/loaddata/test_archive.py` serves archives from a local HTTP server.

A cohort of 50–60 patients leaves fewer than 10 in the test split, which is enough to
verify the pipeline runs but not to produce a meaningful metric. Report such runs as
verification, not as results.

There is no GPU in a Codespace or dev container. Use it for development, tests, and
small demonstration runs; train full cohorts on a GPU machine.

## Layout

```
kalecancer/
├── auto/        high-level construction (AutoCancer* classes, planned)
├── loaddata/    access APIs per modality, plus generic splitting
├── prepdata/    transforms
├── model/       layers/ blocks, embed/ encoders and fusion, predict/ heads
├── pipeline/    the trainer and its tasks
├── evaluate/    metrics and prediction scoring, by task
├── interpret/   attention export
├── utils/       seeding, artefact writing, and the cross-package protocols
└── config.py    configuration schema and defaults
```

```
pipeline/
├── task.py     PredictionTask, SurvivalTask, ClassificationTask
└── trainer.py  CohortTrainer -- the only trainer
```

There is **one** trainer. `CohortTrainer` takes a set of modality embedders and a
task; neither constrains the other, so a whole-slide survival model is one bag
modality with a `SurvivalTask`, and a multimodal classifier is two modalities with a
`ClassificationTask`. Do not add a trainer per modality or per endpoint — there is
nothing for one to add.

A task decides exactly four things: the head, the loss, whether a batch carries a
gradient, and the epoch metric. A new endpoint is a new `PredictionTask`, never a new
trainer.

`loaddata/` is one access API per kind of data, and none of them knows a dataset:

```
loaddata/
├── tabular_access.py     Cohort, CohortView, TabularCohort
├── wsi_access.py         patch bags from HDF5: discovery, reading, validation
├── multimodal_access.py  MultimodalDataset, ModalitySource, and the record types
├── archive_access.py     remote ZIP access, for datasets published as one
└── splitting.py          HoldOut, CrossValidation, Predefined -- scikit-learn's shape
```

One module per kind of data it reads, named for it.

The `Target` and `Preprocessor` contracts live in `multimodal_access.py`, beside the
targets that implement them. They are not re-exported from any package root: nobody
calls a protocol, they are implemented and type-checked against. `check_target` is
the runtime check, since a `Protocol` alone verifies nothing — `tests/loaddata/test_targets.py`
covers it. Do not add a protocol for something with one implementation.

`evaluate/` is named the same way — by the task it scores and the thing it produces:

```
evaluate/
├── classification_metrics.py  ROC-AUC, average precision, F1, mean ROC curve
├── survival_metrics.py        C-index, IPCW time-dependent AUC, integrated Brier
├── survival_predictions.py    running a model over a loader, and scoring the result
└── cross_validation.py        refitting across folds, and resampled intervals
```

Describing a *cohort* is not in `evaluate/`: what counts as an excluded patient
depends on how the cohort was built, so it lives with the builder in `examples/`.

**Splitting stays in `loaddata/`, not `prepdata/`.** A splitter decides which samples
load into which loader; `prepdata/` is fold-local *fitted state*, and the thing that
creates the folds cannot be state belonging to one. scikit-learn draws the same line
between `model_selection` and `preprocessing`.

A modality is a named `ModalitySource`, so imaging + tabular, imaging + imaging and
four of each are the same `MultimodalDataset` with a different dictionary. Adding a
combination is never a new class; adding a *kind* of data is one `ModalitySource`.

A bag of patches is an ordinary modality whose value is a list rather than a stacked
tensor; `BagEncoder` pools it. That is why there is no whole-slide trainer.

`model/` is three stages of one pipeline, and a class belongs to exactly one:

```
model/
├── layers/    MLP, AttentionMIL, GatedAttention -- blocks that transform tensors
├── embed/     MLPEmbedder, BagEncoder, TabICLEmbedder, MultimodalFusion
└── predict/   heads.py: LinearHead, CoxHead;  losses.py: every objective
```

**There is no `survival/` stage.** Time-to-event support is not a pipeline stage, it
is one endpoint among others, so its pieces sit with their kind: the head and loss in
`model/predict/`, the C-index and baseline hazard in `evaluate/survival_metrics.py`,
and `SurvivalTarget` beside `ColumnTarget` in `loaddata/multimodal_access.py`. A head
and its loss live in the same package because neither is meaningful alone — `CoxHead`
is bias-free *because* of its partial likelihood.

A **layer** transforms tensors and knows nothing about modalities; an **embedder**
adapts a layer to the contract fusion relies on (`out_dim`, `needs_full_batch`, a
`mask` argument); a **head** turns the fused vector into a score. `MLPEmbedder` is a
thin subclass of `MLP` and `BagEncoder` wraps `AttentionMIL` for exactly this reason
— put a new block in `layers/` and adapt it in `embed/`, never both at once.

`CoxHead` and `LinearHead` sit together in `predict/heads.py`, and every loss in
`predict/losses.py` — including the `multimodal_*` wrappers, which used to live in
the fusion module.

**Orchestration does not belong here.** Assembling a cohort, choosing splits, naming
an endpoint and writing a report are experiment concerns, so they live in
`examples/<name>/runner.py`. If a piece of code names a dataset, an endpoint or a
configuration key, it is an experiment, not a library component.

## Examples

Named `<dataset>_<modality>_<task>`, with `survival` for right-censored time-to-event
endpoints and `classification` for binary ones. See [examples/README.md](examples/README.md)
for the index. Shared HANCOCK archive access lives in `examples/hancock/`, imported by
every HANCOCK example rather than duplicated into each.

Examples run as modules from the repository root — `python -m examples.<name>.main` —
so their imports resolve as ordinary packages. Do not add `sys.path` manipulation to an
example.

Tests mirror this layout: code in `kalecancer/loaddata/splitting.py` is tested in
`tests/loaddata/test_splitting.py`, and a dataset's own code in
`tests/examples/`.

## Conventions

- Follow [PyKale](https://github.com/pykale/pykale) conventions: verb-oriented stages,
  Google-style docstrings, type hints, YACS configuration.
- Reuse PyKale APIs where one exists rather than reimplementing.
- Line length 120. Formatting is enforced by ruff-format; do not hand-format.
- Comments explain why, not what. Do not restate the code.
- New configuration belongs in `kalecancer/config.py`, never hardcoded in a module.
- Dataset paths appear only in example configs and command-line arguments.
- Nothing in `kalecancer/` may name a dataset, an endpoint or an experiment. If it
  does, it belongs in `examples/`. That covers cohort construction, endpoint column
  names, slide-filename patterns, and which patients a published split assigns
  where: `loaddata/` supplies the mechanisms, the dataset supplies the decisions.
- A predefined partition -- a published split, internal against external, one site
  held out -- is applied with `Predefined`, but read and named by the dataset.
- `persistent_workers` loaders must be released with `release_workers` when the
  split that built them is finished; see its docstring for what happens otherwise.

## Constraints

- Splitting is patient-level. Never split on slides or patches; a patient's slides
  must never span two splits -- including the train/validation carve, not only
  train/test.
- The dataset's published train/test assignment is the default everywhere
  (`DATASET.SPLIT_MODE="published"`). Re-drawing the test set makes a result
  incomparable with other work on the cohort, so `cv` and `random` are opt-in.
  Validation is always carved out of the training half.
- Censoring-aware quantities (IPCW weights, baseline hazard) are estimated from the
  training split only. So is every fitted transform: scalers, encoders, and the
  TabICL context. When cross-validating, all of them are rebuilt per fold -- fitting
  once outside the fold loop carries every fold's test patients into the others.
- The event convention is `1` observed, `0` censored, normalised at load time.
- Risk scores are log partial hazards where higher means higher risk.

## Adding a component

1. Implement it in the matching `kalecancer/` subpackage.
2. Export it from that subpackage's `__init__.py`.
3. Add tests in the mirrored `tests/` path, using synthetic data.
4. Add any new setting to `kalecancer/config.py` with a comment.
5. Run the format, lint, and test commands above.
