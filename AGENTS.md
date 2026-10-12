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

There is **one** estimator, `Pipeline`. It takes a model, transforms for each modality,
and training settings, and it works the same for every model and target. Do not add a
pipeline or trainer for a particular modality, fusion strategy or endpoint.

A model is a `Unimodal`, `EarlyFusion`, `IntermediateFusion` or `LateFusion`, built from
a list of encoders for each modality, a fusion method, and a head. Extend it
like this:

- **A new combination of modalities** is a different dictionary, never a new class.
- **A new kind of data** is one class in `loaddata/modalities.py` that sets `ids` and
  implements `_load(id)`. It subclasses `FixedShapeModality` when every patient's item has
  the same shape, so a batch is stacked into one tensor, or `BagModality` when each item is
  an `(N_i, d)` bag whose size varies, so a batch is a list; a bag also implements
  `instances`. A kind that is neither subclasses `Modality` and implements `collate` too.
  A modality that accepts sklearn transforms implements `transform_input` and
  `with_transform`, as `Tabular` does. Tables are passed as `Tabular(frame)`, never as a
  bare DataFrame.
- **A new encoder** is a plain `nn.Module` in `model/encode/encoders.py`, or in its own file under `model/encode/`
  when it needs an optional extra, as `tabicl.py` does. Its `forward(x)` takes the modality's batch or the previous
  encoder's output, and it checks its own input. Stock torch modules such as `nn.Linear` are encoders too. An encoder
  fitted on the training patients subclasses `FittedEncoder`, and one that reports attention subclasses
  `AttentionEncoder`, both from `model/encode/base_encoders.py`.
- **A new fusion method** is one `FusionMethod` subclass in `model/fusion.py`. It declares the `stages` it supports,
  implements `defined` and `forward`, and adds its own rules by overriding `check(context)`. Models
  never test for a particular fusion method.
- **A new endpoint** is a `BaseTarget` subclass in `loaddata/targets.py`, implementing
  `tensors`, `strata` and `counts`, plus a head in `model/heads.py`. The head declares the
  target class it predicts as `target_type` and owns the output, prediction, loss and
  target check.

## Component contracts

The code checks these at construction or during `fit`, so new components must follow them:

- An encoder checks its own input in `forward`; nothing reads declared widths. Before training, `fit` runs the model
  once on up to two training patients per modality (the dry run), so a wrong width or input fails before the first
  optimizer step. Errors get notes naming the encoder (`encoders['wsi'][1]`), the head, and the late fusion branch.
- An encoder list must end in a finite, floating-point `(len(ids), d)` tensor: one embedding per patient.
  `EncoderList` checks this in every forward pass and says whether NaN or inf came from the input or from the
  encoders. It does not check its input, so an encoder that handles missing values needs no flag; otherwise impute
  them in the modality's transform. When the encoders turn NaN or inf in the input into finite embeddings, `fit`,
  `predict` and `run_encoders` each raise one `NonFiniteInputWarning` per modality, and `fit` records the counts in
  `fit_report_["non_finite_inputs"]`.
- Only the first encoder may be a `FittedEncoder`. It implements `fit(x, target, ids)` and `forward(x, ids)`, and is
  fitted on the training patients who have the modality before training starts. It keeps fitted state in buffers, or
  in `extra_state` for anything that is not a tensor, so the state loads under `weights_only=True`. It owns the rows
  it is fitted on, including whether they may hold NaN.
- A module is initialised when it is built, and `fit` never re-initialises it: it trains a copy of the weights the
  model was built with. `Pipeline(random_state=...)` seeds only what `fit` does, so to vary the initial weights, seed
  before building the model. `load_pipeline` builds the model under the config's `random_state`.
- Store every `__init__` argument under an attribute with the same name.
  `sklearn.base.clone` and `dump_config` both depend on this.
- Encoders run only on patients who have the modality, and their embeddings are scattered back
  with NaN for the others. Fusion selects the defined rows. Never multiply by a mask:
  selecting rows is what keeps gradients finite.
- `EarlyFusion` fuses raw inputs, so every modality it reads must be a `FixedShapeModality`.
- An encoder that samples a bag's instances at random does so only when `self.training` is
  true. A fixed selection, such as dropping background patches, belongs in the modality, so
  that its features and `instances` describe the same rows. Attention export needs a
  `BagModality`, and raises an error if an encoder before the attending one changes the number
  of instances.
- A fusion method's `defined(present)` decides which patients it combines, both in `check` and in the forward
  pass, and its `forward` only receives those rows. Its rules read the `FusionContext` the model passes to
  `check`, never the dataset.
- A head's `loss` returns `None` when a batch has no signal (for Cox, no event with anyone
  else at risk). The Pipeline skips those batches and reports them.

## Constraints

- Split by patient, never by slide or patch. `subset` and `train_test_split` take patient ids.
- `fit` trains a copy (`model_`) and leaves `model` unchanged. A model that has already
  been fitted is rejected, so no fold can start from weights trained on other patients.
  `evaluate` rejects patients used in `fit` unless you pass `allow_seen=True`.
- Patient ids are strings. Read JSON with `dtype={"patient_id": str}`; otherwise `"001"`
  becomes `1`.
- `TimeToEvent.event` is a bool: `True` means observed and `False` means censored. A
  missing value raises an error rather than being treated as censored.
- `CoxHead` predicts `log_hazard`, where a higher value means higher risk.

## Conventions

- Nothing in `kalecancer/` may name a dataset, an endpoint column, a file pattern or a
  published split. These are the dataset's decisions and belong in `examples/`; the library
  supplies only the mechanisms.
- Components are configured through their constructor arguments, not a global config.
  YAML configs use jsonargparse's `class_path`/`init_args` format and are read with
  `load_pipeline` and written with `dump_config`.
- Use scikit-learn, Lightning, torchsurv or PyKale where they already cover what you need,
  instead of writing your own. Import Lightning as `lightning.pytorch as L`; ruff bans
  `pytorch_lightning`.
- scikit-survival is GPL-licensed. Use it only as a reference implementation in tests, and
  never import it from `kalecancer/`. Unless you think it would be hugely beneficial - then we could discuss its import.
- `tabicl` is pinned to an exact version because `TabICLEncoder` relies on its private
  internals. Bump it only if `tests/test_tabicl.py` passes with the pretrained checkpoint
  cached, so the comparison with upstream runs rather than being skipped.
- Export public classes from their subpackage's `__init__.py`. Users import from
  `kalecancer.<subpackage>`, not from the package root.
- Use Google-style docstrings, type hints and a line length of 120. ruff-format decides
  formatting; do not format by hand. Comments explain why, not what.

## Examples

```text
examples/
├── synthetic/  synthetic_survival: generated cohort, CPU only, downloads nothing
└── hancock/    survival_intermediate, from_config, cross_validation, classification_late
```

Examples are standalone scripts. Run them from the repository root with
`uv run python examples/<dir>/<script>.py`.

The HANCOCK examples read a local copy of the data from `data/hancock/`, which git
ignores. They use one of the published splits `in`, `out` or `Oropharynx`. Do not use
`treatment_outcome` for survival, because its test set was selected by outcome.

The dev container has no GPU, so train full cohorts on a GPU machine.

## Tests

Tests live in a flat `tests/` directory with one file per area, such as
`test_loaddata.py`, `test_model_fusion.py` and `test_pipeline.py`. They build synthetic
HDF5 cohorts with `make_cohort` from `tests/conftest.py` and never access the network.

- `test_tabicl.py` runs `TabICLEncoder` on a tiny, randomly initialised TabICL checkpoint
  that a fixture builds, so it needs only the `tabicl` extra. The one test that compares
  with upstream `TabICLClassifier` is skipped unless the pretrained checkpoint is already in
  the Hugging Face cache. The `gpu` test is skipped when CUDA is not available.
- `test_synthetic_example.py` runs the synthetic example and requires a Harrell's C above
  0.7, so any change to that example must keep it learning.

## Adding a component

1. Implement it in the matching `kalecancer/` subpackage and export it from that
   subpackage's `__init__.py`.
2. Follow the component contracts above.
3. Add tests with synthetic data to the matching `tests/test_<area>.py`.
4. Run pre-commit and pytest.
