# ADR 0001: Modality refactor

- **Status:** accepted
- **Date:** 2026-10-07
- **Scope:** the decisions behind the changes made in this commit

## TL;DR

The core change is how modalities are built. Before, a modality was any object with `ids`, `load` and `collate`.
Extra abilities were found with `hasattr`, and a DataFrame was wrapped silently. So a new modality's author could not
see what to implement, and each one had to get batching right on its own.

Now `Modality` is an abstract base class with two kinds:

- `FixedShapeModality`: every patient's item has the same shape (a table row), so a batch is stacked.
- `BagModality`: each item is an `(N_i, d)` bag of varying size (a slide's patches), so a batch is a list, and each
  instance can be described with `instances(id)`.

A new modality picks a kind and implements only `_load(id)`. The kind decides batching, `modality[ids]` returns a
batch, and tables are passed explicitly as `Tabular(frame)`. Most other decisions follow from this: the standard
PyTorch dataset pattern, the transform hooks, NaN checks moving into the model, and `EarlyFusion` accepting only
fixed-shape modalities.

Separately, interpretation now builds on the fitted pipeline through `Pipeline.run_modality`, and the pipeline no
longer imports `kalecancer.interpret`.

Each section below gives the context, the options considered, and the choice.

## 1. `Modality` is an abstract base class

**Context:** Modalities were duck-typed (`ids`, `load`, `collate`). Optional hooks such as `check`, `with_transform`
and `coords` were found with `hasattr`, so the author of a new modality could not see them.

**Options:**

- `typing.Protocol` (before): shares no code, and the hooks stay hidden.
- Subclass `torch.utils.data.Dataset`: it is indexed by integers, and `a + b` builds a `ConcatDataset`.
- Subclass `collections.abc.Mapping`: its inherited `in` and `==` load the data.
- A plain `ABC` (chosen).

**Choice:** A plain `ABC`. A subclass with a missing method fails when it is created. Subclasses implement the
private `_load(id)`, and callers read data with `modality[...]`, so the shared id checks always run.

## 2. Indexing a modality follows pandas

**Context:** Users wanted to read one or several patients straight from a modality.

**Options:**

- PyTorch's convention: `[]` returns one sample, and a method such as `batch(ids)` returns a batch.
- A batch-only `load(ids)`: less conventional, and only speeds up tables, which are already fast.
- pandas' convention: the type of the key decides the result (chosen).

**Choice:** pandas' convention, because it is more natural for users. `m["001"]` returns one item, and
`m[["001", "002"]]` returns a batch, even for a list of one. An empty list raises `ValueError`. Returning an empty
batch was rejected because it needs more code.

## 3. Two kinds of modality decide how a batch is formed

**Context:** A table row is `(d,)`, so a batch stacks into `(n, d)`. A bag of patch features is `(N_i, d)` with
`N_i` varying, so a batch must stay a list. Every modality used to write its own `collate`. The goal was to make
adding a modality as easy as possible.

**Options:**

- Each modality writes `collate` (before): easy to get wrong.
- Every batch is a list, and the model stacks it: plain `nn.Linear` can no longer be a first stage, and
  `EarlyFusion` and TabICL fitting would each need their own stacking.
- The dataset stacks 1-D items and keeps the rest as lists: wrong for fixed-size items such as `(C, H, W)` images.
- A default `collate` that returns a list, which tables override: every new vector modality must remember to
  override it.
- Padding with masks, or nested tensors: memory grows with the largest bag, and nested tensors cannot be indexed
  by patient.
- Kinds that load a whole batch from ids (`_load_batch`): the dataset could no longer load one patient at a time
  (see [5](#5-multimodaldataset-follows-the-standard-pytorch-pattern)).
- Two kinds of modality, each with its own `collate` (chosen).

**Choice:** Two kinds. `FixedShapeModality` stacks, because every patient's item has the same shape. `BagModality`
keeps a list. A new modality picks a kind and implements only `_load`, plus `instances` for a bag. Data that is
neither, such as a graph, subclasses `Modality` and implements `collate`. "Fixed shape" covers images as well as
vectors; "vector" and "matrix" were rejected, since images are not vectors and bags are the matrices. Knowing the kind
also lets `EarlyFusion` reject bags before training, and gives `instances` a home on bags only (see
[4](#4-bags-describe-their-instances-with-instancesid)).

## 4. Bags describe their instances with `instances(id)`

**Context:** This follows from the split into two kinds in [3](#3-two-kinds-of-modality-decide-how-a-batch-is-formed):
only bags have instances. Attention export joined weights to patch coordinates with `PatchFeatures.coords()`, found with
`getattr`. A weight carries no identity: weight *i* belongs to row *i* of the loaded bag. So the description must
follow the exact order of `_load`.

**Options:**

- Return coordinates from `_load`, always or behind a CLAM-style switch: every batch and stage would carry them.
- `attention.py` reads the files itself: the file layout and order would live in two places.
- Read the coordinates in `__init__`: opens every file up front.
- Rename the export to something WSI-specific, or add a `HasCoords` protocol: puts modality-specific code in a
  general module.
- A `details()` method on every modality: too loose to rely on, or over-built as a dataclass.
- Separate `PatchFeatures` and `PatchCoords` objects over shared files: users wire up three objects.
- `instances()` on every modality, returning `None` by default (the first version): fixed-shape data has no
  instances.
- An abstract `instances(id)` on `BagModality` (chosen).

**Choice:** Every bag implements `instances(id)`: one row per instance, in `_load`'s order, never read by fit or
predict. It suits any bag: patches, CT slices, report chunks. Attention export requires a `BagModality`.

Stages must keep the instances aligned. A stage samples instances at random only when `self.training` is true, and a
fixed selection, such as dropping background patches, belongs in the modality. Attention export checks the instance
count after each stage and names the stage that changed it.

## 5. `MultimodalDataset` follows the standard PyTorch pattern

**Context:** Once a modality could load a batch, loading could move out of `MultimodalDataset.__getitem__` and into
`collate`.

**Options:**

- `__getitem__` returns only the id, and `collate` loads: `data[0]` becomes a bare string.
- `__getitems__` returns a finished batch: breaks PyTorch's documented contract and needs a pass-through
  `collate_fn`.
- `batch_size=None` with a `BatchSampler`: an unfamiliar `DataLoader` setup.
- `__getitem__` loads one patient, and `collate` builds the batch (chosen).

**Choice:** The standard pattern, so anyone can use `MultimodalDataset` in their own training loop with
`DataLoader(data, collate_fn=data.collate)`. `collate` asks each modality to combine its items. A modality that no
patient in the batch has gets no entry in `inputs`.

## 6. Tables are passed explicitly as `Tabular(frame)`

**Context:** A DataFrame was wrapped automatically in a private `_Table`, and its data was only reachable through
undocumented attributes.

**Options:**

- `data.table(name)` on the dataset: the access belongs on the table.
- Subclass `pd.DataFrame`: pandas operations return plain DataFrames, extra attributes need `_metadata`, and
  `table["age"]` would be ambiguous (a patient or a column?).
- A pandas accessor: registers on every DataFrame.
- A wrapper whose `frame` is the live DataFrame: in-place edits would leave the cached array stale.
- A wrapper whose `frame` returns a copy (chosen).

**Choice:** A public `Tabular` wrapper. It copies the frame it is given, and `frame` returns a copy, so its cached
float32 array always matches the data. `MultimodalDataset` rejects a bare DataFrame, so there is one way to pass a
table.

## 7. Transforms stay scikit-learn, fitted by the Pipeline

**Context:** We reviewed whether the transform workflow was right, and how the Pipeline should get the raw rows that
a transform is fitted on.

**Options:**

- Our own transforms: rewrites scikit-learn and loses `clone`, feature names and YAML configs.
- Preprocess in the example script: leaks unless redone in every fold. It suits only steps that learn nothing from the
  data.
- Transforms as torch stages: torch cannot encode strings.
- The dataset fits its own transforms: gives the dataset per-fold state, which is how leakage creeps back.
- The Pipeline checks `isinstance(source, Tabular)`: modality-specific code in the Pipeline.
- Two transform hooks declared on `Modality` (chosen).

**Choice:** scikit-learn style transforms, fitted by the Pipeline on training rows only. `Modality` declares
`transform_input(ids)` and `with_transform(fitted, ids)`, which raise `TypeError` by default. `Tabular` implements
both, and a future modality can accept transforms the same way.

## 8. The model, not the modality, rejects NaN

**Context:** `Table` rejected NaN and infinite values, but whether NaN is acceptable depends on the encoder:
TabPFN-style models accept it. Without any check, Cox blamed "divergence", classification trained without error and
predicted NaN for every patient, and NaN in test rows gave NaN predictions without an error.

**Options:**

- Keep the check in the modality: the data cannot know what the model accepts.
- Check in each encoder: stock `nn.Module`s cannot, and authors forget.
- No check: the failures above.
- Check where the data is used, with an opt-out, as scikit-learn's `allow_nan` tag does (chosen).

**Choice:** `StageList` checks its input before the first stage, in training, prediction and `InContextModule`
fitting, and names the patients. A first stage that handles NaN sets `allow_nan = True`. The cost is that the error
comes at the first bad batch, not before training starts.

- `Tabular` keeps its non-numeric check, because a table cannot hold strings, and dates would silently become huge
  numbers.
- `TabICLEncoder` keeps its own check, so it is safe to use on its own. It does not set `allow_nan`, because it
  bypasses upstream TabICL's imputer and NaN still fails.

## 9. Interpretation builds on the Pipeline, never the reverse

**Context:** `Pipeline.attention` imported `kalecancer.interpret`, which then called private Pipeline methods. Each
new interpretation tool would need another Pipeline method or more private calls.

**Options:**

- Keep `Pipeline.attention`.
- Free functions that call private Pipeline methods.
- One public, general Pipeline method for interpretation tools (chosen).

**Choice:** Interpretation is an outer layer that takes a fitted pipeline, as in scikit-learn's `inspection`, Captum
and SHAP. `Pipeline.run_modality(data, modality, fn)` runs one modality's fitted stages with a callback, and `encode`
uses it too. `kalecancer.interpret.attention(pipe, data, modality)` replaces `Pipeline.attention`, and
`kalecancer.pipeline` no longer imports `kalecancer.interpret`.
