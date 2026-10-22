# Adding a modality

A modality holds one kind of data for a set of patients, keyed by patient id. A new kind of data is one class in
`kalecancer/loaddata/modalities.py`. Nothing else in the library changes: `MultimodalDataset`, the models and the
`Pipeline` work with it as they are.

## 1. Choose a kind

The kind decides how a batch is formed, so pick it first. Only inherit from Modality directly if strictly necessary, prefer FixedShapeModality or BagModality where possible.

| Subclass | Use it when | Batch | You implement |
| --- | --- | --- | --- |
| `FixedShapeModality` | every patient's item has the same shape (a vector, an image, a resampled volume) | one stacked `(n, *shape)` tensor | `ids`, `_view`, `_load` |
| `BagModality` | each item is an `(N_i, d)` bag whose size varies (patches, CT slices, report chunks) | a list of `(N_i, d)` tensors | `ids`, `_view`, `_load`, `instances` |
| `Modality` | neither, e.g. a graph | whatever your `collate` returns | `ids`, `_view`, `_load`, `collate` |

`EarlyFusion` fuses raw inputs, so it accepts only fixed-shape modalities.

## 2. Implement it

The base class checks the ids and provides `[ids]` from your `_view` and `tensors(ids)` from your `_load`, so a
fixed-shape modality sets `ids` and implements `_view(ids)` and `_load(id)`:

```python
class NpyVectors(FixedShapeModality):
    """One ``.npy`` vector per patient, such as an expression profile.

    Args:
        files: Paths indexed by patient id.
    """

    def __init__(self, files: pd.Series):
        self.files = files
        self.ids = _as_ids(files.index, type(self).__name__)

    def _view(self, ids: pd.Index) -> pd.DataFrame:
        rows = np.stack([np.load(self.files[pid]) for pid in ids])
        return pd.DataFrame(rows, index=ids, columns=[f"feature_{j}" for j in range(rows.shape[1])])

    def _load(self, id: str) -> Tensor:
        return torch.from_numpy(np.load(self.files[id])).float()
```

- `ids` is a `pd.Index` of unique strings. `_as_ids` from `loaddata/identifiers.py` checks this and gives the
  standard error messages.
- `_view(ids)` returns the readable view, which fit and predict never read. It receives a non-empty Index of known
  ids, in the order requested. Return a DataFrame indexed by patient id, **even for one id**, unless another form
  suits the data better, such as text. Never return tensors, and don't override `__getitem__`, which checks the ids.
- `_load(id)` returns one patient's item as a tensor.
- Load lazily in `_load`, as `PatchFeatures` does, rather than reading every file in `__init__`.
- Follow the [Conventions](../../CONTRIBUTING.md#conventions).

A bag also implements `instances(id)`: a DataFrame with one row per instance, **in the same order as `_load`**, describing each one. Attention export joins it to the attention weights, and fit and predict never read it. Its view also has one row per instance, in the same order, indexed by `(id, instance)`. See `PatchFeatures` for example.

```python
class NpyBags(BagModality):
    ...  # __init__, _view and _load as above, with _load returning an (N_i, d) tensor

    def instances(self, id: str) -> pd.DataFrame:
        n = np.load(self.files[id], mmap_mode="r").shape[0]
        return pd.DataFrame({"file": self.files[id], "row": range(n)})
```

A fixed selection, such as dropping background patches, belongs in `_load`, applied in the same way in
`instances`, so that the two describe the same rows. Random sampling belongs in an encoder (see
[encoders.md](encoders.md)).

## 3. Accept transforms (optional)

By default a modality rejects transforms. To accept scikit-learn transforms, which the `Pipeline` fits on the training
patients only, implement both hooks as `Tabular` does:

- `transform_input(ids)` returns the raw data of `ids` that a transform is fitted on;
- `with_transform(fitted, ids)` returns a new modality holding the rows of `ids` after the fitted transform.

Missing values are not checked when data is loaded. Impute them in the transform, unless the encoder handles them
itself.

## 4. Use it

```python
data = MultimodalDataset(
    modalities={"rna": NpyVectors(files), "clinical": Tabular(clinical)},
    target=TimeToEvent(time=outcome["time"], event=outcome["event"]),
    required_modalities=["rna"],
)
```

Give the modality's encoders input of the right shape: an `(n, *shape)` tensor for a fixed-shape modality, or a list
of bags for a bag modality.

`data["001"]` shows patient 001's views of every modality, and its target.

## Checklist

- [ ] The class lives in `kalecancer/loaddata/modalities.py` and is exported from `kalecancer/loaddata/__init__.py`.
- [ ] Tests in `tests/test_loaddata.py` cover the view and `tensors`, for one id and several, unknown ids and, for a
  bag, that `instances` and the view have one row per loaded instance. Build the data synthetically in `tmp_path`.
- [ ] The modality table in `docs/architecture.md` lists it.

See [ADR 0001](../adr/0001-modality-refactor.md) for why modalities are designed this way.
