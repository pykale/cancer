# ADR 0003: Patient ids

- **Date:** 2026-10-21
- **Scope:** the type, dtype, order and name of the `ids` that modalities, targets and `MultimodalDataset` expose

## TL;DR

- Every `ids` is a `pd.Index` of unique `str` values, sorted, named `"id"`, and built with `dtype="str"`.
- All of them are built by `_as_ids`. Methods that take ids still accept any iterable of `str`.

## Context

Modalities and targets exposed a `pd.Index` of dtype `object`. `MultimodalDataset` exposed a `list[str]`. Index names differed: `Tabular` kept the user's, `PatchFeatures` had none, and `present` used `"id"`. Target ids kept their input order, while modality and dataset ids were sorted. Ids are how users align their own tables with the library's, so the type is part of the public contract and hard to change later.

## Options

- **`list[str]`:** simple, but mutable, so it can fall out of step with the dataset's id set and `present`.
- **`tuple[str, ...]`:** immutable, and `==` returns a single bool. However, `df.loc[ids]` reads a tuple as a (row,
  column) key. It raises a confusing `KeyError`, or, with a single id, silently returns a Series.
- **`pd.Index` (chosen):** immutable, and hashed, and it works directly with `.loc`, `reindex`, `isin` and set operations. Like the rest of the API (`Tabular` frames, `Target` Series, `present`, predictions), it is pandas.
- **A custom `Ids` class:** a new public type for little gain.

## Choice

`pd.Index`. The expected users, bioinformaticians, know pandas. With a tuple, writing `df.loc[data.ids]` would fail. Deep learning users mostly iterate over ids, take their length, slice them and pass them to splitters, and those work the same for an Index and a tuple.

The contract that ids provide is what really matters. ids are unique `str` values, sorted and immutable, and two ids match only when the strings are identical.

## Dtype

`_as_ids` builds the Index with `dtype="str"`. Ids are already checked to be `str`, so the dtype should reflect this. On pandas 3 this is the default string dtype and prints as `dtype='str'`. On pandas 2.x, `"str"` is stored as `object`.

kalecancer is held on pandas 2.2 because `pykale>=0.2.0` pins `pandas<=2.2.2`. Once that pin is relaxed, ids show as `dtype='str'` with no code change.

## Consequences

- `MultimodalDataset.ids` changes from a list to an Index. Tests compare ids with `.equals` or `list(...)`.
- Target ids are sorted, like modality and dataset ids.
- Saved attributes such as `train_ids_` are written out with `list(...)`, because an Index is not JSON-serialisable.
