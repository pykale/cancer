# ADR 0004: Patient data access

- **Date:** 2026-10-22
- **Scope:** how people and the library read data from modalities, targets and `MultimodalDataset`

## TL;DR

- `[ids]` (`__getitem__()`) on a `Modality`, `Target` or `MultimodalDataset` is a readable **view** for users. `tensors(ids)` on a modality or target, and `data.as_torch()` on the dataset, provide the tensors for models.
- Every `[]` takes patient ids only, one `str` or an ordered collection, and not a position.
- Table-shaped views are DataFrames indexed by patient id. `MultimodalDataset[ids]` is `{modality1: view, modality2: view, ..., "target": view}`.

## Context

`MultimodalDataset[i]` took a position and `Modality`'s took ids, and both returned tensors. Readable data was spread across `Tabular.frame`, `PatchFeatures.instances(id)` and `target.frame`. The dataset, meant to hold a cohort aligned across modalities, was hard to work with.

## Options

- `[]` returns tensors and `.view(ids)` returns readable data. `DataLoader(data)` keeps working, but the quick look in a notebook, `data["p1"]`, gives tensors.
- `[]` returns readable data and `.tensors(ids)` returns tensors (chosen). Only library code needs tensors, so the short syntax goes to people. The cost is an adapter for `DataLoader`.

For patch features, the view could be coordinates only (hides the features), AnnData (the scverse standard, but a hard dependency many users won't know), or a DataFrame with the features as columns (chosen: pandas only, and ready for plotting and statistics).

## Choice

- `tensors` names its output and matches the target's existing `tensors`. `load` was rejected, because a `Tabular` loads nothing.
- `data.as_torch()` returns a `MultimodalTorchDataset`, which loads each patient's tensors and holds `collate`. Torch's `DataLoader` and samplers are unchanged.
- One helper, `_requested_ids`, checks every key, so every object gives the same errors. An int or a set raises `TypeError`: ids such as `"0001"` make positions and labels easy to confuse.
- A view's type never depends on the key, so one id still gives a DataFrame, unlike pandas' `df.loc["a"]`.
- `data[ids]` is keyed by modality, so one id and many give the same shape. A missing modality is `None`, so a `KeyError` only ever means an unknown modality name.

## Consequences

- `DataLoader(data, collate_fn=data.collate)` becomes `DataLoader(torch_data, collate_fn=torch_data.collate)`, with `torch_data = data.as_torch()`.
- `data[i]` (where i is an int index) raises `TypeError`, and `modality[ids]` returns a view: code that wants tensors calls `tensors(ids)`.
- Every modality implements `_view`, which the base `__getitem__` calls after checking the ids, and a modality cannot be called `"target"`.
- Supersedes ADR 0001 §5. ADR 0001 §2 now describes `tensors`.
