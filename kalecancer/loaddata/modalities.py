"""Modalities: where each patient's inputs come from, keyed by patient id."""

from __future__ import annotations

import glob
import re
from collections.abc import Sequence
from typing import Any, Literal, Protocol

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin
from torch import Tensor

from kalecancer.loaddata.identifiers import _as_ids


class Modality(Protocol):
    """What a modality provides. New kinds of data (slides, volumes, text) implement this."""

    ids: pd.Index

    def load(self, id: str) -> Any: ...

    def collate(self, items: list) -> Any: ...


class _Table:
    """A DataFrame modality: one float32 vector per patient."""

    def __init__(self, frame: pd.DataFrame, name: str):
        self.frame = frame
        self.name = name
        self.ids = _as_ids(frame.index, f"modality {name!r}")
        self._position = {pid: k for k, pid in enumerate(self.ids)}
        self._values: np.ndarray | None = None

    def _array(self) -> np.ndarray:
        if self._values is None:
            non_numeric = [c for c in self.frame.columns if not pd.api.types.is_numeric_dtype(self.frame[c])]
            if non_numeric:
                raise TypeError(
                    f"modality {self.name!r}: columns {non_numeric[:5]} are not numeric; give this modality a transform"
                )
            self._values = self.frame.to_numpy(dtype=np.float32)
        return self._values

    def check(self, ids: Sequence[str]) -> None:
        rows = self._array()[[self._position[pid] for pid in ids]]
        bad = ~np.isfinite(rows).all(axis=1)
        if bad.any():
            examples = [pid for pid, b in zip(ids, bad, strict=True) if b][:5]
            raise ValueError(
                f"modality {self.name!r}: NaN or infinite values for {int(bad.sum())} patients (e.g. {examples}); "
                "impute explicitly in a transform"
            )

    def load(self, id: str) -> Tensor:
        row = self._array()[self._position[id]]
        if not np.isfinite(row).all():
            raise ValueError(
                f"modality {self.name!r}: NaN or infinite values for {id!r}; impute explicitly in a transform"
            )
        return torch.tensor(row)

    def collate(self, items: list[Tensor]) -> Tensor:
        return torch.stack(items) if items else torch.empty(0, self.frame.shape[1])

    def transform_input(self, ids: Sequence[str]) -> pd.DataFrame:
        return self.frame.loc[list(ids)]

    def with_transform(self, fitted: TransformerMixin, ids: Sequence[str]) -> _Table:
        raw = self.frame.loc[list(ids)]
        values = fitted.transform(raw)
        if hasattr(values, "toarray"):
            values = values.toarray()
        try:
            values = np.asarray(values, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise TypeError(f"modality {self.name!r}: the transform output is not numeric ({error})") from error
        if values.ndim != 2 or len(values) != len(raw):
            raise ValueError(
                f"modality {self.name!r}: the transform must return one row per patient, got shape {values.shape}"
            )
        try:
            columns = [str(c) for c in fitted.get_feature_names_out()]
        except (AttributeError, ValueError):
            columns = None
        if columns is not None and len(columns) != values.shape[1]:
            columns = None
        return _Table(pd.DataFrame(values, index=raw.index, columns=columns), self.name)


class PatchFeatures:
    """Pre-extracted patch features: one h5 file (or several) per patient, read lazily per patient.

    Args:
        files: Paths indexed by patient id. A repeated id means that patient has several files.
        features_key: Dataset holding the (N, D) features in each file.
        coords_key: Dataset holding the (N, 2) patch coordinates, row-aligned with the features.
        multiple_files: What to do when a patient has several files. ``"error"`` raises;
            ``"concatenate"`` reads them as one bag, in sorted path order.
    """

    def __init__(
        self,
        files: pd.Series,
        features_key: str = "features",
        coords_key: str = "coords",
        multiple_files: Literal["error", "concatenate"] = "error",
    ):
        if multiple_files not in ("error", "concatenate"):
            raise ValueError(f"multiple_files must be 'error' or 'concatenate', got {multiple_files!r}")
        self.files = files
        self.features_key = features_key
        self.coords_key = coords_key
        self.multiple_files = multiple_files
        _as_ids(pd.Index(files.index).unique(), "PatchFeatures")
        self._paths = {pid: sorted(str(p) for p in paths) for pid, paths in files.groupby(level=0)}
        several = sorted(pid for pid, paths in self._paths.items() if len(paths) > 1)
        if several and multiple_files == "error":
            raise ValueError(
                f"{len(several)} patients have several files (e.g. {several[:3]}); "
                "pass multiple_files='concatenate' to read each patient's files as one bag"
            )
        self.ids = pd.Index(sorted(self._paths))

    @classmethod
    def from_glob(
        cls,
        pattern: str,
        id_pattern: str,
        features_key: str = "features",
        coords_key: str = "coords",
        multiple_files: Literal["error", "concatenate"] = "error",
    ) -> PatchFeatures:
        """Find files with a glob pattern; group 1 of ``id_pattern`` (searched in each path) is the patient id."""
        paths = sorted(glob.glob(pattern, recursive=True))
        if not paths:
            raise FileNotFoundError(f"no files match {pattern!r}")
        regex = re.compile(id_pattern)
        if regex.groups < 1:
            raise ValueError(f"id_pattern {id_pattern!r} needs a capturing group for the patient id")
        ids = []
        for path in paths:
            match = regex.search(path)
            if match is None:
                raise ValueError(f"{path} does not match id_pattern {id_pattern!r}")
            ids.append(match.group(1))
        return cls(pd.Series(paths, index=pd.Index(ids, name="id")), features_key, coords_key, multiple_files)

    @staticmethod
    def _dataset(f: h5py.File, path: str, key: str) -> h5py.Dataset:
        if key not in f:
            raise KeyError(f"{path}: no dataset {key!r} (found {sorted(f.keys())})")
        dataset = f[key]
        if dataset.ndim != 2 or dataset.shape[0] == 0:
            raise ValueError(f"{path}:{key} must be a non-empty 2-D array, got shape {dataset.shape}")
        return dataset

    def load(self, id: str) -> Tensor:
        arrays = []
        for path in self._paths[id]:
            with h5py.File(path, "r") as f:
                arrays.append(self._dataset(f, path, self.features_key)[:])
        return torch.from_numpy(np.concatenate(arrays)).float()

    def coords(self, id: str) -> pd.DataFrame:
        """Patch coordinates in the same file and row order as :meth:`load`."""
        frames = []
        for path in self._paths[id]:
            with h5py.File(path, "r") as f:
                n_features = self._dataset(f, path, self.features_key).shape[0]
                xy = self._dataset(f, path, self.coords_key)[:]
            if len(xy) != n_features:
                raise ValueError(f"{path}: {len(xy)} coordinates for {n_features} feature rows")
            frames.append(pd.DataFrame({"file": path, "x": xy[:, 0], "y": xy[:, 1]}))
        return pd.concat(frames, ignore_index=True)

    def collate(self, items: list[Tensor]) -> list[Tensor]:
        return list(items)


def _as_modality(name: str, value: Any) -> Any:
    if not isinstance(name, str):
        raise TypeError(f"modality names must be str, got {name!r}")
    if isinstance(value, pd.DataFrame):
        return _Table(value, name)
    missing = [attr for attr in ("ids", "load", "collate") if not hasattr(value, attr)]
    if missing:
        raise TypeError(f"modality {name!r}: {type(value).__name__} is not a DataFrame and lacks {missing}")
    return value
