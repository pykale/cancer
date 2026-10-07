"""Modalities: where each patient's inputs come from, keyed by patient id."""

from __future__ import annotations

import glob
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from functools import cached_property
from typing import Any, Literal, Self

import h5py
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.base import TransformerMixin
from torch import Tensor

from kalecancer.loaddata.identifiers import _as_ids


class Modality(ABC):
    """Per-patient data, keyed by patient id. Subclass ``FixedShapeModality`` or ``BagModality`` and implement
    ``_load``."""

    ids: pd.Index

    def __getitem__(self, ids: Iterable[str] | str) -> Tensor | list[Tensor]:
        """One patient's item for a single id, or a batch for several ids, in the order given."""
        if isinstance(ids, str):
            if ids not in self.ids:
                raise KeyError(f"{type(self).__name__}: unknown id {ids!r}")
            return self._load(ids)

        requested = _as_ids(ids, type(self).__name__)
        if len(requested) == 0:
            raise ValueError(f"{type(self).__name__}: no ids requested")
        if missing := [pid for pid in requested if pid not in self.ids]:
            raise KeyError(f"{type(self).__name__}: {len(missing)} unknown ids, e.g. {missing[:5]}")
        return self.collate([self._load(pid) for pid in requested])

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)

    def with_transform(self, fitted_transforms: Any, ids: Sequence[str]) -> Self:
        """Given some fitted transforms, return the data of the specified ids with transforms applied."""
        raise TypeError(
            f"{self.__class__.__name__} has no implemented with_transform() method, so cannot accept any transformations"
        )

    def transform_input(self, ids: Sequence[str]) -> Any:
        """The data of ``ids`` that a transform for this modality is fitted on."""
        raise TypeError(f"{type(self).__name__} does not accept transforms")

    @abstractmethod
    def _load(self, id: str) -> Tensor:
        """Return the data as a tensor for the given id.
        Used by Modality.__getitem__()."""

    @abstractmethod
    def collate(self, items: list[Tensor]) -> Tensor | list[Tensor]:
        """Given a list of items returned by _load(), combine one item per patient, in order, into a batch.
        Used by Modality.__getitem__() when multiple IDs are passed.
        Used by MultimodalDataset.collate() to combine the items of each modality into a batch."""


class FixedShapeModality(Modality):
    """Every patient's item has the same shape, e.g. one vector per patient."""

    def collate(self, items: list[Tensor]) -> Tensor:
        """Stack the items into one ``(n, *item_shape)`` tensor."""
        return torch.stack(items)


class BagModality(Modality):
    """Each patient's item is an ``(N_i, d)`` bag of instances, where N_i varies between patients."""

    def collate(self, items: list[Tensor]) -> list[Tensor]:
        """Keep the bags as a list, since their sizes differ."""
        return items

    @abstractmethod
    def instances(self, id: str) -> pd.DataFrame:
        """One row per instance of ``_load(id)``, in the same order, describing each one (e.g. file, x, y).
        Never read by fit or predict.
        Used by interpret module."""


class Tabular(FixedShapeModality):
    """A DataFrame modality: one float32 vector per patient."""

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame.copy()
        self.ids = _as_ids(self._frame.index, f"{self.__class__.__name__}")
        self._position = {pid: k for k, pid in enumerate(self.ids)}

    @property
    def frame(self) -> pd.DataFrame:
        """A copy of the table."""
        return self._frame.copy()

    @cached_property
    def _array(self) -> np.ndarray:
        """The table as a float32 array; raises if a column is not numeric."""
        non_numeric = [c for c in self._frame.columns if not pd.api.types.is_numeric_dtype(self._frame[c])]
        if non_numeric:
            raise TypeError(
                f"{self.__class__.__name__!r}: columns {non_numeric[:5]} are not numeric; give this modality a transform"
            )
        return self._frame.to_numpy(dtype=np.float32)

    def _load(self, id: str) -> Tensor:
        """The patient's row as a float32 vector."""
        return torch.tensor(self._array[self._position[id]], dtype=torch.float32)

    def with_transform(self, fitted_transforms: TransformerMixin, ids: Sequence[str]) -> Tabular:
        """A new ``Tabular`` holding the rows of ``ids`` after ``fitted_transforms``."""
        raw = self.transform_input(ids)
        values = fitted_transforms.transform(raw)
        if sparse.issparse(values):
            values = values.toarray()
        values = np.asarray(values, dtype=np.float32)
        try:
            columns = [str(c) for c in fitted_transforms.get_feature_names_out()]
        except (AttributeError, ValueError):
            columns = None
        if columns is not None and len(columns) != values.shape[1]:
            columns = None
        return Tabular(pd.DataFrame(values, index=raw.index, columns=columns))

    def transform_input(self, ids: Sequence[str]) -> pd.DataFrame:
        return self._frame.loc[list(ids)]


class PatchFeatures(BagModality):
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
        _as_ids(pd.Index(files.index).unique(), f"{self.__class__.__name__}")
        self._paths = {str(pid): sorted(str(p) for p in paths) for pid, paths in files.groupby(level=0)}
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
        """The dataset ``key`` in ``f``, checked to be a non-empty 2-D array."""
        if key not in f:
            raise KeyError(f"{path}: no dataset {key!r} (found {sorted(f.keys())})")
        dataset = f[key]
        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"{path}:{key} is a {type(dataset).__name__}, not a dataset")
        if dataset.ndim != 2 or dataset.shape[0] == 0:
            raise ValueError(f"{path}:{key} must be a non-empty 2-D array, got shape {dataset.shape}")
        return dataset

    def _load(self, id: str) -> Tensor:
        """The patient's patch features as one ``(N, D)`` float32 tensor, joined across their files."""
        arrays = []
        for path in self._paths[id]:
            with h5py.File(path, "r") as f:
                arrays.append(self._dataset(f, path, self.features_key)[:])
        return torch.from_numpy(np.concatenate(arrays)).float()

    def instances(self, id: str) -> pd.DataFrame:
        """Patch coordinates in the same file and row order as :meth:`_load`."""
        frames = []
        for path in self._paths[id]:
            with h5py.File(path, "r") as f:
                n_features = self._dataset(f, path, self.features_key).shape[0]
                xy = self._dataset(f, path, self.coords_key)[:]
            if len(xy) != n_features:
                raise ValueError(f"{path}: {len(xy)} coordinates for {n_features} feature rows")
            frames.append(pd.DataFrame({"file": path, "x": xy[:, 0], "y": xy[:, 1]}))
        return pd.concat(frames, ignore_index=True)
