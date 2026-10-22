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

from kalecancer.loaddata.identifiers import _as_ids, _requested_ids


class Modality(ABC):
    """Per-patient data, keyed by patient id.

    Subclass ``FixedShapeModality`` or ``BagModality`` and implement ``_view`` and ``_load``.
    """

    ids: pd.Index

    def __getitem__(self, ids: Iterable[str] | str) -> Any:
        """Checks provided `ids` and then returns the modality of the given id(s) in a convenient format."""
        return self._view(_requested_ids(ids, self.ids, type(self).__name__))

    @abstractmethod
    def _view(self, ids: pd.Index) -> Any:
        """Returns the modality of the given id(s) in a convenient format.

        ``ids`` is already checked by ``__getitem__``: a non-empty Index of known ids, in the order requested.
        Return a `pd.DataFrame` indexed by patient id, even for one id, unless another format is standard for the modality, e.g. free text. If the raw data is too large to hold in memory, such as image volumes, return `pd.DataFrame` pointing to its files instead.
        """

    def tensors(self, ids: Iterable[str] | str) -> Tensor | list[Tensor]:
        """One patient's item for a single id, or a batch for several ids, in the order given."""
        requested = _requested_ids(ids, self.ids, type(self).__name__)
        if isinstance(ids, str):
            return self._load(ids)
        return self.collate([self._load(pid) for pid in requested])

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)

    def with_transform(self, fitted_transforms: Any, ids: Sequence[str]) -> Self:
        """A new modality holding the data of ``ids`` after ``fitted_transforms``.

        Modalities that accept transforms override this. By default, it raises ``TypeError``.
        """
        raise TypeError(
            f"{self.__class__.__name__} has no implemented with_transform() method, so cannot accept any transformations"
        )

    def transform_input(self, ids: Sequence[str]) -> Any:
        """The data of ``ids`` that a transform for this modality is fitted on."""
        raise TypeError(f"{type(self).__name__} does not accept transforms")

    @abstractmethod
    def _load(self, id: str) -> Tensor:
        """Return the data as a tensor for the given id.

        Used by ``Modality.tensors()``.
        """

    @staticmethod
    @abstractmethod
    def collate(items: list[Tensor]) -> Tensor | list[Tensor]:
        """Combine items returned by ``_load``, one per patient and in order, into a batch.

        Used by ``Modality.tensors()`` when several ids are passed, and by ``MultimodalDataset.collate`` to batch
        each modality's items.
        """


class FixedShapeModality(Modality, ABC):
    """Every patient's item has the same shape, e.g. one vector per patient."""

    @staticmethod
    def collate(items: list[Tensor]) -> Tensor:
        """Stack the items into one ``(n, *item_shape)`` tensor."""
        return torch.stack(items)


class BagModality(Modality, ABC):
    """Each patient's item is an ``(N_i, d)`` bag of instances, where N_i varies between patients."""

    @staticmethod
    def collate(items: list[Tensor]) -> list[Tensor]:
        """Keep the bags as a list, since their sizes differ."""
        return items

    @abstractmethod
    def instances(self, id: str) -> pd.DataFrame:
        """One row per instance of ``tensors(id)``, in the same order, describing each one (e.g. file, x, y).

        Used by the interpret module, never by fit or predict.
        """


class Tabular(FixedShapeModality):
    """A DataFrame modality: one float32 vector per patient.

    Args:
        frame: One row per patient, indexed by patient id as a string. Its columns must be numeric, unless the
            pipeline gives this modality a transform that makes them numeric.
    """

    def __init__(self, frame: pd.DataFrame):
        self._frame = frame.copy()
        self._frame.index = _as_ids(self._frame.index, f"{self.__class__.__name__}")
        self.ids = self._frame.index.sort_values()
        self._position = {pid: k for k, pid in enumerate(self._frame.index)}

    def _view(self, ids: pd.Index) -> pd.DataFrame:
        """The rows of ``ids``, in the order given, as a DataFrame even for one id."""
        return self._frame.loc[ids]

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
        """The rows of ``ids``, before any transform."""
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
        self.ids = _as_ids(pd.Index(files.index).unique(), f"{self.__class__.__name__}").sort_values()
        self._paths = {pid: sorted(str(p) for p in paths) for pid, paths in files.groupby(level=0)}
        several = sorted(pid for pid, paths in self._paths.items() if len(paths) > 1)
        if several and multiple_files == "error":
            raise ValueError(
                f"{len(several)} patients have several files (e.g. {several[:3]}); "
                "pass multiple_files='concatenate' to read each patient's files as one bag"
            )

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

    def _view(self, ids: pd.Index) -> pd.DataFrame:
        """Patches of the given patient(s), indexed by (id, patch), with file, x, y and feature_0..feature_{D-1}."""
        ids = list(ids)

        # Pass 1: shapes only (reads HDF5 headers, not data).
        plan = []  # (id, path, n_rows)
        n_features = None
        for pid in ids:
            for path in self._paths[pid]:
                with h5py.File(path, "r") as f:
                    rows, d = self._dataset(f, path, self.features_key).shape
                    n_coords = self._dataset(f, path, self.coords_key).shape[0]
                if n_coords != rows:
                    raise ValueError(f"{path}: {n_coords} coordinates for {rows} feature rows")
                if n_features is None:
                    n_features = d
                elif d != n_features:
                    raise ValueError(f"{path}: {d} features, expected {n_features}")
                plan.append((pid, path, rows))
        total = sum(rows for _, _, rows in plan)
        assert n_features is not None  # for pre-commit only - it cannot be None due to _requrested_ids()

        # Pass 2: read straight into preallocated buffers.
        features = np.empty((total, n_features), dtype=np.float32)
        xy = np.empty((total, 2), dtype=np.int64)
        file_codes = np.empty(total, dtype=np.int32)
        patch = np.empty(total, dtype=np.int64)
        id_codes = np.empty(total, dtype=np.int32)

        paths = [path for _, path, _ in plan]
        id_position = {pid: k for k, pid in enumerate(ids)}
        start = 0
        next_patch = dict.fromkeys(ids, 0)
        for k, (pid, path, rows) in enumerate(plan):
            stop = start + rows
            with h5py.File(path, "r") as f:
                f[self.features_key].read_direct(features, dest_sel=np.s_[start:stop])
                f[self.coords_key].read_direct(xy, source_sel=np.s_[:, :2], dest_sel=np.s_[start:stop])
            file_codes[start:stop] = k
            id_codes[start:stop] = id_position[pid]
            patch[start:stop] = np.arange(next_patch[pid], next_patch[pid] + rows)
            next_patch[pid] += rows
            start = stop

        index = pd.MultiIndex(
            levels=[pd.Index(ids, name="id"), pd.RangeIndex(patch.max() + 1 if total else 0)],
            codes=[id_codes, patch],
            names=["id", "patch"],
        )
        df = pd.DataFrame(features, index=index, columns=[f"feature_{j}" for j in range(n_features)], copy=False)
        df.insert(0, "y", xy[:, 1])
        df.insert(0, "x", xy[:, 0])
        df.insert(0, "file", pd.Categorical.from_codes(file_codes, categories=paths))
        return df

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
