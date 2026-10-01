"""Targets: what is predicted for each patient."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from kalecancer.loaddata.identifiers import _as_ids


class BaseTarget(ABC):
    frame: pd.DataFrame

    @property
    def ids(self) -> pd.Index:
        return self.frame.index

    @abstractmethod
    def tensors(self, ids: Sequence[str]) -> dict[str, Tensor]:
        """The target for ``ids``, in that order, as the tensors the head's loss reads."""

    @abstractmethod
    def strata(self, ids: Sequence[str]) -> np.ndarray:
        """One discrete label per patient for stratified splitters"""

    @abstractmethod
    def counts(self, ids: Sequence[str]) -> dict[str, int]:
        """Summary counts for ``ids`` to report in fits and folds, such as events or patients per class."""


class TimeToEvent(BaseTarget):
    """Right-censored time-to-event target.

    Args:
        time: Follow-up time per patient, finite and > 0.
        event: ``True`` where the event was observed, ``False`` where censored. Missing values raise:
            an unknown outcome is not a censored one.
    """

    def __init__(self, time: pd.Series, event: pd.Series):
        self.time = time
        self.event = event
        ids = _as_ids(time.index, "TimeToEvent.time")
        if set(_as_ids(event.index, "TimeToEvent.event")) != set(ids):
            raise ValueError("TimeToEvent: time and event must be indexed by the same ids")
        event = event.loc[ids]
        if event.isna().any():
            examples = event.index[event.isna()][:5].tolist()
            raise ValueError(
                f"TimeToEvent: event is missing for {int(event.isna().sum())} patients (e.g. {examples}); "
                "map every status explicitly, an unknown outcome is not censoring"
            )
        if not (pd.api.types.is_bool_dtype(event) or all(isinstance(v, bool | np.bool_) for v in event)):
            raise TypeError(f"TimeToEvent: event must be bool (True = event observed), got dtype {event.dtype}")
        numeric_time = pd.to_numeric(time, errors="coerce")
        bad = ~np.isfinite(numeric_time.to_numpy(dtype=float)) | (numeric_time.to_numpy(dtype=float) <= 0)
        if bad.any():
            raise ValueError(
                f"TimeToEvent: time must be finite and > 0; {int(bad.sum())} patients violate this "
                f"(e.g. {time.index[bad][:5].tolist()})"
            )
        self.frame = pd.DataFrame(
            {"time": numeric_time.to_numpy(dtype=np.float32), "event": event.to_numpy(dtype=bool)}, index=ids
        )

    def tensors(self, ids: Sequence[str]) -> dict[str, Tensor]:
        rows = self.frame.loc[list(ids)]
        return {
            "time": torch.tensor(rows["time"].to_numpy(dtype=np.float32)),
            "event": torch.tensor(rows["event"].to_numpy(dtype=bool)),
        }

    def strata(self, ids: Sequence[str]) -> np.ndarray:
        return self.frame.loc[list(ids), "event"].to_numpy()

    def counts(self, ids: Sequence[str]) -> dict[str, int]:
        return {"events": int(self.frame.loc[list(ids), "event"].sum())}


class Classification(BaseTarget):
    """Class-label target.

    Args:
        labels: Label per patient; every value must be one of ``classes``.
        classes: All classes, in the order that defines class indices and prediction columns.
    """

    def __init__(self, labels: pd.Series, classes: Sequence[Hashable]):
        self.labels = labels
        self.classes = classes
        ordered = list(classes)
        if len(ordered) < 2 or len(set(ordered)) != len(ordered):
            raise ValueError(f"Classification: classes must hold at least two distinct values, got {ordered}")
        ids = _as_ids(labels.index, "Classification")
        if labels.isna().any():
            examples = labels.index[labels.isna()][:5].tolist()
            raise ValueError(f"Classification: {int(labels.isna().sum())} labels are missing (e.g. {examples})")
        unknown = sorted({str(v) for v in labels if v not in set(ordered)})
        if unknown:
            raise ValueError(f"Classification: labels {unknown[:5]} are not in classes {ordered}")
        self._index = {c: k for k, c in enumerate(ordered)}
        self.frame = pd.DataFrame({"label": labels.to_numpy(dtype=object)}, index=ids)

    def tensors(self, ids: Sequence[str]) -> dict[str, Tensor]:
        codes = [self._index[v] for v in self.frame.loc[list(ids), "label"]]
        return {"label": torch.tensor(codes, dtype=torch.int64)}

    def strata(self, ids: Sequence[str]) -> np.ndarray:
        return self.frame.loc[list(ids), "label"].to_numpy()

    def counts(self, ids: Sequence[str]) -> dict[str, int]:
        observed = Counter(self.frame.loc[list(ids), "label"])
        return {f"label: {c}": observed.get(c, 0) for c in self.classes}
