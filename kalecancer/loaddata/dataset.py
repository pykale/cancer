"""The multimodal dataset: modalities and a target joined by patient id, and patient-level splitting."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd
import torch
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split as _sklearn_train_test_split

from kalecancer.loaddata.identifiers import _as_ids, _check_leading_zeros, _NotFoundError
from kalecancer.loaddata.modalities import _as_modality
from kalecancer.loaddata.targets import Classification, TimeToEvent


class MultimodalDataset(torch.utils.data.Dataset):
    """Modalities and a target joined by patient id.

    Args:
        modalities: Name to modality. A DataFrame indexed by id is a table modality.
        target: The target, or ``None`` for unlabelled data.
        required_modalities: Modalities every included patient must have. Patients also need the target
            (when given) and at least one modality. ``summary()`` reports who was excluded and why.
    """

    def __init__(
        self,
        modalities: Mapping[str, Any],
        target: TimeToEvent | Classification | None,
        required_modalities: Sequence[str],
    ):
        if not modalities:
            raise ValueError("MultimodalDataset needs at least one modality")
        if isinstance(required_modalities, str):
            raise TypeError("required_modalities must be a list of modality names, not a single string")
        self.modalities = {name: _as_modality(name, value) for name, value in modalities.items()}
        self.target = target
        self.required_modalities = list(required_modalities)
        unknown = [m for m in self.required_modalities if m not in self.modalities]
        if unknown:
            raise _NotFoundError(
                f"required_modalities names unknown modalities {unknown}; available: {list(self.modalities)}"
            )
        sources = {f"modality {name!r}": modality.ids for name, modality in self.modalities.items()}
        if target is not None:
            sources["target"] = target.ids
        _check_leading_zeros(sources)
        self._modality_ids = {name: set(modality.ids) for name, modality in self.modalities.items()}
        self._target_ids = set(target.ids) if target is not None else None
        candidates = set().union(*self._modality_ids.values(), self._target_ids or set())
        self._excluded = {pid: reason for pid in candidates if (reason := self._exclusion_reason(pid))}
        self._set_ids(sorted(candidates - self._excluded.keys()))

    def _exclusion_reason(self, pid: str) -> str | None:
        if self._target_ids is not None and pid not in self._target_ids:
            return "no target"
        for name in self.required_modalities:
            if pid not in self._modality_ids[name]:
                return f"missing required modality {name!r}"
        if not any(pid in ids for ids in self._modality_ids.values()):
            return "no modality"
        return None

    def _set_ids(self, ids: list[str]) -> None:
        self.ids = ids
        self._id_set = set(ids)
        self.present = pd.DataFrame(
            {name: [pid in self._modality_ids[name] for pid in ids] for name in self.modalities},
            index=pd.Index(ids, name="id"),
            dtype=bool,
        )

    def subset(self, ids: Iterable[str]) -> MultimodalDataset:
        """Keep exactly these ids. Any id this dataset does not include raises, with the reason it was excluded."""
        if isinstance(ids, pd.DataFrame):
            raise TypeError("subset takes ids (a list, Index or Series of str), not a DataFrame")
        requested = _as_ids(ids, "subset")
        missing = [pid for pid in requested if pid not in self._id_set]
        if missing:
            reasons = Counter(self._excluded.get(pid, "not in any modality or target") for pid in missing)
            raise _NotFoundError(
                f"subset: {len(missing)} ids are not in this dataset {dict(reasons)}, e.g. {missing[:5]}"
            )
        kept = set(requested)
        new = copy.copy(self)
        new._excluded = {**self._excluded, **{pid: "outside subset" for pid in self.ids if pid not in kept}}
        new._set_ids(sorted(kept))
        return new

    def summary(self) -> pd.DataFrame:
        counts = {f"available: {name}": len(ids) for name, ids in self._modality_ids.items()}
        if self._target_ids is not None:
            counts["available: target"] = len(self._target_ids)
        counts["included"] = len(self.ids)
        counts.update({f"included with {name}": int(self.present[name].sum()) for name in self.modalities})
        counts.update({f"excluded: {reason}": n for reason, n in sorted(Counter(self._excluded.values()).items())})
        if self.target is not None:
            counts.update(self.target.counts(self.ids))
        return pd.DataFrame({"count": counts})

    def present_ids(self, modality: str) -> list[str]:
        return [pid for pid in self.ids if pid in self._modality_ids[modality]]

    def with_transforms(self, fitted: Mapping[str, TransformerMixin]) -> MultimodalDataset:
        """The same dataset with fitted transforms applied to the named modalities."""
        new = copy.copy(self)
        new.modalities = dict(self.modalities)
        for name, transform in fitted.items():
            source = self.modalities[name]
            if not hasattr(source, "with_transform"):
                raise TypeError(f"modality {name!r} ({type(source).__name__}) does not accept transforms")
            new.modalities[name] = source.with_transform(transform, self.present_ids(name))
        return new

    def check_inputs(self) -> None:
        """Fail before batching if a table modality still has non-numeric or missing values."""
        for name, source in self.modalities.items():
            if hasattr(source, "check"):
                source.check(self.present_ids(name))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        pid = self.ids[index]
        inputs = {name: source.load(pid) for name, source in self.modalities.items() if pid in self._modality_ids[name]}
        return {"id": pid, "inputs": inputs}

    def collate(self, samples: list[dict]) -> dict:
        """Batch: ids, per-modality presence masks, inputs of present patients only (batch order), target."""
        ids = [sample["id"] for sample in samples]
        batch = {
            "ids": ids,
            "present": {
                name: torch.tensor([name in sample["inputs"] for sample in samples], dtype=torch.bool)
                for name in self.modalities
            },
            "inputs": {
                name: source.collate([sample["inputs"][name] for sample in samples if name in sample["inputs"]])
                for name, source in self.modalities.items()
            },
        }
        if self.target is not None:
            batch["target"] = self.target.tensors(ids)
        return batch


def train_test_split(
    data: MultimodalDataset, test_size: float, stratify: bool, random_state: int | None
) -> tuple[MultimodalDataset, MultimodalDataset]:
    """Patient-level split; ``stratify=True`` stratifies on the target's strata (events or labels)."""
    if stratify and data.target is None:
        raise ValueError("stratify=True needs a dataset with a target")
    strata = data.target.strata(data.ids) if stratify and data.target is not None else None
    train_ids, test_ids = _sklearn_train_test_split(
        data.ids, test_size=test_size, stratify=strata, random_state=random_state, shuffle=True
    )
    return data.subset(train_ids), data.subset(test_ids)
