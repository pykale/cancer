"""The item type: one patient in, one batch out.

A :class:`CohortView` yields :class:`PatientSample`; :func:`collate_samples` turns a
list of them into a :class:`PatientBatch`. Two types because padding only exists
once things are batched.

``default_collate`` rejects dataclasses, so :func:`collate_samples` is mandatory --
no loss, since ragged slide bags need custom padding anyway. Lightning's
``apply_to_collection`` *is* dataclass-aware, so device transfer needs no help.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(slots=True)
class PatientSample:
    """One patient, every modality, ready for a model.

    Tensors only -- paths are resolved inside ``payload()``, in the DataLoader worker.

    Attributes:
        patient_id (str): Sample identifier, so a prediction traces back to a patient.
        modalities (dict[str, Tensor]): Features by modality, e.g. ``{"clinical": (d,),
            "wsi_primary": (n_tiles, d)}``.
        present (dict[str, Tensor]): 0-d bool per modality. An absent modality is still
            present in ``modalities``, zero-filled, to keep batches uniform; this is
            what tells a fusion layer to ignore those zeros.
        target (dict[str, Tensor]): Supervision values. Empty for a cohort with no target.
    """

    patient_id: str
    modalities: dict[str, Tensor]
    present: dict[str, Tensor]
    target: dict[str, Tensor] = field(default_factory=dict)


@dataclass(slots=True)
class PatientBatch:
    """A collated batch. Same fields as :class:`PatientSample`, batched.

    ``present`` and ``pad_mask`` are different axes, which is why neither is just
    called "mask": availability per modality, versus which tiles within a bag are real.

    Attributes:
        patient_id (list[str]): Identifiers, in batch order.
        modalities (dict[str, Tensor]): Leading dimension ``B``, ragged modalities
            zero-padded to the batch maximum.
        present (dict[str, Tensor]): ``(B,)`` bool per modality.
        pad_mask (dict[str, Tensor]): ``(B, n_max)`` bool, only for modalities that
            needed padding -- empty for fixed-width data such as a clinical table.
        target (dict[str, Tensor]): ``(B,)`` per key.
    """

    patient_id: list[str]
    modalities: dict[str, Tensor]
    present: dict[str, Tensor]
    pad_mask: dict[str, Tensor] = field(default_factory=dict)
    target: dict[str, Tensor] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.patient_id)


def _stack_or_pad(tensors: list[Tensor], name: str) -> tuple[Tensor, Tensor | None]:
    """Stack same-shaped tensors, or zero-pad along axis 0 when they are ragged.

    Returns the stacked tensor and a ``(B, n_max)`` mask of real entries, or ``None``
    when nothing needed padding.

    Raises:
        ValueError: If they differ in any axis but the first -- a feature-width
            mismatch, not a ragged bag, and padding it would hide the cause.
    """
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) == 1:
        return torch.stack(tensors), None

    trailing = {tuple(t.shape[1:]) for t in tensors}
    if len(trailing) != 1:
        raise ValueError(
            f"Modality '{name}' has samples differing in more than the first axis: "
            f"{sorted(shapes)}. Only the leading axis may be ragged -- that is a bag "
            f"of variable length. A differing trailing shape is a feature-width "
            f"mismatch and must be fixed upstream, not padded over."
        )

    longest = max(t.shape[0] for t in tensors)
    padded = tensors[0].new_zeros((len(tensors), longest, *tensors[0].shape[1:]))
    mask = torch.zeros((len(tensors), longest), dtype=torch.bool)
    for i, tensor in enumerate(tensors):
        padded[i, : tensor.shape[0]] = tensor
        mask[i, : tensor.shape[0]] = True
    return padded, mask


def _require_same_keys(samples: list[PatientSample], attribute: str) -> list[str]:
    """Return the shared keys of ``attribute`` across samples, or raise.

    Disagreement means a cohort built them inconsistently; collating anyway would
    silently drop a modality for the whole batch.
    """
    first = list(getattr(samples[0], attribute))
    expected = set(first)
    for sample in samples[1:]:
        found = set(getattr(sample, attribute))
        if found != expected:
            raise ValueError(
                f"Samples disagree about {attribute}: {sample.patient_id!r} has "
                f"{sorted(found)}, {samples[0].patient_id!r} has {sorted(expected)}. "
                f"Every sample from one cohort must carry the same keys; an absent "
                f"modality is zero-filled with present=False, not omitted."
            )
    return first


def collate_samples(samples: list[PatientSample]) -> PatientBatch:
    """Collate samples into a :class:`PatientBatch`, padding ragged modalities.

    Pass as ``collate_fn`` to a ``DataLoader``; ``CohortDataModule`` does so by default.

    Raises:
        ValueError: If ``samples`` is empty, if they disagree about which modalities
            or target keys they carry, or if a modality is ragged beyond its first axis.
    """
    if not samples:
        raise ValueError("Cannot collate an empty list of samples.")

    modality_names = _require_same_keys(samples, "modalities")
    target_names = _require_same_keys(samples, "target")

    modalities: dict[str, Tensor] = {}
    pad_mask: dict[str, Tensor] = {}
    for name in modality_names:
        stacked, mask = _stack_or_pad([s.modalities[name] for s in samples], name)
        modalities[name] = stacked
        if mask is not None:
            pad_mask[name] = mask

    return PatientBatch(
        patient_id=[s.patient_id for s in samples],
        modalities=modalities,
        present={name: torch.stack([s.present[name] for s in samples]) for name in modality_names},
        pad_mask=pad_mask,
        target={key: torch.stack([s.target[key] for s in samples]) for key in target_names},
    )
