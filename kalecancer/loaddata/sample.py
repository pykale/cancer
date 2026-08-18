"""The item type: one patient in, one batch out.

A :class:`PatientSample` is what a :class:`~kalecancer.loaddata.view.CohortView`
yields, and :class:`PatientBatch` is what :func:`collate_samples` turns a list of
them into. Keeping the two types distinct is deliberate -- a batch of samples is
not itself a sample, and the fields that only exist once things are batched
(padding) have nowhere sensible to live on a single patient.

Two measured facts shape this module:

* ``torch.utils.data.default_collate`` **rejects dataclasses** outright, so
  :func:`collate_samples` is mandatory rather than an optimisation. That cost is
  already unavoidable -- variable-length slide bags cannot be stacked without
  padding -- and one named function owning it beats scattering padding logic.
* Lightning's ``apply_to_collection`` *is* dataclass-aware, so
  ``transfer_batch_to_device`` moves a :class:`PatientBatch` to the GPU with no
  extra work from us.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(slots=True)
class PatientSample:
    """One patient, every modality, ready for a model.

    No paths and no lazy handles: by the time this object exists, payload loading
    has happened. Identifiers and file paths live in a cohort's index and are
    resolved inside ``payload()``, which runs in the DataLoader worker where the
    parallelism already is.

    Attributes:
        patient_id (str): Sample identifier. Carried through so a prediction can
            always be traced back to a patient.
        modalities (dict[str, Tensor]): Feature tensors by modality name, e.g.
            ``{"clinical": (d,), "wsi_primary": (n_tiles, d)}``.
        present (dict[str, Tensor]): 0-d bool per modality -- whether this patient
            actually has it. A modality that is absent still appears in
            ``modalities``, zero-filled, so that batches keep a uniform structure;
            ``present`` is what tells a fusion layer to ignore those zeros.
        target (dict[str, Tensor]): Supervision values, e.g. ``{"time": ...,
            "event": ...}``. Empty for a cohort with no target.
    """

    patient_id: str
    modalities: dict[str, Tensor]
    present: dict[str, Tensor]
    target: dict[str, Tensor] = field(default_factory=dict)


@dataclass(slots=True)
class PatientBatch:
    """A collated batch. Same fields as :class:`PatientSample`, batched.

    Attributes:
        patient_id (list[str]): Identifiers, in batch order.
        modalities (dict[str, Tensor]): Leading dimension ``B``. Ragged modalities
            are zero-padded to the batch maximum.
        present (dict[str, Tensor]): ``(B,)`` bool per modality -- is this modality
            available for each patient at all.
        pad_mask (dict[str, Tensor]): ``(B, n_max)`` bool per *padded* modality --
            which entries along the ragged axis are real. Only modalities that
            actually needed padding appear here, so it stays empty for fixed-width
            data such as a clinical table.
        target (dict[str, Tensor]): ``(B,)`` per key.

    Note:
        ``present`` and ``pad_mask`` are different axes and are deliberately not
        both called "mask". ``present`` is per-modality availability; ``pad_mask``
        is per-tile within one bag and is what an attention-based aggregator needs.
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

    Args:
        tensors (list[Tensor]): One per sample.
        name (str): Modality name, for error messages.

    Returns:
        tuple[Tensor, Tensor | None]: The stacked tensor, and a ``(B, n_max)`` bool
        mask marking real entries -- or ``None`` when no padding was needed.

    Raises:
        ValueError: If the tensors differ in any axis other than the first. That
            is a genuine mismatch (a feature width that changed between folds, say)
            rather than a ragged bag, and padding it would hide the cause.
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

    Samples disagreeing about which modalities they carry means a cohort built
    them inconsistently. Collating anyway would silently drop a modality for the
    whole batch.
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

    Pass this as ``collate_fn`` to a ``DataLoader``;
    :class:`~kalecancer.loaddata.module.CohortDataModule` does so by default.

    Args:
        samples (list[PatientSample]): One batch worth of samples.

    Returns:
        PatientBatch: Batched tensors, plus a ``pad_mask`` entry for each modality
        that needed padding.

    Raises:
        ValueError: If ``samples`` is empty, if samples disagree about which
            modalities or target keys they carry, or if a modality is ragged in
            more than its leading axis.
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
