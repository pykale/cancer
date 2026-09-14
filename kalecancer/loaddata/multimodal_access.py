"""Generic access to any combination of per-patient modalities.

A cohort is a set of patients and, for each, some number of *sources* that can be
asked for that patient's data. What those sources are is the caller's business:

    >>> dataset = MultimodalDataset(
    ...     identifiers,
    ...     sources={
    ...         "clinical": VectorSource(clinical_vectors),
    ...         "blood": VectorSource(blood_vectors),
    ...         "primary": FeatureBagSource(primary_slides, feature_dim=1024),
    ...         "lymph_node": FeatureBagSource(lymph_slides, feature_dim=1024),
    ...     },
    ...     target=target,
    ... )

Nothing about that is specific to two modalities, or to imaging plus tabular. Two
tabular sources, two imaging sources, four of each, or a single one, are the same
call with a different dictionary, because the sources are named rather than
positional and each is asked for one patient at a time. Adding a modality is adding
a dictionary entry; adding a *kind* of modality is implementing
:class:`ModalitySource`.

**Missing data is expected, not exceptional.** A source returns ``None`` for a
patient it has nothing for, and the dataset substitutes a zero placeholder of the
right shape while recording the absence in ``present``. The placeholder is never
evidence: fusion reads ``present`` and drops the modality for that patient. That is
what lets one cohort hold patients with every modality and patients with one.

The record types every access API in this package produces live here too --
:class:`PatientSample`, :class:`PatientBatch`, and the two ways of collating between
them -- because assembling a patient is what they exist for. Two types rather than
one because padding only exists once things are batched.

The contracts a source or a target must satisfy live in
:mod:`~kalecancer.utils.protocols`, which is separate because it must import
nothing from ``kalecancer``.
"""

from __future__ import annotations

import gc
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from kalecancer.loaddata.wsi_access import read_feature_bag

#: One modality's value in a batch: a stacked tensor, or a list of per-patient bags
#: that were left ragged rather than padded. Bag-pooling embedders consume the list
#: form directly, which avoids padding to the largest bag in the batch.
ModalityValue: TypeAlias = Tensor | list[Tensor]


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
        metadata (dict[str, Any]): Provenance that is *not* model input -- patch
            coordinates and source slide ids, say. Carried so a prediction or an
            attention weight can be traced back to where it came from, and ignored by
            every model.
    """

    patient_id: str
    modalities: dict[str, Tensor]
    present: dict[str, Tensor]
    target: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PatientBatch:
    """A collated batch. Same fields as :class:`PatientSample`, batched.

    ``present`` and ``pad_mask`` are different axes, which is why neither is just
    called "mask": availability per modality, versus which tiles within a bag are real.

    Attributes:
        patient_id (list[str]): Identifiers, in batch order.
        modalities (dict[str, ModalityValue]): Leading dimension ``B``. A ragged
            modality is either zero-padded to the batch maximum, or left as a list of
            per-patient bags for an embedder that pools them itself.
        present (dict[str, Tensor]): ``(B,)`` bool per modality.
        pad_mask (dict[str, Tensor]): ``(B, n_max)`` bool, only for modalities that
            needed padding -- empty for fixed-width data such as a clinical table.
        target (dict[str, Tensor]): ``(B,)`` per key.
        metadata (dict[str, list]): Per-patient provenance in batch order, one list
            per key. Never model input; see :class:`PatientSample`.
    """

    patient_id: list[str]
    modalities: dict[str, ModalityValue]
    present: dict[str, Tensor]
    pad_mask: dict[str, Tensor] = field(default_factory=dict)
    target: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, list] = field(default_factory=dict)

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

    # Typed as the batch field is: this collate always pads, but the batch also
    # accepts the ragged list form that bag-pooling embedders build directly.
    modalities: dict[str, ModalityValue] = {}
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


def collate_ragged(samples: list[PatientSample]) -> PatientBatch:
    """Collate samples, leaving ragged modalities as a list of per-patient tensors.

    The alternative to :func:`collate_samples`, which pads. Padding to the largest
    bag in the batch wastes most of the tensor when patch counts vary by orders of
    magnitude, and a bag-pooling embedder such as
    :class:`~kalecancer.model.embed.BagEncoder` pools each bag independently anyway,
    so it never needs them stacked.

    Whether a modality stays a list is decided by its *rank*, not by whether this
    batch's bags happen to be the same length: a per-patient value with more than one
    axis is a bag, a flat vector is a fixed-width feature. Deciding it by shape would
    make a bag arrive as a list or as a stacked tensor depending on the data, and an
    embedder cannot be written against a contract that changes underneath it.

    Raises:
        ValueError: If ``samples`` is empty, or they disagree about which modalities
            or target keys they carry.
    """
    if not samples:
        raise ValueError("Cannot collate an empty list of samples.")

    modality_names = _require_same_keys(samples, "modalities")
    target_names = _require_same_keys(samples, "target")

    modalities: dict[str, ModalityValue] = {}
    for name in modality_names:
        values = [sample.modalities[name] for sample in samples]
        modalities[name] = values if values[0].dim() > 1 else torch.stack(values)

    return PatientBatch(
        patient_id=[sample.patient_id for sample in samples],
        modalities=modalities,
        present={name: torch.stack([s.present[name] for s in samples]) for name in modality_names},
        target={key: torch.stack([s.target[key] for s in samples]) for key in target_names},
        metadata={key: [s.metadata[key] for s in samples] for key in samples[0].metadata},
    )


# --------------------------------------------------------------------------- #
# DataLoader lifecycle
# --------------------------------------------------------------------------- #


def release_workers(*loaders: DataLoader) -> None:
    """Shut down each loader's worker processes now, rather than at interpreter exit.

    ``persistent_workers=True`` keeps a pool alive for its loader's lifetime, which
    is what makes epoch-heavy training fast -- on the whole-slide example it is the
    difference between 2m16s and 5m33s, because respawning workers that reopen HDF5
    files dominates otherwise. The cost is that the pool outlives whatever created
    it, so a run that builds several loaders per fold leaves several pools for the
    garbage collector.

    Collected together at interpreter shutdown, each pool's queue-feeder thread
    races the connection handles it is still writing to; on Windows that surfaces as
    ``ValueError: semaphore or lock released too many times``. Dropping the iterator
    here runs exactly the same finaliser, but one pool at a time and while the
    interpreter is still healthy.

    Loaders without worker processes are unaffected, so this is safe to call on any.
    """
    for loader in loaders:
        # The iterator owns the pool; PyTorch exposes no public shutdown, but
        # releasing the only reference to it runs its finaliser deterministically.
        loader._iterator = None
    gc.collect()


# --------------------------------------------------------------------------- #
# Modality sources
# --------------------------------------------------------------------------- #


class ModalitySource(ABC):
    """One modality's data, answered per patient.

    Implement this to add a kind of modality. The dataset asks only two things of a
    source -- what a patient's value is, and what shape to substitute when there
    isn't one -- so a source can read a file, index a dictionary, or compute
    something, without the dataset knowing which.
    """

    @abstractmethod
    def get(self, identifier: str, index: int) -> Tensor | None:
        """This patient's value, or ``None`` if the source has nothing for them.

        Args:
            identifier: Which patient.
            index: Position in the dataset, for a source whose reads are stochastic
                and want a reproducible per-sample seed.
        """

    @abstractmethod
    def placeholder(self) -> Tensor:
        """A zero-filled stand-in of the right shape, for a patient with no value.

        Needed so a batch stays rectangular. It is never read as evidence: the
        dataset marks the modality absent in ``present``, and fusion drops it.
        """

    def provenance(self, identifier: str, index: int) -> dict[str, Any]:
        """Where this patient's value came from, for a batch's ``metadata``.

        Never model input. Empty unless a source has something to trace, which is why
        this is not abstract.
        """
        return {}


class VectorSource(ModalitySource):
    """Precomputed per-patient vectors, held in memory.

    The general case for anything already reduced to one array per patient: an
    encoded clinical row, a radiomics vector, a frozen embedding.

    Args:
        values: One tensor per patient, keyed by identifier. Patients absent from the
            mapping are treated as missing.
        width: Length of the placeholder. Inferred from the first value when omitted,
            which requires at least one.

    Raises:
        ValueError: If ``width`` is omitted and ``values`` is empty, leaving no shape
            to place a missing patient against.
    """

    def __init__(self, values: Mapping[str, Tensor], width: int | None = None) -> None:
        if width is None:
            if not values:
                raise ValueError("width must be given when values is empty; there is no shape to infer")
            width = int(next(iter(values.values())).shape[-1])
        self.values = values
        self.width = width

    def get(self, identifier: str, index: int) -> Tensor | None:
        return self.values.get(identifier)

    def placeholder(self) -> Tensor:
        return torch.zeros(self.width)


class FeatureBagSource(ModalitySource):
    """A patient's patch features, pooled from however many slides they have.

    One entry per patient, so a patient with three slides of one region is a single
    bag; a *second region* is a second source, which is what makes imaging + imaging
    an ordinary two-source cohort rather than a special case.

    Args:
        paths: Feature files per patient, keyed by identifier.
        feature_dim: Width of a patch embedding, for the placeholder.
        max_patches: Cap on patches per bag, bounding memory during training. Leave
            ``None`` for evaluation and interpretation so attention covers whole
            slides.
        seed: Base seed for subsampling, combined with the sample index so a given
            patient draws the same patches on every epoch of a given run.
        with_coordinates: Carry patch coordinates and their source slide as
            provenance. Needed to place attention back on a slide, and skipped when
            nothing will read it.
    """

    def __init__(
        self,
        paths: Mapping[str, Sequence[str | Path]],
        feature_dim: int,
        max_patches: int | None = None,
        seed: int = 0,
        with_coordinates: bool = False,
    ) -> None:
        self.paths = paths
        self.feature_dim = feature_dim
        self.max_patches = max_patches
        self.seed = seed
        self.with_coordinates = with_coordinates

    def _read(self, identifier: str, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Concatenated features, coordinates and slide index for one patient."""
        files = self.paths.get(identifier)
        if not files:
            return None

        parts = [read_feature_bag(path, expected_dim=self.feature_dim) for path in files]
        features = np.concatenate([features for features, _ in parts])
        coords = np.concatenate([coords for _, coords in parts])
        slide_index = np.concatenate(
            [np.full(len(f), position, dtype=np.int64) for position, (f, _) in enumerate(parts)]
        )

        if self.max_patches is not None and len(features) > self.max_patches:
            generator = np.random.default_rng(self.seed + index)
            keep = np.sort(generator.choice(len(features), size=self.max_patches, replace=False))
            features, coords, slide_index = features[keep], coords[keep], slide_index[keep]
        return features, coords, slide_index

    def get(self, identifier: str, index: int) -> Tensor | None:
        read = self._read(identifier, index)
        return None if read is None else torch.from_numpy(read[0]).float()

    def placeholder(self) -> Tensor:
        # One zero patch: a bag pooler needs something to pool, and ``present`` says
        # not to believe the result.
        return torch.zeros(1, self.feature_dim)

    def provenance(self, identifier: str, index: int) -> dict[str, Any]:
        if not self.with_coordinates:
            return {}
        read = self._read(identifier, index)
        if read is None:
            return {}
        _, coords, slide_index = read
        return {
            "coords": torch.from_numpy(coords),
            "slide_index": torch.from_numpy(slide_index),
            "slide_ids": tuple(Path(path).stem for path in self.paths[identifier]),
        }


class ColumnTarget:
    """Supervision read from named columns of a table, keyed by identifier.

    Satisfies the :class:`~kalecancer.loaddata.multimodal_access.Target` contract for any
    endpoint whose values are already columns: ``{"time": "duration", "event":
    "event"}`` for a survival model, ``{"label": "recurrence"}`` for a binary one.
    Which columns mean what is the caller's declaration, so the same class serves an
    endpoint this package has never heard of.

    Renaming is explicit because the batch key and the column name are different
    vocabularies: a cohort table calls it ``duration``, a batch target calls it
    ``time``, and writing the map out is what keeps that from being a silent guess.

    Args:
        frame: Table carrying ``id_column`` and every named column.
        columns: Batch target key mapped to the column supplying it.
        id_column: Column holding the identifier.

    Raises:
        KeyError: If a named column is absent, or an identifier is duplicated.
    """

    def __init__(self, frame: pd.DataFrame, columns: Mapping[str, str], id_column: str = "patient_id") -> None:
        missing = [column for column in (id_column, *columns.values()) if column not in frame.columns]
        if missing:
            raise KeyError(f"table has no column(s) {missing}; available: {list(frame.columns)}")

        identifiers = frame[id_column].astype(str)
        if identifiers.duplicated().any():
            repeated = sorted(identifiers[identifiers.duplicated()].unique())
            raise KeyError(f"{id_column} must be unique, but {repeated[:5]} repeat")

        self.columns = dict(columns)
        self._values = {
            key: torch.as_tensor(frame[column].to_numpy(dtype=float), dtype=torch.float32)
            for key, column in self.columns.items()
        }
        self._row_of = {identifier: row for row, identifier in enumerate(identifiers)}

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(self.columns.values())

    def bind(self, identifiers: Sequence[str], values: Mapping[str, Any]) -> None:
        """No-op: the values were bound from the table at construction."""

    def for_(self, identifier: str) -> dict[str, Tensor]:
        row = self._row_of[identifier]
        return {key: values[row] for key, values in self._values.items()}

    def values_for(self, identifiers: Sequence[str]) -> dict[str, Tensor]:
        rows = torch.tensor([self._row_of[identifier] for identifier in identifiers])
        return {key: values[rows] for key, values in self._values.items()}


class MultimodalDataset(Dataset[PatientSample]):
    """A cohort of patients, each assembled from any number of modality sources.

    Yields the :class:`~kalecancer.loaddata.multimodal_access.PatientSample` every trainer here
    reads, so a one-source cohort and a four-source cohort are the same object with
    different arguments.

    Args:
        identifiers: Patients in this split, in the order they will be indexed.
        sources: One :class:`ModalitySource` per modality, keyed by the name the
            modality carries into the batch and into the model's embedders.
        target: Supervision, keyed by identifier. ``None`` for an unlabelled cohort,
            such as one being embedded for inspection.
        metadata_from: Which sources contribute provenance. Defaults to every source
            that offers any. Naming a subset avoids re-reading a large bag purely to
            record where it came from.

    Raises:
        ValueError: If ``sources`` is empty, or ``metadata_from`` names a source that
            was not given.
    """

    def __init__(
        self,
        identifiers: Sequence[str],
        sources: Mapping[str, ModalitySource],
        target: Target | None = None,
        metadata_from: Sequence[str] | None = None,
    ) -> None:
        if not sources:
            raise ValueError("a cohort needs at least one modality source")
        if metadata_from is not None:
            unknown = set(metadata_from) - set(sources)
            if unknown:
                raise ValueError(f"metadata_from names {sorted(unknown)}, which are not sources: {sorted(sources)}")

        self.identifiers = list(identifiers)
        self.sources = dict(sources)
        self.target = target
        self.metadata_from = tuple(sources if metadata_from is None else metadata_from)

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, index: int) -> PatientSample:
        identifier = self.identifiers[index]

        modalities: dict[str, Tensor] = {}
        present: dict[str, Tensor] = {}
        for name, source in self.sources.items():
            value = source.get(identifier, index)
            # A placeholder keeps the batch rectangular; ``present`` is what a model
            # reads, so the zeros are never treated as evidence.
            modalities[name] = value if value is not None else source.placeholder()
            present[name] = torch.tensor(value is not None)

        metadata: dict[str, Any] = {}
        for name in self.metadata_from:
            metadata.update(self.sources[name].provenance(identifier, index))

        return PatientSample(
            patient_id=identifier,
            modalities=modalities,
            present=present,
            target={} if self.target is None else self.target.for_(identifier),
            metadata=metadata,
        )

    def __repr__(self) -> str:
        modalities = ", ".join(f"{name}={type(source).__name__}" for name, source in self.sources.items())
        return f"MultimodalDataset({len(self)} patients | {modalities})"


# --------------------------------------------------------------------------- #
# Right-censored supervision
# --------------------------------------------------------------------------- #


def _is_missing(value: Any) -> bool:
    """Whether one value is absent: ``None`` or float ``NaN``, whatever the column dtype."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _as_float(value: Any) -> float:
    """Coerce to float, returning NaN rather than raising, so the caller can name
    every offending sample at once."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


class SurvivalTarget:
    """Right-censored survival target: a time and an event indicator.

    Columns are named rather than positional, and the value meaning "the event
    happened" is declared. Both stop the same silent bug: a positional ``[time,
    status]`` pair or an assumed ``1 == event`` coding runs cleanly when reversed, and
    predicts survival inverted.

    Args:
        time (str): Column holding follow-up time.
        event (str): Column holding the event indicator.
        event_value (Any, optional): Value(s) meaning the event was observed. Any other
            *recorded* value is censored, which is what makes competing risks work:
            with ``0/1/2`` and ``event_value=1``, other-cause deaths are censored --
            the cause-specific convention. A *missing* value raises. Defaults to ``1``.
        unit (str | None, optional): What one unit of ``time`` is called. **A display
            label only** -- nothing is converted, so it cannot make a result wrong.
            ``None`` reports follow-up in "time units", right whatever the column holds.

    Raises:
        ValueError: On :meth:`bind`, if times or statuses are invalid or absent, or the
            event rate is 0% or 100% -- a mis-specified target, not an unusual cohort.
    """

    #: Used when no ``unit`` was given. Reporting the median in the column's own
    #: scale is correct whatever that scale is, so there is nothing left to assume.
    GENERIC_UNIT = "time units"

    def __init__(self, time: str, event: str, event_value: Any = 1, unit: str | None = None):
        self.time = time
        self.event = event
        self.event_values = list(event_value) if isinstance(event_value, list | tuple | set) else [event_value]
        self.unit = None if unit is None else str(unit).strip()
        self._row_of: dict[str, int] = {}
        self._times: Tensor = torch.empty(0)
        self._events: Tensor = torch.empty(0)

    @property
    def required_columns(self) -> tuple[str, ...]:
        """The time and event columns, which a cohort must supply to :meth:`bind`."""
        return (self.time, self.event)

    def bind(self, identifiers: Sequence[str], values: Mapping[str, ArrayLike]) -> None:
        """Extract and validate times and events, keyed by identifier.

        Raises:
            ValueError: On non-numeric or negative times, a missing event status, or a
                degenerate event rate.
        """
        ids = list(identifiers)
        raw_times = np.asarray(values[self.time], dtype=object).ravel()
        raw_events = np.asarray(values[self.event], dtype=object).ravel()

        times = np.array([_as_float(v) for v in raw_times], dtype=float)
        bad = np.isnan(times)
        if bad.any():
            examples = [ids[i] for i in np.flatnonzero(bad)[:5]]
            raise ValueError(
                f"Column '{self.time}' has {int(bad.sum())} missing or non-numeric "
                f"value(s), e.g. for {examples}. Survival time must be known for every "
                f"sample; drop or impute these before constructing the cohort."
            )
        if (times < 0).any():
            raise ValueError(f"Column '{self.time}' contains negative values.")

        unknown = np.array([_is_missing(v) for v in raw_events])
        if unknown.any():
            examples = [ids[i] for i in np.flatnonzero(unknown)[:5]]
            raise ValueError(
                f"Column '{self.event}' has {int(unknown.sum())} missing value(s), e.g. for "
                f"{examples}. An unknown outcome is not a censored one: censoring asserts "
                f"the sample was event-free throughout its recorded time, which an absent "
                f"status does not establish. Treating these as censored would drop events "
                f"from the numerator while keeping their follow-up in the denominator, "
                f"biasing every estimate towards the null. Decide explicitly before "
                f"constructing the cohort -- drop these rows, or recode them as censored "
                f"if that is what a blank means in your data dictionary."
            )

        events = np.array([v in self.event_values for v in raw_events], dtype=float)
        rate = float(events.mean()) if events.size else 0.0
        if rate in (0.0, 1.0):
            observed = sorted({str(v) for v in raw_events})[:10]
            raise ValueError(
                f"event_value={self.event_values} matches {rate:.0%} of rows in column "
                f"'{self.event}', which cannot be right. Observed values: {observed}"
            )

        # Tensors rather than dicts of floats: for_() then returns a view instead of
        # allocating two tensors per sample per epoch.
        self._row_of = {identifier: i for i, identifier in enumerate(ids)}
        self._times = torch.tensor(times, dtype=torch.float32)
        self._events = torch.tensor(events, dtype=torch.float32)

    def for_(self, identifier: str) -> dict[str, Tensor]:
        """Return ``{"time": Tensor, "event": Tensor}`` for one sample."""
        row = self._row_of[identifier]
        return {"time": self._times[row], "event": self._events[row]}

    def values_for(self, identifiers: Sequence[str]) -> dict[str, Tensor]:
        """The batched sibling of :meth:`for_`: the same keys and dtypes, many samples.

        Lets a caller ask for supervision by name -- ``values_for(ids)["event"]`` --
        without knowing this is a survival target. Tensors rather than arrays, so a
        batch built from this is indistinguishable from one built by collating
        :meth:`for_`; :meth:`events_for` and :meth:`times_for` are the numpy doors,
        for scikit-learn.
        """
        rows = self._rows(identifiers)
        return {"time": self._times[rows], "event": self._events[rows]}

    def events_for(self, identifiers: Sequence[str]) -> np.ndarray:
        """Event indicators for ``identifiers``, as a float array."""
        return self._gather(self._events, identifiers)

    def times_for(self, identifiers: Sequence[str]) -> np.ndarray:
        """Follow-up times for ``identifiers``, as a float array."""
        return self._gather(self._times, identifiers)

    def stratify_labels(self, identifiers: Sequence[str]) -> np.ndarray:
        """Labels to stratify a split on: the event indicator.

        The optional half of the ``Target`` contract, used by ``Cohort.split``.
        """
        return self.events_for(identifiers)

    def _rows(self, identifiers: Sequence[str]) -> Tensor:
        """Row positions for ``identifiers``, as an index tensor."""
        return torch.tensor([self._row_of[i] for i in identifiers], dtype=torch.long)

    def _gather(self, source: Tensor, identifiers: Sequence[str]) -> np.ndarray:
        return source[self._rows(identifiers)].numpy().astype(float)

    def summarise(self, identifiers: Sequence[str]) -> str:
        """One-line description of events and follow-up across ``identifiers``."""
        events = self.events_for(identifiers)
        times = self.times_for(identifiers)
        n_events = int(events.sum())
        parts = [f"{n_events} events ({n_events / max(len(events), 1):.1%})"]

        # Median follow-up among the censored: the simple approximation, not reverse KM.
        censored = times[events == 0]
        if censored.size:
            parts.append(f"median follow-up {np.median(censored):.1f} {self.unit or self.GENERIC_UNIT}")
        return " | ".join(parts)

    def __repr__(self) -> str:
        parts = [f"time={self.time!r}", f"event={self.event!r}", f"event_value={self.event_values}"]
        if len(self._row_of):
            parts.append(self.summarise(list(self._row_of)))
        return f"{type(self).__name__}({' | '.join(parts)})"


# --------------------------------------------------------------------------- #
# The contracts
# --------------------------------------------------------------------------- #


class Target(Protocol):
    """What a cohort requires of a supervision target.

    Bound once, at index-loading time, and keyed by identifier rather than row
    position, so a cohort can be subset and recombined without realignment.

    Anything derived from a *training fold* is not a target and must not live here.
    Discrete-time bin edges are the case to watch: they are quantiles of one fold's
    event times, so they belong in ``prepdata`` under the same fold-local discipline
    as a scaler.
    """

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Columns this target needs, so a cohort can check them before binding."""
        ...

    def bind(self, identifiers: Sequence[str], values: Mapping[str, ArrayLike]) -> None:
        """Extract and validate supervision values, keyed by identifier.

        Takes arrays rather than a ``DataFrame`` to keep pandas out of the contract.
        Validate here and raise; do not defer a mis-specified target to training.

        Args:
            identifiers (Sequence[str]): Sample identifiers, in row order.
            values (Mapping[str, ArrayLike]): One array per entry in
                :attr:`required_columns`, aligned with ``identifiers``.
        """
        ...

    def for_(self, identifier: str) -> dict[str, Tensor]:
        """This sample's values for ``PatientSample.target``.

        Named keys, never a positional pack: a ``tensor([time, event])`` built
        backwards runs perfectly and predicts survival inverted.
        """
        ...

    def values_for(self, identifiers: Sequence[str]) -> dict[str, Tensor]:
        """The same values for many samples: ``for_`` with a leading batch dimension.

        Every consumer that is not a ``DataLoader`` wants this form -- a whole split's
        labels for an embedder to condition on, a full-batch loss, a metric over the
        held-out set. Looping :meth:`for_` costs a Python call per sample; a target
        that stores its values as arrays answers this in one indexing operation.

        Must return the same keys and the same dtypes as :meth:`for_`, and agree with
        it value for value -- the two are read by different paths over the same data.
        """
        ...


class Preprocessor(Protocol):
    """Fitted state belonging to exactly one cross-validation fold.

    Deliberately thin -- a cohort defines and consumes its own preprocessor type.
    What every preprocessor owes the rest of the system is provenance, so a fold's
    held-out rows can be proven untouched.
    """

    @property
    def fitted_on(self) -> frozenset[str]:
        """Identifiers this was fitted on.

        Identifiers rather than positions, so the check survives subsetting and
        multimodal composition.
        """
        ...

    @property
    def feature_names(self) -> dict[str, list[str]]:
        """Post-encoding feature names, keyed by modality.

        Keyed because a composite preprocessor serves several modalities at once.
        A modality whose features have no meaningful names returns an empty list.
        """
        ...

    def describe(self) -> str:
        """Human-readable summary of what this fold applied."""
        ...


def check_target(target: object) -> None:
    """Verify an object satisfies :class:`Target`, or raise naming what is missing.

    Raises:
        TypeError: If any part of the contract is absent.
    """
    missing = [name for name in ("required_columns", "bind", "for_", "values_for") if not hasattr(target, name)]
    if missing:
        raise TypeError(
            f"{type(target).__name__} is not a valid Target: missing {missing}. "
            f"A target must declare required_columns, and implement "
            f"bind(identifiers, values), for_(identifier) and values_for(identifiers). See "
            f"kalecancer.loaddata.multimodal_access.Target."
        )
