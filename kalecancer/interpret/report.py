"""A per-patient interpretability bundle: what the model predicted, and why.

**The critical distinction.** :mod:`~kalecancer.interpret.modality` and
:mod:`~kalecancer.interpret.attribution` are *cohort-level*: they ask whether a
modality or a feature helps the model rank patients, and answer with a change in a
metric (C-index, say) computed over a whole dataloader. This module asks a different
question -- did this modality or feature push *this one patient's* risk score up or
down -- and answers with a change in that patient's own risk, never a metric. A
cohort-level Shapley value and a per-patient one are different numbers answering
different questions; do not call :func:`~kalecancer.interpret.modality.modality_shapley`
or :func:`~kalecancer.interpret.attribution.feature_shapley` and present their cohort
numbers as what drove one patient's score. ``PatientReport.modality_contributions``
and ``PatientReport.feature_contributions`` below are risk deltas, computed with their
own Shapley-style enumeration/sampling over raw risk rather than over a metric.

**Why this module can be cheaper per patient than the cohort-level ones.** Every
embedder and fusion block in this package is row-independent: a patient's embedding
and prediction depend only on their own input, never on which other patients share
the batch (:class:`~kalecancer.model.embed.tabicl.TabICLEmbedder`'s own docstring
notes this explicitly for the rows being embedded; every fusion block here is a
per-row Linear/Gaussian combination with no batch-level statistic). A cohort-level
Shapley value needs the whole evaluation set in every coalition, because the *metric*
is a property of the cohort; a per-patient risk delta does not, because the quantity
being explained is a property of one row. So once the full cohort has supplied one
thing -- ``risk_percentile``, which is a property of the cohort by definition -- every
other computation here runs on just the requested patients, not the whole dataloader.

Reuses :mod:`~kalecancer.interpret.modality` and
:mod:`~kalecancer.interpret.attribution`'s conventions: ``model.eval()`` plus
``torch.no_grad()`` with the training mode restored in a ``finally``, the same
:class:`~kalecancer.pipeline.task.SurvivalTask` check, modality discovery from
``model.embedders``, and the cached-non-tabular-branch trick for feature
perturbations -- but not their return values, which answer the cohort-level question.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn

from kalecancer.evaluate.survival_metrics import breslow_baseline_hazard, predict_survival_function
from kalecancer.evaluate.survival_predictions import SplitPredictions
from kalecancer.interpret.attention import attention_records, bag_attention, top_k_patches
from kalecancer.loaddata.multimodal_access import ModalityValue, PatientBatch
from kalecancer.model.embed.tabicl import TabICLEmbedder
from kalecancer.model.predict.losses import as_event_mask
from kalecancer.pipeline.task import SurvivalTask
from kalecancer.utils.io import write_json


def _require_survival_task(model, func_name: str) -> None:
    if not isinstance(model.task, SurvivalTask):
        raise ValueError(
            f"{func_name} currently supports survival (time-to-event) models only: its survival curve "
            f"needs a Breslow baseline hazard, which assumes a SurvivalTask, but this model's task is "
            f"{type(model.task).__name__}."
        )


def _find_tabular_modality(model) -> str | None:
    """The one modality read by a frozen TabICL embedder, if there is exactly one.

    Unlike :mod:`~kalecancer.interpret.attribution`'s version, this never raises:
    feature contributions are one part of a bundle, not this function's sole job, so
    an absent or ambiguous tabular modality just means that part is skipped (recorded
    in ``metadata``), not that the whole report fails.
    """
    tabular = [name for name, embedder in model.embedders.items() if isinstance(embedder, TabICLEmbedder)]
    return tabular[0] if len(tabular) == 1 else None


def _find_bag_modality(model) -> str | None:
    """The one modality whose embedder records attention, if there is exactly one."""
    bag = [name for name, embedder in model.embedders.items() if hasattr(embedder, "last_attention")]
    return bag[0] if len(bag) == 1 else None


def _discover_feature_names(dataloader, modality: str) -> list[str] | None:
    """Post-encoding column names for ``modality``, or ``None`` if undiscoverable.

    See :func:`kalecancer.interpret.attribution._discover_feature_names` for where
    these come from; this version returns ``None`` instead of raising, for the same
    reason :func:`_find_tabular_modality` does.
    """
    dataset = getattr(dataloader, "dataset", None)
    all_names = getattr(dataset, "feature_names", None) if dataset is not None else None
    discovered = all_names.get(modality) if isinstance(all_names, dict) else None
    return list(discovered) if discovered else None


def _to_device(value: ModalityValue, device: torch.device) -> ModalityValue:
    """See :func:`kalecancer.interpret.attribution._to_device` -- duplicated for the
    same reason: calling an embedder directly bypasses the trainer's own device
    handling."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return [item.to(device) for item in value]


def _cat_modality(values: list[ModalityValue]) -> ModalityValue:
    if isinstance(values[0], torch.Tensor):
        return torch.cat(cast("list[torch.Tensor]", values), dim=0)
    return [item for value in values for item in cast("list[torch.Tensor]", value)]


def _materialise(dataloader) -> PatientBatch:
    """Concatenate every batch the loader yields into one -- see
    :func:`kalecancer.interpret.attribution._materialise`. Also keeps ``metadata``,
    which that function does not need but patch-level attention here does.
    """
    patient_ids: list[str] = []
    modality_rows: dict[str, list[ModalityValue]] = {}
    present_rows: dict[str, list[torch.Tensor]] = {}
    target_rows: dict[str, list[torch.Tensor]] = {}
    metadata_rows: dict[str, list] = {}

    for batch in dataloader:
        patient_ids.extend(batch.patient_id)
        for name, value in batch.modalities.items():
            modality_rows.setdefault(name, []).append(value)
        for name, value in batch.present.items():
            present_rows.setdefault(name, []).append(value)
        for key, value in batch.target.items():
            target_rows.setdefault(key, []).append(value)
        for key, value in batch.metadata.items():
            metadata_rows.setdefault(key, []).extend(value)

    return PatientBatch(
        patient_id=patient_ids,
        modalities={name: _cat_modality(values) for name, values in modality_rows.items()},
        present={name: torch.cat(values) for name, values in present_rows.items()},
        target={key: torch.cat(values) for key, values in target_rows.items()},
        metadata=metadata_rows,
    )


def _subset_batch(batch: PatientBatch, indices: list[int]) -> PatientBatch:
    """The rows at ``indices``, in that order.

    Safe to run every later computation on this smaller batch instead of the whole
    cohort: see the module docstring on row-independence.
    """
    idx = torch.tensor(indices, dtype=torch.long)

    def _select(value: ModalityValue) -> ModalityValue:
        if isinstance(value, torch.Tensor):
            return value[idx]
        return [value[i] for i in indices]

    return PatientBatch(
        patient_id=[batch.patient_id[i] for i in indices],
        modalities={name: _select(value) for name, value in batch.modalities.items()},
        present={name: value[idx] for name, value in batch.present.items()},
        target={key: value[idx] for key, value in batch.target.items()},
        metadata={key: [value[i] for i in indices] for key, value in batch.metadata.items()},
    )


class _CachedOutput(nn.Module):
    """See :class:`kalecancer.interpret.attribution._CachedOutput`."""

    def __init__(self, cached_output: torch.Tensor) -> None:
        super().__init__()
        self._cached_output = cached_output

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self._cached_output


def _cache_other_modalities(model, batch: PatientBatch, tabular_name: str) -> dict[str, nn.Module]:
    """See :func:`kalecancer.interpret.attribution._cache_other_modalities`."""
    device = model.device
    originals: dict[str, nn.Module] = {}
    for name, embedder in list(model.embedders.items()):
        if name == tabular_name:
            continue
        output = embedder(_to_device(batch.modalities[name], device))
        originals[name] = embedder
        model.embedders[name] = _CachedOutput(output)
    return originals


def _restore_modalities(model, originals: dict[str, nn.Module]) -> None:
    for name, embedder in originals.items():
        model.embedders[name] = embedder


def _risk_for(model, batch: PatientBatch) -> np.ndarray:
    output = model.predict(batch)
    return output.prediction.reshape(-1).detach().cpu().double().numpy()


def _score_tabular(model, batch: PatientBatch, tabular_name: str, tabular_x: torch.Tensor) -> np.ndarray:
    return _risk_for(model, replace(batch, modalities={**batch.modalities, tabular_name: tabular_x}))


def _percentile_rank(value: float, population: np.ndarray) -> float:
    """Percentile rank of ``value`` within ``population`` (0-100); a tie counts as
    half a step, the usual convention for a percentile rank."""
    below = float(np.sum(population < value))
    tied = float(np.sum(population == value))
    return 100.0 * (below + 0.5 * tied) / population.size


def _modality_contributions(model, batch: PatientBatch) -> dict[str, np.ndarray]:
    """Per-row Shapley value of each modality's contribution to RISK, not a metric.

    Exactly :func:`kalecancer.interpret.modality.modality_shapley`'s enumeration and
    weighting over ``2**k`` coalitions, duplicated here because the coalition
    "value" is the raw per-row risk vector rather than a cohort metric -- the
    distinction this module exists to keep straight. Exact, not sampled: with ``k``
    modalities (2-4 in practice) this is no more expensive than the cohort version.
    """
    modality_names = list(model.embedders)
    k = len(modality_names)
    n = len(batch.patient_id)

    cache: dict[frozenset[str], np.ndarray] = {}
    for size in range(k + 1):
        for subset in itertools.combinations(modality_names, size):
            coalition = frozenset(subset)
            if coalition not in cache:
                masked_present = {
                    name: (value if name in coalition else torch.zeros_like(value))
                    for name, value in batch.present.items()
                }
                cache[coalition] = _risk_for(model, replace(batch, present=masked_present))

    contributions = {}
    for name in modality_names:
        others = [m for m in modality_names if m != name]
        total = np.zeros(n)
        for size in range(len(others) + 1):
            for subset in itertools.combinations(others, size):
                without = frozenset(subset)
                with_name = without | {name}
                weight = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
                total = total + weight * (cache[with_name] - cache[without])
        contributions[name] = total
    return contributions


def _feature_contributions(
    model,
    background_pool: torch.Tensor,
    requested_batch: PatientBatch,
    tabular_name: str,
    feature_names: list[str],
    n_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Per-row Shapley-style risk delta for each raw clinical feature.

    Same permutation-sampling procedure as
    :func:`kalecancer.interpret.attribution.feature_shapley` -- a random feature
    ordering plus an independent per-column resample as the "absent" background,
    walked one reveal at a time -- except the per-sample contribution is a risk
    delta read off directly, never reduced through a metric, and the background to
    resample from is the *whole* cohort (``background_pool``, representative of the
    population) while the forward passes that matter run only over
    ``requested_batch`` (just the patients being reported on; see the module
    docstring on row-independence).
    """
    requested_x = requested_batch.modalities[tabular_name]
    if not isinstance(requested_x, torch.Tensor):
        raise ValueError(f"modality {tabular_name!r} must be a 2-D tensor for feature attribution")
    d = len(feature_names)
    m = requested_x.shape[0]
    n_pool = background_pool.shape[0]

    originals = _cache_other_modalities(model, requested_batch, tabular_name)
    try:
        rng = np.random.default_rng(seed)
        contributions = {name: np.zeros(m) for name in feature_names}

        for _ in range(n_samples):
            order = rng.permutation(d)
            current = requested_x.clone()
            for column in range(d):
                draw = torch.from_numpy(rng.integers(0, n_pool, size=m))
                current[:, column] = background_pool[draw, column]

            previous_risk = _score_tabular(model, requested_batch, tabular_name, current)
            for column in order:
                current = current.clone()
                current[:, column] = requested_x[:, column]
                risk = _score_tabular(model, requested_batch, tabular_name, current)
                contributions[feature_names[column]] = contributions[feature_names[column]] + (risk - previous_risk)
                previous_risk = risk
    finally:
        _restore_modalities(model, originals)

    return {name: values / n_samples for name, values in contributions.items()}


@dataclass(frozen=True)
class PatientReport:
    """One patient's interpretability bundle.

    Attributes:
        patient_id: Who this is.
        risk: This patient's predicted risk score (log-hazard).
        risk_percentile: Where that risk sits (0-100) within the FULL cohort the
            dataloader yielded, not just the patients requested in the same call --
            adding or removing other requested patients never moves this number.
        survival_times: Horizons the survival curve was evaluated at, shape ``(T,)``.
            Empty when the curve could not be computed; see ``metadata``.
        survival_probs: This patient's predicted survival probability at each of
            ``survival_times``, shape ``(T,)``. Empty under the same condition.
        modality_contributions: Risk delta per modality -- how much removing that
            modality (marking it absent, the same mechanism the model itself uses
            for a missing modality) would have changed THIS patient's own risk
            score. Exact Shapley value over modality coalitions; NOT the cohort-level
            number :func:`~kalecancer.interpret.modality.modality_shapley` reports.
        feature_contributions: Risk delta per raw clinical feature, by the same
            logic, for the model's tabular modality. Sampling-based (see
            ``metadata["n_feature_samples"]``); empty if the model has no single,
            unambiguous tabular modality or its feature names could not be
            discovered (see ``metadata``). NOT the cohort-level number
            :func:`~kalecancer.interpret.attribution.feature_shapley` reports.
        top_patches: The ``top_k`` most-attended patches for this patient's bag
            modality, in :func:`~kalecancer.interpret.attention.attention_records`'
            form. ``None`` -- not an empty list -- when the patient has no bag
            modality at all (no such modality on the model, or this patient is
            genuinely missing it); an empty list would wrongly suggest a bag with
            nothing in it.
        metadata: What was computed and how; see :func:`patient_report`.
    """

    patient_id: str
    risk: float
    risk_percentile: float
    survival_times: np.ndarray
    survival_probs: np.ndarray
    modality_contributions: dict[str, float]
    feature_contributions: dict[str, float]
    top_patches: list[dict] | None
    metadata: dict


def patient_report(
    model,
    dataloader,
    patient_ids: Sequence[str],
    *,
    train_predictions: SplitPredictions | None = None,
    eval_times: Sequence[float] | None = None,
    top_k: int = 20,
    n_feature_samples: int = 50,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> list[PatientReport]:
    """Build an interpretability bundle for each named patient.

    ``patient_ids`` is required, not optional, on purpose: a per-patient Shapley
    computation over the whole cohort the dataloader yields would be the kind of
    thing nobody should run by accident, so the caller must name who they want
    explained.

    Args:
        model: A trainer exposing ``embedders``, ``task``, ``device``, ``training``
            and a gradient-free ``predict``, e.g.
            :class:`~kalecancer.pipeline.CohortTrainer`.
        dataloader: Yields :class:`~kalecancer.loaddata.multimodal_access.PatientBatch`;
            consumed exactly once, to assemble the full cohort ``risk_percentile`` is
            computed against.
        patient_ids: Which patients to build a report for. Every id must be present
            in ``dataloader``.
        train_predictions: Training-split predictions, needed to fit the Breslow
            baseline hazard the survival curve is read off. ``None`` skips the
            curve (see ``metadata``), rather than raising or guessing one.
        eval_times: Horizons for the survival curve. ``None`` or empty skips it, the
            same as a missing ``train_predictions``.
        top_k: Number of top-attended patches to keep per patient with a bag modality.
        n_feature_samples: Permutation-sampling budget for ``feature_contributions``,
            the same meaning as
            :func:`~kalecancer.interpret.attribution.feature_shapley`'s ``n_samples``
            -- but see the module docstring for why this is cheaper per patient.
        seed: Seeds the feature-contribution sampling; deterministic for a fixed seed.
        device: If given, the model is moved there before scoring.

    Returns:
        One :class:`PatientReport` per id in ``patient_ids``, in that order.

    Raises:
        ValueError: If ``model.task`` is not a
            :class:`~kalecancer.pipeline.task.SurvivalTask`, or if any
            ``patient_ids`` entry is not present in ``dataloader``.
    """
    _require_survival_task(model, "patient_report")

    if device is not None:
        model.to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            full_batch = _materialise(dataloader)

            position = {patient_id: index for index, patient_id in enumerate(full_batch.patient_id)}
            missing = [patient_id for patient_id in patient_ids if patient_id not in position]
            if missing:
                raise ValueError(f"patient_id(s) not found in dataloader: {missing}")
            indices = [position[patient_id] for patient_id in patient_ids]

            full_risk = _risk_for(model, full_batch)
            requested_batch = _subset_batch(full_batch, indices)

            modality_names = list(model.embedders)
            tabular_name = _find_tabular_modality(model)
            bag_modality = _find_bag_modality(model)

            modality_contributions_all = _modality_contributions(model, requested_batch)

            feature_names = _discover_feature_names(dataloader, tabular_name) if tabular_name is not None else None
            tabular_pool = full_batch.modalities[tabular_name] if tabular_name is not None else None
            if tabular_name is not None and feature_names is not None and isinstance(tabular_pool, torch.Tensor):
                feature_contributions_all: dict[str, np.ndarray] | None = _feature_contributions(
                    model,
                    tabular_pool,
                    requested_batch,
                    tabular_name,
                    feature_names,
                    n_feature_samples,
                    seed,
                )
                feature_contributions_reason = None
            else:
                feature_contributions_all = None
                feature_contributions_reason = (
                    "no single, unambiguous TabICLEmbedder modality found"
                    if tabular_name is None
                    else "feature names undiscoverable from dataloader.dataset.feature_names"
                )

            metadata_ok = {"coords", "slide_index", "slide_ids"} <= set(requested_batch.metadata)
            attention_weights = (
                bag_attention(model, requested_batch, bag_modality)
                if bag_modality is not None and metadata_ok
                else None
            )

            survival_computable = train_predictions is not None and bool(eval_times)
            survival_reason: str | None = None
            survival_times_arr = np.array([])
            survival_probs_matrix: np.ndarray | None = None
            if not survival_computable:
                survival_reason = "train_predictions not given" if train_predictions is None else "eval_times not given"
            else:
                assert train_predictions is not None  # for mypy: survival_computable implies this
                try:
                    train_times = train_predictions.duration.numpy().astype(float)
                    train_events = as_event_mask(train_predictions.event).numpy()
                    event_times, cumulative_hazard = breslow_baseline_hazard(
                        train_predictions.risk.numpy().astype(float), train_times, train_events
                    )
                    survival_times_arr = np.asarray(eval_times, dtype=float)
                    survival_probs_matrix = predict_survival_function(
                        full_risk[indices], event_times, cumulative_hazard, survival_times_arr
                    )
                except ValueError as error:
                    survival_computable = False
                    survival_reason = f"baseline hazard unavailable: {error}"
                    survival_times_arr = np.array([])

            base_metadata = {
                "seed": seed,
                "n_feature_samples": n_feature_samples,
                "modalities_found": modality_names,
                "tabular_modality": tabular_name,
                "bag_modality": bag_modality,
                "survival_curve_computable": survival_computable,
                "survival_curve_unavailable_reason": survival_reason,
                "feature_contributions_unavailable_reason": feature_contributions_reason,
                "contributions_are_risk_deltas": True,
                "contributions_are_not_cohort_shapley_values": True,
                "cohort_size": len(full_batch.patient_id),
            }

            reports = []
            for row, patient_id in enumerate(patient_ids):
                index = indices[row]

                top_patches = None
                if attention_weights is not None:
                    assert bag_modality is not None  # attention_weights is only ever set alongside it
                    if bool(requested_batch.present[bag_modality][row]):
                        records = attention_records(
                            patient_id,
                            attention_weights[patient_id],
                            requested_batch.metadata["coords"][row],
                            requested_batch.metadata["slide_index"][row],
                            requested_batch.metadata["slide_ids"][row],
                        )
                        top_patches = top_k_patches(records, k=top_k)

                reports.append(
                    PatientReport(
                        patient_id=patient_id,
                        risk=float(full_risk[index]),
                        risk_percentile=_percentile_rank(float(full_risk[index]), full_risk),
                        survival_times=survival_times_arr,
                        survival_probs=(
                            survival_probs_matrix[row] if survival_probs_matrix is not None else np.array([])
                        ),
                        modality_contributions={
                            name: float(values[row]) for name, values in modality_contributions_all.items()
                        },
                        feature_contributions=(
                            {name: float(values[row]) for name, values in feature_contributions_all.items()}
                            if feature_contributions_all is not None
                            else {}
                        ),
                        top_patches=top_patches,
                        metadata=dict(base_metadata),
                    )
                )
    finally:
        model.train(was_training)
    return reports


def save_reports(reports: Sequence[PatientReport], path: str | Path) -> None:
    """Write reports to JSON, arrays included, in a form that round-trips.

    Every field is already JSON-native except ``survival_times``/``survival_probs``
    (``numpy.ndarray``), which :func:`~kalecancer.utils.io.write_json` converts via
    ``.tolist()``. No pickle: the file stays a plain, readable JSON array, portable
    across processes and languages, matching every other artefact this package writes
    (see :func:`~kalecancer.evaluate.survival_predictions.save_predictions`).
    """
    write_json(path, [dataclasses.asdict(report) for report in reports])
