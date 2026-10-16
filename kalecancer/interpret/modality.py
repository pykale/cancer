"""How much each modality contributes to a multimodal model's predictions.

Two ways of asking the same question, at different costs. Leave-one-out ablation
drops one modality at a time and reports the change in a held-out metric -- cheap,
``k + 1`` forward passes over the loader, but a modality's "contribution" by this
measure is its marginal contribution against one fixed coalition: everyone else
present. Shapley values fix that by averaging the marginal contribution over every
coalition the modality could join, which is exact credit assignment but costs one
forward pass per coalition -- ``2**k`` of them, so only tractable while ``k`` stays
small (2-4 modalities, <= 16 passes).

Both read the *absence* a model already knows how to score: no raw input is altered,
only the ``present`` bit :class:`~kalecancer.loaddata.multimodal_access.PatientBatch`
carries per modality, exactly as
:func:`~kalecancer.model.embed.multimodal_fusion.modality_dropout` marks a modality
absent during training (the same mechanism the example prototype,
``examples/hancock_multimodal_classification/main.py``'s ``modality_ablation``, uses
for a single held-out modality). Whatever
:class:`~kalecancer.model.embed.MultimodalFusion` does with an absent modality --
placeholder substitution, a negligible-precision expert, or dropping out of a
weighted vote -- is exactly what scores its contribution here; this module invents no
second notion of "missing".

Currently supports a :class:`~kalecancer.pipeline.task.SurvivalTask` model only: the
default metric (Harrell's C-index) and the fixed three-argument metric signature,
``(risk, durations, events) -> float``, both assume time-to-event targets. A
classification model raises ``ValueError`` naming that limitation rather than
guessing how to adapt the signature to a single binary target.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
import torch

from kalecancer.evaluate.survival_metrics import concordance_index
from kalecancer.model.predict.losses import as_event_mask
from kalecancer.pipeline.task import EVENT_KEY, TIME_KEY, SurvivalTask

logger = logging.getLogger(__name__)

#: Score assigned to the empty coalition -- every modality marked absent -- when the
#: model cannot produce a finite prediction there (e.g. a
#: :class:`~kalecancer.model.embed.multimodal_fusion.ProductOfExpertsFusion` with no
#: prior expert, whose precision-weighted mean is undefined once every expert's
#: precision is negligible). 0.5 is chance-level for a concordance-style ranking
#: metric: what a model with literally no information achieves in expectation. Used
#: only as a fallback, never silently in place of an actual finite score.
EMPTY_COALITION_FALLBACK_SCORE = 0.5

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def _require_survival_task(model, func_name: str) -> None:
    if not isinstance(model.task, SurvivalTask):
        raise ValueError(
            f"{func_name} currently supports survival (time-to-event) models only: its default "
            "metric and its (risk, durations, events) metric signature both assume targets from a "
            f"SurvivalTask, but this model's task is {type(model.task).__name__}. Evaluate a "
            "classification model's modality contribution with a task-appropriate metric directly, "
            "as modality_ablation() in examples/hancock_multimodal_classification/main.py does."
        )


def _resolve_metric(model, metric: MetricFn | None) -> MetricFn:
    return metric if metric is not None else concordance_index


def _discover_modalities(model, modalities: Sequence[str] | None) -> list[str]:
    return list(modalities) if modalities is not None else list(model.embedders)


def _target_arrays(targets: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """Durations and events, in the order :func:`concordance_index` expects."""
    durations = targets[TIME_KEY].double().numpy()
    events = as_event_mask(targets[EVENT_KEY]).numpy()
    return durations, events


def _run_coalition(model, dataloader, coalition: frozenset[str]) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    """Run every batch with only ``coalition``'s modalities marked present.

    A modality outside ``coalition`` has its ``present`` bit cleared; one already in
    the batch's natural ``present`` (including a patient genuinely missing it) is left
    untouched, so only the modalities under test are intervened on. The batch's own
    ``present`` dict is never mutated in place -- a caller's loader may hand back the
    same underlying tensors on a later iteration (e.g. a plain in-memory list), and
    this function is called once per coalition over the same loader.
    """
    scores: list[torch.Tensor] = []
    target_rows: dict[str, list[torch.Tensor]] = {key: [] for key in (TIME_KEY, EVENT_KEY)}

    for batch in dataloader:
        masked_present = {
            name: (value if name in coalition else torch.zeros_like(value)) for name, value in batch.present.items()
        }
        output = model.predict(replace(batch, present=masked_present))
        scores.append(output.prediction.reshape(-1).detach().cpu())
        for key in target_rows:
            target_rows[key].append(batch.target[key].detach().cpu())

    risk = torch.cat(scores).double().numpy()
    targets = {key: torch.cat(rows) for key, rows in target_rows.items()}
    return risk, targets


def _score(model, dataloader, coalition: frozenset[str], metric_fn: MetricFn) -> float:
    """Score one coalition, falling back for an empty coalition the model cannot run.

    A non-empty coalition producing a non-finite risk is a different situation: there
    is no "no information" reading to fall back to the way there is for the empty
    coalition, so it is logged and reported as NaN instead. Handing a contaminated
    risk array straight to ``metric_fn`` would not raise or come back NaN either --
    most ranking metrics (concordance included) treat NaN as neither greater nor
    less than anything, which quietly drops the affected patients from every
    comparison and returns a plausible-looking but wrong number.
    """
    risk, targets = _run_coalition(model, dataloader, coalition)
    if not np.all(np.isfinite(risk)):
        if not coalition:
            logger.warning(
                "the empty coalition (every modality marked absent) produced a non-finite prediction; "
                "falling back to the chance-level score %.3f instead of crashing",
                EMPTY_COALITION_FALLBACK_SCORE,
            )
            return EMPTY_COALITION_FALLBACK_SCORE
        logger.warning(
            "coalition %s produced a non-finite risk for %d of %d patients; scoring it as NaN rather "
            "than handing a contaminated risk array to the metric, which would silently drop those "
            "patients from every comparison instead of raising",
            sorted(coalition),
            int(np.sum(~np.isfinite(risk))),
            risk.size,
        )
        return float("nan")
    durations, events = _target_arrays(targets)
    return float(metric_fn(risk, durations, events))


def leave_one_out_ablation(
    model,
    dataloader,
    *,
    modalities: Sequence[str] | None = None,
    metric: MetricFn | None = None,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Drop each modality in turn and report the change in the metric.

    Args:
        model: A trainer exposing ``embedders``, ``task`` and a gradient-free
            ``predict``, e.g. :class:`~kalecancer.pipeline.CohortTrainer`.
        dataloader: Yields :class:`~kalecancer.loaddata.multimodal_access.PatientBatch`;
            iterated once per modality (plus once for the baseline), so it must
            support being iterated more than once.
        modalities: Which modalities to ablate. ``None`` discovers every modality the
            model declares, from ``model.embedders``, rather than hard-coding names.
        metric: ``(risk, durations, events) -> float``. ``None`` defaults to
            :func:`~kalecancer.evaluate.survival_metrics.concordance_index`.
        device: If given, the model is moved there before scoring.

    Returns:
        ``{modality_name: baseline_metric - ablated_metric}``, plus ``"baseline"`` for
        the unablated score. A positive value means removing that modality hurt the
        metric, i.e. it was contributing; a value near zero means the model's
        predictions barely depend on it.

    Raises:
        ValueError: If ``model.task`` is not a
            :class:`~kalecancer.pipeline.task.SurvivalTask` (see module docstring).
    """
    _require_survival_task(model, "leave_one_out_ablation")
    metric_fn = _resolve_metric(model, metric)
    modality_names = _discover_modalities(model, modalities)
    if device is not None:
        model.to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            full = frozenset(modality_names)
            baseline_score = _score(model, dataloader, full, metric_fn)

            results = {"baseline": baseline_score}
            for name in modality_names:
                ablated_score = _score(model, dataloader, full - {name}, metric_fn)
                results[name] = baseline_score - ablated_score
    finally:
        model.train(was_training)
    return results


def modality_shapley(
    model,
    dataloader,
    *,
    modalities: Sequence[str] | None = None,
    metric: MetricFn | None = None,
    device: str | torch.device | None = None,
    max_modalities: int = 4,
) -> dict[str, float]:
    """Exact Shapley value per modality over all ``2**k`` coalitions.

    Every coalition (every subset of ``modalities``, from empty to full) is scored
    exactly once -- cached by its ``frozenset`` -- then combined with the exact
    Shapley weighting ``|S|! (k - |S| - 1)! / k!`` over subsets ``S`` of the other
    modalities. The values sum to ``full_coalition_score - empty_coalition_score``
    (the standard Shapley efficiency property).

    Args:
        model: A trainer exposing ``embedders``, ``task`` and a gradient-free
            ``predict``, e.g. :class:`~kalecancer.pipeline.CohortTrainer`.
        dataloader: Yields :class:`~kalecancer.loaddata.multimodal_access.PatientBatch`;
            iterated once per distinct coalition, so it must support being iterated
            more than once.
        modalities: Which modalities to value. ``None`` discovers every modality the
            model declares, from ``model.embedders``.
        metric: ``(risk, durations, events) -> float``. ``None`` defaults to
            :func:`~kalecancer.evaluate.survival_metrics.concordance_index`.
        device: If given, the model is moved there before scoring.
        max_modalities: Refuses ``modalities`` larger than this, since the cost is
            ``2**len(modalities)`` forward passes over the whole loader.

    Returns:
        ``{modality_name: shapley_value}``.

    Raises:
        ValueError: If ``model.task`` is not a
            :class:`~kalecancer.pipeline.task.SurvivalTask` (see module docstring), or
            if ``len(modalities) > max_modalities``.
    """
    _require_survival_task(model, "modality_shapley")
    metric_fn = _resolve_metric(model, metric)
    modality_names = _discover_modalities(model, modalities)
    k = len(modality_names)
    if k > max_modalities:
        raise ValueError(
            f"modality_shapley needs one forward pass over the whole loader per coalition (2**k); "
            f"{k} modalities would need {2**k} of them, exceeding max_modalities={max_modalities}. "
            "Pass a smaller `modalities` subset, or raise max_modalities if that cost is acceptable."
        )

    if device is not None:
        model.to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            cache: dict[frozenset[str], float] = {}
            for size in range(k + 1):
                for subset in itertools.combinations(modality_names, size):
                    coalition = frozenset(subset)
                    if coalition not in cache:
                        cache[coalition] = _score(model, dataloader, coalition, metric_fn)

            shapley = {}
            for name in modality_names:
                others = [m for m in modality_names if m != name]
                total = 0.0
                for size in range(len(others) + 1):
                    for subset in itertools.combinations(others, size):
                        without = frozenset(subset)
                        with_name = without | {name}
                        weight = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
                        total += weight * (cache[with_name] - cache[without])
                shapley[name] = total
    finally:
        model.train(was_training)
    return shapley
