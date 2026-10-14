"""Which raw clinical features drive a model's risk predictions.

The counterpart of :mod:`~kalecancer.interpret.modality`, one level down: a modality's
contribution is a property of a whole branch, but a clinician does not recognise a
512-d TabICL embedding dimension -- they recognise ``age`` or ``tumour_stage``. This
module attributes risk to those raw, named columns instead.

**Why this needs re-embedding, not a perturbed embedding.** The tabular modality is
read by a *frozen* :class:`~kalecancer.model.embed.tabicl.TabICLEmbedder`: every row's
512-d representation is a function of the raw feature matrix (and the fold's
in-context rows), produced by a transformer, not a linear readout. Perturbing one
column of the embedding and calling that "feature j's effect" would attribute the
perturbation to whatever direction happens to live in dimension j of an arbitrary
embedding space -- meaningless, and not what either function below does. Perturbing a
*feature* here always means perturbing the raw ``(N, n_features)`` matrix handed to
the embedder and then re-embedding it; the 512-d output is never touched directly.

**Why the other modalities are cached.** A tabular perturbation changes nothing about
any other modality's raw input, and every other embedder is frozen too (or, if not,
still a pure function of unperturbed input), so its output is identical on every
perturbation. Both functions below compute every *other* modality's embedding exactly
once, up front, by temporarily swapping each such embedder for one that replays its
cached output (see :class:`_CachedOutput`); only the tabular embedder re-runs, once
per perturbation. The whole evaluation set is also materialised into one batch before
any of this starts, both so the embedder sees its whole split at once (as
:attr:`~kalecancer.model.embed.tabicl.TabICLEmbedder.needs_full_batch` asks) and so a
perturbation never has to re-touch the caller's ``dataloader``.

**Why "absent" means resampled, not zero.** Zero is a real, meaningful value for most
clinical variables (a dose, a lab count, a one-hot level) -- it is not a stand-in for
"unknown" the way it is for an unseen modality. Marginalising a feature out therefore
means replacing its column with values resampled from its own empirical distribution
in the evaluation set (what the caller passed as ``dataloader`` doubles as the
background here, since no separate reference set is threaded through these
signatures), never with a constant.

**Why sampling, not enumeration.** :mod:`~kalecancer.interpret.modality` enumerates
every modality coalition because ``k`` stays at 2-4. Clinical tables hold dozens of
columns, where ``2**d`` is not a number anyone should compute. Both functions here are
therefore sampling-based and carry sampling error; :func:`feature_shapley` is an
estimate, not the exact value :func:`~kalecancer.interpret.modality.modality_shapley`
returns, and does not satisfy an exact efficiency property for a finite sample.

Reuses :mod:`~kalecancer.interpret.modality`'s conventions: ``model.eval()`` plus
``torch.no_grad()`` with the training mode restored in a ``finally``, modality
discovery from ``model.embedders`` rather than hard-coded names, the same
:class:`~kalecancer.pipeline.task.SurvivalTask` check, and the same
``(risk, durations, events) -> float`` metric signature.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import cast

import numpy as np
import torch
from torch import nn

from kalecancer.evaluate.survival_metrics import concordance_index
from kalecancer.loaddata.multimodal_access import ModalityValue, PatientBatch
from kalecancer.model.embed.tabicl import TabICLEmbedder
from kalecancer.model.predict.losses import as_event_mask
from kalecancer.pipeline.task import EVENT_KEY, TIME_KEY, SurvivalTask

logger = logging.getLogger(__name__)

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


class _CachedOutput(nn.Module):
    """Replays a precomputed output regardless of input.

    Stands in for a non-tabular modality's embedder across every tabular
    perturbation: that embedder's raw input never changes within one call to
    :func:`permutation_importance` or :func:`feature_shapley`, only the tabular
    modality's does, so its output is computed once and replayed rather than
    recomputed on every perturbation.
    """

    def __init__(self, cached_output: torch.Tensor) -> None:
        super().__init__()
        self._cached_output = cached_output

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self._cached_output


def _require_survival_task(model, func_name: str) -> None:
    if not isinstance(model.task, SurvivalTask):
        raise ValueError(
            f"{func_name} currently supports survival (time-to-event) models only: its default "
            "metric and its (risk, durations, events) metric signature both assume targets from a "
            f"SurvivalTask, but this model's task is {type(model.task).__name__}."
        )


def _resolve_metric(metric: MetricFn | None) -> MetricFn:
    return metric if metric is not None else concordance_index


def _find_tabular_modality(model) -> str:
    """The one modality read by a frozen TabICL embedder -- what raw features attach to."""
    tabular = [name for name, embedder in model.embedders.items() if isinstance(embedder, TabICLEmbedder)]
    if len(tabular) != 1:
        raise ValueError(
            "feature-level attribution needs exactly one TabICLEmbedder modality to attribute raw "
            f"clinical columns to; found {len(tabular)} ({sorted(tabular)}). This module cannot guess "
            "which modality a raw feature matrix belongs to when there is not exactly one."
        )
    return tabular[0]


def _discover_feature_names(dataloader, modality: str, feature_names: Sequence[str] | None) -> list[str]:
    """Post-encoding column names for ``modality``, from the caller or the dataset.

    Reads ``dataloader.dataset.feature_names`` -- the
    :class:`~kalecancer.loaddata.multimodal_access.Preprocessor` contract every fold's
    dataset carries -- rather than inventing "feature_0", "feature_1", ... names that
    would not trace back to a column a clinician could recognise.
    """
    if feature_names is not None:
        return list(feature_names)

    dataset = getattr(dataloader, "dataset", None)
    all_names = getattr(dataset, "feature_names", None) if dataset is not None else None
    discovered = all_names.get(modality) if isinstance(all_names, dict) else None
    if not discovered:
        raise ValueError(
            f"could not discover feature names for modality {modality!r}: `dataloader.dataset."
            "feature_names` is unavailable, or has none for this modality. Pass `feature_names` "
            "explicitly."
        )
    return list(discovered)


def _target_arrays(targets: dict[str, torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """Durations and events, in the order :func:`concordance_index` expects."""
    durations = targets[TIME_KEY].double().numpy()
    events = as_event_mask(targets[EVENT_KEY]).numpy()
    return durations, events


def _to_device(value: ModalityValue, device: torch.device) -> ModalityValue:
    """Move one modality's raw input, whether a stacked tensor or a ragged bag list.

    A small duplicate of :func:`kalecancer.pipeline.trainer._to_device`: calling an
    embedder directly (to cache its output) bypasses the trainer's own device
    handling in :meth:`~kalecancer.pipeline.CohortTrainer.forward`, so it is redone
    here rather than reaching into that module's private helper.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return [item.to(device) for item in value]


def _cat_modality(values: list[ModalityValue]) -> ModalityValue:
    if isinstance(values[0], torch.Tensor):
        return torch.cat(cast("list[torch.Tensor]", values), dim=0)
    return [item for value in values for item in cast("list[torch.Tensor]", value)]


def _materialise(dataloader) -> PatientBatch:
    """Concatenate every batch the loader yields into one.

    TabICL re-embeds its whole context on every call, and mini-batching only wastes
    that work without changing the result (see
    :mod:`kalecancer.model.embed.tabicl`'s module docstring); re-iterating the
    dataloader once per perturbation would waste it many times over instead. This
    runs the loader exactly once, up front, and every later perturbation reuses the
    single batch it returns.
    """
    patient_ids: list[str] = []
    modality_rows: dict[str, list[ModalityValue]] = {}
    present_rows: dict[str, list[torch.Tensor]] = {}
    target_rows: dict[str, list[torch.Tensor]] = {}

    for batch in dataloader:
        patient_ids.extend(batch.patient_id)
        for name, value in batch.modalities.items():
            modality_rows.setdefault(name, []).append(value)
        for name, value in batch.present.items():
            present_rows.setdefault(name, []).append(value)
        for key, value in batch.target.items():
            target_rows.setdefault(key, []).append(value)

    return PatientBatch(
        patient_id=patient_ids,
        modalities={name: _cat_modality(values) for name, values in modality_rows.items()},
        present={name: torch.cat(values) for name, values in present_rows.items()},
        target={key: torch.cat(values) for key, values in target_rows.items()},
    )


def _cache_other_modalities(model, batch: PatientBatch, tabular_name: str) -> dict[str, nn.Module]:
    """Embed every non-tabular modality once and install a cached replay in its place.

    Returns the original embedders, so they can be restored once scoring is done.
    """
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


def _score_tabular(model, batch: PatientBatch, tabular_name: str, tabular_x: torch.Tensor) -> np.ndarray:
    """Run the model with the tabular modality's raw input replaced by ``tabular_x``.

    Every other modality's embedder is, at this point, a cached replay (see
    :func:`_cache_other_modalities`), so this is the only step that re-embeds anything.
    """
    perturbed = replace(batch, modalities={**batch.modalities, tabular_name: tabular_x})
    output = model.predict(perturbed)
    return output.prediction.reshape(-1).detach().cpu().double().numpy()


def _resolve_tabular_matrix(full_batch: PatientBatch, tabular_name: str, feature_names: Sequence[str]) -> torch.Tensor:
    tabular_x = full_batch.modalities[tabular_name]
    if not isinstance(tabular_x, torch.Tensor) or tabular_x.ndim != 2:
        raise ValueError(f"modality {tabular_name!r} must be a 2-D (N, n_features) tensor for feature attribution")
    if tabular_x.shape[1] != len(feature_names):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but modality {tabular_name!r}'s raw input "
            f"has {tabular_x.shape[1]} columns; they must line up one-to-one."
        )
    return tabular_x


def permutation_importance(
    model,
    dataloader,
    *,
    feature_names: Sequence[str] | None = None,
    metric: MetricFn | None = None,
    n_repeats: int = 10,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> dict[str, dict[str, float]]:
    """Shuffle each clinical feature in turn and measure the metric drop.

    Each feature's column is permuted across patients in the evaluation set --
    breaking whatever link it has to the outcome while preserving its marginal
    distribution exactly, since the permuted values are its own -- the raw tabular
    matrix is then re-embedded by the (frozen) TabICL embedder and re-scored. Every
    other modality is embedded once; see the module docstring.

    Args:
        model: A trainer exposing ``embedders``, ``task``, ``device`` and a
            gradient-free ``predict``, e.g. :class:`~kalecancer.pipeline.CohortTrainer`,
            with exactly one modality read by a
            :class:`~kalecancer.model.embed.tabicl.TabICLEmbedder`.
        dataloader: Yields :class:`~kalecancer.loaddata.multimodal_access.PatientBatch`;
            consumed exactly once (see the module docstring), so a single-use
            iterator works as well as a reusable one.
        feature_names: Post-encoding column names for the tabular modality, matching
            its raw input's column order. ``None`` discovers them from
            ``dataloader.dataset.feature_names``; if that is unavailable, raises
            ``ValueError`` rather than inventing placeholder names.
        metric: ``(risk, durations, events) -> float``. ``None`` defaults to
            :func:`~kalecancer.evaluate.survival_metrics.concordance_index`.
        n_repeats: Independent shuffles per feature. The spread across them (the
            returned ``"std"``) is what makes a single number interpretable --
            without it, a mean alone cannot say whether a feature's effect is
            consistent or one lucky (or unlucky) shuffle.
        seed: Seeds the shuffles; deterministic for a fixed seed.
        device: If given, the model is moved there before scoring.

    Returns:
        ``{feature_name: {"mean": ..., "std": ..., "baseline": ...}}``. ``"mean"`` is
        the average metric drop from shuffling that feature (baseline minus shuffled,
        averaged over ``n_repeats``); positive means the feature was contributing.
        ``"baseline"`` repeats the unshuffled score for every feature, for convenience.

    Raises:
        ValueError: If ``model.task`` is not a
            :class:`~kalecancer.pipeline.task.SurvivalTask`; if the model does not
            expose exactly one :class:`~kalecancer.model.embed.tabicl.TabICLEmbedder`
            modality; or if ``feature_names`` cannot be resolved or does not match the
            tabular modality's column count.
    """
    _require_survival_task(model, "permutation_importance")
    metric_fn = _resolve_metric(metric)
    tabular_name = _find_tabular_modality(model)
    names = _discover_feature_names(dataloader, tabular_name, feature_names)

    if device is not None:
        model.to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            full_batch = _materialise(dataloader)
            tabular_x = _resolve_tabular_matrix(full_batch, tabular_name, names)
            originals = _cache_other_modalities(model, full_batch, tabular_name)
            try:
                durations, events = _target_arrays(full_batch.target)
                n = tabular_x.shape[0]

                baseline_risk = _score_tabular(model, full_batch, tabular_name, tabular_x)
                baseline_score = float(metric_fn(baseline_risk, durations, events))

                rng = np.random.default_rng(seed)
                results: dict[str, dict[str, float]] = {}
                for column, name in enumerate(names):
                    drops = np.empty(n_repeats)
                    for repeat in range(n_repeats):
                        shuffled = tabular_x.clone()
                        shuffled[:, column] = tabular_x[torch.from_numpy(rng.permutation(n)), column]
                        risk = _score_tabular(model, full_batch, tabular_name, shuffled)
                        drops[repeat] = baseline_score - float(metric_fn(risk, durations, events))
                    results[name] = {
                        "mean": float(drops.mean()),
                        "std": float(drops.std()),
                        "baseline": baseline_score,
                    }
            finally:
                _restore_modalities(model, originals)
    finally:
        model.train(was_training)
    return results


def feature_shapley(
    model,
    dataloader,
    *,
    feature_names: Sequence[str] | None = None,
    metric: MetricFn | None = None,
    n_samples: int = 50,
    seed: int = 0,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Sampling-based Shapley values over raw clinical features.

    Uses the permutation-sampling estimator (Strumbelj & Kononenko, 2010): each sample
    draws a random ordering of the features and a fresh, independent per-column
    resample of the whole tabular matrix (the "every feature absent" background, see
    the module docstring), then reveals the true columns one at a time in that order,
    re-embedding and re-scoring after each reveal. A feature's marginal contribution
    at the step it is revealed is accumulated; its Shapley estimate is the mean of
    those contributions over all samples. This is the estimator exact for the true
    coalition value in expectation, and has far lower variance per forward pass than
    sampling random *subsets* directly, because consecutive scores along one ordering
    share most of their randomness.

    **These are estimates, not exact values** -- unlike
    :func:`~kalecancer.interpret.modality.modality_shapley`, which enumerates every
    coalition because a handful of modalities makes ``2**k`` tractable. With ``d``
    features here in the dozens, enumeration is impossible, so this samples instead.
    Standard error of the mean scales as ``O(1/sqrt(n_samples))`` -- quadrupling
    ``n_samples`` roughly halves it -- and is logged per feature (at ``INFO``) as a
    convergence diagnostic rather than returned, to keep this function's return type
    exactly ``{feature_name: float}``.

    Args:
        model: A trainer exposing ``embedders``, ``task``, ``device`` and a
            gradient-free ``predict``, e.g. :class:`~kalecancer.pipeline.CohortTrainer`,
            with exactly one modality read by a
            :class:`~kalecancer.model.embed.tabicl.TabICLEmbedder`.
        dataloader: Yields :class:`~kalecancer.loaddata.multimodal_access.PatientBatch`;
            consumed exactly once (see the module docstring).
        feature_names: Post-encoding column names for the tabular modality. ``None``
            discovers them from ``dataloader.dataset.feature_names``; raises
            ``ValueError`` if that is unavailable.
        metric: ``(risk, durations, events) -> float``. ``None`` defaults to
            :func:`~kalecancer.evaluate.survival_metrics.concordance_index`.
        n_samples: Number of random orderings to sample. Costs ``n_samples * (d + 1)``
            forward passes over the whole cohort, where ``d`` is the number of
            features (every other modality is still embedded only once; see the
            module docstring). The default, 50, is usually enough to stabilise the
            *ranking* of features -- which one is largest, which are near zero --
            even though the *values* themselves keep tightening with more samples.
            Watch the logged standard error (``INFO``) to judge whether 50 is
            enough for what you need, or whether to raise this.
        seed: Seeds both the feature ordering and the background resampling;
            deterministic for a fixed seed.
        device: If given, the model is moved there before scoring.

    Returns:
        ``{feature_name: shapley_value}``.

    Raises:
        ValueError: If ``model.task`` is not a
            :class:`~kalecancer.pipeline.task.SurvivalTask`; if the model does not
            expose exactly one :class:`~kalecancer.model.embed.tabicl.TabICLEmbedder`
            modality; or if ``feature_names`` cannot be resolved or does not match the
            tabular modality's column count.
    """
    _require_survival_task(model, "feature_shapley")
    metric_fn = _resolve_metric(metric)
    tabular_name = _find_tabular_modality(model)
    names = _discover_feature_names(dataloader, tabular_name, feature_names)
    n_features = len(names)

    if device is not None:
        model.to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            full_batch = _materialise(dataloader)
            tabular_x = _resolve_tabular_matrix(full_batch, tabular_name, names)
            originals = _cache_other_modalities(model, full_batch, tabular_name)
            try:
                durations, events = _target_arrays(full_batch.target)
                n = tabular_x.shape[0]
                rng = np.random.default_rng(seed)

                # Logged at roughly every 10% of n_samples: a real run is thousands
                # of forward passes, and silence until the first feature's summary
                # line gives no sign anything is happening.
                log_every = max(1, n_samples // 10)

                contributions: dict[str, list[float]] = {name: [] for name in names}
                for sample in range(n_samples):
                    if (sample + 1) % log_every == 0 or sample + 1 == n_samples:
                        logger.info("feature_shapley: %d/%d permutation samples done", sample + 1, n_samples)
                    order = rng.permutation(n_features)

                    # This sample's background: every column independently resampled
                    # from its own values in the evaluation set -- "absent", not zero.
                    current = tabular_x.clone()
                    for column in range(n_features):
                        current[:, column] = tabular_x[torch.from_numpy(rng.permutation(n)), column]

                    previous_score = float(
                        metric_fn(_score_tabular(model, full_batch, tabular_name, current), durations, events)
                    )
                    for column in order:
                        current = current.clone()
                        current[:, column] = tabular_x[:, column]
                        score = float(
                            metric_fn(_score_tabular(model, full_batch, tabular_name, current), durations, events)
                        )
                        contributions[names[column]].append(score - previous_score)
                        previous_score = score
            finally:
                _restore_modalities(model, originals)
    finally:
        model.train(was_training)

    shapley = {}
    for name, values in contributions.items():
        values_arr = np.asarray(values)
        shapley[name] = float(values_arr.mean())
        if values_arr.size > 1:
            standard_error = float(values_arr.std(ddof=1) / np.sqrt(values_arr.size))
            logger.info("feature_shapley: %s = %.4g (standard error %.4g)", name, shapley[name], standard_error)
    return shapley
