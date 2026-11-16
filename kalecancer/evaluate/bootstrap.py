"""Bootstrap confidence intervals for a metric on one set of held-out predictions."""

from __future__ import annotations

import copy
import dataclasses
import warnings
from dataclasses import dataclass

import numpy as np
import torch

from kalecancer.evaluate.metrics import Metric
from kalecancer.loaddata.targets import BaseTarget
from kalecancer.model.outputs import BaseOutput


@dataclass(frozen=True)
class BootstrapCI:
    """A metric's value on the evaluated patients and its percentile bootstrap interval.

    Args:
        estimate: The metric on the patients as given, not resampled. With a baseline, the metric of the output minus
            that of the baseline.
        lower: Lower bound of the interval.
        upper: Upper bound of the interval.
        confidence: Coverage the interval was computed for, e.g. 0.95.
        n_resamples: Resamples drawn, including undefined ones.
        n_undefined: Resamples on which the metric was undefined; the interval is computed from the others.
    """

    estimate: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    n_undefined: int

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


def _score(
    metric: Metric,
    output: BaseOutput,
    baseline: BaseOutput | None,
    target: BaseTarget,
    train_target: BaseTarget,
    baseline_train_target: BaseTarget,
) -> float:
    value = metric(output, target, train_target)
    return value if baseline is None else value - metric(baseline, target, baseline_train_target)


def _in_order(output: BaseOutput, ids: list[str], what: str) -> BaseOutput:
    """``output``'s patients in the order of ``ids``, which must be exactly its patients, without its branches.

    Metrics never read a ``LateFusion`` model's branch outputs, so they are dropped here rather than resampled.
    """
    position = {pid: k for k, pid in enumerate(output.ids)}
    if len(position) != len(output.ids) or set(position) != set(ids):
        raise ValueError(f"{what} must hold exactly the patients of output, each once")
    fused = dataclasses.replace(output, branches={})
    return fused.select(torch.tensor([position[pid] for pid in ids], dtype=torch.long))


def _resample(output: BaseOutput, rows: np.ndarray, ids: list[str]) -> BaseOutput:
    # metrics require unique ids, so a resampled patient drawn twice gets two new ids
    return dataclasses.replace(output.select(torch.as_tensor(rows)), ids=ids)


def _resample_target(target: BaseTarget, frame_ids: list[str], rows: np.ndarray, ids: list[str]) -> BaseTarget:
    # metrics read only the target's type and frame, so the copy keeps everything else as it is
    resampled = copy.copy(target)
    resampled.frame = target.frame.loc[frame_ids].iloc[rows].set_axis(ids)
    return resampled


def bootstrap_ci(
    metric: Metric,
    output: BaseOutput,
    target: BaseTarget,
    train_target: BaseTarget,
    baseline: BaseOutput | None = None,
    baseline_train_target: BaseTarget | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap interval of ``metric``, resampling patients with replacement.

    Use it on one model's output for held-out patients, never on out-of-fold predictions pooled over
    cross-validation folds, whose scales differ between folds (Cox log-hazards have an arbitrary offset).

    With ``baseline``, the interval is of the paired difference ``metric(output) - metric(baseline)``: both are scored
    on the same resampled patients, so the interval is of the gain itself. Separate intervals for two models overlap
    even when one is consistently better, so compare models this way rather than by eye.

    ``train_target`` is not resampled: IPCW metrics keep estimating censoring from the training patients.

    Args:
        metric: The metric to resample, e.g. ``HarrellC()``.
        output: A model's output for the held-out patients, as ``Pipeline.predict_output`` returns it.
        target: The outcomes of at least those patients.
        train_target: The outcomes of the patients the model was fitted on, ``Pipeline.train_target_``.
        baseline: Another model's output for the same patients, to compare against.
        baseline_train_target: The baseline model's training outcomes; defaults to ``train_target``.
        n_resamples: Number of bootstrap resamples.
        confidence: Coverage of the interval, between 0 and 1.
        random_state: Seeds the resampling.

    Returns:
        The estimate on the patients as given, and the interval.

    Raises:
        ValueError: If the metric fails on the patients as given, or is undefined on every resample.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    baseline_train_target = train_target if baseline_train_target is None else baseline_train_target
    # Raises on misuse (missing outcomes, a wrong output type) before any resampling can mistake it for an undefined
    # value.
    estimate = _score(metric, output, baseline, target, train_target, baseline_train_target)

    # Positions in the target's order, so the resamples do not depend on the order of the output's patients.
    have = set(output.ids)
    patients = [pid for pid in target.ids if pid in have]
    output = _in_order(output, patients, "output")
    baseline = None if baseline is None else _in_order(baseline, patients, "baseline")
    n = len(patients)
    ids = [str(k) for k in range(n)]
    rng = np.random.default_rng(random_state)
    values = []
    for _ in range(n_resamples):
        rows = rng.integers(0, n, size=n)
        try:
            values.append(
                _score(
                    metric,
                    _resample(output, rows, ids),
                    None if baseline is None else _resample(baseline, rows, ids),
                    _resample_target(target, patients, rows, ids),
                    train_target,
                    baseline_train_target,
                )
            )
        except ValueError:
            # Metrics raise ValueError when they are undefined, e.g. a resample without an event or a positive.
            continue
    n_undefined = n_resamples - len(values)
    if not values:
        raise ValueError(f"{metric!r} is undefined on every one of {n_resamples} resamples")
    if n_undefined:
        warnings.warn(
            f"{metric!r} is undefined on {n_undefined} of {n_resamples} resamples; the interval is computed from "
            "the others, which can make it too narrow",
            stacklevel=2,
        )
    tail = 100 * (1 - confidence) / 2
    lower, upper = np.percentile(values, [tail, 100 - tail])
    return BootstrapCI(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        n_resamples=n_resamples,
        n_undefined=n_undefined,
    )
