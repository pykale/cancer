"""Bootstrap confidence intervals for a metric on one set of held-out predictions."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kalecancer.evaluate.metrics import EvalContext, Metric


@dataclass(frozen=True)
class BootstrapCI:
    """A metric's value on the evaluated patients and its percentile bootstrap interval.

    Args:
        estimate: The metric on the patients as given, not resampled. With a baseline, the metric of the prediction
            minus that of the baseline.
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
    prediction: pd.DataFrame,
    baseline: pd.DataFrame | None,
    target: pd.DataFrame,
    context: EvalContext,
    baseline_context: EvalContext,
) -> float:
    value = metric(prediction, target, context)
    return value if baseline is None else value - metric(baseline, target, baseline_context)


def _resample(frame: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    return frame.iloc[rows].set_axis(pd.RangeIndex(len(rows)))


def bootstrap_ci(
    metric: Metric,
    prediction: pd.DataFrame,
    target: pd.DataFrame,
    context: EvalContext,
    baseline: pd.DataFrame | None = None,
    baseline_context: EvalContext | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap interval of ``metric``, resampling patients with replacement.

    Use it on one model's predictions for held-out patients, never on out-of-fold predictions pooled over
    cross-validation folds, whose scales differ between folds (Cox log-hazards have an arbitrary offset).

    With ``baseline``, the interval is of the paired difference ``metric(prediction) - metric(baseline)``: both are
    scored on the same resampled patients, so the interval is of the gain itself. Separate intervals for two models
    overlap even when one is consistently better, so compare models this way rather than by eye.

    ``context`` is not resampled: IPCW metrics keep estimating censoring from the training target.

    Args:
        metric: The metric to resample, e.g. ``HarrellC()``.
        prediction: Predictions indexed by patient id, as ``Pipeline.predict`` returns them.
        target: The target's ``frame`` for the same patients.
        context: The training target and class order of the model that made ``prediction``.
        baseline: Another model's predictions for the same patients, to compare against.
        baseline_context: The baseline model's context; defaults to ``context``.
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
    baseline_context = context if baseline_context is None else baseline_context
    # Raises on misuse (mismatched ids, missing columns) before any resampling can mistake it for an undefined value.
    estimate = _score(metric, prediction, baseline, target, context, baseline_context)

    # Positions in the target's row order; resampled rows get new ids 0..n-1, since metrics require unique ids.
    prediction = prediction.loc[target.index]
    baseline = None if baseline is None else baseline.loc[target.index]
    n = len(target)
    rng = np.random.default_rng(random_state)
    values = []
    for _ in range(n_resamples):
        rows = rng.integers(0, n, size=n)
        try:
            values.append(
                _score(
                    metric,
                    _resample(prediction, rows),
                    None if baseline is None else _resample(baseline, rows),
                    _resample(target, rows),
                    context,
                    baseline_context,
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
