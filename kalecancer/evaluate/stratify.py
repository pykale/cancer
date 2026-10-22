"""Risk-group stratification: quantile groups by ``log_hazard``, and their Kaplan-Meier survival curves."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

from kalecancer.evaluate.metrics import _event_time, _quantile_groups, _require_columns


def risk_groups(
    prediction: pd.DataFrame,
    *,
    n_groups: int = 2,
    labels: Sequence[Hashable] | None = None,
) -> pd.Series:
    """Assign each patient to a risk-quantile group by ``log_hazard``, indexed by id.

    Group 0 is the lowest-risk quantile (lowest ``log_hazard``), group ``n_groups - 1`` the highest. Patients with
    a NaN ``log_hazard`` (``Pipeline.predict`` emits NaN for a patient it could not score) are excluded rather than
    assigned a group; the excluded count is not silently dropped, it is on the returned Series as
    ``.attrs["n_excluded"]``.

    Args:
        prediction: Predictions indexed by patient id, with a ``log_hazard`` column, as ``Pipeline.predict`` returns
            them.
        n_groups: Number of equal-sized risk quantiles. 2 is a median split, 3 tertiles, 4 quartiles.
        labels: Group names, lowest risk first -- any hashable value (strings work, but so do e.g. enum members).
            ``None`` (the default) labels the groups with the plain integers ``0 .. n_groups - 1``.

    Returns:
        Ordered categorical Series named ``"group"``, indexed by the patients with a defined ``log_hazard``.

    Raises:
        ValueError: If ``n_groups < 2``, if ``labels`` does not have ``n_groups`` entries, or if the quantile cuts
            are not unique (``log_hazard`` is too heavily tied to form ``n_groups`` distinct groups).
    """
    _require_columns(prediction, ["log_hazard"], "prediction")
    if n_groups < 2:
        raise ValueError(f"n_groups must be at least 2, got {n_groups}")
    if labels is not None and len(labels) != n_groups:
        raise ValueError(f"labels must have n_groups={n_groups} entries, got {len(labels)}")
    names: Sequence[Hashable] = list(labels) if labels is not None else list(range(n_groups))

    risk = prediction["log_hazard"]
    defined = risk[risk.notna()]
    excluded = int(risk.isna().sum())

    codes = _quantile_groups(defined.to_numpy(dtype=np.float64), n_groups)
    group = pd.Series(
        pd.Categorical.from_codes(codes, categories=names, ordered=True),
        index=defined.index,
        name="group",
    )
    group.attrs["n_excluded"] = excluded
    return group


def _kaplan_meier_curve(time: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    """One row per distinct observed time: ``at_risk``, ``events``, and the right-continuous KM survival estimate.

    Hand-rolled (no scikit-survival at runtime): at each distinct time, the survival drop is ``1 - events/at_risk``,
    cumulatively multiplied; a time with no event (pure censoring) leaves survival unchanged.

    A KM curve conventionally starts at ``(0, 1)``, so a ``time=0`` row with ``survival=1.0`` and the full group
    size as ``at_risk`` is prepended -- unless the data already has an observation at ``time=0``, in which case that
    row already represents the start and a synthetic duplicate would only be confusing.
    """
    distinct_times, inverse, counts = np.unique(time, return_inverse=True, return_counts=True)
    n = time.shape[0]

    events_at_time = np.zeros(distinct_times.shape[0], dtype=np.float64)
    at_event = event.astype(bool)
    np.add.at(events_at_time, inverse[at_event], 1.0)

    left_before = np.concatenate([[0], np.cumsum(counts)[:-1]])
    at_risk = n - left_before

    survival = np.cumprod(1.0 - events_at_time / at_risk)
    curve = pd.DataFrame(
        {
            "time": distinct_times,
            "at_risk": at_risk.astype(np.int64),
            "events": events_at_time.astype(np.int64),
            "survival": survival,
        }
    )
    if distinct_times[0] > 0:
        anchor = pd.DataFrame({"time": [0.0], "at_risk": [n], "events": [0], "survival": [1.0]})
        curve = pd.concat([anchor, curve], ignore_index=True)
    return curve


def kaplan_meier(
    prediction: pd.DataFrame,
    target: pd.DataFrame,
    *,
    n_groups: int = 2,
    labels: Sequence[Hashable] | None = None,
) -> pd.DataFrame:
    """Kaplan-Meier survival curves per risk group.

    Returns one row per (group, time) with columns: group, time, at_risk, events, survival. Long format, ready to
    plot or write out. ``time`` ranges over every distinct observed time within that group (event or censoring),
    plus a leading ``time=0, survival=1.0`` row per group (the conventional KM starting point), unless a group's
    own data already has an observation at ``time=0``. ``survival`` only changes at a time with ``events > 0``.

    Groups come from :func:`risk_groups`: patients with a NaN ``log_hazard`` are excluded the same way, and the
    excluded count is on the returned frame's ``.attrs["n_excluded"]``.

    Args:
        prediction: Predictions indexed by patient id, with a ``log_hazard`` column.
        target: The same patients' ``time``/``event`` columns (or a superset of them).
        n_groups: Number of risk-quantile groups; see :func:`risk_groups`.
        labels: Group names; see :func:`risk_groups`.

    Raises:
        ValueError: Under the same conditions as :func:`risk_groups`, or if ``target`` is missing any risk-grouped
            patient.
    """
    group = risk_groups(prediction, n_groups=n_groups, labels=labels)

    missing = group.index.difference(target.index)
    if len(missing):
        raise ValueError(
            f"target is missing {len(missing)} of the {len(group)} risk-grouped patients, e.g. {missing[:5].tolist()}"
        )
    event, time = _event_time(target.loc[group.index], "target")
    event_arr = event.numpy()
    time_arr = time.numpy().astype(np.float64)

    frames = []
    for name in group.cat.categories:
        mask = (group == name).to_numpy()
        curve = _kaplan_meier_curve(time_arr[mask], event_arr[mask])
        curve.insert(0, "group", name)
        frames.append(curve)
    table = pd.concat(frames, ignore_index=True)
    table.attrs["n_excluded"] = group.attrs.get("n_excluded", 0)
    return table
