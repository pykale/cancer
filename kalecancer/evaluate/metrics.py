"""Evaluation metrics: a score of a prediction frame against a target frame, both indexed by patient id.

Survival metrics come from torchsurv and classification metrics from scikit-learn.
"""

from __future__ import annotations

import math
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

with warnings.catch_warnings():
    # torchsurv applies torch.jit.script when imported, which recent torch deprecates; users cannot act on it
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.* is deprecated", category=FutureWarning)
    from torchsurv.metrics.auc import Auc
    from torchsurv.metrics.brier_score import BrierScore
    from torchsurv.metrics.cindex import ConcordanceIndex
    from torchsurv.stats.ipcw import get_ipcw
    from torchsurv.stats.kaplan_meier import KaplanMeierEstimator


@dataclass(frozen=True)
class EvalContext:
    """What a metric may use besides the evaluated rows.

    Args:
        train_target: Target frame of the ids the model was fitted on; IPCW metrics estimate censoring from it.
        classes: Class values in prediction-column order; ``None`` for time-to-event targets.
    """

    train_target: pd.DataFrame
    classes: Sequence[Hashable] | None


class Metric(ABC):
    """Scores ``prediction`` against ``target`` row by row, matched by id.

    ``__init__`` arguments are stored under their own names, which config dumps rely on.
    """

    higher_is_better: bool

    @abstractmethod
    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float:
        """Score ``prediction`` against ``target``, matching rows by patient id.

        Args:
            prediction: Predictions indexed by patient id, as ``Pipeline.predict`` returns them.
            target: The target's ``frame`` for the same patients.
            context: The training target and class order, for metrics that need them.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in vars(self).items())})"


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} needs columns {list(columns)}; missing {missing} among {list(frame.columns)}")


def _aligned_values(prediction: pd.DataFrame, target: pd.DataFrame, columns: list, dtype: type) -> np.ndarray:
    """``prediction[columns]`` in the target's row order."""
    for name, frame in (("prediction", prediction), ("target", target)):
        if not frame.index.is_unique:
            duplicated = frame.index[frame.index.duplicated()][:5].tolist()
            raise ValueError(f"{name} ids must be unique; duplicated: {duplicated}")
    unpredicted = target.index.difference(prediction.index)
    untargeted = prediction.index.difference(target.index)
    if len(unpredicted) or len(untargeted):
        raise ValueError(
            f"prediction and target ids differ: {len(unpredicted)} target ids lack a prediction "
            f"(e.g. {unpredicted[:5].tolist()}) and {len(untargeted)} predicted ids lack a target "
            f"(e.g. {untargeted[:5].tolist()})"
        )
    values = prediction.loc[target.index, columns]
    undefined = values.isna().any(axis=1)
    if undefined.any():
        examples = values.index[undefined][:5].tolist()
        raise ValueError(
            f"{int(undefined.sum())} of {len(values)} predictions are NaN (e.g. {examples}); "
            "a metric is never computed on the defined subset"
        )
    return values.to_numpy(dtype=dtype)


def _float32_bound(value: float) -> np.float32:
    """The float32 ``b`` for which ``t < b`` holds exactly when ``t < value``, for every float32 ``t``.

    torchsurv casts times and ``tmax`` to float32 before its strict comparison; sksurv compares in float64.
    """
    bound = np.float32(value)
    # float(): under NumPy 2 promotion, float32 < python float would compare in float32 and see them as equal
    return np.nextafter(bound, np.float32(np.inf)) if float(bound) < value else bound


def _defined(value: float, reason: str) -> float:
    if not math.isfinite(value):
        raise ValueError(reason)
    return value


def _event_time(frame: pd.DataFrame, what: str) -> tuple[torch.Tensor, torch.Tensor]:
    _require_columns(frame, ["time", "event"], what)
    if not pd.api.types.is_bool_dtype(frame["event"]):
        raise TypeError(f"{what}: column 'event' must be bool (True = event observed), got {frame['event'].dtype}")
    return torch.tensor(frame["event"].to_numpy(dtype=bool)), torch.tensor(frame["time"].to_numpy(dtype=np.float32))


def _survival_inputs(prediction: pd.DataFrame, target: pd.DataFrame) -> tuple[torch.Tensor, ...]:
    _require_columns(prediction, ["log_hazard"], "survival prediction")
    event, time = _event_time(target, "target")
    risk = torch.tensor(_aligned_values(prediction, target, ["log_hazard"], np.float32)[:, 0])
    return risk, event, time


def _ipcw_training(metric: Metric, name: str, value: float, context: EvalContext) -> tuple[torch.Tensor, ...]:
    """Training event and time, after checking that censoring weights at ``value`` are defined.

    torchsurv gives zero weight where the training censoring survival G is zero, which silently drops test events;
    sksurv raises instead.
    """
    event, time = _event_time(context.train_target, "context.train_target")
    last = float(time.max())
    km = KaplanMeierEstimator()
    km(event, time, censoring_dist=True)
    if not (value < last and float(km.predict(torch.tensor([value], dtype=torch.float32))[0]) > 0):
        raise ValueError(
            f"{metric!r}: {name} must be below the last training time {last}, where the censoring survival G "
            f"estimated on context.train_target is positive (valid: {name} < {last} and G({name}) > 0)"
        )
    return event, time


class HarrellC(Metric):
    """Harrell's concordance of ``log_hazard``.

    Among comparable pairs (the earlier time is an event), the share in which the earlier patient has the higher
    log-hazard. Tied log-hazards count one half.
    """

    higher_is_better = True

    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float:
        """Harrell's C of the ``log_hazard`` column against the target's ``time`` and ``event``."""
        risk, event, time = _survival_inputs(prediction, target)
        return float(ConcordanceIndex()(risk, event, time))


class UnoC(Metric):
    """Uno's concordance of ``log_hazard``, truncated at ``tau``.

    Harrell's pairs whose earlier time is before ``tau``, weighted by 1 / G(earlier time)², with G the Kaplan–Meier
    censoring survival of ``context.train_target``.

    Raises unless ``tau`` is below the last training time with G(tau) > 0.
    """

    higher_is_better = True

    def __init__(self, tau: float):
        self.tau = tau

    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float:
        """Uno's C of the ``log_hazard`` column, truncated at ``tau``."""
        risk, event, time = _survival_inputs(prediction, target)
        train_event, train_time = _ipcw_training(self, "tau", self.tau, context)
        weight = get_ipcw(train_event, train_time, new_time=time)
        tmax = torch.tensor(_float32_bound(self.tau))
        c = float(ConcordanceIndex()(risk, event, time, weight=weight, tmax=tmax))
        return _defined(c, f"{self!r} is undefined: no comparable pair has its earlier time before tau")


class TimeDependentAUC(Metric):
    """Cumulative/dynamic AUC of ``log_hazard`` at ``time``.

    Cases had the event by ``time`` and are weighted by 1 / G(event time), with G the Kaplan–Meier censoring
    survival of ``context.train_target``; controls are still event-free after ``time``.

    Raises unless ``time`` is below the last training time with G(time) > 0, and within the target's follow-up:
    min target time <= ``time`` < max target time.
    """

    higher_is_better = True

    def __init__(self, time: float):
        self.time = time

    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float:
        """The cumulative/dynamic AUC of the ``log_hazard`` column at ``time``."""
        risk, event, time = _survival_inputs(prediction, target)
        train_event, train_time = _ipcw_training(self, "time", self.time, context)
        first, last = float(time.min()), float(time.max())
        if not first <= self.time < last:
            raise ValueError(f"{self!r}: time must lie within the follow-up of the target, [{first}, {last})")
        # float64 so cases are compared with time itself, as in sksurv, not with its float32 rounding.
        new_time = torch.tensor([self.time], dtype=torch.float64)
        auc = Auc()(
            risk,
            event,
            time,
            new_time=new_time,
            weight=get_ipcw(train_event, train_time, new_time=time),
            weight_new_time=get_ipcw(train_event, train_time, new_time=new_time.float()),
        )
        return _defined(float(auc[0]), f"{self!r} is undefined: no target event at or before time")


_SURVIVAL_COLUMN = re.compile(r"survival\[(.+)\]")


class IntegratedBrier(Metric):
    """Integrated Brier score of the ``survival[<time>]`` columns: how far predicted survival is from the outcome.

    At each column's time t, patients with an event by t contribute S(t)², weighted by 1 / G(event time), and patients
    still event-free after t contribute (1 - S(t))², weighted by 1 / G(t), with G the Kaplan–Meier censoring survival
    of ``context.train_target``. These Brier scores are integrated over the column times with the trapezoidal rule and
    divided by their range. Unlike the concordances it scores calibration as well as ranking. ``Pipeline.predict`` and
    ``Pipeline.evaluate`` add the columns when given ``survival_times``.

    Raises unless there are at least two columns, their last time is below the last training time with G > 0, and
    they lie within the target's follow-up: min target time <= first time and last time < max target time.
    """

    higher_is_better = False

    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float:
        """The integrated Brier score of the ``survival[<time>]`` columns."""
        columns = {float(m[1]): c for c in prediction.columns if (m := _SURVIVAL_COLUMN.fullmatch(str(c)))}
        if len(columns) < 2:
            raise ValueError(
                f"{self!r} needs at least two 'survival[<time>]' columns, got {list(prediction.columns)}; set "
                "survival_times with two or more times in Pipeline.predict, evaluate or cross_validate"
            )
        times = sorted(columns)
        event, time = _event_time(target, "target")
        survival = torch.tensor(_aligned_values(prediction, target, [columns[t] for t in times], np.float64))
        train_event, train_time = _ipcw_training(self, "the last survival time", times[-1], context)
        first, last = float(time.min()), float(time.max())
        if not (first <= times[0] and times[-1] < last):
            raise ValueError(
                f"{self!r}: survival times {times[0]}..{times[-1]} must lie within the follow-up of the target, "
                f"[{first}, {last})"
            )
        # float64 so event times are compared with the column times themselves, as in sksurv: torchsurv compares each
        # time as a 0-dim tensor, which would not promote float32 event times
        new_time = torch.tensor(times, dtype=torch.float64)
        brier = BrierScore()
        brier(
            survival,
            event,
            time.double(),
            new_time=new_time,
            weight=get_ipcw(train_event, train_time, new_time=time),
            weight_new_time=get_ipcw(train_event, train_time, new_time=new_time.float()),
        )
        return float(brier.integral())


def _quantile_groups(values: np.ndarray, n_groups: int) -> np.ndarray:
    """Assign each value to one of ``n_groups`` equal-sized quantile groups: 0 the lowest, ``n_groups - 1`` highest.

    Unique cuts are not enough to guarantee every group gets a member: e.g. ``[0, 1, 1, 2]`` with ``n_groups=4``
    gives distinct edges but codes ``[0, 1, 1, 3]`` -- group 2 is empty, because the two tied ``1``s both land in
    the same bin rather than splitting across the one edge between them. Checked here, after assigning codes,
    rather than left for a caller to discover when it tries to estimate something (survival, a group mean) over
    an empty group.

    Raises:
        ValueError: If ``n_groups < 2``, if the quantile cuts are not unique, or if a cut leaves one of the
            ``n_groups`` groups with no members -- both mean the values are too heavily tied to form that many
            distinct, non-empty groups.
    """
    if n_groups < 2:
        raise ValueError(f"n_groups must be at least 2, got {n_groups}")
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_groups + 1))
    if np.unique(edges).size != edges.size:
        raise ValueError(
            f"cannot form {n_groups} risk groups: quantile cuts are not unique among {values.size} values "
            f"({np.unique(values).size} distinct) -- they are too heavily tied"
        )
    codes = np.clip(np.digitize(values, edges[1:-1], right=True), 0, n_groups - 1)
    empty = np.flatnonzero(np.bincount(codes, minlength=n_groups) == 0)
    if empty.size:
        raise ValueError(
            f"cannot form {n_groups} risk groups: group(s) {empty.tolist()} would be empty among {values.size} "
            f"values ({np.unique(values).size} distinct) -- they are too heavily tied"
        )
    return codes


def _log_rank_statistic(time: np.ndarray, event: np.ndarray, group: np.ndarray, n_groups: int) -> float:
    """Mantel-Cox chi-square statistic across groups -- the same statistic ``sksurv.compare.compare_survival`` computes.

    Observed minus expected events per group at each distinct event time, with the hypergeometric
    variance-covariance, reduced to a quadratic form over ``n_groups - 1`` groups (the last is linearly dependent
    on the others).
    """
    distinct_times, inverse = np.unique(time, return_inverse=True)
    n_times = distinct_times.shape[0]

    count_at_time = np.zeros((n_times, n_groups))
    np.add.at(count_at_time, (inverse, group), 1.0)
    events_at_time = np.zeros((n_times, n_groups))
    at_event = event.astype(bool)
    np.add.at(events_at_time, (inverse[at_event], group[at_event]), 1.0)

    total_per_group = count_at_time.sum(axis=0)
    left_before = np.vstack([np.zeros((1, n_groups)), np.cumsum(count_at_time, axis=0)[:-1]])
    at_risk = total_per_group - left_before  # (n_times, n_groups): n_g at each distinct time

    n_j = at_risk.sum(axis=1)
    d_j = events_at_time.sum(axis=1)

    observed = events_at_time.sum(axis=0)
    safe_n_j = np.where(n_j > 0, n_j, 1.0)
    expected = (at_risk * (d_j / safe_n_j)[:, None]).sum(axis=0)

    has_variance = n_j > 1
    factor = np.zeros(n_times)
    factor[has_variance] = (d_j[has_variance] * (n_j[has_variance] - d_j[has_variance])) / (
        (n_j[has_variance] - 1) * n_j[has_variance] ** 2
    )
    weighted_at_risk = at_risk * factor[:, None]
    covariance = np.diag((weighted_at_risk * n_j[:, None]).sum(axis=0)) - weighted_at_risk.T @ at_risk

    diff = (observed - expected)[:-1]
    try:
        return float(diff @ np.linalg.solve(covariance[:-1, :-1], diff))
    except np.linalg.LinAlgError:
        return float("nan")


class LogRank(Metric):
    """Mantel-Cox log-rank statistic across risk-quantile groups of ``log_hazard``.

    Splits patients into ``n_groups`` equal-sized quantile groups by ``log_hazard`` (2 is a median split, 3
    tertiles, 4 quartiles) and computes the Mantel-Cox chi-square statistic for a difference in survival across
    them -- the standard log-rank test, generalised beyond two groups.

    A larger statistic means more separation between the groups' survival curves, which is why
    ``higher_is_better = True``; a p-value would invert that meaning, since a SMALLER p-value means more separation.
    If the p-value itself is wanted: ``scipy.stats.chi2.sf(statistic, n_groups - 1)``.

    This measures SEPARATION between risk groups, not calibration (how close predicted and observed survival are --
    see ``IntegratedBrier``) or ranking accuracy (see ``HarrellC``, ``UnoC``).
    """

    higher_is_better = True

    def __init__(self, n_groups: int = 2):
        self.n_groups = n_groups

    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float:
        """The Mantel-Cox chi-square statistic across ``n_groups`` quantile groups of ``log_hazard``."""
        risk, event, time = _survival_inputs(prediction, target)
        group = _quantile_groups(risk.numpy().astype(np.float64), self.n_groups)
        event_arr = event.numpy()
        time_arr = time.numpy().astype(np.float64)

        if not event_arr.any():
            raise ValueError(f"{self!r} is undefined: no observed event among the evaluated patients")
        occupied = np.bincount(group, minlength=self.n_groups)
        if np.count_nonzero(occupied) < 2:
            raise ValueError(f"{self!r} is undefined: fewer than two non-empty risk groups to compare")

        statistic = _log_rank_statistic(time_arr, event_arr, group, self.n_groups)
        return _defined(statistic, f"{self!r} is undefined: the group covariance is singular")


def _classification_inputs(
    prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext
) -> tuple[np.ndarray, np.ndarray]:
    """Class scores in ``context.classes`` order and labels, both in the target's row order."""
    if context.classes is None:
        raise ValueError("classification metrics need context.classes (class values in prediction-column order)")
    classes = list(context.classes)
    columns = list(prediction.columns)
    if len(columns) != len(classes) or not all(
        str(col).endswith(f"[{c}]") for col, c in zip(columns, classes, strict=True)
    ):
        raise ValueError(
            f"prediction columns {columns} must be one per class, in the order of classes {classes}, "
            "each named '<prefix>[<class>]'"
        )
    _require_columns(target, ["label"], "classification target")
    unknown = target.index[~target["label"].isin(classes)]
    if len(unknown):
        raise ValueError(f"{len(unknown)} target labels are not in classes {classes} (e.g. ids {unknown[:5].tolist()})")
    return _aligned_values(prediction, target, columns, np.float64), target["label"].to_numpy(dtype=object)


class AUROC(Metric):
    """Area under the ROC curve of ``positive_class`` against all other classes.

    It is scored by that class's prediction column. With two classes this is the usual binary AUROC.
    """

    higher_is_better = True

    def __init__(self, positive_class: Hashable):
        self.positive_class = positive_class

    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float:
        """The AUROC of ``positive_class`` against all other classes."""
        scores, labels = _classification_inputs(prediction, target, context)

        # assert is temporary fix to keep mypy quiet
        # real fix requires rethinking EvalContext
        assert context.classes is not None, "classification targets always carry classes"
        classes = list(context.classes)
        if self.positive_class not in classes:
            raise ValueError(f"{self!r}: positive_class is not one of classes {classes}")
        positive = labels == self.positive_class
        if positive.all() or not positive.any():
            raise ValueError(f"{self!r} is undefined: the target needs both {self.positive_class!r} and other labels")
        return float(roc_auc_score(positive, scores[:, classes.index(self.positive_class)]))


class BalancedAccuracy(Metric):
    """Mean per-class recall over the classes in the target, predicting each patient's highest-scoring class.

    Ties go to the class listed first in ``context.classes`` (numpy argmax). With two probability columns a patient is
    predicted ``classes[1]`` only if its probability is above 0.5.
    """

    higher_is_better = True

    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float:
        """The balanced accuracy of each patient's highest-scoring class."""
        scores, labels = _classification_inputs(prediction, target, context)
        predicted = np.array(context.classes, dtype=object)[scores.argmax(axis=1)]
        return float(balanced_accuracy_score(labels, predicted))
