"""Evaluation metrics: a score of a prediction frame against a target frame, both indexed by patient id.

Survival metrics come from torchsurv and classification metrics from scikit-learn.
"""

from __future__ import annotations

import math
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
    classes: tuple | None


class Metric(ABC):
    """Scores ``prediction`` against ``target`` row by row, matched by id. ``__init__`` arguments are stored under
    their own names, which config dumps rely on."""

    higher_is_better: bool

    @abstractmethod
    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float: ...

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
    """Harrell's concordance of ``log_hazard``: among comparable pairs (the earlier time is an event), the share in
    which the earlier patient has the higher log-hazard. Tied log-hazards count one half."""

    higher_is_better = True

    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float:
        risk, event, time = _survival_inputs(prediction, target)
        return float(ConcordanceIndex()(risk, event, time))


class UnoC(Metric):
    """Uno's concordance of ``log_hazard`` truncated at ``tau``: Harrell's pairs whose earlier time is before ``tau``,
    weighted by 1 / G(earlier time)², with G the Kaplan–Meier censoring survival of ``context.train_target``.

    Raises unless ``tau`` is below the last training time with G(tau) > 0.
    """

    higher_is_better = True

    def __init__(self, tau: float):
        self.tau = tau

    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float:
        risk, event, time = _survival_inputs(prediction, target)
        train_event, train_time = _ipcw_training(self, "tau", self.tau, context)
        weight = get_ipcw(train_event, train_time, new_time=time)
        tmax = torch.tensor(_float32_bound(self.tau))
        c = float(ConcordanceIndex()(risk, event, time, weight=weight, tmax=tmax))
        return _defined(c, f"{self!r} is undefined: no comparable pair has its earlier time before tau")


class TimeDependentAUC(Metric):
    """Cumulative/dynamic AUC of ``log_hazard`` at ``time``: cases had the event by ``time`` and are weighted by
    1 / G(event time), with G the Kaplan–Meier censoring survival of ``context.train_target``; controls are still
    event-free after ``time``.

    Raises unless ``time`` is below the last training time with G(time) > 0, and within the target's follow-up:
    min target time <= ``time`` < max target time.
    """

    higher_is_better = True

    def __init__(self, time: float):
        self.time = time

    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float:
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
    """Area under the ROC curve of ``positive_class`` against all other classes, scored by that class's prediction
    column (the usual binary AUROC when there are two classes)."""

    higher_is_better = True

    def __init__(self, positive_class: Hashable):
        self.positive_class = positive_class

    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float:
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
        scores, labels = _classification_inputs(prediction, target, context)
        predicted = np.array(context.classes, dtype=object)[scores.argmax(axis=1)]
        return float(balanced_accuracy_score(labels, predicted))
