"""Cross-validation over patients."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from kalecancer.evaluate.metrics import EvalContext, Metric
from kalecancer.loaddata.dataset import MultimodalDataset
from kalecancer.loaddata.targets import Classification


@dataclass
class CVResult:
    """Per-fold scores and fit reports, out-of-fold predictions, and optionally the fitted estimators."""

    folds: pd.DataFrame
    predictions: pd.DataFrame
    metrics: list[str]
    estimators: list | None

    def summary(self) -> pd.DataFrame:
        """Mean and sample standard deviation of each metric over folds."""
        return self.folds[self.metrics].agg(["mean", "std"]).T


def cross_validate(
    estimator: Any,
    data: MultimodalDataset,
    cv: Any,
    metrics: Mapping[str, Metric],
    return_estimators: bool = False,
) -> CVResult:
    """Fit a clone of ``estimator`` on each training fold of ``cv`` and score it on the held-out patients.

    ``cv`` is any sklearn splitter; it receives the patient ids and the target's strata (events or labels) as ``y``.
    Every fold refits everything learned from data: validation carve, transforms, fit hooks and weights.
    Metrics are reported per fold; they are not pooled over out-of-fold predictions, whose scales differ between
    folds (Cox log-hazards have an arbitrary offset).
    """
    if data.target is None:
        raise ValueError("cross_validate needs a dataset with a target")
    if not metrics:
        raise ValueError("metrics is empty")
    ids = np.array(data.ids, dtype=object)
    strata = data.target.strata(data.ids)
    rows, predictions, estimators = [], [], []
    for fold, (train_index, test_index) in enumerate(cv.split(ids, strata)):
        train_ids, test_ids = sorted(ids[train_index].tolist()), sorted(ids[test_index].tolist())
        fitted = clone(estimator).fit(data.subset(train_ids))
        leaked = (set(fitted.train_ids_) | set(fitted.val_ids_)) & set(test_ids)
        if leaked:
            raise AssertionError(f"fold {fold}: {len(leaked)} test patients were used in fit")
        prediction = fitted.predict(data.subset(test_ids))
        target = data.target.frame.loc[prediction.index]
        context = EvalContext(
            train_target=fitted.train_target_,
            classes=fitted.target_.classes if isinstance(fitted.target_, Classification) else None,
        )
        test_counts = {f"test {key}": value for key, value in data.target.counts(test_ids).items()}
        scores = {name: metric(prediction, target, context) for name, metric in metrics.items()}
        rows.append({"fold": fold, "n_test": len(test_ids), **test_counts, **fitted.fit_report_, **scores})
        predictions.append(prediction.assign(fold=fold))
        if return_estimators:
            estimators.append(fitted)
    if not rows:
        raise ValueError("cv produced no folds")
    return CVResult(
        folds=pd.DataFrame(rows).set_index("fold"),
        predictions=pd.concat(predictions),
        metrics=list(metrics),
        estimators=estimators if return_estimators else None,
    )
