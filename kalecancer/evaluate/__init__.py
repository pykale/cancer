"""Scoring predictions, by task.

Four modules, named for the task they score and the thing they produce:

.. code-block:: text

    classification_metrics.py  ROC-AUC, average precision, F1, mean ROC curve
    survival_metrics.py        C-index, IPCW time-dependent AUC, integrated Brier
    survival_predictions.py    running a model over a loader, and scoring the result
    cross_validation.py        refitting across folds, and resampled intervals

Describing a *cohort* is not here: what counts as an excluded patient depends on how
that cohort was built, so it belongs to the dataset that built it.
"""

from kalecancer.evaluate.classification_metrics import MetricError, binary_metrics, mean_roc_curve, roc_auc
from kalecancer.evaluate.cross_validation import (
    bootstrap_ci,
    compare_models,
    cross_validate_survival,
    patient_stratified_splits,
)
from kalecancer.evaluate.survival_metrics import integrated_brier, kaplan_meier_groups, time_dependent_auc
from kalecancer.evaluate.survival_predictions import (
    SplitPredictions,
    evaluate_predictions,
    predict_split,
    save_predictions,
    summarise_folds,
)

__all__ = [
    "MetricError",
    "SplitPredictions",
    "binary_metrics",
    "bootstrap_ci",
    "compare_models",
    "cross_validate_survival",
    "evaluate_predictions",
    "integrated_brier",
    "kaplan_meier_groups",
    "mean_roc_curve",
    "patient_stratified_splits",
    "predict_split",
    "roc_auc",
    "save_predictions",
    "summarise_folds",
    "time_dependent_auc",
]
