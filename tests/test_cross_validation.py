from functools import partial

import pytest
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from kalecancer.evaluate import AUROC, BalancedAccuracy, HarrellC, cross_validate
from kalecancer.loaddata import Classification, MultimodalDataset, PatchFeatures, TimeToEvent
from kalecancer.model import ABMIL, ClassificationHead, Concat, CoxHead, IntermediateFusion
from kalecancer.pipeline import Pipeline


def dataset(cohort, task="survival"):
    numeric = cohort.clinical[["age"]].assign(age=lambda f: (f["age"] - 60) / 10)
    target = (
        TimeToEvent(cohort.time, cohort.event)
        if task == "survival"
        else Classification(cohort.labels, classes=["low", "high"])
    )
    return MultimodalDataset(
        {"clinical": numeric, "wsi": PatchFeatures(cohort.wsi_files, multiple_files="concatenate")},
        target=target,
        required_modalities=["clinical", "wsi"],
    )


def small_pipeline(task="survival"):
    model = IntermediateFusion(
        encoding={
            "clinical": [nn.Linear(1, 2)],
            "wsi": [ABMIL(in_dim=16, hidden_dim=2, attention_dim=2, dropout=0.0)],
        },
        fusion=Concat(),
        head=CoxHead(in_dim=4, ties="efron") if task == "survival" else ClassificationHead(in_dim=4, n_classes=2),
    )
    return Pipeline(
        model=model,
        transforms={},
        optimizer=partial(torch.optim.AdamW, lr=1e-2),
        batch_size=8,
        drop_last=True,
        max_epochs=1,
        validation=None,
        early_stopping=None,
        accelerator="cpu",
        precision="32-true",
        random_state=0,
    )


def test_every_patient_is_predicted_once_by_a_model_that_never_saw_it(cohort):
    data = dataset(cohort)
    result = cross_validate(
        small_pipeline(),
        data,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        metrics={"harrell_c": HarrellC()},
        return_estimators=True,
    )
    assert len(result.folds) == 3
    assert sorted(result.predictions.index) == data.ids and result.predictions.index.is_unique
    assert list(result.summary().index) == ["harrell_c"] and list(result.summary().columns) == ["mean", "std"]
    assert result.folds["n_train"].sum() == 2 * len(data)
    for fold, estimator in enumerate(result.estimators):
        tested = set(result.predictions.index[result.predictions["fold"] == fold])
        assert tested.isdisjoint(estimator.train_ids_)
    assert result.estimators[0].model_ is not result.estimators[1].model_


def test_stratification_uses_the_target_strata(cohort):
    data = dataset(cohort)
    result = cross_validate(
        small_pipeline(),
        data,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        metrics={"c": HarrellC()},
    )
    events = result.folds["test events"]
    assert events.max() - events.min() <= 1


def test_classification_folds_are_scored_against_the_fitted_classes(cohort):
    result = cross_validate(
        small_pipeline(task="classification"),
        dataset(cohort, task="classification"),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
        metrics={"auroc": AUROC(positive_class="high"), "bacc": BalancedAccuracy()},
    )
    assert list(result.predictions.columns) == ["probability[low]", "probability[high]", "fold"]
    assert result.folds[["auroc", "bacc"]].stack().between(0.0, 1.0).all()
    low = result.folds["test label: low"]
    assert low.max() - low.min() <= 1


def test_arguments_are_checked(cohort):
    data = dataset(cohort)
    with pytest.raises(ValueError, match="metrics is empty"):
        cross_validate(small_pipeline(), data, cv=StratifiedKFold(n_splits=3), metrics={})
    unlabelled = MultimodalDataset({"clinical": cohort.clinical[["age"]]}, target=None, required_modalities=[])
    with pytest.raises(ValueError, match="needs a dataset with a target"):
        cross_validate(small_pipeline(), unlabelled, cv=StratifiedKFold(n_splits=3), metrics={"c": HarrellC()})
