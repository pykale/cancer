from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn

from kalecancer.evaluate import AUROC, BalancedAccuracy, HarrellC
from kalecancer.loaddata import Classification, MultimodalDataset, PatchFeatures, TimeToEvent
from kalecancer.model import (
    ABMIL,
    ClassificationHead,
    Concat,
    CoxHead,
    InContextModule,
    IntermediateFusion,
    LateFusion,
    MaskedMean,
    MeanLogits,
    Unimodal,
)
from kalecancer.pipeline import EarlyStopping, Pipeline
from kalecancer.prepdata import ColumnGroup, TableTransform


def clinical_transform():
    return TableTransform(
        groups={
            "numeric": ColumnGroup(["age"], [StandardScaler()]),
            "categorical": ColumnGroup(
                ["sex", "smoking"],
                [
                    SimpleImputer(strategy="constant", fill_value="missing"),
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ],
            ),
        },
        drop=[],
    )


def dataset(cohort, required=("clinical", "wsi"), task="survival", event=None):
    if task == "survival":
        target = TimeToEvent(cohort.time, cohort.event if event is None else event)
    else:
        target = Classification(cohort.labels, classes=["low", "high"])
    modalities = {"clinical": cohort.clinical, "wsi": PatchFeatures(cohort.wsi_files, multiple_files="concatenate")}
    return MultimodalDataset(modalities, target=target, required_modalities=list(required))


def split(data, n_train=22):
    return data.subset(data.ids[:n_train]), data.subset(data.ids[n_train:])


def intermediate(fusion=None, head_dim=8):
    return IntermediateFusion(
        encoding={
            "clinical": [nn.Linear(3, 4)],
            "wsi": [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.25), nn.Linear(6, 4)],
        },
        fusion=fusion or Concat(),
        head=CoxHead(in_dim=head_dim, ties="efron"),
    )


def late_classifier():
    return LateFusion(
        branches={
            "clinical": Unimodal("clinical", [nn.Linear(3, 4)], ClassificationHead(in_dim=4, n_classes=2)),
            "wsi": Unimodal(
                "wsi",
                [ABMIL(in_dim=16, hidden_dim=4, attention_dim=4, dropout=0.0)],
                ClassificationHead(in_dim=4, n_classes=2),
            ),
        },
        combine=MeanLogits(),
    )


def pipeline(model=None, **overrides):
    arguments = {
        "model": model if model is not None else intermediate(),
        "transforms": {"clinical": clinical_transform()},
        "optimizer": partial(torch.optim.AdamW, lr=1e-2),
        "batch_size": 8,
        "drop_last": True,
        "max_epochs": 2,
        "validation": None,
        "early_stopping": None,
        "accelerator": "cpu",
        "precision": "32-true",
        "random_state": 0,
    }
    arguments.update(overrides)
    return Pipeline(**arguments)


# ---------------------------------------------------------------- fit / predict / evaluate


def test_fit_predict_evaluate(cohort, tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    train, test = split(dataset(cohort))
    pipe = pipeline()
    template = {k: v.clone() for k, v in pipe.model.state_dict().items()}
    pipe.fit(train)

    prediction = pipe.predict(test)
    assert list(prediction.index) == test.ids and list(prediction.columns) == ["log_hazard"]
    assert np.isfinite(prediction.to_numpy()).all()
    scores = pipe.evaluate(test, metrics={"harrell_c": HarrellC()})
    assert 0.0 <= scores["harrell_c"] <= 1.0

    assert len(pipe.history_) == 2 and pipe.fit_report_["n_train"] == 22
    assert pipe.fit_report_["optimizer_steps"] == 4
    assert not pipe.model_.training
    assert all(torch.equal(template[k], v) for k, v in pipe.model.state_dict().items())
    assert not any(workdir.iterdir()), "fit must not write files to the working directory"


def test_evaluating_on_patients_used_in_fit_raises(cohort):
    train, _ = split(dataset(cohort))
    pipe = pipeline().fit(train)
    with pytest.raises(ValueError, match="22 patients were used to fit"):
        pipe.evaluate(train, metrics={"c": HarrellC()})
    assert 0.0 <= pipe.evaluate(train, metrics={"c": HarrellC()}, allow_seen=True)["c"] <= 1.0


def test_random_state_fixes_the_initial_weights(cohort):
    train, test = split(dataset(cohort))
    torch.manual_seed(1)
    first = pipeline(intermediate()).fit(train).predict(test)
    torch.manual_seed(2)
    second = pipeline(intermediate()).fit(train).predict(test)
    pd.testing.assert_frame_equal(first, second)


def test_keep_weights_modules_are_not_reinitialised(cohort):
    class Pretrained(nn.Linear):
        keep_weights = True

    projection = Pretrained(3, 4)
    projection.requires_grad_(False)
    model = IntermediateFusion(
        encoding={"clinical": [projection], "wsi": [ABMIL(16, 4, 4, 0.0)]},
        fusion=Concat(),
        head=CoxHead(in_dim=8, ties="efron"),
    )
    pipe = pipeline(model).fit(split(dataset(cohort))[0])
    torch.testing.assert_close(pipe.model_.encoding["clinical"][0].weight, projection.weight)


def test_parameters_without_reset_raise(cohort):
    class Scale(nn.Module):
        def __init__(self):
            super().__init__()
            self.factor = nn.Parameter(torch.ones(3))

        def forward(self, x):
            return x * self.factor

    model = IntermediateFusion(
        encoding={"clinical": [Scale(), nn.Linear(3, 4)], "wsi": [ABMIL(16, 4, 4, 0.0)]},
        fusion=Concat(),
        head=CoxHead(in_dim=8, ties="efron"),
    )
    with pytest.raises(TypeError, match="no reset_parameters"):
        pipeline(model).fit(split(dataset(cohort))[0])


def test_clone_is_unfitted_and_independent(cohort):
    train, test = split(dataset(cohort))
    pipe = pipeline().fit(train)
    before = pipe.predict(test)
    copy = clone(pipe)
    assert not hasattr(copy, "model_")
    copy.set_params(max_epochs=1).fit(train)
    pd.testing.assert_frame_equal(pipe.predict(test), before)


def test_a_fitted_model_cannot_be_a_template(cohort):
    train, _ = split(dataset(cohort))
    pipe = pipeline().fit(train)
    with pytest.raises(ValueError, match="modules from a fitted pipeline"):
        pipeline(pipe.model_).fit(train)


# ---------------------------------------------------------------- validation and early stopping


def test_early_stopping_restores_the_epoch_it_judged_best(cohort):
    train, _ = split(dataset(cohort, required=["clinical"]), n_train=36)
    pipe = pipeline(
        intermediate(fusion=MaskedMean(), head_dim=4),
        max_epochs=8,
        batch_size=8,
        validation=StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0),
        early_stopping=EarlyStopping(metric="loss", patience=2, restore_best=True),
    ).fit(train)
    history = pipe.history_
    assert "val_loss" in history and pipe.fit_report_["n_val"] == 11
    assert pipe.fit_report_["best_epoch"] == int(history["val_loss"].idxmin())
    assert pipe.fit_report_["best_score"] == pytest.approx(history["val_loss"].min())
    assert set(pipe.val_ids_).isdisjoint(pipe.train_ids_)


def test_validation_arguments_are_checked(cohort):
    train, _ = split(dataset(cohort))
    with pytest.raises(ValueError, match="early_stopping needs validation"):
        pipeline(early_stopping=EarlyStopping(metric="loss", patience=2, restore_best=True)).fit(train)
    with pytest.raises(ValueError, match="n_splits=1"):
        pipeline(validation=StratifiedShuffleSplit(n_splits=5, test_size=0.2)).fit(train)


# ---------------------------------------------------------------- guards


def test_training_guards(cohort):
    train, _ = split(dataset(cohort))
    with pytest.raises(ValueError, match=r"trainer_kwargs cannot set \['max_epochs'\]"):
        pipeline(trainer_kwargs={"max_epochs": 3}).fit(train)
    with pytest.raises(ValueError, match="accumulate_grad_batches"):
        pipeline(trainer_kwargs={"accumulate_grad_batches": 2}).fit(train)
    with pytest.raises(ValueError, match="matches no trainable parameter"):
        pipeline(param_groups={"encoding.clinicl": {"lr": 1e-3}}).fit(train)
    with pytest.raises(ValueError, match="precision"):
        pipeline(precision="16-true").fit(train)
    with pytest.raises(ValueError, match="make no batch"):
        pipeline(batch_size=64).fit(train)


def test_column_transformer_dropping_columns_raises(cohort):
    train, _ = split(dataset(cohort))
    dropping = ColumnTransformer([("numeric", StandardScaler(), ["age"])])
    with pytest.raises(ValueError, match=r"silently drops columns \['sex', 'smoking'\]"):
        pipeline(transforms={"clinical": dropping}).fit(train)


def test_batches_without_signal_are_skipped_and_reported(cohort):
    censored = pd.Series(False, index=cohort.event.index)
    train, _ = split(dataset(cohort, event=censored))
    with pytest.warns(UserWarning, match="4 training batches were skipped"):
        pipe = pipeline().fit(train)
    assert pipe.fit_report_["skipped_batches"] == 4 and pipe.fit_report_["optimizer_steps"] == 0


def test_missing_modality_at_prediction_is_checked(cohort):
    train, _ = split(dataset(cohort))
    pipe = pipeline().fit(train)
    everyone = dataset(cohort, required=["clinical"])
    with pytest.raises(ValueError, match="Concat cannot combine"):
        pipe.predict(everyone.subset(everyone.ids[30:]))


# ---------------------------------------------------------------- late fusion, encode, attention


def test_late_fusion_branch_predictions_and_metrics(cohort):
    train, test = split(dataset(cohort, task="classification"))
    pipe = pipeline(late_classifier()).fit(train)
    fused = pipe.predict(test)
    wsi = pipe.predict(test, branch="wsi")
    assert list(fused.columns) == list(wsi.columns) == ["probability[low]", "probability[high]"]
    np.testing.assert_allclose(fused.sum(axis=1), 1.0, rtol=1e-5)
    scores = pipe.evaluate(test, metrics={"auroc": AUROC(positive_class="high"), "bacc": BalancedAccuracy()})
    assert set(scores.index) == {"auroc", "bacc"}
    with pytest.raises(ValueError, match="branch='pathology'"):
        pipe.predict(test, branch="pathology")


def test_encode_is_deterministic_and_attention_aligns_with_coordinates(cohort):
    train, test = split(dataset(cohort))
    pipe = pipeline().fit(train)
    first = pipe.encode(test, modality="wsi")
    pd.testing.assert_frame_equal(first, pipe.encode(test, modality="wsi"))
    assert list(first.index) == test.ids and first.shape[1] == 4

    attention = pipe.attention(test, modality="wsi")
    assert len(attention) == sum(len(cohort.bags[pid]) for pid in test.ids)
    np.testing.assert_allclose(attention.groupby("patient_id")["attention"].sum(), 1.0, rtol=1e-5)
    patient = attention[attention["patient_id"] == test.ids[0]]
    np.testing.assert_array_equal(patient[["x", "y"]].to_numpy(), cohort.coords[test.ids[0]])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_gpu_training_with_early_stopping_matches_its_own_bookkeeping(cohort):
    train, test = split(dataset(cohort, required=["clinical"]), n_train=36)
    pipe = pipeline(
        intermediate(fusion=MaskedMean(), head_dim=4),
        max_epochs=4,
        validation=StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0),
        early_stopping=EarlyStopping(metric=HarrellC(), patience=2, restore_best=True),
        accelerator="gpu",
    ).fit(train)
    assert pipe.fit_report_["best_epoch"] == int(pipe.history_["val_metric"].idxmax())
    assert next(pipe.model_.parameters()).device.type == "cpu"
    assert np.isfinite(pipe.predict(test).to_numpy()).all()


class ContextStage(InContextModule):
    """An in-context stage that passes rows through, so only its fitted context matters."""

    out_dim = 3

    def __init__(self):
        super().__init__(context_label="event", context_folds=2, random_state=0)

    def embed(self, context, queries):
        return queries


def test_restoring_the_best_epoch_keeps_non_tensor_fitted_state(cohort):
    train, _ = split(dataset(cohort, required=["clinical"]), n_train=36)
    model = IntermediateFusion(
        encoding={"clinical": [ContextStage(), nn.Linear(3, 4)], "wsi": [ABMIL(16, 4, 4, 0.0)]},
        fusion=MaskedMean(),
        head=CoxHead(in_dim=4, ties="efron"),
    )
    pipe = pipeline(
        model,
        max_epochs=3,
        validation=StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0),
        early_stopping=EarlyStopping(metric="loss", patience=2, restore_best=True),
    ).fit(train)
    assert pipe.model_.encoding["clinical"][0].context_ids == pipe.train_ids_
