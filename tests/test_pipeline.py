import warnings
from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import nn

from kalecancer.evaluate import AUROC, BalancedAccuracy, HarrellC, IntegratedBrier, cross_validate
from kalecancer.interpret import attention
from kalecancer.loaddata import Classification, MultimodalDataset, PatchFeatures, Tabular, TimeToEvent
from kalecancer.model import (
    ABMIL,
    BreslowBaseline,
    ClassificationEndpoint,
    Concat,
    CoxEndpoint,
    CoxOutput,
    FittedEncoder,
    IntermediateFusion,
    LateFusion,
    MaskedMean,
    Unimodal,
)
from kalecancer.pipeline import EarlyStopping, NonFiniteInputWarning, Pipeline
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
    elif task == "classification":
        target = Classification(cohort.labels, classes=["low", "high"])
    else:
        target = None
    modalities = {
        "clinical": Tabular(cohort.clinical),
        "wsi": PatchFeatures(cohort.wsi_files, multiple_files="concatenate"),
    }
    return MultimodalDataset(modalities, target=target, required_modalities=list(required))


def split(data, n_train=22):
    return data.subset(data.ids[:n_train]), data.subset(data.ids[n_train:])


def intermediate(fusion=None, head_dim=8):
    return IntermediateFusion(
        encoders={
            "clinical": [nn.Linear(3, 4)],
            "wsi": [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.25), nn.Linear(6, 4)],
        },
        fusion=fusion or Concat(),
        endpoint=CoxEndpoint(nn.Linear(head_dim, 1, bias=False), "efron"),
    )


def late_classifier():
    return LateFusion(
        branches={
            "clinical": Unimodal(
                "clinical", [nn.Linear(3, 4)], ClassificationEndpoint(nn.Linear(4, 2), nn.CrossEntropyLoss())
            ),
            "wsi": Unimodal(
                "wsi",
                [ABMIL(in_dim=16, hidden_dim=4, attention_dim=4, dropout=0.0)],
                ClassificationEndpoint(nn.Linear(4, 2), nn.CrossEntropyLoss()),
            ),
        },
        fusion=MaskedMean(),
        endpoint=ClassificationEndpoint(None, nn.CrossEntropyLoss()),
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
    assert list(prediction.index) == list(test.ids) and list(prediction.columns) == ["log_hazard"]
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


def test_widths_are_checked_by_a_dry_run_before_training(cohort):
    train, _ = split(dataset(cohort))
    # the dry run runs two patients, so a batch of 2 rows shows the error came before training
    with pytest.raises(RuntimeError, match=r"\(2x8 and 7x1\)") as error:
        pipeline(intermediate(head_dim=7)).fit(train)
    assert error.value.__notes__ == ["in the head of CoxEndpoint (Linear), called on a tensor of shape (2, 8)"]
    unequal = IntermediateFusion(
        encoders={"clinical": [nn.Linear(3, 4)], "wsi": [ABMIL(16, 6, 4, 0.0)]},
        fusion=MaskedMean(),
        endpoint=CoxEndpoint(nn.Linear(4, 1, bias=False), "efron"),
    )
    with pytest.raises(ValueError, match=r"MaskedMean needs inputs of one width, got \{'clinical': 4, 'wsi': 6\}"):
        pipeline(unequal).fit(train)


@pytest.mark.parametrize(
    ("task", "endpoint", "match"),
    [
        (
            "survival",
            CoxEndpoint(nn.Linear(8, 2, bias=False), "efron"),
            r"of shape \(2, 2\), but CoxEndpoint expects \(2, 1\)",
        ),
        (
            "classification",
            ClassificationEndpoint(nn.Linear(8, 3), nn.CrossEntropyLoss()),
            r"of shape \(2, 3\), but ClassificationEndpoint expects \(2, 2\)",
        ),
    ],
)
def test_the_dry_run_checks_the_head_against_the_endpoint_width(cohort, task, endpoint, match):
    model = IntermediateFusion(
        encoders={"clinical": [nn.Linear(3, 4)], "wsi": [ABMIL(16, 4, 4, 0.0)]}, fusion=Concat(), endpoint=endpoint
    )
    with pytest.raises(ValueError, match=match):
        pipeline(model).fit(split(dataset(cohort, task=task))[0])


def test_a_late_fusion_error_names_the_branch(cohort):
    train, _ = split(dataset(cohort, task="classification"))
    model = LateFusion(
        {
            "clinical": Unimodal(
                "clinical", [nn.Linear(3, 2)], ClassificationEndpoint(nn.Linear(2, 2), nn.CrossEntropyLoss())
            ),
            "wsi": Unimodal(
                "wsi", [ABMIL(16, 6, 4, 0.0)], ClassificationEndpoint(nn.Linear(5, 2), nn.CrossEntropyLoss())
            ),
        },
        MaskedMean(),
        None,
    )
    with pytest.raises(RuntimeError, match=r"\(2x6 and 5x2\)") as error:
        pipeline(model).fit(train)
    assert error.value.__notes__ == [
        "in the head of ClassificationEndpoint (Linear), called on a tensor of shape (2, 6)",
        "in branch 'wsi' (Unimodal)",
    ]


def test_the_dry_run_runs_the_loss_before_training(cohort):
    class BrokenLoss(nn.Module):
        def forward(self, logits, labels):
            raise RuntimeError("this loss cannot read the target")

    built = []

    def optimizer(groups):
        built.append(True)
        return torch.optim.AdamW(groups, lr=1e-2)

    model = Unimodal("clinical", [nn.Linear(3, 4)], ClassificationEndpoint(nn.Linear(4, 2), BrokenLoss()))
    with pytest.raises(RuntimeError, match="cannot read the target"):
        pipeline(model, optimizer=optimizer).fit(split(dataset(cohort, task="classification"))[0])
    assert not built


def test_models_take_an_endpoint_not_a_head():
    with pytest.raises(TypeError, match="endpoint must be a BaseEndpoint"):
        Unimodal("clinical", [nn.Linear(3, 4)], nn.Linear(4, 1))


def test_fit_does_not_depend_on_the_global_seed(cohort):
    train, test = split(dataset(cohort))
    model = intermediate()
    torch.manual_seed(1)
    first = pipeline(model).fit(train).predict(test)
    torch.manual_seed(2)
    second = pipeline(model).fit(train).predict(test)
    pd.testing.assert_frame_equal(first, second)


def test_fit_starts_from_the_weights_the_model_was_built_with(cohort):
    model = intermediate()
    # with a learning rate of 0 training changes nothing, so model_ holds exactly the weights fit started from
    pipe = pipeline(model, optimizer=partial(torch.optim.AdamW, lr=0.0)).fit(split(dataset(cohort))[0])
    for (name, built), (_, fitted) in zip(model.state_dict().items(), pipe.model_.state_dict().items(), strict=True):
        torch.testing.assert_close(fitted, built, msg=name)


def test_modules_without_reset_parameters_fit_with_a_random_state(cohort):
    class Scale(nn.Module):
        def __init__(self):
            super().__init__()
            self.factor = nn.Parameter(torch.ones(3))

        def forward(self, x):
            return x * self.factor

    model = IntermediateFusion(
        encoders={"clinical": [Scale(), nn.Linear(3, 4)], "wsi": [ABMIL(16, 4, 4, 0.0)]},
        fusion=Concat(),
        endpoint=CoxEndpoint(nn.Linear(8, 1, bias=False), "efron"),
    )
    pipeline(model, random_state=0).fit(split(dataset(cohort))[0])


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


def test_early_stopping_can_monitor_a_classification_metric(cohort):
    train, _ = split(dataset(cohort, task="classification"))
    pipe = pipeline(
        late_classifier(),
        max_epochs=3,
        validation=StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0),
        early_stopping=EarlyStopping(metric=AUROC(positive_class="high"), patience=2, restore_best=True),
    ).fit(train)
    history = pipe.history_
    assert history["val_metric"].between(0.0, 1.0).all()
    assert pipe.fit_report_["best_score"] == pytest.approx(history["val_metric"].max())


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
        pipeline(param_groups={"encoders.clinicl": {"lr": 1e-3}}).fit(train)
    with pytest.raises(ValueError, match="precision"):
        pipeline(precision="16-true").fit(train)
    with pytest.raises(ValueError, match="make no batch"):
        pipeline(batch_size=64).fit(train)


def test_column_transformer_dropping_columns_raises(cohort):
    train, _ = split(dataset(cohort))
    dropping = ColumnTransformer([("numeric", StandardScaler(), ["age"])])
    with pytest.raises(ValueError, match=r"silently drops columns \['sex', 'smoking'\]"):
        pipeline(transforms={"clinical": dropping}).fit(train)


def test_a_transform_for_a_bag_modality_raises(cohort):
    train, _ = split(dataset(cohort))
    with pytest.raises(TypeError, match="PatchFeatures does not accept transforms"):
        pipeline(transforms={"wsi": StandardScaler()}).fit(train)


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


def remove_age(cohort, ids):
    """Make ``age`` missing for ``ids``. ``clinical_transform`` scales age without imputing it, and sklearn's scalers
    pass NaN through, so these patients reach the model with NaN inputs."""
    cohort.clinical.loc[ids, "age"] = np.nan


@pytest.mark.parametrize("task", ["survival", "classification"])
def test_nan_inputs_raise_at_fit_naming_the_patients(cohort, task):
    # without the check, Cox blamed divergence and classification trained to NaN predictions without an error
    remove_age(cohort, ["002", "004", "006"])
    train, _ = split(dataset(cohort, task=task))
    model = intermediate() if task == "survival" else late_classifier()
    with pytest.raises(ValueError, match=r"encoders\['clinical'\] output NaN or infinite values") as error:
        pipeline(model).fit(train)
    assert any(pid in str(error.value) for pid in ["002", "004", "006"])


def test_nan_inputs_at_prediction_raise_naming_the_patients(cohort):
    remove_age(cohort, ["024", "025"])
    train, test = split(dataset(cohort))
    pipe = pipeline().fit(train)
    message = (
        r"encoders\['clinical'\] output NaN or infinite values for 2 patient\(s\) \(e\.g\. \['024', '025'\]\)"
        r".*input already has NaN"
    )
    with pytest.raises(ValueError, match=message):
        pipe.predict(test)
    with pytest.raises(ValueError, match=message):
        pipe.encode(test, modality="clinical")


class NanToZero(nn.Module):
    """An encoder that handles missing values itself."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0)


def nan_handling_model():
    return IntermediateFusion(
        encoders={
            "clinical": [NanToZero(), nn.Linear(3, 4)],
            "wsi": [ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(6, 4)],
        },
        fusion=Concat(),
        endpoint=CoxEndpoint(nn.Linear(8, 1, bias=False), "efron"),
    )


def test_an_encoder_that_handles_nan_receives_nan_inputs_and_a_warning_reports_them(cohort):
    remove_age(cohort, ["002", "024"])
    train, test = split(dataset(cohort))
    fitted = r"1 of 22 patients have NaN or infinite values in 'clinical' \(e\.g\. \['002'\]\)"
    with pytest.warns(NonFiniteInputWarning, match=fitted):
        pipe = pipeline(nan_handling_model()).fit(train)
    assert pipe.fit_report_["non_finite_inputs"] == {"clinical": 1}
    with pytest.warns(NonFiniteInputWarning, match=r"1 of 8 patients .* \['024'\]"):
        prediction = pipe.predict(test)
    assert np.isfinite(prediction.to_numpy()).all()
    # encode and attention export share predict's inference step, so they warn too
    with pytest.warns(NonFiniteInputWarning, match=r"\['024'\]"):
        pipe.encode(test, modality="clinical")


def fit_recording_non_finite_warnings(pipe, train, silence):
    """Fit ``pipe``, and return it with the ``NonFiniteInputWarning``s that reached the user."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if silence:
            warnings.filterwarnings("ignore", category=NonFiniteInputWarning)
        pipe.fit(train)
    return pipe, [w for w in caught if issubclass(w.category, NonFiniteInputWarning)]


def test_clean_inputs_raise_no_warning_and_the_warning_can_be_silenced(cohort):
    pipe, caught = fit_recording_non_finite_warnings(pipeline(), split(dataset(cohort))[0], silence=False)
    assert pipe.fit_report_["non_finite_inputs"] == {} and not caught
    remove_age(cohort, ["002"])
    train = split(dataset(cohort))[0]
    _, caught = fit_recording_non_finite_warnings(pipeline(nan_handling_model()), train, silence=True)
    assert not caught


# ---------------------------------------------------------------- baseline hazard and survival curves


def inner_times(data, quantiles=(0.2, 0.4, 0.6)):
    """Times inside the follow-up of ``data``'s patients, where survival metrics are defined."""
    return [float(t) for t in np.quantile(data.target.frame.loc[data.ids, "time"], quantiles)]


def test_the_baseline_is_fitted_on_the_log_hazards_of_the_training_patients_only(cohort):
    train, test = split(dataset(cohort))
    pipe = pipeline(validation=StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)).fit(train)
    assert pipe.val_ids_
    seen = pipe.predict(train.subset(pipe.train_ids_))
    outcome = train.target.frame.loc[seen.index]
    expected = BreslowBaseline().fit(seen["log_hazard"], outcome["time"], outcome["event"].to_numpy())
    baseline = pipe.model_.endpoint.baseline_
    np.testing.assert_allclose(baseline.cumulative_hazard_, expected.cumulative_hazard_, rtol=1e-6)
    np.testing.assert_array_equal(baseline.event_times_, expected.event_times_)
    assert pipe.fit_report_["baseline_events"] == train.target.counts(pipe.train_ids_)["events"]


def test_survival_functions_come_from_the_endpoint_baseline(cohort):
    train, test = split(dataset(cohort))
    pipe = pipeline().fit(train)
    times = inner_times(test)
    prediction = pipe.predict(test)
    assert list(prediction.columns) == ["log_hazard"]
    curves = pipe.predict_survival_function(test, times)
    assert list(curves.columns) == times and curves.index.equals(prediction.index)
    np.testing.assert_allclose(
        curves.to_numpy(), pipe.model_.endpoint.baseline_.survival(prediction["log_hazard"], times), rtol=1e-12
    )
    assert (np.diff(curves.to_numpy(), axis=1) <= 0).all()
    # the integrated Brier score holds its own times, and asks the output for its survival function
    scores = pipe.evaluate(test, metrics={"ibs": IntegratedBrier(times), "c": HarrellC()})
    assert 0.0 <= scores["ibs"] <= 1.0


def test_predict_output_is_what_metrics_score(cohort):
    train, test = split(dataset(cohort))
    pipe = pipeline().fit(train)
    output = pipe.predict_output(test)
    assert isinstance(output, CoxOutput) and output.baseline is pipe.model_.endpoint.baseline_
    assert HarrellC()(output, test.target, pipe.train_target_) == pipe.evaluate(test, metrics={"c": HarrellC()})["c"]
    assert list(pipe.train_target_.ids) == pipe.train_ids_


def test_metrics_that_cannot_score_the_model_are_refused_before_predicting_or_training(cohort):
    train, test = split(dataset(cohort))
    pipe = pipeline().fit(train)
    with pytest.raises(TypeError, match="AUROC.* needs a model that outputs ClassScores; this one outputs CoxOutput"):
        pipe.evaluate(test, metrics={"c": HarrellC(), "auroc": AUROC(positive_class="high")})
    stopping = EarlyStopping(metric=AUROC(positive_class="high"), patience=2, restore_best=True)
    built = []

    def optimizer(groups):
        built.append(True)
        return torch.optim.AdamW(groups, lr=1e-2)

    validation = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    with pytest.raises(TypeError, match="needs a model that outputs ClassScores"):
        pipeline(optimizer=optimizer, validation=validation, early_stopping=stopping).fit(train)
    assert not built


def test_a_classifier_has_no_baseline_and_no_survival_function(cohort):
    train, test = split(dataset(cohort, task="classification"))
    pipe = pipeline(late_classifier()).fit(train)
    assert "baseline_events" not in pipe.fit_report_
    message = "needs a model that outputs SurvivalFunction; this one outputs ClassProbabilities"
    with pytest.raises(ValueError, match=f"predict_survival_function {message}"):
        pipe.predict_survival_function(test, [1.0, 2.0])


def test_predict_proba_has_a_column_per_class_in_the_target_order(cohort):
    train, test = split(dataset(cohort, task="classification"))
    pipe = pipeline(late_classifier()).fit(train)
    proba = pipe.predict_proba(test)
    assert list(proba.columns) == ["low", "high"] and proba.index.equals(pipe.predict(test).index)
    np.testing.assert_array_equal(proba.to_numpy(), pipe.predict(test).to_numpy())
    assert list(pipe.predict_proba(test, branch="wsi").columns) == ["low", "high"]
    with pytest.raises(ValueError, match="predict_proba needs a model that outputs ClassProbabilities"):
        pipeline().fit(split(dataset(cohort))[0]).predict_proba(test)


@pytest.mark.parametrize(
    ("times", "match"),
    [([], "finite times >= 0"), ([-1.0, 2.0], "finite times >= 0"), ([np.inf], "finite"), ([5.0, 5.0], "increasing")],
)
def test_survival_function_times_are_checked(cohort, times, match):
    train, test = split(dataset(cohort))
    pipe = pipeline().fit(train)
    with pytest.raises(ValueError, match=match):
        pipe.predict_survival_function(test, times)


def test_integrated_brier_cannot_be_monitored():
    with pytest.raises(ValueError, match=r"IntegratedBrier\(times=\[1.0, 2.0\]\) cannot be monitored"):
        EarlyStopping(metric=IntegratedBrier([1.0, 2.0]), patience=2, restore_best=True)


def test_late_fusion_branches_and_fused_output_have_their_own_survival_curves(cohort):
    train, test = split(dataset(cohort))
    model = LateFusion(
        branches={
            "clinical": Unimodal("clinical", [nn.Linear(3, 4)], CoxEndpoint(nn.Linear(4, 1, bias=False), "efron")),
            "wsi": Unimodal(
                "wsi",
                [ABMIL(in_dim=16, hidden_dim=4, attention_dim=4, dropout=0.0)],
                CoxEndpoint(nn.Linear(4, 1, bias=False), "efron"),
            ),
        },
        fusion=MaskedMean(),
        endpoint=CoxEndpoint(None, "efron"),
    )
    times = inner_times(test)
    pipe = pipeline(model).fit(train)
    # each branch and the fused output have their own baseline, fitted on their own log-hazards
    baselines = [pipe.model_.endpoint.baseline_, *(b.endpoint.baseline_ for b in pipe.model_.branches.values())]
    assert len({id(b) for b in baselines}) == 3
    for branch, baseline in ((None, baselines[0]), ("wsi", baselines[2])):
        log_hazard = pipe.predict(test, branch=branch)["log_hazard"]
        curves = pipe.predict_survival_function(test, times, branch=branch)
        np.testing.assert_allclose(curves.to_numpy(), baseline.survival(log_hazard, times), rtol=1e-12)


def test_late_fusion_without_a_fused_endpoint_has_no_fused_survival_curve(cohort):
    train, test = split(dataset(cohort))
    branches = {
        "clinical": Unimodal("clinical", [nn.Linear(3, 4)], CoxEndpoint(nn.Linear(4, 1, bias=False), "efron")),
        "wsi": Unimodal("wsi", [ABMIL(16, 4, 4, 0.0)], CoxEndpoint(nn.Linear(4, 1, bias=False), "efron")),
    }
    pipe = pipeline(LateFusion(branches, MaskedMean(), None)).fit(train)
    assert "baseline_events" not in pipe.fit_report_
    assert 0.0 <= pipe.evaluate(test, metrics={"c": HarrellC()})["c"] <= 1.0
    with pytest.raises(ValueError, match="outputs SurvivalFunction; this one outputs RiskScore"):
        pipe.predict_survival_function(test, inner_times(test))
    # the branches keep their own endpoints, and so their own survival curves
    assert pipe.predict_survival_function(test, inner_times(test), branch="wsi").notna().all().all()


def test_cross_validation_scores_the_integrated_brier_per_fold(cohort):
    data = dataset(cohort, required=["clinical"])
    result = cross_validate(
        pipeline(intermediate(fusion=MaskedMean(), head_dim=4)),
        data,
        StratifiedKFold(n_splits=2, shuffle=True, random_state=0),
        metrics={"ibs": IntegratedBrier(inner_times(data, quantiles=(0.3, 0.5)))},
    )
    assert result.folds["ibs"].between(0.0, 1.0).all()
    assert (result.folds["baseline_events"] > 0).all()


# ---------------------------------------------------------------- late fusion, encode, attention


def test_late_fusion_branch_predictions_and_metrics(cohort):
    train, test = split(dataset(cohort, task="classification"))
    pipe = pipeline(late_classifier()).fit(train)
    fused = pipe.predict(test)
    wsi = pipe.predict(test, branch="wsi")
    assert list(fused.columns) == list(wsi.columns) == ["probability[low]", "probability[high]"]
    np.testing.assert_allclose(fused.sum(axis=1), 1.0, rtol=1e-5)
    # columns come from the target the pipeline was fitted on, so data without a target is named the same way
    unlabelled = dataset(cohort, task=None).subset(test.ids)
    pd.testing.assert_frame_equal(pipe.predict(unlabelled), fused)
    scores = pipe.evaluate(test, metrics={"auroc": AUROC(positive_class="high"), "bacc": BalancedAccuracy()})
    assert set(scores.index) == {"auroc", "bacc"}
    with pytest.raises(ValueError, match="branch='pathology'"):
        pipe.predict(test, branch="pathology")


def test_encode_is_deterministic_and_attention_aligns_with_coordinates(cohort):
    train, test = split(dataset(cohort))
    pipe = pipeline().fit(train)
    first = pipe.encode(test, modality="wsi")
    pd.testing.assert_frame_equal(first, pipe.encode(test, modality="wsi"))
    assert list(first.index) == list(test.ids) and first.shape[1] == 4

    weights = attention(pipe, test, "wsi")
    assert len(weights) == sum(len(cohort.bags[pid]) for pid in test.ids)
    np.testing.assert_allclose(weights.groupby("patient_id")["attention"].sum(), 1.0, rtol=1e-5)
    patient = weights[weights["patient_id"] == test.ids[0]]
    np.testing.assert_array_equal(patient[["x", "y"]].to_numpy(), cohort.coords[test.ids[0]])

    with pytest.raises(TypeError, match="Attention export needs a BagModality. 'clinical' is a Tabular"):
        attention(pipe, test, "clinical")


class DropFirstInstance(nn.Module):
    def forward(self, bags: list[torch.Tensor]) -> list[torch.Tensor]:
        return [bag[1:] for bag in bags]


def test_attention_rejects_an_encoder_that_changes_the_instances(cohort):
    model = IntermediateFusion(
        encoders={
            "clinical": [nn.Linear(3, 4)],
            "wsi": [DropFirstInstance(), ABMIL(in_dim=16, hidden_dim=6, attention_dim=4, dropout=0.0), nn.Linear(6, 4)],
        },
        fusion=Concat(),
        endpoint=CoxEndpoint(nn.Linear(8, 1, bias=False), "efron"),
    )
    train, test = split(dataset(cohort))
    pipe = pipeline(model).fit(train)
    with pytest.raises(ValueError, match=r"encoders\['wsi'\]\[0\] DropFirstInstance changes the number of instances"):
        attention(pipe, test, "wsi")


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


class ContextIds(FittedEncoder):
    """Passes rows through, and keeps the ids it was fitted on as non-tensor fitted state."""

    def __init__(self):
        super().__init__()
        self.context_ids: list[str] = []

    def fit(self, x, target, ids):
        self.context_ids = list(ids)

    def forward(self, x, ids):
        return x

    def get_extra_state(self) -> dict:
        return {"context_ids": self.context_ids}

    def set_extra_state(self, state: dict) -> None:
        self.context_ids = state["context_ids"]


def test_restoring_the_best_epoch_keeps_non_tensor_fitted_state(cohort):
    train, _ = split(dataset(cohort, required=["clinical"]), n_train=36)
    model = IntermediateFusion(
        encoders={"clinical": [ContextIds(), nn.Linear(3, 4)], "wsi": [ABMIL(16, 4, 4, 0.0)]},
        fusion=MaskedMean(),
        endpoint=CoxEndpoint(nn.Linear(4, 1, bias=False), "efron"),
    )
    pipe = pipeline(
        model,
        max_epochs=3,
        validation=StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0),
        early_stopping=EarlyStopping(metric="loss", patience=2, restore_best=True),
    ).fit(train)
    assert pipe.model_.encoders["clinical"][0].context_ids == pipe.train_ids_
