import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

from kalecancer.evaluate import AUROC, BalancedAccuracy, EvalContext, HarrellC, TimeDependentAUC, UnoC


def _ids(prefix: str, n: int) -> pd.Index:
    return pd.Index([f"{prefix}{k:04d}" for k in range(n)])


def _surv(frame: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(frame["event"].to_numpy(), frame["time"].to_numpy(np.float64))


@pytest.fixture
def survival():
    rng = np.random.default_rng(0)
    train_time = rng.integers(1, 80, 300).astype(np.float32)
    train_event = rng.random(300) < 0.6
    # The last training observation is censored, so the training censoring survival G is zero from time 80 on.
    train_time[0], train_event[0] = 80.0, False
    train = pd.DataFrame({"time": train_time, "event": train_event}, index=_ids("tr", 300))

    time = rng.integers(1, 60, 150).astype(np.float32)
    event = rng.random(150) < 0.6
    # An event at the first follow-up time, and events at float32 values just below 3.3 and just above 30.1, where
    # truncating or thresholding at float32(tau or time) would disagree with sksurv.
    time[:3], event[:3] = [1.0, 3.3, 30.1], True
    test = pd.DataFrame({"time": time, "event": event}, index=_ids("te", 150))

    # Rounded, so risks tie as well as times.
    log_hazard = np.round(-time / 20 + rng.normal(0, 1, 150), 1).astype(np.float32)
    prediction = pd.DataFrame({"log_hazard": log_hazard}, index=test.index)
    return train, test, prediction, EvalContext(train_target=train, classes=None)


def _classification(n_classes: int, prefix: str = "probability"):
    rng = np.random.default_rng(1)
    classes = ("alive", "deceased", "lost")[:n_classes]
    codes = rng.integers(0, n_classes, 200)
    logits = rng.normal(size=(200, n_classes)) + np.eye(n_classes)[codes]
    probability = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    ids = _ids("p", 200)
    prediction = pd.DataFrame(probability, index=ids, columns=[f"{prefix}[{c}]" for c in classes])
    target = pd.DataFrame({"label": np.array(classes, dtype=object)[codes]}, index=ids)
    return prediction, target, EvalContext(train_target=target, classes=classes)


def test_arguments_are_stored_under_their_names():
    assert UnoC(tau=5.0).tau == 5.0
    assert TimeDependentAUC(time=5.0).time == 5.0
    assert AUROC(positive_class="deceased").positive_class == "deceased"
    assert repr(UnoC(5.0)) == "UnoC(tau=5.0)"
    assert all(m.higher_is_better for m in (HarrellC(), UnoC(1.0), TimeDependentAUC(1.0), AUROC(1), BalancedAccuracy()))


def test_harrell_c_matches_sksurv(survival):
    _, test, prediction, context = survival
    expected = concordance_index_censored(
        test["event"].to_numpy(), test["time"].to_numpy(np.float64), prediction["log_hazard"].to_numpy()
    )[0]
    assert HarrellC()(prediction, test, context) == pytest.approx(expected, abs=1e-5)


def test_higher_log_hazard_means_higher_risk(survival):
    _, test, _, context = survival
    assert HarrellC()(pd.DataFrame({"log_hazard": -test["time"]}), test, context) > 0.9
    distinct = pd.DataFrame({"log_hazard": np.random.default_rng(2).permutation(150).astype(np.float32)}, test.index)
    c = HarrellC()(distinct, test, context)
    assert HarrellC()(-distinct, test, context) == pytest.approx(1 - c, abs=1e-6)


@pytest.mark.parametrize("tau", [3.3, 20.0, 45.5, 79.5])
def test_uno_c_matches_sksurv(survival, tau):
    train, test, prediction, context = survival
    expected = concordance_index_ipcw(_surv(train), _surv(test), prediction["log_hazard"].to_numpy(), tau=tau)[0]
    assert UnoC(tau)(prediction, test, context) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("time", [1.0, 20.0, 30.1, 45.5])
def test_time_dependent_auc_matches_sksurv(survival, time):
    train, test, prediction, context = survival
    expected = cumulative_dynamic_auc(_surv(train), _surv(test), prediction["log_hazard"].to_numpy(), times=[time])[0]
    assert TimeDependentAUC(time)(prediction, test, context) == pytest.approx(expected[0], abs=1e-5)


@pytest.mark.parametrize("metric", [UnoC(80.0), UnoC(85.0), TimeDependentAUC(80.0), TimeDependentAUC(85.0)], ids=repr)
def test_ipcw_metrics_raise_where_training_censoring_survival_is_zero(survival, metric):
    _, test, prediction, context = survival
    late = test.copy()
    # A test event after the last training time would get weight zero from torchsurv; the follow-up now reaches 90.
    late.loc[late.index[3], ["time", "event"]] = [82.0, True]
    late.loc[late.index[4], ["time", "event"]] = [90.0, False]
    with pytest.raises(ValueError, match=r"below the last training time 80\.0"):
        metric(prediction, late, context)


def test_time_dependent_auc_follow_up_condition_is_sksurv_s(survival):
    train, test, prediction, context = survival
    first, last = float(test["time"].min()), float(test["time"].max())
    outcomes = []
    for time in (first - 0.5, first, last - 0.5, last, last + 0.5):
        try:
            cumulative_dynamic_auc(_surv(train), _surv(test), prediction["log_hazard"].to_numpy(), times=[time])
            outcomes.append(False)
            TimeDependentAUC(time)(prediction, test, context)
        except ValueError as error:
            assert "follow-up time of test data" in str(error)
            outcomes.append(True)
            with pytest.raises(ValueError, match="within the follow-up of the target"):
                TimeDependentAUC(time)(prediction, test, context)
    assert outcomes == [True, False, False, True, True]


def test_undefined_ipcw_metrics_raise(survival):
    _, test, prediction, context = survival
    with pytest.raises(ValueError, match="no comparable pair"):
        UnoC(0.5)(prediction, test, context)
    early_censored = test.copy()
    early_censored.loc[early_censored["time"] <= 2, "event"] = False
    with pytest.raises(ValueError, match="no target event"):
        TimeDependentAUC(1.5)(prediction, early_censored, context)


def test_survival_metrics_emit_no_warnings(survival):
    _, test, prediction, context = survival
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for metric in (HarrellC(), UnoC(45.5), TimeDependentAUC(20.0)):
            metric(prediction, test, context)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_auroc_matches_sklearn_one_vs_rest(n_classes):
    prediction, target, context = _classification(n_classes)
    for k, c in enumerate(context.classes):
        expected = roc_auc_score(target["label"] == c, prediction.iloc[:, k])
        assert AUROC(c)(prediction, target, context) == pytest.approx(expected)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_balanced_accuracy_matches_sklearn_on_argmax(n_classes):
    prediction, target, context = _classification(n_classes, prefix="vote_score")
    classes = np.array(context.classes, dtype=object)
    if n_classes == 2:
        predicted = np.where(prediction.iloc[:, 1] > 0.5, classes[1], classes[0])
    else:
        predicted = classes[prediction.to_numpy().argmax(axis=1)]
    expected = balanced_accuracy_score(target["label"], predicted)
    assert BalancedAccuracy()(prediction, target, context) == pytest.approx(expected)


# scickit-learn warning expected from second `assert`` line
@pytest.mark.filterwarnings("ignore:y_pred contains classes not in y_true")
def test_balanced_accuracy_ties_go_to_the_first_class():
    ids = pd.Index(["a", "b"])
    columns = ["vote_score[alive]", "vote_score[deceased]"]
    prediction = pd.DataFrame([[0.5, 0.5], [0.0, 1.0]], index=ids, columns=columns)
    context = EvalContext(train_target=pd.DataFrame(), classes=("alive", "deceased"))
    assert BalancedAccuracy()(prediction, pd.DataFrame({"label": ["alive", "deceased"]}, index=ids), context) == 1.0
    assert BalancedAccuracy()(prediction, pd.DataFrame({"label": ["deceased", "deceased"]}, index=ids), context) == 0.5


def test_prediction_row_order_does_not_matter(survival):
    _, test, prediction, context = survival
    shuffled = prediction.sample(frac=1.0, random_state=0)
    assert not shuffled.index.equals(prediction.index)
    for metric in (HarrellC(), UnoC(45.5), TimeDependentAUC(20.0)):
        assert metric(shuffled, test, context) == metric(prediction, test, context)

    prediction, target, context = _classification(3)
    shuffled = prediction.sample(frac=1.0, random_state=0)
    for metric in (AUROC("lost"), BalancedAccuracy()):
        assert metric(shuffled, target, context) == metric(prediction, target, context)


def test_ids_must_match(survival):
    _, test, prediction, context = survival
    renamed = prediction.rename(index={prediction.index[0]: "stranger"})
    with pytest.raises(ValueError, match=r"1 target ids lack a prediction .*'te0000'.* 1 predicted ids lack a target"):
        HarrellC()(renamed, test, context)
    with pytest.raises(ValueError, match="prediction ids must be unique"):
        HarrellC()(pd.concat([prediction, prediction.iloc[:1]]), test, context)


def test_nan_predictions_raise(survival):
    _, test, prediction, context = survival
    undefined = prediction.copy()
    undefined.iloc[:3, 0] = np.nan
    with pytest.raises(ValueError, match="3 of 150 predictions are NaN"):
        UnoC(45.5)(undefined, test, context)

    prediction, target, context = _classification(2)
    prediction.iloc[5, 1] = np.nan
    with pytest.raises(ValueError, match="1 of 200 predictions are NaN"):
        BalancedAccuracy()(prediction, target, context)


def test_wrong_target_or_prediction_format_raises(survival):
    train, test, prediction, context = survival
    class_prediction, class_target, class_context = _classification(2)
    with pytest.raises(ValueError, match=r"target needs columns \['time', 'event'\]"):
        HarrellC()(prediction.set_axis(class_target.index[:150]), class_target.iloc[:150], context)
    with pytest.raises(ValueError, match="log_hazard"):
        HarrellC()(class_prediction, class_target, context)
    with pytest.raises(ValueError, match="context.train_target needs columns"):
        UnoC(45.5)(prediction, test, EvalContext(train_target=class_target, classes=None))
    with pytest.raises(ValueError, match=r"classification target needs columns \['label'\]"):
        AUROC("deceased")(class_prediction.iloc[:150].set_axis(test.index), test, class_context)
    with pytest.raises(TypeError, match="must be bool"):
        HarrellC()(prediction, test.astype({"event": int}), context)


def test_classification_guards(survival):
    prediction, target, context = _classification(3)
    with pytest.raises(ValueError, match="not one of classes"):
        AUROC("unknown")(prediction, target, context)
    with pytest.raises(ValueError, match="need context.classes"):
        AUROC("lost")(prediction, target, EvalContext(train_target=target, classes=None))
    for wrong in (prediction[prediction.columns[::-1]], prediction.iloc[:, :2]):
        with pytest.raises(ValueError, match="must be one per class"):
            BalancedAccuracy()(wrong, target, context)
    with pytest.raises(ValueError, match="needs both 'lost' and other labels"):
        AUROC("lost")(prediction, target.assign(label="alive"), context)
    with pytest.raises(ValueError, match="1 target labels are not in classes"):
        AUROC("lost")(prediction, target.assign(label=["dead"] + target["label"].tolist()[1:]), context)
