import dataclasses
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

from kalecancer.evaluate import (
    AUROC,
    BalancedAccuracy,
    HarrellC,
    IntegratedBrier,
    TimeDependentAUC,
    UnoC,
)
from kalecancer.loaddata import Classification, TimeToEvent
from kalecancer.model import (
    BreslowBaseline,
    ClassProbabilities,
    ClassScores,
    CoxOutput,
    RiskScore,
    SurvivalFunction,
    VoteScores,
)


def _ids(prefix: str, n: int) -> pd.Index:
    return pd.Index([f"{prefix}{k:04d}" for k in range(n)])


def _surv(frame: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(frame["event"].to_numpy(), frame["time"].to_numpy(np.float64))


def _tte(frame: pd.DataFrame) -> TimeToEvent:
    return TimeToEvent(frame["time"], frame["event"])


def _risk(prediction: pd.DataFrame, baseline: BreslowBaseline | None = None) -> CoxOutput:
    """The ``log_hazard`` column as a model's output, every patient defined."""
    log_hazard = torch.tensor(prediction["log_hazard"].to_numpy(np.float32))
    defined = torch.ones(len(prediction), dtype=torch.bool)
    return CoxOutput(ids=list(prediction.index), defined=defined, risk=log_hazard, baseline=baseline)


def _undefine(output, rows):
    defined = output.defined.clone()
    defined[rows] = False
    return dataclasses.replace(output, defined=defined)


def _permuted(output, seed=0):
    return output.select(torch.as_tensor(np.random.default_rng(seed).permutation(len(output.ids))))


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
    return train, test, prediction


def _classification(n_classes: int, probabilities: bool = True):
    rng = np.random.default_rng(1)
    classes = ("alive", "deceased", "lost")[:n_classes]
    codes = rng.integers(0, n_classes, 200)
    logits = rng.normal(size=(200, n_classes)) + np.eye(n_classes)[codes]
    probability = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    ids = _ids("p", 200)
    labels = pd.Series(np.array(classes, dtype=object)[codes], index=ids)
    everyone = torch.ones(200, dtype=torch.bool)
    output: ClassScores
    if probabilities:
        # log-probabilities are logits whose softmax is the probabilities
        logits = torch.tensor(np.log(probability), dtype=torch.float32)
        output = ClassProbabilities(ids=list(ids), defined=everyone, logits=logits, classes=classes)
    else:
        votes = torch.tensor(probability, dtype=torch.float32)
        output = VoteScores(ids=list(ids), defined=everyone, votes=votes, classes=classes)
    return output, probability, Classification(labels, classes=classes)


def test_arguments_are_stored_under_their_names():
    assert UnoC(tau=5.0).tau == 5.0
    assert TimeDependentAUC(time=5.0).time == 5.0
    assert AUROC(positive_class="deceased").positive_class == "deceased"
    assert repr(UnoC(5.0)) == "UnoC(tau=5.0)"
    assert all(m.higher_is_better for m in (HarrellC(), UnoC(1.0), TimeDependentAUC(1.0), AUROC(1), BalancedAccuracy()))


def test_harrell_c_matches_sksurv(survival):
    train, test, prediction = survival
    expected = concordance_index_censored(
        test["event"].to_numpy(), test["time"].to_numpy(np.float64), prediction["log_hazard"].to_numpy()
    )[0]
    assert HarrellC()(_risk(prediction), _tte(test), _tte(train)) == pytest.approx(expected, abs=1e-5)


def test_higher_log_hazard_means_higher_risk(survival):
    train, test, _ = survival
    target, reference = _tte(test), _tte(train)
    assert HarrellC()(_risk(pd.DataFrame({"log_hazard": -test["time"]})), target, reference) > 0.9
    distinct = pd.DataFrame({"log_hazard": np.random.default_rng(2).permutation(150).astype(np.float32)}, test.index)
    c = HarrellC()(_risk(distinct), target, reference)
    assert HarrellC()(_risk(-distinct), target, reference) == pytest.approx(1 - c, abs=1e-6)


@pytest.mark.parametrize("tau", [3.3, 20.0, 45.5, 79.5])
def test_uno_c_matches_sksurv(survival, tau):
    train, test, prediction = survival
    expected = concordance_index_ipcw(_surv(train), _surv(test), prediction["log_hazard"].to_numpy(), tau=tau)[0]
    assert UnoC(tau)(_risk(prediction), _tte(test), _tte(train)) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("time", [1.0, 20.0, 30.1, 45.5])
def test_time_dependent_auc_matches_sksurv(survival, time):
    train, test, prediction = survival
    expected = cumulative_dynamic_auc(_surv(train), _surv(test), prediction["log_hazard"].to_numpy(), times=[time])[0]
    assert TimeDependentAUC(time)(_risk(prediction), _tte(test), _tte(train)) == pytest.approx(expected[0], abs=1e-5)


@pytest.mark.parametrize("metric", [UnoC(80.0), UnoC(85.0), TimeDependentAUC(80.0), TimeDependentAUC(85.0)], ids=repr)
def test_ipcw_metrics_raise_where_training_censoring_survival_is_zero(survival, metric):
    train, test, prediction = survival
    late = test.copy()
    # A test event after the last training time would get weight zero from torchsurv; the follow-up now reaches 90.
    late.loc[late.index[3], ["time", "event"]] = [82.0, True]
    late.loc[late.index[4], ["time", "event"]] = [90.0, False]
    with pytest.raises(ValueError, match=r"below the last training time 80\.0"):
        metric(_risk(prediction), _tte(late), _tte(train))


def test_time_dependent_auc_follow_up_condition_is_sksurv_s(survival):
    train, test, prediction = survival
    first, last = float(test["time"].min()), float(test["time"].max())
    outcomes = []
    for time in (first - 0.5, first, last - 0.5, last, last + 0.5):
        try:
            cumulative_dynamic_auc(_surv(train), _surv(test), prediction["log_hazard"].to_numpy(), times=[time])
            outcomes.append(False)
            TimeDependentAUC(time)(_risk(prediction), _tte(test), _tte(train))
        except ValueError as error:
            assert "follow-up time of test data" in str(error)
            outcomes.append(True)
            with pytest.raises(ValueError, match="within the follow-up of the target"):
                TimeDependentAUC(time)(_risk(prediction), _tte(test), _tte(train))
    assert outcomes == [True, False, False, True, True]


def test_undefined_ipcw_metrics_raise(survival):
    train, test, prediction = survival
    with pytest.raises(ValueError, match="no comparable pair"):
        UnoC(0.5)(_risk(prediction), _tte(test), _tte(train))
    early_censored = test.copy()
    early_censored.loc[early_censored["time"] <= 2, "event"] = False
    with pytest.raises(ValueError, match="no target event"):
        TimeDependentAUC(1.5)(_risk(prediction), _tte(early_censored), _tte(train))


def test_survival_metrics_emit_no_warnings(survival):
    train, test, prediction = survival
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for metric in (HarrellC(), UnoC(45.5), TimeDependentAUC(20.0)):
            metric(_risk(prediction), _tte(test), _tte(train))
        IntegratedBrier([5.0, 20.0])(_with_baseline(train, prediction), _tte(test), _tte(train))


@pytest.mark.parametrize("n_classes", [2, 3])
def test_auroc_matches_sklearn_one_vs_rest(n_classes):
    output, probability, target = _classification(n_classes)
    for k, c in enumerate(output.classes):
        expected = roc_auc_score(target.frame["label"] == c, probability[:, k])
        assert AUROC(c)(output, target, target) == pytest.approx(expected)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_balanced_accuracy_matches_sklearn_on_argmax(n_classes):
    output, probability, target = _classification(n_classes, probabilities=False)
    classes = np.array(output.classes, dtype=object)
    if n_classes == 2:
        predicted = np.where(probability[:, 1] > 0.5, classes[1], classes[0])
    else:
        predicted = classes[probability.argmax(axis=1)]
    expected = balanced_accuracy_score(target.frame["label"], predicted)
    assert BalancedAccuracy()(output, target, target) == pytest.approx(expected)


def _labels(ids, labels, classes=("alive", "deceased")):
    return Classification(pd.Series(labels, index=ids), classes=classes)


# scickit-learn warning expected from second `assert`` line
@pytest.mark.filterwarnings("ignore:y_pred contains classes not in y_true")
def test_balanced_accuracy_ties_go_to_the_first_class():
    ids = ["a", "b"]
    scores = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
    output = VoteScores(ids=ids, defined=torch.ones(2, dtype=torch.bool), votes=scores, classes=("alive", "deceased"))
    assert BalancedAccuracy()(output, _labels(ids, ["alive", "deceased"]), _labels(ids, ["alive", "deceased"])) == 1.0
    assert (
        BalancedAccuracy()(output, _labels(ids, ["deceased", "deceased"]), _labels(ids, ["alive", "deceased"])) == 0.5
    )


def test_patient_order_does_not_matter(survival):
    train, test, prediction = survival
    output = _risk(prediction)
    assert _permuted(output).ids != output.ids
    for metric in (HarrellC(), UnoC(45.5), TimeDependentAUC(20.0)):
        assert metric(_permuted(output), _tte(test), _tte(train)) == metric(output, _tte(test), _tte(train))

    output, _, target = _classification(3)
    for metric in (AUROC("lost"), BalancedAccuracy()):
        assert metric(_permuted(output), target, target) == metric(output, target, target)


def test_the_target_may_hold_more_patients_than_the_output(survival):
    train, test, prediction = survival
    part = _risk(prediction.iloc[:100])
    assert HarrellC()(part, _tte(test), _tte(train)) == HarrellC()(part, _tte(test.iloc[:100]), _tte(train))


def test_every_patient_of_the_output_needs_an_outcome_once(survival):
    train, test, prediction = survival
    renamed = _risk(prediction.rename(index={prediction.index[0]: "stranger"}))
    with pytest.raises(ValueError, match=r"1 patients of the output have no outcome in the target .*'stranger'"):
        HarrellC()(renamed, _tte(test), _tte(train))
    output = _risk(prediction)
    twice = output.select(torch.tensor([0, 0, 1]))
    with pytest.raises(ValueError, match="output ids must be unique"):
        HarrellC()(twice, _tte(test), _tte(train))


def test_undefined_patients_raise(survival):
    train, test, prediction = survival
    with pytest.raises(ValueError, match="3 of 150 patients have no prediction"):
        UnoC(45.5)(_undefine(_risk(prediction), [0, 1, 2]), _tte(test), _tte(train))

    output, _, target = _classification(2)
    with pytest.raises(ValueError, match="1 of 200 patients have no prediction"):
        BalancedAccuracy()(_undefine(output, [5]), target, target)


def test_metrics_check_the_output_type_and_the_targets(survival):
    train, test, prediction = survival
    class_output, _, class_target = _classification(2)
    with pytest.raises(TypeError, match=r"HarrellC\(\) needs a model that outputs RiskScore; this one outputs"):
        HarrellC()(class_output, class_target, class_target)
    with pytest.raises(TypeError, match="needs a TimeToEvent target, got Classification"):
        HarrellC()(_risk(prediction.set_axis(class_target.ids[:150])), class_target, class_target)
    with pytest.raises(TypeError, match="needs a TimeToEvent train_target"):
        UnoC(45.5)(_risk(prediction), _tte(test), class_target)
    with pytest.raises(TypeError, match="needs a Classification target"):
        AUROC("deceased")(class_output, _tte(test), _tte(test))
    with pytest.raises(TypeError, match="needs a model that outputs SurvivalFunction"):
        IntegratedBrier([5.0, 20.0]).check_output_type(RiskScore)
    HarrellC().check_output_type(CoxOutput)
    AUROC("alive").check_output_type(ClassProbabilities)
    with pytest.raises(TypeError, match="needs a model that outputs ClassScores"):
        BalancedAccuracy().check_output_type(CoxOutput)


def test_classification_guards():
    output, _, target = _classification(3)
    with pytest.raises(ValueError, match="not one of classes"):
        AUROC("unknown")(output, target, target)
    alive = Classification(pd.Series("alive", index=target.ids), classes=output.classes)
    with pytest.raises(ValueError, match="needs both 'lost' and other labels"):
        AUROC("lost")(output, alive, alive)
    labels = target.frame["label"].copy()
    labels.iloc[0] = "dead"
    wider = Classification(labels, classes=[*output.classes, "dead"])
    with pytest.raises(ValueError, match="1 target labels are not in classes"):
        AUROC("lost")(output, wider, wider)


def _with_baseline(train, prediction):
    """``prediction`` as a Cox output whose baseline is fitted on ``train``.

    The training log-hazards follow the fixture's model for the test ones, so the curves are roughly calibrated.
    """
    log_hazard = -train["time"].to_numpy() / 20 + np.random.default_rng(3).normal(0, 1, len(train))
    return _risk(prediction, BreslowBaseline().fit(log_hazard, train["time"], train["event"].to_numpy()))


@dataclasses.dataclass(kw_only=True)
class _Flat(RiskScore, SurvivalFunction):
    """A survival function of 0.5 everywhere: uninformative, and badly calibrated."""

    def survival(self, times: Sequence[float]) -> torch.Tensor:
        return torch.full((len(self.ids), len(times)), 0.5, dtype=torch.float64)


@pytest.mark.parametrize(
    "times", [[5.0, 20.0, 45.5], [1.0, 3.3, 30.1, 58.0], list(np.linspace(2.0, 55.0, 30))], ids=["3", "edges", "30"]
)
def test_integrated_brier_matches_sksurv(survival, times):
    train, test, prediction = survival
    output = _with_baseline(train, prediction)
    estimate = output.survival(times).numpy()
    expected = integrated_brier_score(_surv(train), _surv(test), estimate, times)
    assert IntegratedBrier(times)(output, _tte(test), _tte(train)) == pytest.approx(expected, abs=1e-6)


def test_integrated_brier_is_lower_for_better_calibration(survival):
    train, test, prediction = survival
    metric = IntegratedBrier([5.0, 20.0, 45.5])
    good = _with_baseline(train, prediction)
    flat = _Flat(ids=good.ids, defined=good.defined, risk=good.risk)
    assert not IntegratedBrier.higher_is_better
    assert metric(good, _tte(test), _tte(train)) < metric(flat, _tte(test), _tte(train))
    assert metric(_permuted(good), _tte(test), _tte(train)) == pytest.approx(metric(good, _tte(test), _tte(train)))


def test_integrated_brier_guards(survival):
    train, test, prediction = survival
    for times in ([20.0], [20.0, 20.0], [30.0, 20.0]):
        with pytest.raises(ValueError, match="at least two strictly increasing times"):
            IntegratedBrier(times)
    assert IntegratedBrier([5.0, 20.0]).times == [5.0, 20.0]
    output = _with_baseline(train, prediction)
    first, last = float(test["time"].min()), float(test["time"].max())
    with pytest.raises(ValueError, match="within the follow-up of the target"):
        IntegratedBrier([first - 0.5, 20.0])(output, _tte(test), _tte(train))
    with pytest.raises(ValueError, match="within the follow-up of the target"):
        IntegratedBrier([20.0, last])(output, _tte(test), _tte(train))
    late = test.copy()
    late.loc[late.index[3], ["time", "event"]] = [90.0, False]
    with pytest.raises(ValueError, match=r"below the last training time 80\.0"):
        IntegratedBrier([20.0, 85.0])(output, _tte(late), _tte(train))
    with pytest.raises(ValueError, match="2 of 150 patients have no prediction"):
        IntegratedBrier([5.0, 20.0])(_undefine(output, [0, 1]), _tte(test), _tte(train))
    with pytest.raises(RuntimeError, match="fitted when training ends"):
        IntegratedBrier([5.0, 20.0])(_risk(prediction), _tte(test), _tte(train))
