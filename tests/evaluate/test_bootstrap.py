import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from sksurv.metrics import concordance_index_censored

from kalecancer.evaluate import AUROC, BootstrapCI, HarrellC, bootstrap_ci
from kalecancer.loaddata import Classification, TimeToEvent
from kalecancer.model import ClassProbabilities, CoxOutput


def _ids(n: int) -> pd.Index:
    return pd.Index([f"p{k:04d}" for k in range(n)])


def _risk(log_hazard: pd.Series) -> CoxOutput:
    defined = torch.ones(len(log_hazard), dtype=torch.bool)
    return CoxOutput(ids=list(log_hazard.index), defined=defined, risk=torch.tensor(log_hazard.to_numpy()))


def _probabilities(ids, score) -> ClassProbabilities:
    p = torch.tensor(np.stack([1 - score, score], axis=1), dtype=torch.float32)
    defined = torch.ones(len(ids), dtype=torch.bool)
    return ClassProbabilities(ids=list(ids), defined=defined, logits=p.clamp(min=1e-6).log(), classes=("no", "yes"))


@pytest.fixture
def survival():
    rng = np.random.default_rng(0)
    n = 120
    time = rng.integers(1, 60, n).astype(np.float32)
    event = rng.random(n) < 0.6
    frame = pd.DataFrame({"time": time, "event": event}, index=_ids(n))
    # A strong model and a weak one, rounded so that risks tie.
    strong = np.round(-time / 10 + rng.normal(0, 0.5, n), 1).astype(np.float32)
    weak = np.round(-time / 10 + rng.normal(0, 4.0, n), 1).astype(np.float32)
    # Patients in a different order from the target, which bootstrap_ci must align by id.
    order = rng.permutation(n)
    strong = pd.Series(strong, index=frame.index).iloc[order]
    weak = pd.Series(weak, index=frame.index).iloc[order]
    target = TimeToEvent(frame["time"], frame["event"])
    return target, _risk(strong), _risk(weak), target


def test_matches_sksurv_on_the_same_resamples(survival):
    target, prediction, _, context = survival
    result = bootstrap_ci(HarrellC(), prediction, target, context, n_resamples=200, random_state=3)

    risk = pd.Series(prediction.risk.numpy(), index=prediction.ids).loc[target.ids].to_numpy(np.float64)
    event, time = target.frame["event"].to_numpy(), target.frame["time"].to_numpy(np.float64)
    rng = np.random.default_rng(3)
    values = []
    for _ in range(200):
        rows = rng.integers(0, len(target.ids), size=len(target.ids))
        values.append(concordance_index_censored(event[rows], time[rows], risk[rows])[0])

    assert result.estimate == pytest.approx(concordance_index_censored(event, time, risk)[0], abs=1e-6)
    assert result.lower == pytest.approx(np.percentile(values, 2.5), abs=1e-6)
    assert result.upper == pytest.approx(np.percentile(values, 97.5), abs=1e-6)
    assert (result.n_resamples, result.n_undefined, result.confidence) == (200, 0, 0.95)


def test_is_reproducible_and_seeded(survival):
    target, prediction, _, context = survival
    first = bootstrap_ci(HarrellC(), prediction, target, context, n_resamples=100, random_state=1)
    again = bootstrap_ci(HarrellC(), prediction, target, context, n_resamples=100, random_state=1)
    other = bootstrap_ci(HarrellC(), prediction, target, context, n_resamples=100, random_state=2)
    assert first == again
    assert (first.lower, first.upper) != (other.lower, other.upper)
    assert first.lower <= first.estimate <= first.upper


def test_narrower_confidence_gives_a_nested_interval(survival):
    target, prediction, _, context = survival
    wide = bootstrap_ci(HarrellC(), prediction, target, context, n_resamples=200, confidence=0.95)
    narrow = bootstrap_ci(HarrellC(), prediction, target, context, n_resamples=200, confidence=0.5)
    assert wide.lower <= narrow.lower <= narrow.upper <= wide.upper


def test_paired_difference(survival):
    target, strong, weak, context = survival
    result = bootstrap_ci(HarrellC(), strong, target, context, baseline=weak, n_resamples=300)
    assert result.estimate == pytest.approx(HarrellC()(strong, target, context) - HarrellC()(weak, target, context))
    assert result.lower > 0, "the strong model beats the weak one on every patient sample"

    same = bootstrap_ci(HarrellC(), strong, target, context, baseline=strong, n_resamples=50)
    assert (same.estimate, same.lower, same.upper) == (0.0, 0.0, 0.0)


def test_paired_interval_is_narrower_than_the_unpaired_one(survival):
    # The two models share the time signal, so their resampling errors cancel in the difference.
    target, strong, weak, context = survival
    paired = bootstrap_ci(HarrellC(), strong, target, context, baseline=weak, n_resamples=300)
    a = bootstrap_ci(HarrellC(), strong, target, context, n_resamples=300)
    b = bootstrap_ci(HarrellC(), weak, target, context, n_resamples=300)
    assert paired.upper - paired.lower < (a.upper - a.lower) + (b.upper - b.lower)


def test_undefined_resamples_are_counted_and_warned():
    # One positive among 15 patients: about a third of resamples leave it out, where AUROC is undefined.
    ids = _ids(15)
    target = Classification(pd.Series(["yes"] + ["no"] * 14, index=ids), classes=["no", "yes"])
    prediction, context = _probabilities(ids, np.linspace(1, 0, 15)), target
    with pytest.warns(UserWarning, match=r"undefined on \d+ of 200 resamples"):
        result = bootstrap_ci(AUROC(positive_class="yes"), prediction, target, context, n_resamples=200)
    assert 30 < result.n_undefined < 120
    assert (result.estimate, result.lower, result.upper) == pytest.approx((1.0, 1.0, 1.0))


def test_no_warning_when_every_resample_is_defined(survival):
    target, prediction, _, context = survival
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bootstrap_ci(HarrellC(), prediction, target, context, n_resamples=20)


def test_misuse_raises_before_resampling(survival):
    target, prediction, _, context = survival
    stranger = prediction.select(torch.arange(3))
    stranger.ids[0] = "stranger"
    with pytest.raises(ValueError, match="no outcome in the target"):
        bootstrap_ci(HarrellC(), stranger, target, context)
    with pytest.raises(ValueError, match="exactly the patients of output"):
        bootstrap_ci(HarrellC(), prediction, target, context, baseline=prediction.select(torch.arange(5)))
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(HarrellC(), prediction, target, context, confidence=95)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci(HarrellC(), prediction, target, context, n_resamples=0)


def test_raises_when_undefined_on_every_resample():
    ids = _ids(1)
    target = Classification(pd.Series(["yes"], index=ids), classes=["no", "yes"])
    prediction, context = _probabilities(ids, np.array([0.8])), target
    # Undefined on the patients as given already, so this is the estimate's error, not the resamples'.
    with pytest.raises(ValueError, match="undefined"):
        bootstrap_ci(AUROC(positive_class="yes"), prediction, target, context, n_resamples=5)


def test_str():
    result = BootstrapCI(estimate=0.7, lower=0.65, upper=0.75, confidence=0.95, n_resamples=10, n_undefined=0)
    assert str(result) == "0.7000 [0.6500, 0.7500]"
