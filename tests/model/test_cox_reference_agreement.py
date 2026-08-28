"""Cross-check ``neg_partial_log_likelihood`` against lifelines' ``CoxPHFitter``.

lifelines fits its Cox model with Efron-corrected ties (see its
``_get_efron_values_batch``/``_newton_raphson_for_efron_model``), so its
fitted partial log-likelihood is a reference implementation independent of
ours. This test only runs if ``lifelines`` (a dev-only optional dependency,
see ``pyproject.toml``'s ``dev`` extra) is installed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kalecancer.model.predict.losses import neg_partial_log_likelihood

lifelines = pytest.importorskip("lifelines")


def test_matches_lifelines_efron_log_likelihood() -> None:
    pd = pytest.importorskip("pandas")
    from lifelines import CoxPHFitter

    rng = np.random.default_rng(0)
    n_samples = 200
    covariates = rng.standard_normal((n_samples, 2))
    true_coef = np.array([0.8, -0.5])
    risk = covariates @ true_coef

    event_times = rng.exponential(scale=np.exp(-risk))
    censor_times = rng.exponential(scale=2.0, size=n_samples)
    observed_times = np.minimum(event_times, censor_times)
    events = event_times <= censor_times
    # Coarsen to integers so tied event times actually occur.
    integer_times = np.ceil(observed_times * 10).astype(np.int64)
    assert len(np.unique(integer_times)) < n_samples, "test setup must produce tied times"

    frame = pd.DataFrame(
        {
            "time": integer_times,
            "event": events,
            "x1": covariates[:, 0],
            "x2": covariates[:, 1],
        }
    )
    cph = CoxPHFitter()
    cph.fit(frame, duration_col="time", event_col="event")

    fitted_coef = cph.params_.loc[["x1", "x2"]].to_numpy()
    log_hazard = torch.tensor(covariates @ fitted_coef, dtype=torch.float64)
    times = torch.tensor(integer_times, dtype=torch.float64)
    events_t = torch.tensor(events)

    our_log_likelihood = -neg_partial_log_likelihood(log_hazard, times, events_t, reduction="sum").item()

    assert abs(our_log_likelihood - cph.log_likelihood_) < 1e-4
