"""Breslow baseline hazard, turning Cox risk scores into survival probabilities.

A Cox model predicts a *relative* risk; the partial likelihood deliberately cancels
the baseline hazard. Absolute survival probabilities - needed for the Brier score -
require estimating that baseline separately.

The estimator must be fitted on **training** data only. Fitting it on the evaluation
set would leak the test outcome distribution into the predictions being scored.
"""

from __future__ import annotations

import torch

from kalecancer.survival.loss import as_event_mask


class BreslowBaselineHazard:
    """Breslow estimator of the baseline cumulative hazard.

    For event times :math:`t_i` with :math:`d_i` events and risk set :math:`R(t_i)`,

    .. math::
        H_0(t) = \\sum_{t_i \\le t} \\frac{d_i}{\\sum_{j \\in R(t_i)} \\exp(\\eta_j)}

    and the survival function follows as
    :math:`S(t \\mid x) = \\exp(-H_0(t)\\exp(\\eta_x))`.
    """

    def __init__(self) -> None:
        self.event_times: torch.Tensor | None = None
        self.cumulative_hazard: torch.Tensor | None = None

    def fit(self, log_hz: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> BreslowBaselineHazard:
        """Estimate the baseline cumulative hazard from training predictions.

        Args:
            log_hz: ``(n,)`` training log partial hazards.
            event: ``(n,)`` indicator, ``1`` observed and ``0`` censored.
            time: ``(n,)`` training event or censoring times.

        Returns:
            ``self``.

        Raises:
            ValueError: If the training data contain no observed events.
        """
        log_hz = log_hz.detach().flatten().double()
        time = time.detach().flatten().double()
        mask = as_event_mask(event).detach().flatten()

        if not mask.any():
            raise ValueError("cannot estimate a baseline hazard without observed events")

        event_times = torch.unique(time[mask])
        hazards = torch.exp(log_hz)

        # At each event time: events observed there, divided by the total hazard of
        # everyone still at risk (i.e. with follow-up at least that long).
        at_risk = time.unsqueeze(0) >= event_times.unsqueeze(1)
        num_events = (time[mask].unsqueeze(0) == event_times.unsqueeze(1)).sum(dim=1)
        risk_totals = (at_risk * hazards.unsqueeze(0)).sum(dim=1)

        self.event_times = event_times
        self.cumulative_hazard = torch.cumsum(num_events / risk_totals.clamp_min(torch.finfo(torch.double).tiny), dim=0)
        return self

    def baseline_cumulative_hazard(self, new_time: torch.Tensor) -> torch.Tensor:
        """Evaluate the step-function baseline cumulative hazard at ``new_time``."""
        if self.event_times is None or self.cumulative_hazard is None:
            raise RuntimeError("BreslowBaselineHazard.fit must be called before evaluation")

        new_time = new_time.detach().flatten().double()
        # Number of fitted event times at or before each query time indexes the step.
        steps = torch.searchsorted(self.event_times, new_time, right=True)
        padded = torch.cat([torch.zeros(1, dtype=torch.double), self.cumulative_hazard])
        return padded[steps]

    def survival_function(self, log_hz: torch.Tensor, new_time: torch.Tensor) -> torch.Tensor:
        """Predict survival probabilities.

        Args:
            log_hz: ``(n,)`` log partial hazards for the patients being scored.
            new_time: ``(t,)`` times at which to evaluate survival.

        Returns:
            ``(n, t)`` survival probabilities.
        """
        baseline = self.baseline_cumulative_hazard(new_time)
        hazards = torch.exp(log_hz.detach().flatten().double())
        return torch.exp(-baseline.unsqueeze(0) * hazards.unsqueeze(1)).float()
