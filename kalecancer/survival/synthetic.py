"""Synthetic survival data with known ground truth.

Generates embeddings, censored event times, and event indicators from a
linear Cox model so that downstream components (losses, heads, metrics)
can be tested against a known answer before any real data exists.

Boundary rules: this module imports only ``torch`` (and stdlib).
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class SyntheticSurvival(NamedTuple):
    """Container for one synthetic cohort.

    Attributes:
        embeddings: Float tensor of shape ``(n, d)``; stand-in for encoder or fusion output.
        times: Float tensor of shape ``(n,)``; observed time = min(event time, censoring time).
        events: Bool tensor of shape ``(n,)``; True if the event was observed, False if censored.
        true_risk: Float tensor of shape ``(n,)``; the true log-hazard used to generate times.
        weights: Float tensor of shape ``(d,)``; the true linear coefficients.
    """

    embeddings: torch.Tensor
    times: torch.Tensor
    events: torch.Tensor
    true_risk: torch.Tensor
    weights: torch.Tensor


def make_synthetic_survival(
    n_samples: int = 512,
    n_features: int = 16,
    baseline_rate: float = 1.0e-3,
    censoring_rate: float = 1.5e-3,
    risk_scale: float = 0.5,
    seed: int | None = 0,
) -> SyntheticSurvival:
    """Sample a cohort from a linear Cox proportional-hazards model.

    Event times follow an exponential distribution with hazard
    ``baseline_rate * exp(risk)`` where ``risk = embeddings @ weights``.
    Censoring times are drawn independently from an exponential with rate
    ``censoring_rate``; the observed time is the minimum of the two.

    Args:
        n_samples: Number of patients ``n``.
        n_features: Embedding dimension ``d``.
        baseline_rate: Baseline hazard of the event-time distribution.
        censoring_rate: Rate of the independent censoring distribution.
            Larger values censor more patients.
        risk_scale: Standard deviation of the true coefficient vector;
            controls how strongly embeddings determine survival.
        seed: Seed for reproducibility. ``None`` draws fresh randomness.

    Returns:
        A :class:`SyntheticSurvival` tuple.
    """
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2, got {n_samples}")
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    embeddings = torch.randn(n_samples, n_features, generator=generator)
    weights = risk_scale * torch.randn(n_features, generator=generator)
    true_risk = embeddings @ weights

    uniform_event = torch.rand(n_samples, generator=generator).clamp_min(1.0e-12)
    event_times = -torch.log(uniform_event) / (baseline_rate * torch.exp(true_risk))

    uniform_censor = torch.rand(n_samples, generator=generator).clamp_min(1.0e-12)
    censor_times = -torch.log(uniform_censor) / censoring_rate

    times = torch.minimum(event_times, censor_times)
    events = event_times <= censor_times

    return SyntheticSurvival(
        embeddings=embeddings,
        times=times,
        events=events,
        true_risk=true_risk,
        weights=weights,
    )
