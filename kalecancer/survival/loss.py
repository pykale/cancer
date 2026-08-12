"""Cox partial-likelihood loss for time-to-event training."""

from __future__ import annotations

import torch
from torchsurv.loss.cox import neg_partial_log_likelihood


def as_event_mask(event: torch.Tensor) -> torch.Tensor:
    """Normalise an event indicator to the boolean mask the Cox routines require.

    Accepts the ``1 = event observed`` / ``0 = censored`` convention used across
    ``kalecancer`` in any numeric dtype.
    """
    return event.bool() if event.dtype != torch.bool else event


def has_risk_set(event: torch.Tensor) -> bool:
    """Whether a batch contains at least one observed event.

    The Cox partial likelihood is a sum over observed events, so a batch of purely
    censored patients contributes no gradient. Callers should skip such batches
    rather than stepping the optimiser on a constant loss.
    """
    return bool(as_event_mask(event).any())


def cox_ph_loss(
    log_hz: torch.Tensor,
    event: torch.Tensor,
    time: torch.Tensor,
    ties_method: str = "efron",
) -> torch.Tensor:
    """Negative Cox partial log likelihood.

    Censored patients are not discarded: they enter the risk set of every event that
    happens before their censoring time, which is what distinguishes this from a
    regression loss on observed times.

    The risk set spans the given batch, so batches must be large enough to contain
    events. If none do, the loss is zero and carries no gradient - guard with
    :func:`has_risk_set`.

    Args:
        log_hz: ``(batch,)`` log partial hazards, higher meaning higher risk.
        event: ``(batch,)`` indicator, ``1`` observed and ``0`` censored.
        time: ``(batch,)`` event or censoring times.
        ties_method: ``"efron"`` (default) or ``"breslow"``. Efron's approximation is
            more accurate when several patients share an event time, which is common
            with day-resolution follow-up.

    Returns:
        Scalar loss.
    """
    return neg_partial_log_likelihood(
        log_hz,
        as_event_mask(event),
        time.float(),
        ties_method=ties_method,
        reduction="mean",
    )
