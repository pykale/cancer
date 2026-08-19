"""Cox proportional-hazards partial likelihood.

Implements the Efron-corrected negative partial log-likelihood used to
train survival heads on (possibly right-censored) time-to-event data.

Boundary rules: this module imports only ``torch`` (and stdlib).
"""

from __future__ import annotations

from typing import Literal, get_args

import torch
from torch import nn

Reduction = Literal["mean", "sum", "none"]
_VALID_REDUCTIONS = get_args(Reduction)


def neg_partial_log_likelihood(
    log_hazard: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Efron-corrected Cox negative partial log-likelihood.

    Subjects are sorted by observed time and the risk-set sum for each
    distinct time is obtained from a single ``logcumsumexp`` pass (no
    ``O(B^2)`` loop over sample pairs). Tied event times are grouped and
    corrected with Efron's approximation, computed via the identity
    ``log(R_k - (l / d_k) * T_k) = log(R_k) + log1p(-(l / d_k) * T_k / R_k)``
    so that the (potentially large) risk-set sum ``R_k`` is never
    exponentiated back out of log-space; only the ratio ``T_k / R_k``,
    which is bounded in ``[0, 1]``, is.

    Args:
        log_hazard: Model output, shape ``(B,)`` or ``(B, 1)``.
        times: Observed (event or censoring) time, shape ``(B,)``.
        events: ``True`` where the event was observed, ``False`` if censored,
            shape ``(B,)``.
        reduction: ``"mean"`` averages over observed events, ``"sum"`` totals
            over observed events, ``"none"`` returns the per-event vector
            (length equal to the number of observed events, in their
            original relative order) whose sum/mean reproduce the other two.

    Returns:
        A scalar tensor for ``"mean"``/``"sum"``, or a ``(num_events,)``
        tensor for ``"none"``.

    Raises:
        TypeError: If dtypes are wrong (``log_hazard``/``times`` not
            floating point, ``events`` not bool).
        ValueError: If shapes are inconsistent, ``reduction`` is not one of
            ``"mean"``, ``"sum"``, ``"none"``, or no event is observed.
    """
    if reduction not in _VALID_REDUCTIONS:
        raise ValueError(f"reduction must be one of {_VALID_REDUCTIONS}, got {reduction!r}")

    if log_hazard.dim() == 2 and log_hazard.shape[1] == 1:
        log_hazard = log_hazard.squeeze(-1)
    if log_hazard.dim() != 1:
        raise ValueError(f"log_hazard must have shape (B,) or (B, 1), got {tuple(log_hazard.shape)}")
    if times.dim() != 1:
        raise ValueError(f"times must have shape (B,), got {tuple(times.shape)}")
    if events.dim() != 1:
        raise ValueError(f"events must have shape (B,), got {tuple(events.shape)}")
    if not (log_hazard.shape[0] == times.shape[0] == events.shape[0]):
        raise ValueError(
            f"log_hazard, times and events must share batch size, "
            f"got {log_hazard.shape[0]}, {times.shape[0]}, {events.shape[0]}"
        )
    if not torch.is_floating_point(log_hazard):
        raise TypeError(f"log_hazard must be a floating-point tensor, got dtype {log_hazard.dtype}")
    if not torch.is_floating_point(times):
        raise TypeError(f"times must be a floating-point tensor, got dtype {times.dtype}")
    if events.dtype != torch.bool:
        raise TypeError(f"events must be a bool tensor, got dtype {events.dtype}")
    if not torch.any(events):
        raise ValueError("neg_partial_log_likelihood requires at least one observed event; all samples are censored")

    device, dtype = log_hazard.device, log_hazard.dtype

    order = torch.argsort(times, descending=True)
    s_log_hazard = log_hazard[order]
    s_events = events[order]
    s_times = times[order]

    # Prefix log-sum-exp in descending-time order: at any position this is
    # log(sum of exp(log_hazard) over everyone with time >= that position's time).
    risk_logcumsumexp = torch.logcumsumexp(s_log_hazard, dim=0)

    _, inverse, counts = torch.unique_consecutive(s_times, return_inverse=True, return_counts=True)
    num_groups = counts.shape[0]
    # A tie group's true risk-set sum is only complete once every member of the
    # group (event or censored) has been folded in, i.e. at the group's last index.
    last_idx = torch.cumsum(counts, dim=0) - 1
    risk_log_per_group = risk_logcumsumexp.index_select(0, last_idx)

    d_k = torch.zeros(num_groups, dtype=dtype, device=device).scatter_add(0, inverse, s_events.to(dtype))

    masked_log_hazard = torch.where(s_events, s_log_hazard, torch.full_like(s_log_hazard, float("-inf")))
    group_max = torch.full((num_groups,), float("-inf"), dtype=dtype, device=device)
    group_max = group_max.scatter_reduce(0, inverse, masked_log_hazard, reduce="amax", include_self=True)
    # Groups with no tied events would otherwise subtract -inf from -inf (nan);
    # substitute a finite placeholder since those groups are never used below.
    safe_group_max = torch.where(d_k > 0, group_max, torch.zeros_like(group_max))
    exp_shifted = torch.exp(masked_log_hazard - safe_group_max.index_select(0, inverse))
    sum_exp = torch.zeros(num_groups, dtype=dtype, device=device).scatter_add(0, inverse, exp_shifted)
    log_tied_hazard = safe_group_max + torch.log(sum_exp.clamp_min(torch.finfo(dtype).tiny))

    ratio = torch.exp(log_tied_hazard - risk_log_per_group).clamp(min=0.0, max=1.0)
    d_k_safe = d_k.clamp_min(1.0)

    total_correction_per_group = torch.zeros(num_groups, dtype=dtype, device=device)
    max_ties = int(d_k.max().item())
    for tie_index in range(max_ties):
        active = d_k > tie_index
        # Mask the log1p argument itself, not just its output: for inactive
        # groups -(tie_index / d_k_safe) * ratio can be <= -1, and log1p of
        # that is nan/-inf, which torch.where would otherwise still route a
        # nan gradient through.
        z = torch.where(active, -(tie_index / d_k_safe) * ratio, torch.zeros_like(ratio))
        term = risk_log_per_group + torch.log1p(z)
        total_correction_per_group = total_correction_per_group + torch.where(active, term, torch.zeros_like(term))

    per_event_correction = total_correction_per_group.index_select(0, inverse) / d_k_safe.index_select(0, inverse)
    nll_sorted = -s_log_hazard + per_event_correction

    full_nll = torch.zeros(log_hazard.shape[0], dtype=dtype, device=device).scatter(0, order, nll_sorted)
    per_event_nll = full_nll[events]

    if reduction == "none":
        return per_event_nll
    if reduction == "sum":
        return per_event_nll.sum()
    return per_event_nll.mean()


class CoxHead(nn.Module):
    """Linear Cox proportional-hazards head.

    Maps a fixed-width embedding to a single log-hazard score. Bias-free:
    the Cox partial likelihood is invariant to an additive shift in
    log-hazard, so a bias term would be unidentifiable and untrained.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, 1, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log-hazard scores.

        Args:
            z: Embeddings, shape ``(B, D)`` with ``D == self.in_features``.

        Returns:
            Log-hazard, shape ``(B, 1)``.

        Raises:
            ValueError: If ``z`` is not 2-D or its last dimension does not
                match ``in_features``.
        """
        if z.dim() != 2 or z.shape[1] != self.in_features:
            raise ValueError(f"CoxHead expects input of shape (B, {self.in_features}), got {tuple(z.shape)}")
        return self.linear(z)

"""Cox proportional-hazards prediction head."""

from __future__ import annotations

import torch
from torch import nn


class CoxHead(nn.Module):
    """Map a patient representation to a log partial hazard.

    The output is the linear predictor of a Cox model: an unbounded score where
    **higher means higher risk** (shorter expected time to event). It has no
    intercept, because the Cox partial likelihood cancels any baseline hazard; use
    :class:`~kalecancer.survival.baseline.BreslowBaselineHazard` to recover absolute
    survival probabilities.

    Args:
        in_features: Dimension of the patient representation.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.risk = nn.Linear(in_features, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict log partial hazards.

        Args:
            x: ``(batch, in_features)`` patient representations.

        Returns:
            ``(batch,)`` log partial hazards.
        """
        return self.risk(x).squeeze(-1)
