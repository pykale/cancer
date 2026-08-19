"""Evaluation metrics for survival predictions.

Boundary rules: this module imports only ``numpy`` (and stdlib). Metrics
operate on plain arrays after training/inference, so there is no torch
dependency here.
"""

from __future__ import annotations

import numpy as np


def concordance_index(risk: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """Harrell's concordance index (C-index).

    A pair ``(i, j)`` is comparable when subject ``i`` had an observed
    event and its time is strictly earlier than subject ``j``'s
    (``t_i < t_j``) -- ``j`` may itself be censored or not, its later time
    is known regardless. Among comparable pairs, a pair is concordant if
    the model ranked ``i`` (the earlier event) as higher risk than ``j``;
    ties in risk score count as half a concordant pair. A censored subject
    can only ever be the later member ``j`` of a pair, never the earlier
    reference ``i``.

    Computed via one ``(N, N)`` vectorised broadcast; no Python-level
    ``O(N^2)`` loop.

    Args:
        risk: Predicted risk score, higher means higher hazard, shape
            ``(N,)`` or ``(N, 1)``.
        times: Observed (event or censoring) time, shape ``(N,)``.
        events: ``True`` where the event was observed, ``False`` if
            censored, shape ``(N,)``.

    Returns:
        The concordance index in ``[0, 1]``.

    Raises:
        TypeError: If dtypes are wrong (``risk``/``times`` not floating
            point, ``events`` not bool).
        ValueError: If shapes are inconsistent or there is no comparable
            pair.
    """
    risk = np.asarray(risk)
    times = np.asarray(times)
    events = np.asarray(events)

    if risk.ndim == 2 and risk.shape[1] == 1:
        risk = risk.squeeze(-1)
    if risk.ndim != 1:
        raise ValueError(f"risk must have shape (N,) or (N, 1), got {risk.shape}")
    if times.ndim != 1:
        raise ValueError(f"times must have shape (N,), got {times.shape}")
    if events.ndim != 1:
        raise ValueError(f"events must have shape (N,), got {events.shape}")
    if not (risk.shape[0] == times.shape[0] == events.shape[0]):
        raise ValueError(
            f"risk, times and events must share length, got {risk.shape[0]}, {times.shape[0]}, {events.shape[0]}"
        )
    if not np.issubdtype(risk.dtype, np.floating):
        raise TypeError(f"risk must be a floating-point array, got dtype {risk.dtype}")
    if not np.issubdtype(times.dtype, np.floating):
        raise TypeError(f"times must be a floating-point array, got dtype {times.dtype}")
    if events.dtype != np.bool_:
        raise TypeError(f"events must be a bool array, got dtype {events.dtype}")

    # comparable[i, j]: subject i had an event and is strictly earlier than j.
    comparable = events[:, None] & (times[:, None] < times[None, :])
    n_comparable = comparable.sum()
    if n_comparable == 0:
        raise ValueError("concordance_index requires at least one comparable pair")

    concordant = (risk[:, None] > risk[None, :]).astype(np.float64)
    tied = (risk[:, None] == risk[None, :]).astype(np.float64) * 0.5
    score = np.where(comparable, concordant + tied, 0.0)

    return float(score.sum() / n_comparable)
