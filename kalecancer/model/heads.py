"""Prediction heads: output, prediction, loss and target compatibility for each task."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Final, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kalecancer.loaddata.targets import BaseTarget, Classification, TimeToEvent

with warnings.catch_warnings():
    # torchsurv applies torch.jit.script when imported, which recent torch deprecates; users cannot act on it
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.* is deprecated", category=FutureWarning)
    from torchsurv.loss.cox import neg_partial_log_likelihood


def _has_comparable_event(time: Tensor, event: Tensor) -> bool:
    if not bool(event.any()):
        return False
    at_risk = (time.unsqueeze(0) >= time[event].unsqueeze(1)).sum(dim=1)
    return bool((at_risk > 1).any())


class CoxHead(nn.Module):
    """Linear log-hazard trained with the Cox partial likelihood over the patients in each batch."""

    target_type: Final[type[TimeToEvent]] = TimeToEvent
    decomposable_loss = False

    def __init__(self, in_dim: int, ties: Literal["efron", "breslow"]):
        super().__init__()
        if ties not in ("efron", "breslow"):
            raise ValueError(f"ties must be 'efron' or 'breslow', got {ties!r}")
        self.in_dim = in_dim
        self.ties = ties
        self.out_dim = 1
        # the partial likelihood is invariant to an additive constant, so a bias would never be learned
        self.linear = nn.Linear(in_dim, 1, bias=False)

    def forward(self, z: Tensor) -> Tensor:
        return self.linear(z)

    def predict(self, output: Tensor) -> Tensor:
        return output

    def columns(self, target: TimeToEvent) -> list[str]:
        return ["log_hazard"]

    def check_target(self, target: BaseTarget) -> None:
        if not isinstance(target, self.target_type):
            raise TypeError(f"CoxHead needs a time-to-event target, got {type(target)}")

    def loss(self, output: Tensor, target: Mapping[str, Tensor]) -> Tensor | None:
        """Negative partial log-likelihood averaged over events; ``None`` when no event has anyone else at risk."""
        event, time = target["event"], target["time"]
        if event.dtype != torch.bool:
            raise TypeError(f"target 'event' must be bool, got {event.dtype}")
        with torch.autocast(device_type=output.device.type, enabled=False):
            log_hazard, time = output.float().reshape(-1), time.float()
            if not bool(torch.isfinite(log_hazard).all()):
                raise FloatingPointError("Cox loss received non-finite log-hazards: the model has diverged")
            if not _has_comparable_event(time, event):
                return None
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*[Tt]ies.*")
                # torchsurv's reduction="mean" averages over distinct event times when there are ties
                total = neg_partial_log_likelihood(log_hazard, event, time, ties_method=self.ties, reduction="sum")
            if not torch.isfinite(total):
                raise FloatingPointError("Cox loss is not finite: log-hazards have diverged")
            return total / event.sum()


class ClassificationHead(nn.Module):
    """Linear class logits trained with cross-entropy."""

    target_type: Final[type[Classification]] = Classification
    decomposable_loss = True

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"n_classes must be at least 2, got {n_classes}")
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.out_dim = n_classes
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, z: Tensor) -> Tensor:
        return self.linear(z)

    def predict(self, output: Tensor) -> Tensor:
        return torch.softmax(output.float(), dim=-1)

    def columns(self, target: Classification) -> list[str]:
        return [f"probability[{c}]" for c in target.classes]

    def check_target(self, target: BaseTarget) -> None:
        if not isinstance(target, self.target_type):
            raise TypeError(f"ClassificationHead needs a classification target, got {type(target)}")

        if len(target.classes) != self.n_classes:
            raise ValueError(
                f"ClassificationHead has n_classes={self.n_classes} but the target has classes {target.classes}"
            )

    def loss(self, output: Tensor, target: Mapping[str, Tensor]) -> Tensor | None:
        with torch.autocast(device_type=output.device.type, enabled=False):
            return F.cross_entropy(output.float(), target["label"])
