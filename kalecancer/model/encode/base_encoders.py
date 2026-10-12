from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn

from kalecancer.loaddata import BaseTarget


class AttentionEncoder(nn.Module, ABC):
    """An encoder over bags that can report what it attends to, which attention export reads."""

    @abstractmethod
    def attention(self, bags: list[Tensor]) -> list[Tensor]:
        """One tensor per bag whose last dimension is aligned with the bag's instances, e.g. ``(N_i,)`` or
        ``(heads, N_i)``."""


class FittedEncoder(nn.Module, ABC):
    """An encoder fitted on the training patients before training starts.

    Only the first encoder in a list may be fitted: the inputs of later encoders change as the encoders before them
    train. The encoder list fits it on the training patients who have the modality, and passes the patients' ids to
    every ``forward``. Fitted state kept in buffers may have any size; an unfitted encoder loads a fitted one's state.
    """

    @abstractmethod
    def fit(self, x: Any, target: BaseTarget, ids: list[str]) -> None:
        """Fit on the training patients.

        Args:
            x: The loaded, transformed input of ``ids``, as a batch of them would be.
            target: The training dataset's target.
            ids: The training patients who have the modality, in the order of ``x``.
        """

    @abstractmethod
    def forward(self, x: Any, ids: list[str]) -> Tensor:
        """Embed a batch of the patients ``ids``."""

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Fitted buffers have data-dependent sizes: resize so an unfitted encoder can load a fitted one.
        for name, buffer in list(self._buffers.items()):
            if (incoming := state_dict.get(prefix + name)) is not None and buffer is not None:
                setattr(self, name, torch.empty_like(incoming, device=buffer.device))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
