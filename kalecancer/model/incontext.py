"""In-context stages: embed rows against a fold-local context of labelled training rows."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import Tensor, nn


class InContextModule(nn.Module):
    """A stage conditioned on the training rows and their labels, which ``fit`` stores before training.

    A row whose id is in the context is embedded against the context minus its own stratified fold; any other id sees
    the whole context. Subclasses set ``out_dim`` and implement ``embed``.
    """

    needs_ids = True

    context_x: Tensor
    context_y: Tensor
    context_fold: Tensor

    out_dim: int

    def __init__(self, context_label: Literal["event", "label"], context_folds: int, random_state: int):
        super().__init__()
        if context_label not in ("event", "label"):
            raise ValueError(f"context_label must be 'event' or 'label', got {context_label!r}")
        self.context_label, self.context_folds, self.random_state = context_label, context_folds, random_state
        for name in ("context_x", "context_y", "context_fold"):
            self.register_buffer(name, torch.empty(0))
        self._set_context_ids([])

    def fit(self, x: Tensor, target: dict[str, Tensor], ids: list[str]) -> None:
        if self.context_label not in target:
            raise ValueError(
                f"context_label={self.context_label!r} needs a target with {self.context_label!r}, "
                f"but this target has {sorted(target)}"
            )
        labels = target[self.context_label].detach().cpu().long()
        x = x.detach()
        if x.ndim != 2 or len(x) == 0 or len(x) != len(ids) or len(labels) != len(ids):
            raise ValueError(
                f"expected a non-empty (n, F) context aligned with {len(ids)} ids, "
                f"got rows {tuple(x.shape)} and labels {tuple(labels.shape)}"
            )
        classes, counts = np.unique(labels.numpy(), return_counts=True)
        if (rare := counts < self.context_folds).any():
            raise ValueError(
                f"context classes {classes[rare].tolist()} have fewer rows than context_folds={self.context_folds}, "
                "so some fold-excluded contexts would lack them"
            )
        folds = np.empty(len(labels), dtype=np.int64)
        splitter = StratifiedKFold(self.context_folds, shuffle=True, random_state=self.random_state)
        for fold, (_, rows) in enumerate(splitter.split(labels.numpy(), labels.numpy())):
            folds[rows] = fold
        device = self.context_x.device
        self.context_x, self.context_y = x.to(device), labels.to(device)
        self.context_fold = torch.as_tensor(folds, device=device)
        self._set_context_ids(ids)

    def embed(self, context: Tensor, queries: Tensor) -> Tensor:
        """Embed ``queries`` (n, F) against the context rows selected by the boolean mask ``context``.

        Queries must not attend to each other: ``forward`` embeds each fold's rows in one call.
        """
        raise NotImplementedError

    def forward(self, x: Tensor, ids: list[str]) -> Tensor:
        if not self.context_ids:
            raise RuntimeError(f"{type(self).__name__}.fit must be called before forward")
        if x.ndim != 2 or x.shape[1] != self.context_x.shape[1] or len(x) != len(ids):
            raise ValueError(f"expected x of shape ({len(ids)}, {self.context_x.shape[1]}), got {tuple(x.shape)}")
        position = torch.tensor([self._position.get(pid, -1) for pid in ids], dtype=torch.long, device=x.device)
        known = position >= 0
        differs = torch.zeros_like(known)
        differs[known] = (x[known] != self.context_x[position[known]]).any(dim=1)
        if differs.any():
            bad = [ids[k] for k in differs.nonzero().flatten().tolist()]
            raise ValueError(
                f"ids {bad[:5]} are context ids but their rows differ from the context rows: "
                "an id shared with another cohort, or a different transform"
            )
        # A training row with itself in the context reads its own label, and a context minus exactly the batch
        # leaks labels through its class mix. Excluding the row's whole stratified fold avoids both.
        fold = torch.where(known, self.context_fold[position.clamp(min=0)], -1)
        out = x.new_empty(len(ids), self.out_dim)
        for f in fold.unique().tolist():
            rows = fold == f
            out[rows] = self.embed(self.context_fold != f, x[rows])
        return out

    def _set_context_ids(self, ids: list[str]) -> None:
        self.context_ids = list(ids)
        self._position = {pid: k for k, pid in enumerate(self.context_ids)}

    def get_extra_state(self) -> dict:
        return {"context_ids": self.context_ids}

    def set_extra_state(self, state: dict) -> None:
        self._set_context_ids(state["context_ids"])

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Fitted buffers have data-dependent sizes: resize so an unfitted module can load a fitted one.
        for name, buffer in list(self._buffers.items()):
            if (incoming := state_dict.get(prefix + name)) is not None and buffer is not None:
                setattr(self, name, torch.empty_like(incoming, device=buffer.device))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
