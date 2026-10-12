"""Fitted encoders: encoders fitted on the training patients before training, such as TabICL's in-context encoder."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from sklearn.model_selection import StratifiedKFold
from tabicl._model.tabicl import TabICL
from tabicl._sklearn.preprocessing import PreprocessingPipeline
from torch import Tensor

from kalecancer.loaddata.targets import BaseTarget
from kalecancer.model.encode.base_encoders import FittedEncoder


class TabICLEncoder(FittedEncoder):
    """Embed table rows with pretrained TabICL, conditioned on the training rows and their labels.

    The context is the training rows, labelled with the target's strata.
    A training patient is embedded against the context without its own stratified fold, and any other
    patient against the whole context. This is too avoid data leakage, as removing just the patient
    from the context would allow the model to predict the patients outcome based purely on the other
    patients outcomes. For example, if there are 30 positive outcomes total, but when embedding a given
    patient there are 29 positive outcomes in the context remaining, the model can infer that the
    patient has a positive outcome (purely as it knows the total number of outcomes).

    Args:
        checkpoint: A TabICL classifier checkpoint file in the ``jingang/TabICL`` Hugging Face repository.
        output: ``"row"`` for the row embeddings, ``"icl"`` for the in-context transformer's output.
        trainable: The TabICL stages to train; the others stay frozen.
        context_folds: The number of stratified folds the context is split into.
        random_state: Seeds the fold assignment.
    """

    @property
    def _REPO(self) -> str:
        return "jingang/TabICL"

    @property
    def _STAGES(self):
        return {"col": "col_embedder", "row": "row_interactor", "icl": "icl_predictor"}

    context_x: Tensor
    context_z: Tensor
    context_y: Tensor
    context_fold: Tensor
    mean: Tensor
    scale: Tensor
    lower: Tensor
    upper: Tensor

    def __init__(
        self,
        checkpoint: str,
        output: Literal["row", "icl"],
        trainable: list[Literal["col", "row", "icl"]],
        context_folds: int = 5,
        random_state: int = 0,
    ):
        super().__init__()
        if output not in ("row", "icl"):
            raise ValueError(f"output must be 'row' or 'icl', got {output!r}")
        if unknown := set(trainable) - self._STAGES.keys():
            raise ValueError(f"unknown trainable stages {sorted(unknown)}. Choose from {list(self._STAGES)}")
        if output == "row" and "icl" in trainable:
            raise ValueError("output='row' never runs the ICL stage, so 'icl' cannot be trainable")
        if context_folds < 2:
            raise ValueError(f"context_folds must be at least 2, got {context_folds}")

        self.checkpoint, self.output, self.trainable = checkpoint, output, trainable
        self.context_folds, self.random_state = context_folds, random_state

        try:
            file = hf_hub_download(self._REPO, checkpoint, local_files_only=True)
        except LocalEntryNotFoundError:
            file = hf_hub_download(self._REPO, checkpoint)

        ckpt = torch.load(file, map_location="cpu", weights_only=True)
        config = ckpt["config"]
        if config["dropout"] != 0:
            raise ValueError(
                f"{checkpoint} has dropout={config['dropout']}; forward runs TabICL's train-mode branches, "
                "which equal its inference branches only without dropout"
            )
        if config["max_classes"] == 0:
            raise ValueError(f"{checkpoint} is a regressor; only classifier checkpoints (class context labels) work")
        self.tabicl = TabICL(**config)
        self.tabicl.load_state_dict(ckpt["state_dict"])
        self.tabicl.requires_grad_(False)
        for stage in trainable:
            getattr(self.tabicl, self._STAGES[stage]).requires_grad_(True)
        self.tabicl.icl_predictor.decoder.requires_grad_(False)

        # context_x is the rows as given, which the id check compares against; context_z is them normalised.
        for name in ("context_x", "context_z", "context_y", "context_fold", "mean", "scale", "lower", "upper"):
            self.register_buffer(name, torch.empty(0))
        self._set_context_ids([])

    def fit(self, x: Tensor, target: BaseTarget, ids: list[str]) -> None:
        x = x.detach()
        if x.ndim != 2 or len(x) == 0 or len(x) != len(ids):
            raise ValueError(f"expected a non-empty (n, F) context aligned with {len(ids)} ids, got {tuple(x.shape)}")
        rows = x.cpu().double()
        if not torch.isfinite(rows).all():
            raise ValueError("context rows contain NaN or inf; impute them in the modality's transform")
        if constant := (rows == rows[0]).all(dim=0).nonzero().flatten().tolist():
            raise ValueError(
                f"context columns {constant} are constant; upstream TabICL would drop them silently, "
                "so drop them in the modality's transform"
            )
        classes, labels, counts = np.unique(np.asarray(target.strata(ids)), return_inverse=True, return_counts=True)
        if len(classes) > self.tabicl.max_classes:
            raise ValueError(
                f"the context has {len(classes)} classes, but {self.checkpoint} handles at most "
                f"{self.tabicl.max_classes}"
            )
        if (rare := counts < self.context_folds).any():
            raise ValueError(
                f"context classes {classes[rare].tolist()} have fewer rows than context_folds={self.context_folds}, "
                "so some fold-excluded contexts would lack them"
            )

        # see the class dcostring for explainer on why a splitter is required
        folds = np.empty(len(labels), dtype=np.int64)
        splitter = StratifiedKFold(self.context_folds, shuffle=True, random_state=self.random_state)
        for fold, (_, members) in enumerate(splitter.split(labels, labels)):
            folds[members] = fold

        # TabICLClassifier's preprocessing for its "none" ensemble member, fitted on the context only.
        prep = PreprocessingPipeline(normalization_method="none", outlier_threshold=4.0).fit(rows.numpy())
        device = self.context_x.device
        self.mean = torch.as_tensor(prep.standard_scaler_.mean_, device=device)
        self.scale = torch.as_tensor(prep.standard_scaler_.scale_, device=device)
        self.lower = torch.as_tensor(prep.outlier_remover_.lower_bounds_, device=device)
        self.upper = torch.as_tensor(prep.outlier_remover_.upper_bounds_, device=device)
        self.context_x = x.to(device)
        self.context_z = self._normalise(self.context_x)
        self.context_y = torch.as_tensor(labels, device=device)
        self.context_fold = torch.as_tensor(folds, device=device)
        self._set_context_ids(ids)

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
        queries = self._normalise(x)
        out = queries.new_empty(len(ids), self.tabicl.embed_dim * self.tabicl.row_num_cls)
        for f in fold.unique().tolist():
            rows = fold == f
            out[rows] = self._embed(self.context_fold != f, queries[rows])
        return out

    def _normalise(self, x: Tensor) -> Tensor:
        # PreprocessingPipeline("none").transform, in float64 as upstream computes it: in float32 the subtraction
        # loses precision on columns whose mean is large relative to their spread.
        z = ((x.to(self.mean.dtype) - self.mean) / self.scale).clamp(-100, 100)
        z = torch.maximum(-torch.log1p(z.abs()) + self.lower, z)
        return torch.minimum(torch.log1p(z.abs()) + self.upper, z).float()

    def _embed(self, context: Tensor, queries: Tensor) -> Tensor:
        """Embed the normalised ``queries`` against the context rows selected by the boolean mask ``context``.

        Queries never attend to each other, so a patient's embedding does not depend on the rest of its batch.
        """
        # An outer bf16 autocast changes TabICL's outputs materially (predicted probabilities by up to 0.58).
        with torch.autocast(device_type=queries.device.type, enabled=False):
            m, n = self.tabicl, int(context.sum())
            X = torch.cat([self.context_z[context], queries])[None]
            y = self.context_y[context].float()[None]
            # TabICL's public forward dispatches on self.training, and its inference branch runs under no_grad on
            # CUDA. For a dropout-free checkpoint the train branches compute the same function, in any module mode.
            R = m.row_interactor._train_forward(m.col_embedder._train_forward(X, y, d=None, embed_with_test=False))
            if self.output == "row":
                return R[0, n:]
            icl = m.icl_predictor
            R = torch.cat([R[:, :n] + icl.y_encoder(y), R[:, n:]], dim=1)
            H = icl.tf_icl(R, train_size=n)
            return (icl.ln(H) if icl.norm_first else H)[0, n:]

    def _set_context_ids(self, ids: list[str]) -> None:
        self.context_ids = list(ids)
        self._position = {pid: k for k, pid in enumerate(self.context_ids)}

    def get_extra_state(self) -> dict:
        return {"context_ids": self.context_ids}

    def set_extra_state(self, state: dict) -> None:
        self._set_context_ids(state["context_ids"])
