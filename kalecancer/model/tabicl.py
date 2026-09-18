"""TabICL v2 as a trainable row encoder conditioned on a fold-local context of training rows."""

from __future__ import annotations

from typing import Literal

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError
from tabicl._model.tabicl import TabICL
from tabicl._sklearn.preprocessing import PreprocessingPipeline
from torch import Tensor

from kalecancer.model.incontext import InContextModule

_REPO = "jingang/TabICL"


_STAGES = {"col": "col_embedder", "row": "row_interactor", "icl": "icl_predictor"}


def _download(filename: str) -> str:
    # Cache first: the file name pins the version, and compute nodes often have no internet.
    try:
        return hf_hub_download(_REPO, filename, local_files_only=True)
    except LocalEntryNotFoundError:
        return hf_hub_download(_REPO, filename)


class TabICLEncoder(InContextModule):
    """Embed table rows with pretrained TabICL, conditioned on the training rows and their labels."""

    keep_weights = True

    def __init__(
        self,
        checkpoint: str,
        output: Literal["row", "icl"],
        trainable: list[Literal["col", "row", "icl"]],
        context_label: Literal["event", "label"],
        context_folds: int,
        random_state: int,
    ):
        super().__init__(context_label, context_folds, random_state)
        if output not in ("row", "icl"):
            raise ValueError(f"output must be 'row' or 'icl', got {output!r}")
        if unknown := set(trainable) - _STAGES.keys():
            raise ValueError(f"unknown trainable stages {sorted(unknown)}; choose from {list(_STAGES)}")
        if output == "row" and "icl" in trainable:
            raise ValueError("output='row' never runs the ICL stage, so 'icl' cannot be trainable")
        self.checkpoint, self.output, self.trainable = checkpoint, output, trainable

        ckpt = torch.load(_download(checkpoint), map_location="cpu", weights_only=True)
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
            getattr(self.tabicl, _STAGES[stage]).requires_grad_(True)
        self.tabicl.icl_predictor.decoder.requires_grad_(False)
        self.out_dim: int = config["embed_dim"] * config["row_num_cls"]
        for name in ("mean", "scale", "lower", "upper"):
            self.register_buffer(name, torch.empty(0))

    def fit(self, x: Tensor, target: dict[str, Tensor], ids: list[str]) -> None:
        super().fit(x, target, ids)
        x = x.detach().cpu().double()
        if not torch.isfinite(x).all():
            raise ValueError("context rows contain NaN or inf; impute them in the modality's transform")
        if constant := (x == x[0]).all(dim=0).nonzero().flatten().tolist():
            raise ValueError(
                f"context columns {constant} are constant; upstream TabICL would drop them silently, "
                "so drop them in the modality's transform"
            )
        classes = self.context_y.unique().tolist()
        if classes[0] < 0 or classes[-1] >= self.tabicl.max_classes:
            raise ValueError(f"context labels must be class ids in [0, {self.tabicl.max_classes}), got {classes}")

        # TabICLClassifier's preprocessing for its "none" ensemble member, fitted on the context only.
        prep = PreprocessingPipeline(normalization_method="none", outlier_threshold=4.0).fit(x.numpy())
        stats = {
            "mean": prep.standard_scaler_.mean_,
            "scale": prep.standard_scaler_.scale_,
            "lower": prep.outlier_remover_.lower_bounds_,
            "upper": prep.outlier_remover_.upper_bounds_,
        }
        for name, value in stats.items():
            setattr(self, name, torch.as_tensor(value, device=self.context_x.device))

    def _normalise(self, x: Tensor) -> Tensor:
        # PreprocessingPipeline("none").transform, in float64 as upstream computes it: in float32 the subtraction
        # loses precision on columns whose mean is large relative to their spread.
        z = ((x.to(self.mean.dtype) - self.mean) / self.scale).clamp(-100, 100)
        z = torch.maximum(-torch.log1p(z.abs()) + self.lower, z)
        return torch.minimum(torch.log1p(z.abs()) + self.upper, z).float()

    def embed(self, context: Tensor, queries: Tensor) -> Tensor:
        # An outer bf16 autocast changes TabICL's outputs materially (predicted probabilities by up to 0.58).
        with torch.autocast(device_type=queries.device.type, enabled=False):
            m, n = self.tabicl, int(context.sum())
            X = self._normalise(torch.cat([self.context_x[context], queries.float()]))[None]
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
