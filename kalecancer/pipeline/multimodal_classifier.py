"""Training pipeline for multimodal binary classification.

The classification counterpart of
:class:`~kalecancer.pipeline.MultimodalSurvivalTrainer`, composing the same
:class:`~kalecancer.model.embed.MultimodalFusion` with a linear logit head and
training it with binary cross-entropy. The fusion stage and method come from
configuration, so switching between feature-level and prediction-level fusion needs
no change here.

Nothing in this module names a modality or a dataset: it reads whatever modalities a
:class:`~kalecancer.loaddata.sample.PatientBatch` carries.

Two differences from the survival trainer are worth knowing, because both are
consequences of the objective rather than choices:

* No batch is skipped. Cross-entropy is a per-sample loss, so a batch of purely
  negative patients still carries a gradient, unlike a Cox risk set.
* Batch size is a free hyperparameter. It does not have to be large enough to hold
  events, because no batch-spanning risk set is being formed.

As with the C-index, discrimination is a ranking metric over a whole cohort, so the
validation and test AUC are computed once per epoch over accumulated predictions
rather than averaged over batches.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import partial

import torch
from kale.pipeline.base_nn_trainer import BaseNNTrainer
from torch import nn

from kalecancer.evaluate.classification_metrics import MetricError, roc_auc
from kalecancer.loaddata.sample import PatientBatch
from kalecancer.model.embed.multimodal_fusion import MultimodalFusion, multimodal_bce_loss
from kalecancer.pipeline.multimodal_trainer import _to_device

logger = logging.getLogger(__name__)

LABEL_KEY = "label"

#: One logit per patient. The bias is kept, unlike
#: :class:`~kalecancer.survival.cox.CoxHead`, which drops it because the Cox partial
#: likelihood is shift-invariant and an intercept would be unidentifiable. Here the
#: intercept is what sets the base rate, so it has to be learned.
binary_head = partial(nn.Linear, out_features=1)


class MultimodalClassificationTrainer(BaseNNTrainer):
    """Multimodal fusion + binary classification trainer.

    Args:
        embedders: One module per modality, each exposing ``out_dim``. Built by the
            caller, so this trainer stays independent of any particular modality.
        stage: Fusion stage, see :class:`~kalecancer.model.embed.MultimodalFusion`.
        method: Fusion method, see :data:`~kalecancer.model.embed.FUSION_METHODS`.
        fusion_dim: Width every modality is projected to.
        auxiliary_weight: Weight on the per-modality loss, for stages that produce
            per-modality predictions.
        modality_dropout: Probability of dropping each present modality in training.
        pos_weight: Weight on the positive class, for an imbalanced endpoint. 0
            disables it.
        optimizer: PyKale optimizer spec, e.g. ``{"type": "AdamW", "optim_params": {...}}``.
        max_epochs: Maximum training epochs.
        init_lr: Initial learning rate.
        **fusion_kwargs: Passed to :class:`~kalecancer.model.embed.MultimodalFusion`,
            e.g. ``rank`` or ``combine_predictions``.
    """

    def __init__(
        self,
        embedders: Mapping[str, nn.Module],
        stage: str = "intermediate",
        method: str = "concat",
        fusion_dim: int = 256,
        auxiliary_weight: float = 0.0,
        modality_dropout: float = 0.0,
        pos_weight: float = 0.0,
        optimizer: dict | None = None,
        max_epochs: int = 50,
        init_lr: float = 1e-4,
        **fusion_kwargs,
    ) -> None:
        super().__init__(optimizer=optimizer, max_epochs=max_epochs, init_lr=init_lr)

        self.model = MultimodalFusion(
            embedders,
            binary_head,
            stage=stage,
            method=method,
            fusion_dim=fusion_dim,
            modality_dropout=modality_dropout,
            **fusion_kwargs,
        )
        self.auxiliary_weight = auxiliary_weight
        # Registered as a buffer so it follows the module to its device.
        self.register_buffer("pos_weight", torch.tensor(pos_weight) if pos_weight > 0 else None, persistent=False)
        self._epoch_outputs: dict[str, list[torch.Tensor]] = {}

    def forward(self, batch: PatientBatch):
        """Predict from every modality the batch carries."""
        modalities = {name: _to_device(value, self.device) for name, value in batch.modalities.items()}
        present = {name: value.to(self.device) for name, value in batch.present.items()}
        return self.model(modalities, present)

    def compute_loss(self, batch: PatientBatch, split_name: str = "valid") -> tuple[torch.Tensor, dict]:
        """Binary cross-entropy over the batch."""
        labels = batch.target[LABEL_KEY].to(self.device)

        output = self.forward(batch)
        loss = multimodal_bce_loss(output, labels, auxiliary_weight=self.auxiliary_weight, pos_weight=self.pos_weight)

        if split_name != "train":
            self._accumulate(split_name, output.prediction.reshape(-1), labels)
        return loss, {f"{split_name}_loss": loss}

    def training_step(self, batch: PatientBatch, batch_idx: int) -> torch.Tensor:
        loss, log_metrics = self.compute_loss(batch, split_name="train")
        self.log_dict(log_metrics, on_step=False, on_epoch=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch: PatientBatch, batch_idx: int) -> None:
        self._evaluation_step(batch, "valid")

    def test_step(self, batch: PatientBatch, batch_idx: int) -> None:
        self._evaluation_step(batch, "test")

    def on_validation_epoch_end(self) -> None:
        self._log_auc("valid")

    def on_test_epoch_end(self) -> None:
        self._log_auc("test")

    def predict_logits(self, batch: PatientBatch):
        """Fusion output for a batch, without tracking gradients.

        The training/evaluation mode is restored on return, so calling this during
        training does not silently disable dropout for subsequent steps.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self.forward(batch)
        finally:
            self.train(was_training)

    def _evaluation_step(self, batch: PatientBatch, split_name: str) -> None:
        loss, log_metrics = self.compute_loss(batch, split_name=split_name)
        self.log_dict(log_metrics, on_step=False, on_epoch=True, batch_size=len(batch))

    def _accumulate(self, split_name: str, scores: torch.Tensor, labels: torch.Tensor) -> None:
        store = self._epoch_outputs.setdefault(split_name, [])
        store.append(torch.stack([scores.detach().float(), labels.detach().float()]).cpu())

    def _log_auc(self, split_name: str) -> None:
        """Compute the AUC over the epoch's accumulated predictions."""
        outputs = self._epoch_outputs.pop(split_name, [])
        if not outputs:
            return

        scores, labels = torch.cat(outputs, dim=1)
        try:
            area = roc_auc(labels.numpy(), scores.double().numpy())
        except MetricError as error:
            logger.warning("%s AUC unavailable: %s", split_name, error)
            return
        self.log(f"{split_name}_auc", area, prog_bar=True)
