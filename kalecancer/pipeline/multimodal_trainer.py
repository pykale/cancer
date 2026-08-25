"""Training pipeline for multimodal survival prediction.

Composes any set of modality embedders with :class:`~kalecancer.model.embed.MultimodalFusion`
and a Cox head, and trains it with the Cox partial-likelihood loss. The fusion stage
and method come from configuration, so switching between feature-level and
prediction-level fusion needs no change here.

Nothing in this module names a modality or a dataset: it reads whatever modalities a
:class:`~kalecancer.loaddata.sample.PatientBatch` carries, which is what lets one
trainer serve tabular, imaging, or any later combination.

Because the Cox risk set spans a mini-batch, batches must be large enough to contain
observed events; batches of purely censored patients carry no gradient and are
skipped. Discrimination is a ranking metric over a whole cohort, so the validation
and test C-index are computed once per epoch over accumulated predictions rather than
averaged over batches.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
from kale.pipeline.base_nn_trainer import BaseNNTrainer
from torch import nn

from kalecancer.loaddata.sample import PatientBatch
from kalecancer.model.embed.multimodal_fusion import MultimodalFusion, multimodal_cox_loss
from kalecancer.survival.cox import CoxHead, as_event_mask, has_risk_set
from kalecancer.survival.metrics import concordance_index

logger = logging.getLogger(__name__)

TIME_KEY = "time"
EVENT_KEY = "event"


class MultimodalSurvivalTrainer(BaseNNTrainer):
    """Multimodal fusion + Cox proportional-hazards trainer.

    Args:
        embedders: One module per modality, each exposing ``out_dim``. Built by the
            caller, so this trainer stays independent of any particular modality.
        stage: Fusion stage, see :class:`~kalecancer.model.embed.MultimodalFusion`.
        method: Fusion method, see :data:`~kalecancer.model.embed.FUSION_METHODS`.
        fusion_dim: Width every modality is projected to.
        auxiliary_weight: Weight on the per-modality loss, for stages that produce
            per-modality predictions.
        modality_dropout: Probability of dropping each present modality in training.
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
        optimizer: dict | None = None,
        max_epochs: int = 50,
        init_lr: float = 1e-4,
        **fusion_kwargs,
    ) -> None:
        super().__init__(optimizer=optimizer, max_epochs=max_epochs, init_lr=init_lr)

        self.model = MultimodalFusion(
            embedders,
            CoxHead,
            stage=stage,
            method=method,
            fusion_dim=fusion_dim,
            modality_dropout=modality_dropout,
            **fusion_kwargs,
        )
        self.auxiliary_weight = auxiliary_weight
        self._epoch_outputs: dict[str, list[torch.Tensor]] = {}

    def forward(self, batch: PatientBatch):
        """Predict from every modality the batch carries."""
        modalities = {name: _to_device(value, self.device) for name, value in batch.modalities.items()}
        present = {name: value.to(self.device) for name, value in batch.present.items()}
        return self.model(modalities, present)

    def compute_loss(self, batch: PatientBatch, split_name: str = "valid") -> tuple[torch.Tensor, dict]:
        """Cox partial-likelihood loss over the batch's risk set."""
        times = batch.target[TIME_KEY].to(self.device)
        events = batch.target[EVENT_KEY].to(self.device)

        output = self.forward(batch)
        loss = multimodal_cox_loss(output, times, events, auxiliary_weight=self.auxiliary_weight)

        if split_name != "train":
            self._accumulate(split_name, output.prediction.reshape(-1), events, times)
        return loss, {f"{split_name}_loss": loss}

    def training_step(self, batch: PatientBatch, batch_idx: int) -> torch.Tensor | None:
        if not has_risk_set(batch.target[EVENT_KEY]):
            logger.debug("skipping batch %d: no observed events, so no Cox risk set", batch_idx)
            return None
        loss, log_metrics = self.compute_loss(batch, split_name="train")
        self.log_dict(log_metrics, on_step=False, on_epoch=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch: PatientBatch, batch_idx: int) -> None:
        self._evaluation_step(batch, "valid")

    def test_step(self, batch: PatientBatch, batch_idx: int) -> None:
        self._evaluation_step(batch, "test")

    def on_validation_epoch_end(self) -> None:
        self._log_concordance("valid")

    def on_test_epoch_end(self) -> None:
        self._log_concordance("test")

    def predict_risk(self, batch: PatientBatch):
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
        if has_risk_set(batch.target[EVENT_KEY]):
            self.log_dict(log_metrics, on_step=False, on_epoch=True, batch_size=len(batch))

    def _accumulate(self, split_name: str, risk: torch.Tensor, events: torch.Tensor, times: torch.Tensor) -> None:
        store = self._epoch_outputs.setdefault(split_name, [])
        store.append(torch.stack([risk.detach(), events.detach().float(), times.detach()]).cpu())

    def _log_concordance(self, split_name: str) -> None:
        """Compute the C-index over the epoch's accumulated predictions."""
        outputs = self._epoch_outputs.pop(split_name, [])
        if not outputs:
            return

        risk, events, times = torch.cat(outputs, dim=1)
        if not has_risk_set(events):
            logger.warning("no events in the %s split; skipping the C-index", split_name)
            return

        index = concordance_index(risk.double().numpy(), times.double().numpy(), as_event_mask(events).numpy())
        self.log(f"{split_name}_c_index", index, prog_bar=True)


def _to_device(value, device):
    """Move a modality to the device, whether it is a tensor or a list of ragged bags."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return [item.to(device) for item in value]
