"""Training pipeline for WSI survival prediction.

Composes attention MIL over precomputed patch features with a Cox head, and trains
it with the Cox partial-likelihood loss.

Because the Cox risk set spans a mini-batch, batches must be large enough to contain
observed events; batches of purely censored patients carry no gradient and are
skipped. Discrimination is a ranking metric over a whole cohort, so the validation
and test C-index are computed once per epoch over accumulated predictions rather
than averaged over batches.
"""

from __future__ import annotations

import logging

import torch
from kale.pipeline.base_nn_trainer import BaseNNTrainer

from kalecancer.loaddata.wsi_dataset import BagBatch
from kalecancer.model.embed.attention_mil import AttentionMIL
from kalecancer.survival.cox import CoxHead
from kalecancer.survival.loss import cox_ph_loss, has_risk_set
from kalecancer.survival.metrics import concordance_index

logger = logging.getLogger(__name__)


class WSISurvivalTrainer(BaseNNTrainer):
    """Attention MIL + Cox proportional-hazards trainer.

    Args:
        input_dim: Dimension of the precomputed patch features.
        hidden_dim: Width of the patient representation.
        attention_dim: Width of the attention hidden layer.
        dropout: Dropout after the patch projection.
        gated: Use gated attention.
        optimizer: PyKale optimizer spec, e.g. ``{"type": "AdamW", "optim_params": {...}}``.
        max_epochs: Maximum training epochs.
        init_lr: Initial learning rate.
        ties_method: Tie handling for the Cox loss, ``"efron"`` or ``"breslow"``.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.25,
        gated: bool = True,
        optimizer: dict | None = None,
        max_epochs: int = 50,
        init_lr: float = 1e-4,
        ties_method: str = "efron",
    ) -> None:
        super().__init__(optimizer=optimizer, max_epochs=max_epochs, init_lr=init_lr)
        self.save_hyperparameters()

        self.encoder = AttentionMIL(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout,
            gated=gated,
        )
        self.cox_head = CoxHead(self.encoder.output_dim)
        self.ties_method = ties_method
        self._epoch_outputs: dict[str, list[torch.Tensor]] = {}

    def forward(self, bags: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Predict a risk score per patient.

        Args:
            bags: Per-patient ``(num_patches, input_dim)`` feature tensors.

        Returns:
            ``(risk, attentions)`` where ``risk`` is ``(batch,)`` log partial hazards
            and ``attentions`` holds one weight vector per bag, aligned with its patches.
        """
        embeddings, attentions = self.encoder.forward_bags(bags)
        return self.cox_head(embeddings), attentions

    def compute_loss(self, batch: BagBatch, split_name: str = "valid") -> tuple[torch.Tensor, dict]:
        """Cox partial-likelihood loss over the batch's risk set."""
        bags = [sample.features.to(self.device) for sample in batch.samples]
        duration = batch.duration.to(self.device)
        event = batch.event.to(self.device)

        risk, _ = self.forward(bags)
        loss = cox_ph_loss(risk, event, duration, ties_method=self.ties_method)

        # Only the evaluation splits report an epoch-level C-index.
        if split_name != "train":
            self._accumulate(split_name, risk, event, duration)
        return loss, {f"{split_name}_loss": loss}

    def training_step(self, batch: BagBatch, batch_idx: int) -> torch.Tensor | None:
        if not has_risk_set(batch.event):
            logger.debug("skipping batch %d: no observed events, so no Cox risk set", batch_idx)
            return None
        loss, log_metrics = self.compute_loss(batch, split_name="train")
        self.log_dict(log_metrics, on_step=False, on_epoch=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch: BagBatch, batch_idx: int) -> None:
        self._evaluation_step(batch, "valid")

    def test_step(self, batch: BagBatch, batch_idx: int) -> None:
        self._evaluation_step(batch, "test")

    def on_validation_epoch_end(self) -> None:
        self._log_concordance("valid")

    def on_test_epoch_end(self) -> None:
        self._log_concordance("test")

    def predict_risk(self, batch: BagBatch) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Risk scores and attention weights for a batch, without tracking gradients.

        The training/evaluation mode is restored on return, so calling this during
        training does not silently disable dropout for subsequent steps.
        """
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                bags = [sample.features.to(self.device) for sample in batch.samples]
                return self.forward(bags)
        finally:
            self.train(was_training)

    def _evaluation_step(self, batch: BagBatch, split_name: str) -> None:
        loss, log_metrics = self.compute_loss(batch, split_name=split_name)
        if has_risk_set(batch.event):
            self.log_dict(log_metrics, on_step=False, on_epoch=True, batch_size=len(batch))

    def _accumulate(self, split_name: str, risk: torch.Tensor, event: torch.Tensor, duration: torch.Tensor) -> None:
        store = self._epoch_outputs.setdefault(split_name, [])
        store.append(torch.stack([risk.detach(), event.detach(), duration.detach()]).cpu())

    def _log_concordance(self, split_name: str) -> None:
        """Compute the C-index over the epoch's accumulated predictions."""
        outputs = self._epoch_outputs.pop(split_name, [])
        if not outputs:
            return

        stacked = torch.cat(outputs, dim=1)
        risk, event, duration = stacked[0], stacked[1], stacked[2]
        if not has_risk_set(event):
            logger.warning("no events in the %s split; skipping the C-index", split_name)
            return

        self.log(f"{split_name}_c_index", concordance_index(risk, event, duration), prog_bar=True)
