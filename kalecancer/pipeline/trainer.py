"""The trainer.

One trainer serves every experiment in this package. It takes a set of modality
embedders and a :class:`~kalecancer.pipeline.task.PredictionTask`, and neither axis
constrains the other: a whole-slide survival model is one bag modality with a
:class:`~kalecancer.pipeline.task.SurvivalTask`, a multimodal classifier is two
modalities with a :class:`~kalecancer.pipeline.task.ClassificationTask`, and a
tabular baseline is one fixed-width modality. There is nothing left for a
per-modality or per-endpoint trainer subclass to add.

That works because both variable parts were pushed out of the training loop. The
task supplies the head, the loss, the epoch metric, and whether a batch carries a
gradient at all. The modality is whatever a
:class:`~kalecancer.loaddata.multimodal_access.PatientBatch` happens to carry, and an embedder
that pools ragged bags is just an embedder. Nothing here names a modality, a dataset
or an endpoint.

Epoch metrics are accumulated over the whole split rather than averaged over
batches. Both supplied tasks report a ranking metric, and a mean of per-batch ranks
is not the rank of the cohort: with a batch size of 8 it would average 8-patient
orderings.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
from kale.pipeline.base_nn_trainer import BaseNNTrainer
from torch import Tensor, nn

from kalecancer.evaluate.classification_metrics import MetricError
from kalecancer.loaddata.multimodal_access import PatientBatch
from kalecancer.model.embed.multimodal_fusion import MultimodalFusion, MultimodalOutput
from kalecancer.pipeline.task import PredictionTask

logger = logging.getLogger(__name__)


class CohortTrainer(BaseNNTrainer):
    """Train a model over a cohort of patients, for whatever task is given.

    Args:
        embedders: One module per modality, each exposing ``out_dim``. Built by the
            caller, so this trainer stays independent of any particular modality. A
            bag modality uses :class:`~kalecancer.model.embed.BagEncoder`, which
            pools a patient's tiles into one vector and keeps the attention weights
            for interpretation.
        task: What is being predicted, e.g.
            :class:`~kalecancer.pipeline.task.SurvivalTask` or
            :class:`~kalecancer.pipeline.task.ClassificationTask`.
        stage: Fusion stage, see :class:`~kalecancer.model.embed.MultimodalFusion`.
            Ignored in effect with one modality, which has nothing to fuse.
        method: Fusion method, see :data:`~kalecancer.model.embed.FUSION_METHODS`.
        fusion_dim: Width every modality is projected to.
        auxiliary_weight: Weight on the mean per-modality loss, for stages that
            produce per-modality predictions. 0 disables it.
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
        task: PredictionTask,
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
        self.task = task
        self.auxiliary_weight = auxiliary_weight
        self._epoch_outputs: dict[str, list[Tensor]] = {}

        self.model = MultimodalFusion(
            embedders,
            task.build_head,
            stage=stage,
            method=method,
            fusion_dim=fusion_dim,
            modality_dropout=modality_dropout,
            **fusion_kwargs,
        )

    @property
    def embedders(self) -> nn.ModuleDict:
        """The modality embedders, for reading interpretation state off one of them."""
        return self.model.embedders

    def forward(self, batch: PatientBatch) -> MultimodalOutput:
        """Predict from every modality the batch carries."""
        modalities = {name: _to_device(value, self.device) for name, value in batch.modalities.items()}
        present = {name: value.to(self.device) for name, value in batch.present.items()}
        return self.model(modalities, present)

    def read_targets(self, batch: PatientBatch) -> dict[str, Tensor]:
        """Supervision for one batch, keyed by the task's ``target_keys``."""
        return {key: batch.target[key].to(self.device) for key in self.task.target_keys}

    def predict(self, batch: PatientBatch) -> MultimodalOutput:
        """Model output for a batch, without tracking gradients.

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

    def compute_loss(self, batch: PatientBatch, split_name: str = "valid") -> tuple[Tensor, dict]:
        """Loss for one batch, accumulating predictions on the evaluation splits.

        Raises:
            ValueError: If the task cannot form a loss from this batch -- a Cox risk
                set with no observed event, say. Callers that may see such a batch
                should ask the task's ``has_signal`` first, as the steps below do.
        """
        output, targets = self._run(batch, split_name)
        loss = self.task.loss(output, targets, auxiliary_weight=self.auxiliary_weight)
        return loss, {f"{split_name}_loss": loss}

    def training_step(self, batch: PatientBatch, batch_idx: int) -> Tensor | None:
        if not self.task.has_signal(self.read_targets(batch)):
            logger.debug("skipping batch %d: it carries no gradient for a %s", batch_idx, type(self.task).__name__)
            return None
        loss, log_metrics = self.compute_loss(batch, split_name="train")
        self.log_dict(log_metrics, on_step=False, on_epoch=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch: PatientBatch, batch_idx: int) -> None:
        self._evaluation_step(batch, "valid")

    def test_step(self, batch: PatientBatch, batch_idx: int) -> None:
        self._evaluation_step(batch, "test")

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metric("valid")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metric("test")

    def _run(self, batch: PatientBatch, split_name: str) -> tuple[MultimodalOutput, dict[str, Tensor]]:
        """Forward pass, accumulating predictions on the evaluation splits."""
        targets = self.read_targets(batch)
        output = self.forward(batch)
        if split_name != "train":
            self._accumulate(split_name, output.prediction.reshape(-1), targets)
        return output, targets

    def _evaluation_step(self, batch: PatientBatch, split_name: str) -> None:
        output, targets = self._run(batch, split_name)

        # A batch the task cannot form a loss from is not an error here: with heavy
        # censoring and small batches, a validation batch of purely censored patients
        # is ordinary. Its predictions are still accumulated above, because a censored
        # patient is comparable to everyone who failed before they were censored and
        # so counts towards the epoch's C-index.
        if not self.task.has_signal(targets):
            logger.debug("no %s loss for this batch: no signal for a %s", split_name, type(self.task).__name__)
            return

        loss = self.task.loss(output, targets, auxiliary_weight=self.auxiliary_weight)
        self.log_dict({f"{split_name}_loss": loss}, on_step=False, on_epoch=True, batch_size=len(batch))

    def _accumulate(self, split_name: str, scores: Tensor, targets: dict[str, Tensor]) -> None:
        """Hold one batch's predictions and targets until the epoch ends."""
        rows = [scores.detach().float()] + [targets[key].detach().float() for key in self.task.target_keys]
        self._epoch_outputs.setdefault(split_name, []).append(torch.stack(rows).cpu())

    def _log_epoch_metric(self, split_name: str) -> None:
        """Compute the task's metric over everything accumulated this epoch."""
        outputs = self._epoch_outputs.pop(split_name, [])
        if not outputs:
            return

        rows = torch.cat(outputs, dim=1)
        scores, target_rows = rows[0], rows[1:]
        targets = dict(zip(self.task.target_keys, target_rows, strict=True))

        try:
            value = self.task.metric(scores, targets)
        except MetricError as error:
            logger.warning("%s %s unavailable: %s", split_name, self.task.metric_name, error)
            return
        self.log(f"{split_name}_{self.task.metric_name}", value, prog_bar=True)


def _to_device(value, device):
    """Move a modality to the device, whether it is a tensor or a list of ragged bags."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return [item.to(device) for item in value]
