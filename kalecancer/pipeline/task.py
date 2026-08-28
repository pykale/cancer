"""What distinguishes one prediction task from another.

A trainer's structure is the same whether it predicts a binary outcome or a time to
event: run the batch forward, take a loss, accumulate predictions, summarise the
epoch. Only four things change with the task -- the head, the loss, whether a given
batch carries a gradient at all, and which metric summarises an epoch -- and those
four live here.

That is what lets one trainer serve every task: a :class:`PredictionTask` is passed
to a trainer rather than baked into it, so adding a task means adding a subclass
here, not another trainer. The two supplied cover the endpoints this package targets:
:class:`SurvivalTask` for right-censored time-to-event outcomes and
:class:`ClassificationTask` for binary ones.

Their differences are consequences of the objective rather than choices, and each is
documented on the class that makes it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from kalecancer.evaluate.classification_metrics import MetricError, roc_auc
from kalecancer.evaluate.survival_metrics import concordance_index
from kalecancer.model.embed.multimodal_fusion import MultimodalOutput
from kalecancer.model.predict import CoxHead, LinearHead
from kalecancer.model.predict.losses import as_event_mask, has_risk_set, multimodal_bce_loss, multimodal_cox_loss

#: Batch target keys for a time-to-event endpoint. :class:`SurvivalTarget` emits this
#: pair, so everything that supervises a survival model reads the same two names.
TIME_KEY = "time"
EVENT_KEY = "event"

#: Batch target key for a binary endpoint.
LABEL_KEY = "label"


class PredictionTask(nn.Module, ABC):
    """The task-specific parts of training: head, loss, and epoch metric.

    Subclasses are :class:`torch.nn.Module` so a task may own tensors --
    :class:`ClassificationTask` holds a class weight -- and have them follow the
    trainer onto its device without the trainer knowing they exist.

    Attributes:
        target_keys: Batch target keys this task supervises on. A trainer reads these
            from each batch and passes them to :meth:`loss`, :meth:`metric` and
            :meth:`has_signal`.
        metric_name: Suffix of the epoch-level metric, logged as
            ``{split}_{metric_name}``.
    """

    target_keys: tuple[str, ...]
    metric_name: str

    @abstractmethod
    def build_head(self, in_features: int) -> nn.Module:
        """Build a prediction head for embeddings of width ``in_features``.

        Passed to :class:`~kalecancer.model.embed.MultimodalFusion` as its head
        factory, which may call it more than once: late and hybrid fusion give each
        modality a head of its own.
        """

    @abstractmethod
    def loss(
        self,
        output: MultimodalOutput,
        targets: Mapping[str, Tensor],
        auxiliary_weight: float = 0.0,
    ) -> Tensor:
        """Training objective for one batch.

        Args:
            output: What the model returned for the batch.
            targets: Supervision, keyed by :attr:`target_keys`.
            auxiliary_weight: Weight on the mean per-modality loss, for fusion stages
                that produce per-modality predictions. 0 disables it.
        """

    @abstractmethod
    def metric(self, scores: Tensor, targets: Mapping[str, Tensor]) -> float:
        """Summarise one epoch's accumulated predictions.

        Both supplied tasks report a ranking metric, which is a property of a whole
        cohort rather than of a batch -- hence one value over an accumulated epoch
        rather than a mean over batches.

        Args:
            scores: ``(N,)`` predictions for the epoch, in accumulation order.
            targets: Supervision for the same patients, keyed by :attr:`target_keys`.

        Raises:
            MetricError: If this epoch's targets cannot support the metric. A trainer
                treats that as "skip this epoch's metric", not as a failure.
        """

    def has_signal(self, targets: Mapping[str, Tensor]) -> bool:
        """Whether a batch carries a usable gradient.

        True unless a task says otherwise: only a batch-coupled objective can be
        handed a batch it cannot learn anything from.
        """
        return True


class SurvivalTask(PredictionTask):
    """Right-censored time-to-event prediction by Cox proportional hazards.

    Three things here follow from the partial likelihood rather than from preference:

    * The head is bias-free. The partial likelihood is invariant to an additive shift
      in log-hazard, so an intercept would be unidentifiable and never train.
    * A batch holding no observed event is skipped. The risk set spans the
      mini-batch, so such a batch carries no gradient, and batches must be large
      enough to contain events.
    * The epoch metric is Harrell's C-index, which needs at least one comparable pair
      and so at least one observed event.
    """

    target_keys = (TIME_KEY, EVENT_KEY)
    metric_name = "c_index"

    def build_head(self, in_features: int) -> nn.Module:
        return CoxHead(in_features)

    def loss(
        self,
        output: MultimodalOutput,
        targets: Mapping[str, Tensor],
        auxiliary_weight: float = 0.0,
    ) -> Tensor:
        return multimodal_cox_loss(
            output,
            targets[TIME_KEY],
            targets[EVENT_KEY],
            auxiliary_weight=auxiliary_weight,
        )

    def has_signal(self, targets: Mapping[str, Tensor]) -> bool:
        return bool(has_risk_set(targets[EVENT_KEY]))

    def metric(self, scores: Tensor, targets: Mapping[str, Tensor]) -> float:
        events = targets[EVENT_KEY]
        if not has_risk_set(events):
            raise MetricError("no observed events in this split, so no pair of patients is comparable")
        return concordance_index(
            scores.double().numpy(),
            targets[TIME_KEY].double().numpy(),
            as_event_mask(events).numpy(),
        )


class ClassificationTask(PredictionTask):
    """Binary outcome prediction from the fused embedding.

    The mirror of :class:`SurvivalTask`, and its differences are again consequences
    of the objective:

    * The head keeps its bias. Cross-entropy is not shift-invariant, and the
      intercept is what sets the predicted base rate.
    * No batch is skipped. Cross-entropy is a per-sample loss, so a batch of purely
      negative patients still carries a gradient, which leaves batch size a free
      hyperparameter rather than a constraint.

    Heads emit logits rather than probabilities: the fused sigmoid-and-log form is
    the numerically stable one, and ROC-AUC is unchanged by the sigmoid.

    Args:
        pos_weight: Weight on the positive class, for an imbalanced endpoint. It
            changes calibration and the optimisation path but not the ranking, so its
            effect on ROC-AUC is indirect. 0 disables it.
    """

    target_keys = (LABEL_KEY,)
    metric_name = "auc"

    #: Declared for the type checker: a registered buffer is otherwise seen through
    #: ``nn.Module.__getattr__`` and typed as ``Tensor | Module``.
    pos_weight: Tensor | None

    def __init__(self, pos_weight: float = 0.0) -> None:
        super().__init__()
        # A buffer rather than a plain attribute so it follows the trainer to its
        # device, and non-persistent so it stays out of the checkpointed state.
        self.register_buffer("pos_weight", torch.tensor(pos_weight) if pos_weight > 0 else None, persistent=False)

    def build_head(self, in_features: int) -> nn.Module:
        return LinearHead(in_features)

    def loss(
        self,
        output: MultimodalOutput,
        targets: Mapping[str, Tensor],
        auxiliary_weight: float = 0.0,
    ) -> Tensor:
        return multimodal_bce_loss(
            output,
            targets[LABEL_KEY],
            auxiliary_weight=auxiliary_weight,
            pos_weight=self.pos_weight,
        )

    def metric(self, scores: Tensor, targets: Mapping[str, Tensor]) -> float:
        return roc_auc(targets[LABEL_KEY].numpy(), scores.double().numpy())
