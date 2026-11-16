"""Endpoints: the head, the loss, and the state that turn an embedding into a typed output.

A model holds one endpoint, and ``LateFusion`` holds one more for its fused output. The endpoint owns, in the order a
``fit`` uses them:

1. ``output_type``: what it produces, known when the model is built, so fusion methods and metrics are checked then.
2. ``check_target`` and ``bind``: at the start of ``fit``, raise if the target does not suit it, then take what it
   needs from the target, such as the class order.
3. ``forward``: run the head, any ``nn.Module``, on the embedding. ``to_output`` then builds the typed output and
   attaches the endpoint's state.
4. ``loss``: the loss of a batch, or ``None`` when the batch has no signal. The dry run calls it too.
5. ``fit_state``: once training ends, fit any state on the training patients' outputs, such as a Breslow baseline.

Classification (and later regression) takes any torch-style ``loss_fn(input, target)``, so users choose a loss without
writing an endpoint. In survival the loss defines what the output means (the Cox partial likelihood is what makes a
Breslow baseline valid), so each family of survival models is its own endpoint.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from typing import ClassVar, Literal

import torch
from torch import Tensor, nn

from kalecancer.loaddata.targets import BaseTarget, Classification, TimeToEvent
from kalecancer.model.baseline import BreslowBaseline
from kalecancer.model.outputs import BaseOutput, ClassProbabilities, CoxOutput, RiskScore

with warnings.catch_warnings():
    # torchsurv applies torch.jit.script when imported, which recent torch deprecates; users cannot act on it
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.* is deprecated", category=FutureWarning)
    from torchsurv.loss.cox import neg_partial_log_likelihood


class BaseEndpoint(nn.Module, ABC):
    """The head, the loss and the state that turn an embedding into a typed output.

    The endpoint of a ``LateFusion`` model's fused output has no head: it takes the fusion method's output, which must
    be an ``accepts``, and makes it an ``output_type`` with ``adopt``.

    Args:
        head: Any ``nn.Module`` from the ``(n, d)`` embedding to the raw output, e.g. ``nn.Linear(64, 1)``. ``None``
            for the fused endpoint of a ``LateFusion`` model.
    """

    output_type: ClassVar[type[BaseOutput]]
    accepts: ClassVar[type[BaseOutput]]

    def __init__(self, head: nn.Module | None):
        super().__init__()
        self.head = head

    @property
    @abstractmethod
    def decomposable_loss(self) -> bool:
        """Whether the loss is a mean over patients, so accumulating gradients reproduces a larger batch.

        A property, not a class attribute, because it can depend on the loss the endpoint was given.
        """

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """The width of the ``(n, out_dim)`` raw output the head must produce.

        The dry run checks the head against it, and a batch in which no patient is defined gets NaN rows of this width.
        """

    @abstractmethod
    def check_target(self, target: BaseTarget) -> None:
        """Raise if this endpoint cannot be trained on ``target``. Called when ``fit`` starts."""

    def bind(self, target: BaseTarget) -> None:
        """Take what the endpoint needs from the training target, after ``check_target``. Nothing by default."""

    def forward(self, z: Tensor) -> Tensor:
        """The raw output of the head, or ``z`` itself without a head."""
        return z if self.head is None else self.head(z)

    @abstractmethod
    def to_output(self, raw: Tensor, ids: list[str], defined: Tensor) -> BaseOutput:
        """The typed output of ``raw``, with the endpoint's state attached. ``raw`` has a row for every patient."""

    @abstractmethod
    def adopt(self, output: BaseOutput) -> BaseOutput:
        """A fusion method's ``output``, an ``accepts``, as this endpoint's output, with the endpoint's state attached."""

    @abstractmethod
    def loss(self, output: BaseOutput, target: Mapping[str, Tensor]) -> Tensor | None:
        """The loss over the patients of ``output``, all defined; ``None`` when the batch carries no signal."""

    def fit_state(self, output: BaseOutput, target: BaseTarget) -> dict[str, int]:
        """Fit any state on the output of the training patients, once training ends. Nothing by default.

        Returns:
            Counts to add to ``Pipeline.fit_report_``, such as the events a baseline was fitted on.
        """
        return {}


# ------------------------------------------------------------------------------------------------------- survival


def _has_comparable_event(time: Tensor, event: Tensor) -> bool:
    if not bool(event.any()):
        return False
    at_risk = (time.unsqueeze(0) >= time[event].unsqueeze(1)).sum(dim=1)
    return bool((at_risk > 1).any())


class CoxEndpoint(BaseEndpoint):
    """Log-hazards trained with the Cox partial likelihood over the patients in each batch.

    The head's output is one column, and the head needs no bias: the partial likelihood is invariant to an additive
    constant, so a bias would never be learned.

    Args:
        head: E.g. ``nn.Linear(in_dim, 1, bias=False)``. It must output one column, ``out_dim``, which the dry run
            checks.
        ties: How the loss handles tied event times, ``"efron"`` or ``"breslow"``.

    Attributes:
        baseline_: The Breslow baseline, fitted on the training patients once training ends; ``None`` before.
            Validation patients are left out: they chose the model, they did not fit it.
    """

    output_type = CoxOutput
    accepts = RiskScore

    def __init__(self, head: nn.Module | None, ties: Literal["efron", "breslow"]):
        super().__init__(head)
        if ties not in ("efron", "breslow"):
            raise ValueError(f"ties must be 'efron' or 'breslow', got {ties!r}")
        self.ties = ties
        self.baseline_: BreslowBaseline | None = None

    @property
    def decomposable_loss(self) -> bool:
        """``False``: the risk set of the partial likelihood is the whole batch."""
        return False

    @property
    def out_dim(self) -> int:
        """One log-hazard per patient."""
        return 1

    def check_target(self, target: BaseTarget) -> None:
        """Raise if ``target`` is not a ``TimeToEvent``."""
        if not isinstance(target, TimeToEvent):
            raise TypeError(f"CoxEndpoint needs a time-to-event target, got {type(target).__name__}")

    def to_output(self, raw: Tensor, ids: list[str], defined: Tensor) -> CoxOutput:
        """The log-hazards, with the baseline once it is fitted."""
        return CoxOutput(ids=ids, defined=defined, risk=raw.reshape(-1), baseline=self.baseline_)

    def adopt(self, output: BaseOutput) -> CoxOutput:
        """The fused log-hazards, with this endpoint's own baseline once it is fitted."""
        assert isinstance(output, RiskScore), "LateFusion checks the fusion method's output type"
        return CoxOutput(ids=output.ids, defined=output.defined, risk=output.risk, baseline=self.baseline_)

    def loss(self, output: BaseOutput, target: Mapping[str, Tensor]) -> Tensor | None:
        """Negative partial log-likelihood averaged over events; ``None`` when no event has anyone else at risk."""
        assert isinstance(output, CoxOutput)
        event, time = target["event"], target["time"]
        if event.dtype != torch.bool:
            raise TypeError(f"target 'event' must be bool, got {event.dtype}")
        with torch.autocast(device_type=output.risk.device.type, enabled=False):
            log_hazard, time = output.risk.float(), time.float()
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

    def fit_state(self, output: BaseOutput, target: BaseTarget) -> dict[str, int]:
        """Fit the Breslow baseline on the defined training patients' log-hazards.

        Returns:
            ``baseline_events``, the number of events the baseline was fitted on.
        """
        assert isinstance(output, CoxOutput) and isinstance(target, TimeToEvent)
        output = output.detach_cpu()
        keep = output.defined.numpy()
        outcome = target.frame.loc[[pid for pid, k in zip(output.ids, keep, strict=True) if k]]
        self.baseline_ = BreslowBaseline().fit(
            output.risk.numpy()[keep], outcome["time"].to_numpy(), outcome["event"].to_numpy()
        )
        return {"baseline_events": int(outcome["event"].sum())}


# ------------------------------------------------------------------------------------------------- classification


class ClassificationEndpoint(BaseEndpoint):
    """Class logits, trained with any torch-style classification loss.

    Named ``ClassificationEndpoint`` to avoid clashing with the ``Classification`` target (ADR 0005, open questions).

    Args:
        head: E.g. ``nn.Linear(in_dim, n_classes)``. Its output width must be the number of classes in the target,
            ``out_dim``, which the dry run checks.
        loss_fn: Called as ``loss_fn(logits, labels)`` with class indices, e.g. ``nn.CrossEntropyLoss()``,
            ``nn.CrossEntropyLoss(weight=..., label_smoothing=0.1)`` or a focal loss.
        decomposable_loss: Whether ``loss_fn`` is a mean over patients, as ``nn.CrossEntropyLoss()`` is, so that
            ``accumulate_grad_batches`` reproduces a larger batch. Only gradient accumulation reads it, and the
            ``Pipeline`` refuses accumulation when it is ``False``, so leave it ``False`` unless you accumulate.
    """

    output_type = ClassProbabilities
    accepts = ClassProbabilities

    def __init__(self, head: nn.Module | None, loss_fn: nn.Module, decomposable_loss: bool = False):
        super().__init__(head)
        self.loss_fn = loss_fn
        self._decomposable_loss = decomposable_loss
        self.classes_: tuple[Hashable, ...] = ()

    @property
    def decomposable_loss(self) -> bool:
        """As given: only the user knows whether their loss averages over patients."""
        return self._decomposable_loss

    @property
    def out_dim(self) -> int:
        """One logit per class of the target it is bound to."""
        return len(self.classes_)

    def check_target(self, target: BaseTarget) -> None:
        """Raise if ``target`` is not a ``Classification``."""
        if not isinstance(target, Classification):
            raise TypeError(f"ClassificationEndpoint needs a classification target, got {type(target).__name__}")

    def bind(self, target: BaseTarget) -> None:
        """Take the class order from the target."""
        assert isinstance(target, Classification)
        self.classes_ = tuple(target.classes)

    def to_output(self, raw: Tensor, ids: list[str], defined: Tensor) -> ClassProbabilities:
        """The logits, with the class order."""
        return ClassProbabilities(ids=ids, defined=defined, logits=raw, classes=self.classes_)

    def adopt(self, output: BaseOutput) -> ClassProbabilities:
        """The fused logits, with the class order of the target this endpoint is bound to."""
        assert isinstance(output, ClassProbabilities), "LateFusion checks the fusion method's output type"
        return ClassProbabilities(ids=output.ids, defined=output.defined, logits=output.logits, classes=self.classes_)

    def loss(self, output: BaseOutput, target: Mapping[str, Tensor]) -> Tensor | None:
        """``loss_fn`` of the logits against the labels, in float32. Every batch has signal, so never ``None``."""
        assert isinstance(output, ClassProbabilities)
        with torch.autocast(device_type=output.logits.device.type, enabled=False):
            return self.loss_fn(output.logits.float(), target["label"])
