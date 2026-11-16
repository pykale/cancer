"""Model outputs: what a model returns for a batch, typed by output space.

Everything before the endpoint is tensors, so any ``nn.Module`` works as an encoder or a head. Everything after it is
a typed output. Outputs are always model outputs: a head returns a raw tensor, and the model's endpoint
(``kalecancer.model.endpoints``) turns it into an output, attaching the state it holds (class order, a fitted
baseline). Outputs are data plus the views computed from it, and they fit nothing themselves.

Consumers declare the weakest type they need (a metric's ``requires``, an endpoint's ``accepts``), and compatibility
is ``issubclass(produced, required)``. A late fusion method names the types it can combine, as a metric names the
types it scores; the one thing a type declares for every method that combines it is ``comparable_across_models``
(ADR 0005 §2).

Convention: a field is per patient (indexed by ``select``, ``scatter`` and ``cat``) exactly when it is a ``Tensor``.
Other fields, such as ``classes`` and ``baseline``, are shared by every row and copied by reference.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Self, TypeVar

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from kalecancer.model.baseline import BreslowBaseline

OutputT = TypeVar("OutputT", bound="BaseOutput")


def _detach_cpu(t: Tensor) -> Tensor:
    """``t`` detached on the CPU, cast to float32 unless it is bool, so that numpy can read it."""
    t = t.detach().cpu()
    return t if t.dtype == torch.bool else t.float()


@dataclass(kw_only=True)
class BaseOutput(ABC):
    """What a model returns for one batch: one row per patient, NaN where ``defined`` is ``False``.

    Args:
        ids: The patients, in batch order.
        defined: ``(n,)`` bool, ``True`` where the model predicts for the patient.
        branches: In a ``LateFusion`` model, each branch's own output. Empty for other models.
    """

    ids: list[str]
    defined: Tensor
    branches: dict[str, BaseOutput] = field(default_factory=dict)

    # True where outputs of different models are on one scale, so they may be mixed patient by patient and pooled
    # across cross-validation folds. False by default, so that a type that does not say is never mixed by mistake.
    comparable_across_models: ClassVar[bool] = False

    def _per_patient(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self) if isinstance(getattr(self, f.name), Tensor)]

    def _shared(self) -> list[str]:
        """The fields shared by every row, such as ``classes`` or ``baseline``: not per patient, ``ids`` or ``branches``."""
        per_patient = set(self._per_patient())
        return [f.name for f in dataclasses.fields(self) if f.name not in {"ids", "branches", *per_patient}]

    def select(self, rows: Tensor) -> Self:
        """The output of the patients at ``rows``, e.g. the defined ones the loss runs on.

        Args:
            rows: Integer positions, in the order wanted, or an ``(n,)`` bool mask, as tensors are indexed.
        """
        if rows.dtype == torch.bool:
            rows = rows.nonzero().squeeze(1)
        index = rows.tolist()
        return dataclasses.replace(
            self,
            ids=[self.ids[k] for k in index],
            branches={name: out.select(rows) for name, out in self.branches.items()},
            **{name: getattr(self, name)[rows] for name in self._per_patient()},
        )

    def scatter(self, mask: Tensor, ids: list[str]) -> Self:
        """Put this output back where ``mask`` is ``True`` in a batch of ``ids``, with NaN rows elsewhere.

        ``defined`` becomes ``False`` for the other patients. The inverse of ``select``.
        """
        position = mask.nonzero().squeeze(1)

        def spread(t: Tensor) -> Tensor:
            fill = False if t.dtype == torch.bool else float("nan")
            return t.new_full((mask.shape[0], *t.shape[1:]), fill).index_put((position,), t)

        return dataclasses.replace(
            self,
            ids=list(ids),
            branches={name: out.scatter(mask, ids) for name, out in self.branches.items()},
            **{name: spread(getattr(self, name)) for name in self._per_patient()},
        )

    def detach_cpu(self) -> Self:
        """Copies on the CPU, detached from the autograd graph, with every tensor but ``defined`` cast to float32.

        Collecting each batch's output this way frees its graph and device memory, and the float32 cast lets numpy
        read outputs of mixed-precision training (numpy has no bfloat16). On an output already on the CPU it costs
        almost nothing, so consumers that need numpy, such as metrics, call it once on entry.
        """
        return dataclasses.replace(
            self,
            branches={name: out.detach_cpu() for name, out in self.branches.items()},
            **{name: _detach_cpu(getattr(self, name)) for name in self._per_patient()},
        )

    @staticmethod
    def cat(outputs: Sequence[OutputT]) -> OutputT:
        """The outputs of several batches as one, of their own type.

        Raises:
            ValueError: If ``outputs`` is empty, or the outputs differ in type, branches or a shared field such as
                ``classes`` or ``baseline``.
        """
        if not outputs:
            raise ValueError("cat needs at least one output")
        first = outputs[0]
        shared = first._shared()
        for out in outputs[1:]:
            if type(out) is not type(first):
                raise ValueError(f"cannot concatenate a {type(out).__name__} with a {type(first).__name__}")
            if out.branches.keys() != first.branches.keys():
                raise ValueError(
                    f"cannot concatenate outputs of branches {list(out.branches)} and {list(first.branches)}"
                )
            for name in shared:
                # != falls back to identity for objects such as a BreslowBaseline: batches share the endpoint's own
                if getattr(out, name) != getattr(first, name):
                    raise ValueError(f"cannot concatenate outputs with different {name}")
        return dataclasses.replace(
            first,
            ids=[pid for out in outputs for pid in out.ids],
            branches={name: BaseOutput.cat([out.branches[name] for out in outputs]) for name in first.branches},
            **{name: torch.cat([getattr(out, name) for out in outputs]) for name in first._per_patient()},
        )

    @abstractmethod
    def frame(self) -> pd.DataFrame:
        """What ``Pipeline.predict`` returns: one row per patient, indexed by id, NaN where not defined."""


# ------------------------------------------------------------------------------------------------------- survival


@dataclass(kw_only=True)
class RiskScore(BaseOutput):
    """Risk scores with an arbitrary offset: only the ranking of patients by one model is meaningful.

    So risk scores of different models, or folds, must not be mixed patient by patient: averaged over different
    subsets of branches, or pooled across folds. ``comparable_across_models`` is ``False``, as it is by default.

    Args:
        risk: ``(n,)``, higher means higher risk. For a Cox model, the log-hazard, which the Cox loss reads.
    """

    risk: Tensor

    def frame(self) -> pd.DataFrame:
        """One column, ``risk``."""
        return pd.DataFrame({"risk": _detach_cpu(self.risk).numpy()}, index=pd.Index(self.ids, name="id"))


class SurvivalFunction(ABC):
    """An output that gives absolute survival probabilities."""

    @abstractmethod
    def survival(self, times: Sequence[float]) -> Tensor:
        """``S(t)`` for each patient at each of ``times``, as an ``(n, len(times))`` tensor."""


@dataclass(kw_only=True)
class CoxOutput(RiskScore, SurvivalFunction):
    """A Cox model's log-hazards, as risk scores, which give survival curves once a Breslow baseline is fitted on them.

    Args:
        baseline: The endpoint's Breslow baseline, fitted on the training patients when training ends; ``None``
            before that.
    """

    baseline: BreslowBaseline | None = None

    def survival(self, times: Sequence[float]) -> Tensor:
        """``exp(-H0(t) exp(log_hazard))``, from the baseline."""
        if self.baseline is None:
            raise RuntimeError("survival needs the Breslow baseline, which is fitted when training ends")
        return torch.as_tensor(self.baseline.survival(_detach_cpu(self.risk).numpy(), np.asarray(times)))

    def frame(self) -> pd.DataFrame:
        """One column, ``log_hazard``."""
        return pd.DataFrame({"log_hazard": _detach_cpu(self.risk).numpy()}, index=pd.Index(self.ids, name="id"))


# ------------------------------------------------------------------------------------------------- classification


@dataclass(kw_only=True)
class ClassScores(BaseOutput):
    """One score per class that ranks patients for that class, such as a probability or a vote share.

    Rank-based metrics (AUROC, balanced accuracy) need no more than this. Subclasses say what the scores are.

    Args:
        classes: The classes, in column order. Shared by every row.
    """

    classes: tuple[Hashable, ...]

    # the prefix of the prediction columns, ``<prefix>[<class>]``
    column_prefix: ClassVar[str]

    @property
    @abstractmethod
    def scores(self) -> Tensor:
        """``(n, n_classes)``, higher means more likely, comparable across patients for each class."""

    def frame(self) -> pd.DataFrame:
        """One ``<column_prefix>[<class>]`` column per class."""
        # string columns are for display only; Pipeline.predict_proba gives the classes themselves as columns
        columns = [f"{self.column_prefix}[{c}]" for c in self.classes]
        return pd.DataFrame(_detach_cpu(self.scores).numpy(), index=pd.Index(self.ids, name="id"), columns=columns)


@dataclass(kw_only=True)
class ClassProbabilities(ClassScores):
    """Class probabilities, the softmax of the logits an endpoint was trained on.

    A subclass must keep ``probabilities`` the softmax of ``logits``: consumers read whichever they need, such as
    ``MaskedMean``, which averages the logits. Probabilities computed any other way, such as with a temperature, are a
    sibling type under ``ClassScores``, not a subclass.

    Args:
        logits: ``(n, n_classes)``, which the loss reads.
    """

    logits: Tensor

    column_prefix = "probability"
    # probabilities are on one scale whichever model gives them
    comparable_across_models = True

    @property
    def probabilities(self) -> Tensor:
        """The softmax of the logits, computed when asked for rather than on every training step."""
        return torch.softmax(self.logits.float(), dim=-1)

    @property
    def scores(self) -> Tensor:
        """The probabilities: logits rank the classes of one patient, but not one class across patients."""
        return self.probabilities


@dataclass(kw_only=True)
class VoteScores(ClassScores):
    """Vote scores, such as ``MajorityVote``'s: they rank patients and sum to 1, but are not probabilities.

    Args:
        votes: ``(n, n_classes)``.
    """

    votes: Tensor

    column_prefix = "score"
    # vote scores are on one scale, from 0 to 1, whichever model gives them
    comparable_across_models = True

    @property
    def scores(self) -> Tensor:
        """The vote scores."""
        return self.votes
