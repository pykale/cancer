"""Fusion methods: how early, intermediate and late fusion combine their inputs.

A fusion method combines named inputs that a patient may lack: modality vectors in ``EarlyFusion`` and
``IntermediateFusion``, and branch outputs in ``LateFusion``. Each method owns the rules for when it can be used. A
model describes the experiment in a ``FusionContext`` and calls ``check`` with it, once when the model is built and
again for each dataset. Models never test for a particular method, so adding one needs no change to ``models.py``.

``defined`` says which patients a method can combine. Models use the same answer in ``check`` and in ``forward``, so
the two cannot disagree, and ``forward`` only ever receives rows it accepts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kalecancer.loaddata.targets import BaseTarget, Classification, TimeToEvent

Stage = Literal["early", "intermediate", "late"]
_STAGES: tuple[Stage, ...] = ("early", "intermediate", "late")


@dataclass(frozen=True)
class FusionContext:
    """What a fusion method may use to decide whether it can be fitted.

    The model builds it from the dataset and from itself, so methods never see the dataset.

    Args:
        stage: Which kind of model combines the inputs.
        present: One row per patient and one bool column per input, ``True`` where the patient has it. The inputs are
            modalities in early and intermediate fusion, and branches in late fusion.
        required: Patients who must get a combined value, indexed like ``present``. Every patient, except in a
            ``LateFusion`` branch, where only the patients with at least one of the branch's modalities.
        widths: Width of each input, ``None`` where it is not known yet.
        head_kinds: In late fusion, the ``target_type`` of each branch head. Empty otherwise.
    """

    stage: Stage
    present: pd.DataFrame
    required: pd.Series
    widths: Mapping[str, int | None]
    head_kinds: Mapping[str, type[BaseTarget] | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a context whose ``required`` or ``widths`` do not line up with ``present``."""
        if not self.required.index.equals(self.present.index) or set(self.widths) != set(self.present.columns):
            raise ValueError("FusionContext: required must be indexed like present, and widths keyed like its columns")

    @classmethod
    def without_data(
        cls,
        stage: Stage,
        widths: Mapping[str, int | None],
        head_kinds: Mapping[str, type[BaseTarget] | None] | None = None,
    ) -> FusionContext:
        """A context with no patients, for the checks a model can run when it is built."""
        present = pd.DataFrame({name: pd.Series(dtype=bool) for name in widths})
        required = pd.Series(dtype=bool, index=present.index)
        return cls(stage, present, required, dict(widths), dict(head_kinds or {}))


def _check_one_width(method: FusionMethod, widths: Mapping[str, int | None]) -> None:
    """Raise if the known ``widths`` are not all equal."""
    known = {name: width for name, width in widths.items() if width is not None}
    if len(set(known.values())) > 1:
        raise ValueError(f"{type(method).__name__} needs inputs of one width, got {known}")


def _heads_predict(context: FusionContext, target_type: type[BaseTarget]) -> bool:
    """Whether every branch head in ``context`` predicts ``target_type`` or a subclass of it."""
    kinds = context.head_kinds.values()
    return bool(kinds) and all(kind is not None and issubclass(kind, target_type) for kind in kinds)


def _stack_defined(values: Mapping[str, Tensor], defined: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    """Stack the inputs into ``(n_inputs, n, d)`` values and an ``(n_inputs, n)`` presence mask."""
    stacked = torch.stack([values[name] for name in values])
    mask = torch.stack([defined[name] for name in values])
    return stacked, mask


def _masked_mean(stacked: Tensor, mask: Tensor) -> Tensor:
    """Mean of each row over the inputs it has; 0 for a row that has none."""
    # where() rather than multiplication: absent rows hold NaN, and NaN * 0 is NaN
    filled = torch.where(mask.unsqueeze(-1), stacked, torch.zeros((), dtype=stacked.dtype, device=stacked.device))
    count = mask.sum(dim=0).clamp(min=1).unsqueeze(-1).to(stacked.dtype)
    return filled.sum(dim=0) / count


class FusionMethod(nn.Module, ABC):
    """Combines named ``(n, d)`` inputs, which a patient may lack, into one ``(n, d')`` tensor.

    Subclasses set ``stages`` and implement ``defined``, ``output_dim`` and ``forward``. To add rules, override
    ``check`` and call ``super().check(context)`` first.

    In late fusion, ``input_space`` is what the method receives from each branch. With ``"output"`` it receives the raw
    head outputs (logits or log-hazards), and the first branch's head turns the result into a prediction. With
    ``"prediction"`` it receives the heads' predictions, its result is the prediction, and it implements ``columns``.
    """

    stages: ClassVar[frozenset[Stage]]
    input_space: ClassVar[Literal["output", "prediction"]] = "output"

    @abstractmethod
    def defined(self, present: Tensor) -> Tensor:
        """Which patients this method can combine.

        Args:
            present: ``(n_patients, n_inputs)`` bool, ``True`` where the patient has the input.

        Returns:
            ``(n_patients,)`` bool.
        """

    def defined_rows(self, present: pd.DataFrame) -> pd.Series:
        """``defined`` for a presence table with one row per patient, as a bool Series indexed like it."""
        return pd.Series(self.defined(torch.tensor(present.to_numpy(dtype=bool))).numpy(), index=present.index)

    @abstractmethod
    def output_dim(self, widths: Mapping[str, int]) -> int:
        """Width of the combined tensor, given the width of each input."""

    def check(self, context: FusionContext) -> None:
        """Raise if this method cannot be fitted in ``context``.

        By default, the method must support ``context.stage``, and every required patient must be defined.
        """
        name = type(self).__name__
        if context.stage not in self.stages:
            allowed = " or ".join(stage for stage in _STAGES if stage in self.stages)
            raise TypeError(f"{name} is for {allowed} fusion, not {context.stage} fusion")
        present = context.present
        lacking = context.required & ~self.defined_rows(present)
        if lacking.any():
            counts = {col: int((lacking & ~present[col]).sum()) for col in present.columns}
            inputs, advice = (
                ("branches", "each patient needs the modalities of at least one branch")
                if context.stage == "late"
                else (
                    "modalities",
                    "add the modalities to required_modalities, or use a fusion method that accepts patients who "
                    "lack some",
                )
            )
            raise ValueError(
                f"{name} cannot combine {int(lacking.sum())} patients "
                f"(how many of them lack each of the {inputs}: {counts}); {advice}"
            )

    @abstractmethod
    def forward(self, values: Mapping[str, Tensor], present: Mapping[str, Tensor]) -> Tensor:
        """Combine the rows that ``defined`` accepts.

        Args:
            values: ``(n, d)`` tensor for each input, with NaN in rows that lack it. An input that no row has may be
                left out.
            present: ``(n,)`` bool for every input, ``True`` where the row has it.
        """


class Concat(FusionMethod):
    """Concatenate the inputs. Every patient must have every input.

    Not for late fusion: concatenated branch outputs are not a prediction, and no head follows late fusion.
    """

    stages = frozenset({"early", "intermediate"})

    def defined(self, present: Tensor) -> Tensor:
        """The patients who have every input."""
        return present.all(dim=1)

    def output_dim(self, widths: Mapping[str, int]) -> int:
        """The sum of the input widths."""
        return sum(widths.values())

    def forward(self, values: Mapping[str, Tensor], present: Mapping[str, Tensor]) -> Tensor:
        """Concatenate the inputs along the feature dimension, in input order."""
        return torch.cat([values[name] for name in values], dim=-1)


class MaskedMean(FusionMethod):
    """Mean of equal-width inputs over the ones each patient has.

    In late fusion the inputs are the branch head outputs (logits or log-hazards).
    """

    stages = frozenset({"early", "intermediate", "late"})

    def defined(self, present: Tensor) -> Tensor:
        """The patients who have at least one input."""
        return present.any(dim=1)

    def output_dim(self, widths: Mapping[str, int]) -> int:
        """The width the inputs share."""
        _check_one_width(self, widths)
        return next(iter(widths.values()))

    def check(self, context: FusionContext) -> None:
        """Also needs inputs of one width and, in late fusion, heads of one kind; Cox heads need every branch
        for every patient."""
        super().check(context)
        kinds = set(context.head_kinds.values())
        if len(kinds) > 1:
            raise TypeError(f"MaskedMean needs branch heads of one kind, got {kinds}")
        _check_one_width(self, context.widths)
        incomplete = int((~context.present.all(axis=1)).sum())

        # NOTE: this checks for TimeToEvent heads, but the issue is with Cox heads
        # specifically. This will need fixing if another TimeToEvent head is added
        # that does not have an issue with averaging over branches.
        if _heads_predict(context, TimeToEvent) and incomplete:
            raise ValueError(
                f"MaskedMean over Cox Head needs every patient to have every branch, but {incomplete} do not: each "
                "branch's log-hazard has an arbitrary offset, so averaging different subsets of branches would "
                "reorder patients"
            )

    def forward(self, values: Mapping[str, Tensor], present: Mapping[str, Tensor]) -> Tensor:
        """Average each row over the inputs it has."""
        return _masked_mean(*_stack_defined(values, present))


class MajorityVote(FusionMethod):
    """Each branch votes for its most probable class.

    Scores are ``(votes_c + 0.5 * mean_probability_c) / (n_branches + 0.5)``: they sum to 1, their argmax is the
    majority class, and ties between classes with equal votes are broken by mean probability. The probability
    term can never outweigh a whole vote.
    """

    stages = frozenset({"late"})
    input_space = "prediction"

    def __init__(self, tie_break: Literal["mean_probability", "error"]):
        super().__init__()
        if tie_break not in ("mean_probability", "error"):
            raise ValueError(f"tie_break must be 'mean_probability' or 'error', got {tie_break!r}")
        self.tie_break = tie_break

    def defined(self, present: Tensor) -> Tensor:
        """The patients who have at least one input."""
        return present.any(dim=1)

    def output_dim(self, widths: Mapping[str, int]) -> int:
        """The width the inputs share."""
        _check_one_width(self, widths)
        return next(iter(widths.values()))

    def check(self, context: FusionContext) -> None:
        """Also needs classification heads with one output width."""
        super().check(context)
        if not _heads_predict(context, Classification):
            raise TypeError(f"MajorityVote needs classification heads, got {set(context.head_kinds.values())}")
        _check_one_width(self, context.widths)

    def columns(self, target: Classification) -> list[str]:
        """One vote-score column per class."""
        return [f"vote_score[{c}]" for c in target.classes]

    def forward(self, values: Mapping[str, Tensor], present: Mapping[str, Tensor]) -> Tensor:
        """Score each class from the votes of the branches each row has; with ``tie_break="error"``, raise on a tie."""
        stacked, mask = _stack_defined(values, present)
        n_classes = stacked.shape[-1]
        choice = torch.nan_to_num(stacked, nan=-1.0).argmax(dim=-1)
        votes = (F.one_hot(choice, n_classes).to(stacked.dtype) * mask.unsqueeze(-1)).sum(dim=0)
        n_branches = mask.sum(dim=0).to(stacked.dtype).unsqueeze(-1)
        if self.tie_break == "error":
            top = votes.max(dim=-1, keepdim=True).values
            tied = ((votes == top).sum(dim=-1) > 1) & (n_branches.squeeze(-1) > 0)
            if bool(tied.any()):
                raise ValueError(f"MajorityVote: {int(tied.sum())} patients have tied votes and tie_break='error'")
        return (votes + 0.5 * _masked_mean(stacked, mask)) / (n_branches + 0.5)
