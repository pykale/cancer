"""Fusion methods: how early, intermediate and late fusion combine their inputs.

A fusion method combines named inputs that a patient may lack. A ``TensorFusion`` combines ``(n, d)`` tensors (raw
inputs or embeddings) in ``EarlyFusion`` and ``IntermediateFusion``; an ``OutputFusion`` combines the branches' typed
outputs in ``LateFusion``. A method that does both, such as ``MaskedMean``, subclasses both. Models test only which of
the two interfaces a method has, never for a particular method, so adding one needs no change to ``models.py``.

``defined`` says which patients a method can combine. Models use the same answer to check a dataset and in
``forward``, so the two cannot disagree, and ``forward`` only ever receives rows it accepts. Each other rule has its own
hook, called when the information it needs first exists (ADR 0005 §7): ``OutputFusion.output_type`` when the model is
built, ``OutputFusion.check_branches`` for each dataset, and ``TensorFusion.check_widths`` in the dry run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Literal, cast, overload

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kalecancer.model.outputs import BaseOutput, ClassProbabilities, OutputT, RiskScore, VoteScores


def _check_one_width(method: FusionMethod, widths: Mapping[str, int]) -> None:
    """Raise if ``widths`` are not all equal."""
    if len(set(widths.values())) > 1:
        raise ValueError(f"{type(method).__name__} needs inputs of one width, got {dict(widths)}")


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
    """What every fusion method shares: which patients it can combine.

    A method fuses tensors (``TensorFusion``, in early and intermediate fusion), typed outputs (``OutputFusion``, in late
    fusion), or both, and subclasses those rather than this class: no model takes a bare ``FusionMethod``. The model
    checks that every patient it must predict for is ``defined``, so a method cannot skip that rule.
    """

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


class TensorFusion(FusionMethod):
    """Combines ``(n, d)`` tensors, raw inputs in ``EarlyFusion`` or embeddings in ``IntermediateFusion``."""

    def check_widths(self, widths: Mapping[str, int]) -> None:
        """Raise if inputs of these widths cannot be combined. Called in the dry run; accepts any widths by default.

        Args:
            widths: The width of each input that the dry run's patients have.
        """

    @abstractmethod
    def forward(self, values: Mapping[str, Tensor], present: Mapping[str, Tensor]) -> Tensor:
        """Combine the rows that ``defined`` accepts.

        Args:
            values: An ``(n, d)`` tensor for each input, with NaN in rows that lack it. An input that no row has may
                be left out.
            present: ``(n,)`` bool for every input, ``True`` where the row has it.

        Returns:
            An ``(n, d')`` tensor.
        """


class OutputFusion(FusionMethod):
    """Combines the branches' typed outputs in ``LateFusion`` into one typed output, of the type ``output_type`` says."""

    @abstractmethod
    def output_type(self, input_types: Mapping[str, type[BaseOutput]]) -> type[BaseOutput]:
        """The type of output this method makes from branch outputs of ``input_types``.

        Raises:
            TypeError: If it cannot fuse those types.
        """

    def check_branches(self, present: pd.DataFrame, input_types: Mapping[str, type[BaseOutput]]) -> None:
        """Raise if branch outputs of ``input_types`` cannot be fused for these patients. Accepts any by default.

        Called for each dataset, after the model has checked that every patient has a branch.

        Args:
            present: One row per patient and one bool column per branch, ``True`` where the branch predicts.
            input_types: Each branch's output type, which ``output_type`` accepted when the model was built.
        """

    @abstractmethod
    def forward(self, values: Mapping[str, BaseOutput], present: Mapping[str, Tensor]) -> BaseOutput:
        """Combine the rows that ``defined`` accepts.

        Args:
            values: Each branch's output, with NaN in rows the branch does not predict. A branch that predicts no row
                may be left out.
            present: ``(n,)`` bool for every branch, ``True`` where the branch predicts the row.

        Returns:
            An output of ``output_type`` for the same rows, all of them defined.
        """


def _as(outputs: Mapping[str, BaseOutput], kind: type[OutputT]) -> dict[str, OutputT]:
    """``outputs``, each of which the method's ``output_type`` has already checked is a ``kind``."""
    assert all(isinstance(out, kind) for out in outputs.values()), f"check accepts only {kind.__name__} branches"
    return cast(dict[str, OutputT], dict(outputs))


class Concat(TensorFusion):
    """Concatenate the inputs. Every patient must have every input.

    Not for late fusion: concatenated branch outputs are not a prediction.
    """

    def defined(self, present: Tensor) -> Tensor:
        """The patients who have every input."""
        return present.all(dim=1)

    def forward(self, values: Mapping[str, Tensor], present: Mapping[str, Tensor]) -> Tensor:
        """Concatenate the inputs along the feature dimension, in input order."""
        return torch.cat([values[name] for name in values], dim=-1)


class MaskedMean(TensorFusion, OutputFusion):
    """Mean of equal-width inputs over the ones each patient has, in any stage.

    In late fusion it averages the branches' risk scores, giving a ``RiskScore``, or their logits, giving
    ``ClassProbabilities``. An average of Cox log-hazards is a ``RiskScore`` with no baseline, because each branch's
    baseline belongs to its own log-hazards; a fused endpoint such as ``CoxEndpoint`` fits one for it. Averaging logits
    rather than probabilities takes the normalised geometric mean of the branches' probabilities.
    """

    def defined(self, present: Tensor) -> Tensor:
        """The patients who have at least one input."""
        return present.any(dim=1)

    def check_widths(self, widths: Mapping[str, int]) -> None:
        """Inputs must have one width."""
        _check_one_width(self, widths)

    def check_branches(self, present: pd.DataFrame, input_types: Mapping[str, type[BaseOutput]]) -> None:
        """Outputs that are not comparable across models, such as risk scores, need every branch for every patient."""
        incomplete = int((~present.all(axis=1)).sum())
        relative = sorted(kind.__name__ for kind in input_types.values() if not kind.comparable_across_models)
        if relative and incomplete:
            raise ValueError(
                f"MaskedMean over {relative[0]} needs every patient to have every branch, but {incomplete} do not: each "
                "branch's output has an arbitrary offset, so averaging different subsets of branches would reorder "
                "patients"
            )

    def output_type(self, input_types: Mapping[str, type[BaseOutput]]) -> type[BaseOutput]:
        """``RiskScore`` from risk scores, such as Cox outputs, and ``ClassProbabilities`` from class probabilities."""
        kinds = set(input_types.values())
        for family in (RiskScore, ClassProbabilities):
            if kinds and all(issubclass(kind, family) for kind in kinds):
                return family
        raise TypeError(
            "MaskedMean needs branches of one output type that it can average, risk scores or class probabilities, "
            f"got {sorted(kind.__name__ for kind in kinds)}"
        )

    @overload
    def forward(self, values: Mapping[str, Tensor], present: Mapping[str, Tensor]) -> Tensor: ...

    @overload
    def forward(self, values: Mapping[str, BaseOutput], present: Mapping[str, Tensor]) -> BaseOutput: ...

    def forward(
        self, values: Mapping[str, Tensor] | Mapping[str, BaseOutput], present: Mapping[str, Tensor]
    ) -> Tensor | BaseOutput:
        """Average each row over the inputs it has: tensors, or the risk scores or logits of branch outputs."""
        first = next(iter(values.values()), None)
        if first is None:
            raise ValueError("MaskedMean needs at least one input")
        if isinstance(first, Tensor):
            return _masked_mean(*_stack_defined(cast(Mapping[str, Tensor], values), present))
        return self._average_outputs(cast(Mapping[str, BaseOutput], values), present)

    def _average_outputs(self, outputs: Mapping[str, BaseOutput], present: Mapping[str, Tensor]) -> BaseOutput:
        first = next(iter(outputs.values()))
        if isinstance(first, RiskScore):
            risks = {name: out.risk.unsqueeze(-1) for name, out in _as(outputs, RiskScore).items()}
            mean = _masked_mean(*_stack_defined(risks, present)).squeeze(-1)
            return RiskScore(ids=first.ids, defined=_everyone(first, mean), risk=mean)
        if isinstance(first, ClassProbabilities):
            logits = {name: out.logits for name, out in _as(outputs, ClassProbabilities).items()}
            mean = _masked_mean(*_stack_defined(logits, present))
            return ClassProbabilities(ids=first.ids, defined=_everyone(first, mean), logits=mean, classes=first.classes)
        raise AssertionError(f"check refuses {type(first).__name__} branches")


def _everyone(like: BaseOutput, values: Tensor) -> Tensor:
    """A ``defined`` mask that is ``True`` for every row of ``like``, on the device of ``values``."""
    return torch.ones(len(like.ids), dtype=torch.bool, device=values.device)


class MajorityVote(OutputFusion):
    """Each branch votes for its most probable class.

    Scores are ``(votes_c + 0.5 * mean_probability_c) / (n_branches + 0.5)``: they sum to 1, their argmax is the
    majority class, and ties between classes with equal votes are broken by mean probability. The probability
    term can never outweigh a whole vote.

    Args:
        tie_break: ``"mean_probability"`` breaks ties as above. ``"error"`` raises when a patient's votes are tied.
    """

    def __init__(self, tie_break: Literal["mean_probability", "error"]):
        super().__init__()
        if tie_break not in ("mean_probability", "error"):
            raise ValueError(f"tie_break must be 'mean_probability' or 'error', got {tie_break!r}")
        self.tie_break = tie_break

    def defined(self, present: Tensor) -> Tensor:
        """The patients who have at least one input."""
        return present.any(dim=1)

    def output_type(self, input_types: Mapping[str, type[BaseOutput]]) -> type[BaseOutput]:
        """``VoteScores``: vote scores are not probabilities. The branches must output class probabilities."""
        if not (input_types and all(issubclass(kind, ClassProbabilities) for kind in input_types.values())):
            kinds = sorted(k.__name__ for k in input_types.values())
            raise TypeError(f"MajorityVote needs branches that output class probabilities, got {kinds}")
        return VoteScores

    def forward(self, values: Mapping[str, BaseOutput], present: Mapping[str, Tensor]) -> VoteScores:
        """Score each class from the votes of the branches each row has; with ``tie_break="error"``, raise on a tie."""
        branches = _as(values, ClassProbabilities)
        first = next(iter(branches.values()))
        stacked, mask = _stack_defined({name: out.probabilities for name, out in branches.items()}, present)
        n_classes = stacked.shape[-1]
        choice = torch.nan_to_num(stacked, nan=-1.0).argmax(dim=-1)
        votes = (F.one_hot(choice, n_classes).to(stacked.dtype) * mask.unsqueeze(-1)).sum(dim=0)
        n_branches = mask.sum(dim=0).to(stacked.dtype).unsqueeze(-1)
        if self.tie_break == "error":
            top = votes.max(dim=-1, keepdim=True).values
            tied = ((votes == top).sum(dim=-1) > 1) & (n_branches.squeeze(-1) > 0)
            if bool(tied.any()):
                raise ValueError(f"MajorityVote: {int(tied.sum())} patients have tied votes and tie_break='error'")
        scores = (votes + 0.5 * _masked_mean(stacked, mask)) / (n_branches + 0.5)
        return VoteScores(ids=first.ids, defined=_everyone(first, scores), votes=scores, classes=first.classes)
