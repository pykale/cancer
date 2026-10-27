"""Fusion methods: how early, intermediate and late fusion combine their inputs.

A fusion method combines named inputs that a patient may lack: modality vectors in ``EarlyFusion`` and
``IntermediateFusion``, and branch outputs in ``LateFusion``. Each method owns the rules for when it can be used. A
model describes the experiment in a ``FusionContext`` and calls ``check`` with it, once when the model is built and
again for each dataset. Models never test for a particular method, so adding one needs no change to ``models.py``.

``defined`` says which patients a method can combine. Models use the same answer in ``check`` and in ``forward``, so
the two cannot disagree, and ``forward`` only ever receives rows it accepts.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import pandas as pd
import torch
from kale.embed.multimodal_fusion import FusionMethod as CoreFusionMethod
from kale.embed.multimodal_fusion import MajorityVote as CoreMajorityVote
from kale.embed.multimodal_fusion import MaskedMean as CoreMaskedMean
from kale.embed.multimodal_fusion import NamedInputConcat as CoreNamedInputConcat
from kale.embed.multimodal_fusion import masked_mean, stack_present
from torch import Tensor

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
    """Stack named inputs and match their presence masks by name.

    Args:
        values: Floating ``(n, d)`` tensors in input order. Entirely absent inputs may be omitted.
        defined: Boolean ``(n,)`` masks keyed by input name.

    Returns:
        Values shaped ``(n_inputs, n, d)`` and boolean masks shaped ``(n_inputs, n)``.
    """
    # preserve the helper used by existing custom fusion methods
    return stack_present(values, defined)


def _masked_mean(stacked: Tensor, mask: Tensor) -> Tensor:
    """Average available inputs without propagating absent NaNs.

    Args:
        stacked: Floating values shaped ``(n_inputs, n, d)``.
        mask: Boolean availability shaped ``(n_inputs, n)``.

    Returns:
        Means shaped ``(n, d)`` with the input dtype and device; zero for rows with no inputs.
        Absent entries have zero gradient. Half-precision inputs accumulate in float32.
    """
    return masked_mean(stacked, mask)


class FusionMethod(CoreFusionMethod):
    """Combines named ``(n, d)`` inputs, which a patient may lack, into one ``(n, d')`` tensor.

    Subclasses set ``stages`` and implement ``defined`` and ``forward``. To add rules, override
    ``check`` and call ``super().check(context)`` first.

    In late fusion, ``input_space`` is what the method receives from each branch. With ``"output"`` it receives the raw
    head outputs (logits or log-hazards), and the first branch's head turns the result into a prediction. With
    ``"prediction"`` it receives the heads' predictions, its result is the prediction, and it implements ``columns``.
    """

    stages: ClassVar[frozenset[Stage]]
    input_space: ClassVar[Literal["output", "prediction"]] = "output"
    # models validate in check and the dry run, and pass forward only defined rows, so skip core's per-batch checks
    validate_inputs: bool = False

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

    def check(self, context: FusionContext) -> None:  # type: ignore[override]
        """Raise if this method cannot be fitted in ``context``.

        It takes this module's ``FusionContext``, not core's, and keeps the messages that name patients and inputs.
        By default, the method must support ``context.stage``, and every required patient must be defined.

        Args:
            context: KaleCancer's stage, input availability, widths and branch head kinds.

        Raises:
            TypeError: If the method does not support the model's fusion stage.
            ValueError: If a required patient cannot be combined.
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

        Returns:
            A floating ``(n, d_out)`` tensor of combined inputs on the input device.
        """


class Concat(FusionMethod):
    """Concatenate the inputs. Every patient must have every input.

    Not for late fusion: concatenated branch outputs are not a prediction, and no head follows late fusion.
    """

    stages = frozenset({"early", "intermediate"})

    # borrow stateless operations without introducing a second FusionMethod inheritance path
    defined = CoreNamedInputConcat.defined
    forward = CoreNamedInputConcat.forward


class MaskedMean(FusionMethod):
    """Mean of equal-width inputs over the ones each patient has.

    In late fusion the inputs are the branch head outputs (logits or log-hazards).
    Float16 and bfloat16 inputs accumulate in float32 before returning the input dtype.
    """

    stages = frozenset({"early", "intermediate", "late"})

    defined = CoreMaskedMean.defined

    def check(self, context: FusionContext) -> None:
        """Also needs inputs of one width and, in late fusion, heads of one kind.

        Cox heads also need every branch for every patient.
        """
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

    forward = CoreMaskedMean.forward


class MajorityVote(FusionMethod):
    """Each branch votes for its most probable class.

    Scores are ``(votes_c + 0.5 * mean_probability_c) / (n_branches + 0.5)``: they sum to 1, their argmax is the
    majority class, and ties between classes with equal votes are broken by mean probability. The probability
    term can never outweigh a whole vote.
    Scores are not calibrated probabilities. Float16 and bfloat16 inputs produce float32 scores;
    float32 and float64 inputs retain their dtype.

    Args:
        tie_break: ``"mean_probability"`` breaks ties as above. ``"error"`` raises when a patient's votes are tied.
    """

    stages = frozenset({"late"})
    input_space = "prediction"

    def __init__(self, tie_break: Literal["mean_probability", "error"]) -> None:
        super().__init__()
        if tie_break not in ("mean_probability", "error"):
            raise ValueError(f"tie_break must be 'mean_probability' or 'error', got {tie_break!r}")
        self.tie_break = tie_break

    defined = CoreMajorityVote.defined

    def check(self, context: FusionContext) -> None:
        """Also needs classification heads with one output width."""
        super().check(context)
        if not _heads_predict(context, Classification):
            raise TypeError(f"MajorityVote needs classification heads, got {set(context.head_kinds.values())}")
        _check_one_width(self, context.widths)

    def columns(self, target: Classification) -> list[str]:
        """One vote-score column per class."""
        return [f"vote_score[{c}]" for c in target.classes]

    forward = CoreMajorityVote.forward
