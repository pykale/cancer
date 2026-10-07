"""Models: per-modality stage lists wired into unimodal, early, intermediate and late fusion.

Execution rule shared by every model: stages run on the patients that have the modality, their outputs are
scattered back into the batch with NaN for absent patients, fusion methods select defined rows (never multiply
by a mask), and heads run on defined rows only. This keeps gradients finite when modalities are missing.

Models never test for a particular fusion method: they describe the experiment in a ``FusionContext`` and let the
method's ``check`` decide, and they ask the method's ``defined`` which patients it combines.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch
from torch import Tensor, nn

from kalecancer.loaddata.dataset import MultimodalDataset
from kalecancer.loaddata.modalities import FixedShapeModality, Modality
from kalecancer.loaddata.targets import BaseTarget
from kalecancer.model.fusion import FusionContext, FusionMethod, Stage
from kalecancer.model.incontext import InContextModule


@dataclass
class ModelOutput:
    """Raw head output (what the loss sees), prediction, and which patients have a prediction."""

    output: Tensor
    prediction: Tensor
    defined: Tensor
    branches: dict[str, ModelOutput] = field(default_factory=dict)


def _declared(stage: nn.Module, names: tuple[str, str]) -> int | None:
    """The first of ``names`` that ``stage`` declares as an int, or ``None`` when it declares none."""
    for name in names:
        value = getattr(stage, name, None)
        if isinstance(value, int):
            return value
    return None


def _scatter(rows: Tensor, mask: Tensor) -> Tensor:
    """Put ``rows`` back where ``mask`` is ``True``, with NaN rows for everyone else."""
    if rows.shape[0] == mask.shape[0]:
        return rows
    full = rows.new_full((mask.shape[0], *rows.shape[1:]), float("nan"))
    return full.index_put((mask.nonzero().squeeze(1),), rows)


def _selected_ids(ids: Sequence[str], mask: Tensor) -> list[str]:
    """The ids where ``mask`` is ``True``, in batch order."""
    return [pid for pid, keep in zip(ids, mask.tolist(), strict=True) if keep]


def _vectors(z: Any, name: str) -> Tensor:
    """Return ``z`` if it is an ``(n, d)`` tensor; otherwise raise, naming the stage list."""
    if not isinstance(z, Tensor) or z.ndim != 2:
        shape = tuple(z.shape) if isinstance(z, Tensor) else type(z).__name__
        raise TypeError(f"encoding[{name!r}] must end with (n, d) vectors, got {shape}")
    return z


def _describe(x: Any) -> str:
    """Describe a stage's input for the note added to that stage's errors."""
    if isinstance(x, Tensor):
        return f"a tensor of shape {tuple(x.shape)}"
    if isinstance(x, list) and all(isinstance(item, Tensor) for item in x):
        shapes = ", ".join(str(tuple(item.shape)) for item in x[:3]) + (", ..." if len(x) > 3 else "")
        return f"a list of {len(x)} tensor{'' if len(x) == 1 else 's'} shaped {shapes}"
    return f"a {type(x).__name__}"


class StageList(nn.ModuleList):
    """Stages applied in order. A stage with ``needs_ids = True`` is called as ``stage(x, ids)``. The input must be
    finite unless the first stage sets ``allow_nan = True``."""

    def __init__(self, stages: Sequence[nn.Module], name: str):
        if isinstance(stages, nn.Module | str):
            raise TypeError(f"encoding[{name!r}] must be a list of stages, got {type(stages).__name__}")
        stages = list(stages)
        for k, stage in enumerate(stages):
            if not isinstance(stage, nn.Module):
                raise TypeError(f"encoding[{name!r}][{k}] must be an nn.Module, got {type(stage).__name__}")
            if k > 0 and isinstance(stage, InContextModule):
                raise ValueError(
                    f"encoding[{name!r}][{k}] {type(stage).__name__} is an InContextModule; "
                    "only the first stage may be fitted"
                )
        super().__init__(stages)
        self.name = name
        self.output_width(None)

    def output_width(self, input_width: int | None) -> int | None:
        """Walk declared widths (in_dim/out_dim or in_features/out_features); raise on a broken link."""
        width = input_width
        for k, stage in enumerate(self):
            expected = _declared(stage, ("in_dim", "in_features"))
            if expected is not None and width is not None and expected != width:
                raise ValueError(
                    f"encoding[{self.name!r}][{k}] {type(stage).__name__} expects width {expected} but receives {width}"
                )
            produced = _declared(stage, ("out_dim", "out_features"))
            width = produced if produced is not None else width
        return width

    def check_input(self, x: Any, ids: list[str]) -> None:
        """Raise if a patient's input has NaN or infinite values, unless the first stage sets ``allow_nan = True``.

        Args:
            x: The stage list's input: an ``(n, ...)`` tensor, or a list of one tensor per patient (bags).
            ids: The patients of ``x``, in row order.
        """
        if self and getattr(self[0], "allow_nan", False):
            return
        if isinstance(x, Tensor):
            finite = torch.isfinite(x).reshape(len(x), -1).all(dim=1).tolist()
        elif isinstance(x, list | tuple) and all(isinstance(item, Tensor) for item in x):
            finite = [bool(torch.isfinite(item).all()) for item in x]
        else:
            return  # an input this cannot inspect; the first stage is responsible for it
        if bad := [pid for pid, ok in zip(ids, finite, strict=True) if not ok]:
            raise ValueError(
                f"encoding[{self.name!r}] input has NaN or infinite values for {len(bad)} "
                f"patient{'' if len(bad) == 1 else 's'} (e.g. {bad[:5]}); impute them in the modality's transform, "
                "or start the stage list with a stage that sets allow_nan = True"
            )

    def fit(self, train: MultimodalDataset, ids: list[str], load: Callable[[list[str]], Any]) -> None:
        """Fit the first stage on the rows ``load(ids)`` if it is an ``InContextModule``."""
        if self and isinstance(self[0], InContextModule):
            assert train.target is not None, "an InContextModule stage needs a dataset with a target"
            x = load(ids)
            self.check_input(x, ids)
            self[0].fit(x, train.target.tensors(ids), ids)

    def forward(self, x: Any, ids: list[str]) -> Any:
        """Check the input, then run the stages in order; an error gets a note naming the stage that raised it and
        its input."""
        self.check_input(x, ids)
        for k, stage in enumerate(self):
            try:
                x = stage(x, ids) if getattr(stage, "needs_ids", False) else stage(x)
            except Exception as error:
                error.add_note(f"in encoding[{self.name!r}][{k}] {type(stage).__name__}, called on {_describe(x)}")
                raise
        return x


def _loader(source: Modality) -> Callable[[list[str]], Any]:
    """A function that loads the given patients from ``source`` and collates them into one batch."""
    return lambda ids: source[ids]


def _check_modalities(data: MultimodalDataset, names: Sequence[str]) -> None:
    """Raise if the dataset lacks a modality the model reads."""
    missing = [name for name in names if name not in data.modalities]
    if missing:
        raise ValueError(f"the model uses modalities {missing} that the dataset lacks; it has {list(data.modalities)}")


def _require_fusion_method(fusion: Any, argument: str) -> None:
    """Raise if the model argument ``argument`` is not a ``FusionMethod``."""
    if not isinstance(fusion, FusionMethod):
        raise TypeError(
            f"{argument} must be a FusionMethod, such as Concat, MaskedMean or MajorityVote, "
            f"got {type(fusion).__name__}"
        )


def _presence(present: Mapping[str, Tensor]) -> Tensor:
    """Stack per-input ``(n,)`` masks into the ``(n, n_inputs)`` matrix that ``FusionMethod.defined`` takes."""
    return torch.stack(list(present.values()), dim=1)


def _check_fusion(
    fusion: FusionMethod,
    stage: Stage,
    data: MultimodalDataset,
    widths: Mapping[str, int | None],
    allow_undefined: bool,
) -> None:
    """Describe ``data`` to ``fusion`` in a ``FusionContext``, and let it raise if it cannot be fitted."""
    present = data.present[list(widths)]
    # a LateFusion branch need not cover patients with none of its modalities: another branch predicts for them
    required = present.any(axis=1) if allow_undefined else pd.Series(True, index=present.index)
    fusion.check(FusionContext(stage, present, required, widths))


class _SingleHeadModel(nn.Module):
    """Base for the models that end in one head: ``Unimodal``, ``EarlyFusion`` and ``IntermediateFusion``."""

    head: nn.Module

    @property
    def decomposable_loss(self) -> bool:
        """Whether the head's loss is a sum over patients, so accumulating gradients reproduces a larger batch."""
        return bool(getattr(self.head, "decomposable_loss", False))

    def check(self, data: MultimodalDataset) -> None:
        """Raise if this dataset cannot be used with this model (modalities, target, missing modalities)."""
        self._check(data, allow_undefined=False)

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        """Raise if ``data`` cannot be used with this model.

        Args:
            data: The dataset to check.
            allow_undefined: ``True`` for a ``LateFusion`` branch, which need not predict for every patient.
        """
        raise NotImplementedError

    def _defined_for(self, data: MultimodalDataset) -> pd.Series:
        """Which patients in ``data`` this model predicts for."""
        raise NotImplementedError

    def _check_head_width(self, width: int | None) -> None:
        """Raise if the head declares an input width other than ``width``."""
        expected = _declared(self.head, ("in_dim", "in_features"))
        if width is not None and expected is not None and width != expected:
            raise ValueError(f"head {type(self.head).__name__} expects width {expected} but receives {width}")

    def _check_target(self, data: MultimodalDataset) -> None:
        """Raise if the head cannot predict the dataset's target."""
        if data.target is not None:
            self.head.check_target(data.target)

    def _finish(self, rows: Tensor | None, defined: Tensor) -> ModelOutput:
        """Run the head on the rows of the defined patients (``None`` when there are none) and scatter its output
        back over the batch, with NaN for the other patients."""
        if rows is None:
            output = torch.full((defined.shape[0], self.head.out_dim), float("nan"), device=defined.device)
        else:
            output = _scatter(self.head(rows), defined)
        return ModelOutput(output=output, prediction=self.head.predict(output), defined=defined)

    def loss(self, output: ModelOutput, target: Mapping[str, Tensor]) -> dict[str, Tensor] | None:
        """The head's loss over the defined patients, or ``None`` when none is defined or the batch has no signal."""
        defined = output.defined
        if not bool(defined.any()):
            return None
        if bool(defined.all()):
            value = self.head.loss(output.output, target)
        else:
            index = defined.nonzero().squeeze(1)
            value = self.head.loss(output.output[index], {key: t[index] for key, t in target.items()})
        return None if value is None else {"loss": value}

    def columns(self, target: BaseTarget) -> list[str]:
        """Names of the prediction columns, as the head gives them."""
        return self.head.columns(target)


class Unimodal(_SingleHeadModel):
    """One modality's stages followed by a head."""

    def __init__(self, modality: str, encoding: list[nn.Module], head: nn.Module):
        super().__init__()
        self.modality = modality
        self.encoding = StageList(encoding, modality)
        self.head = head
        self._check_head_width(self.encoding.output_width(None))

    @property
    def modalities(self) -> list[str]:
        """The one modality this model reads."""
        return [self.modality]

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        """Raise if the dataset lacks the modality, its target does not suit the head, or, when this model is
        used on its own, some patients lack the modality."""
        _check_modalities(data, self.modalities)
        self._check_target(data)
        missing = int((~data.present[self.modality]).sum())
        if missing and not allow_undefined:
            raise ValueError(f"{missing} patients lack {self.modality!r}; add it to required_modalities")

    def _defined_for(self, data: MultimodalDataset) -> pd.Series:
        """The patients who have the modality."""
        return data.present[self.modality]

    def fit(self, train: MultimodalDataset) -> None:
        """Fit an ``InContextModule`` first stage on the training patients who have the modality."""
        self.encoding.fit(train, train.present_ids(self.modality), _loader(train.modalities[self.modality]))

    def forward(self, batch: dict) -> ModelOutput:
        """Encode the patients who have the modality and run the head on them."""
        present = batch["present"][self.modality]
        rows = None
        if bool(present.any()):
            z = self.encoding(batch["inputs"][self.modality], _selected_ids(batch["ids"], present))
            rows = _vectors(z, self.modality)
        return self._finish(rows, present)

    def stages_for(self, modality: str) -> StageList:
        """The stage list, if ``modality`` is this model's modality."""
        if modality != self.modality:
            raise KeyError(f"this model uses {self.modality!r}, not {modality!r}")
        return self.encoding


class IntermediateFusion(_SingleHeadModel):
    """Each modality's stages, then a fusion method, then one head."""

    def __init__(self, encoding: dict[str, list[nn.Module]], fusion: FusionMethod, head: nn.Module):
        super().__init__()
        if not encoding:
            raise ValueError("IntermediateFusion needs at least one modality in encoding")
        _require_fusion_method(fusion, "fusion")
        self.encoding = nn.ModuleDict({name: StageList(stages, name) for name, stages in encoding.items()})
        self.fusion = fusion
        self.head = head
        widths = self._widths()
        fusion.check(FusionContext.without_data("intermediate", widths))
        known = {name: width for name, width in widths.items() if width is not None}
        if len(known) == len(widths):
            self._check_head_width(fusion.output_dim(known))

    @property
    def modalities(self) -> list[str]:
        """The modalities in ``encoding``, in order."""
        return list(self.encoding)

    def _widths(self) -> dict[str, int | None]:
        """The output width of each modality's stage list, ``None`` where its stages do not declare one."""
        return {name: stages.output_width(None) for name, stages in self.encoding.items()}

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        """Raise if the dataset lacks a modality, its target does not suit the head, or the fusion method
        cannot be fitted on it."""
        _check_modalities(data, self.modalities)
        self._check_target(data)
        _check_fusion(self.fusion, "intermediate", data, self._widths(), allow_undefined)

    def _defined_for(self, data: MultimodalDataset) -> pd.Series:
        """The patients the fusion method can combine."""
        return self.fusion.defined_rows(data.present[self.modalities])

    def fit(self, train: MultimodalDataset) -> None:
        """Fit each ``InContextModule`` first stage on the training patients who have its modality."""
        for name, stages in self.encoding.items():
            stages.fit(train, train.present_ids(name), _loader(train.modalities[name]))

    def forward(self, batch: dict) -> ModelOutput:
        """Encode each modality on the patients who have it, fuse the defined patients, and run the head."""
        present = {name: batch["present"][name] for name in self.encoding}
        defined = self.fusion.defined(_presence(present))
        if not bool(defined.any()):
            return self._finish(None, defined)
        z = {}
        for name, stages in self.encoding.items():
            if bool(present[name].any()):
                rows = _vectors(stages(batch["inputs"][name], _selected_ids(batch["ids"], present[name])), name)
                z[name] = _scatter(rows, present[name])
        index = defined.nonzero().squeeze(1)
        fused = self.fusion({name: v[index] for name, v in z.items()}, {name: m[index] for name, m in present.items()})
        return self._finish(fused, defined)

    def stages_for(self, modality: str) -> StageList:
        """The stage list for ``modality``."""
        if modality not in self.encoding:
            raise KeyError(f"this model uses {self.modalities}, not {modality!r}")
        return self.encoding[modality]


class EarlyFusion(_SingleHeadModel):
    """Fuse raw vector modalities first, then one stage list, then one head."""

    def __init__(self, modalities: list[str], fusion: FusionMethod, encoding: list[nn.Module], head: nn.Module):
        super().__init__()
        if isinstance(modalities, str) or len(modalities) < 2:
            raise ValueError("EarlyFusion needs a list of at least two modalities; use Unimodal for one")
        _require_fusion_method(fusion, "fusion")
        self.modalities = list(modalities)
        self.fusion = fusion
        self.encoding = StageList(encoding, "fused")
        self.head = head
        fusion.check(FusionContext.without_data("early", self._widths()))
        self._check_head_width(self.encoding.output_width(None))

    def _widths(self) -> dict[str, int | None]:
        """Unknown raw widths: they depend on the table transforms, which are fitted after the model is checked."""
        return dict.fromkeys(self.modalities)

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        """Raise if the dataset lacks a modality, its target does not suit the head, or the fusion method
        cannot be fitted on it."""
        _check_modalities(data, self.modalities)
        bags = [name for name in self.modalities if not isinstance(data.modalities[name], FixedShapeModality)]
        if bags:
            raise TypeError(
                f"EarlyFusion fuses raw inputs, so its modalities must be FixedShapeModality. {bags} are not"
            )
        self._check_target(data)
        _check_fusion(self.fusion, "early", data, self._widths(), allow_undefined)

    def _defined_for(self, data: MultimodalDataset) -> pd.Series:
        """The patients the fusion method can combine."""
        return self.fusion.defined_rows(data.present[self.modalities])

    def fit(self, train: MultimodalDataset) -> None:
        """Fit an ``InContextModule`` first stage on the training patients who have every modality."""
        # the context is built from complete rows, fused exactly as forward fuses them
        ids = [pid for pid in train.ids if all(train.present.at[pid, name] for name in self.modalities)]

        def load(batch_ids: list[str]) -> Tensor:
            inputs = {name: _loader(train.modalities[name])(batch_ids) for name in self.modalities}
            everyone = torch.ones(len(batch_ids), dtype=torch.bool)
            return self.fusion(inputs, {name: everyone for name in self.modalities})

        self.encoding.fit(train, ids, load)

    def forward(self, batch: dict) -> ModelOutput:
        """Fuse the raw inputs of the defined patients, run the stage list on the result, then the head."""
        present = {name: batch["present"][name] for name in self.modalities}
        defined = self.fusion.defined(_presence(present))
        if not bool(defined.any()):
            return self._finish(None, defined)
        index = defined.nonzero().squeeze(1)
        inputs = {
            name: _scatter(batch["inputs"][name], present[name])[index]
            for name in self.modalities
            if bool(present[name].any())
        }
        fused = self.fusion(inputs, {name: present[name][index] for name in self.modalities})
        rows = _vectors(self.encoding(fused, _selected_ids(batch["ids"], defined)), "fused")
        return self._finish(rows, defined)

    def stages_for(self, modality: str) -> StageList:
        """The one stage list, named ``"fused"``."""
        if modality != "fused":
            raise KeyError("EarlyFusion has one stage list, named 'fused', which runs on the fused input")
        return self.encoding


class LateFusion(nn.Module):
    """Branch models trained jointly (loss = unweighted sum of branch losses); ``fusion`` builds the prediction from
    the branch outputs."""

    def __init__(self, branches: dict[str, nn.Module], fusion: FusionMethod):
        super().__init__()
        if len(branches) < 2:
            raise ValueError("LateFusion needs at least two branches")
        for name, branch in branches.items():
            if not isinstance(branch, _SingleHeadModel):
                raise TypeError(
                    f"branches[{name!r}] must be Unimodal, EarlyFusion or IntermediateFusion, "
                    f"got {type(branch).__name__}"
                )
        _require_fusion_method(fusion, "fusion")
        self.branches = nn.ModuleDict(branches)
        self.fusion = fusion
        fusion.check(self._fusion_context(None))

    @property
    def modalities(self) -> list[str]:
        """Every modality any branch reads, in order of first use."""
        return list(dict.fromkeys(name for branch in self.branches.values() for name in branch.modalities))

    @property
    def decomposable_loss(self) -> bool:
        """Whether every branch loss is a sum over patients, so accumulating gradients reproduces a larger batch."""
        return all(branch.decomposable_loss for branch in self.branches.values())

    @property
    def _reference_head(self) -> nn.Module:
        """The first branch's head, which turns a fused ``"output"`` into a prediction."""
        return next(iter(self.branches.values())).head

    def _fusion_context(self, data: MultimodalDataset | None) -> FusionContext:
        """The context for ``fusion``: each branch head's width and kind and, given ``data``, which patients each
        branch predicts for."""
        heads = {name: branch.head for name, branch in self.branches.items()}
        widths = {name: getattr(head, "out_dim", None) for name, head in heads.items()}
        kinds = {name: getattr(head, "target_type", None) for name, head in heads.items()}
        if data is None:
            return FusionContext.without_data("late", widths, kinds)
        # the inputs of late fusion are branches: a patient has one when that branch predicts for them
        covered = pd.DataFrame({name: branch._defined_for(data) for name, branch in self.branches.items()})
        return FusionContext("late", covered, pd.Series(True, index=covered.index), widths, kinds)

    def check(self, data: MultimodalDataset) -> None:
        """Raise if a branch cannot be used with ``data``, or ``fusion`` cannot fuse the branches."""
        for branch in self.branches.values():
            branch._check(data, allow_undefined=True)
        self.fusion.check(self._fusion_context(data))

    def fit(self, train: MultimodalDataset) -> None:
        """Fit each branch's ``InContextModule`` first stages on the training patients."""
        for branch in self.branches.values():
            branch.fit(train)

    def forward(self, batch: dict) -> ModelOutput:
        """Run every branch, then fuse the branch outputs of the patients ``fusion`` defines."""
        outputs = {name: branch(batch) for name, branch in self.branches.items()}
        present = {name: out.defined for name, out in outputs.items()}
        defined = self.fusion.defined(_presence(present))
        by_output = self.fusion.input_space == "output"
        values = {name: out.output if by_output else out.prediction for name, out in outputs.items()}
        index = defined.nonzero().squeeze(1)
        combined = self.fusion(
            {name: v[index] for name, v in values.items()}, {n: m[index] for n, m in present.items()}
        )
        output = _scatter(combined, defined)
        prediction = self._reference_head.predict(output) if by_output else output
        prediction = torch.where(defined.unsqueeze(-1), prediction, torch.full_like(prediction, float("nan")))
        return ModelOutput(output=output, prediction=prediction, defined=defined, branches=outputs)

    def loss(self, output: ModelOutput, target: Mapping[str, Tensor]) -> dict[str, Tensor] | None:
        """The unweighted sum of the branch losses, with each also reported as ``loss/<branch>``."""
        parts = {}
        for name, branch in self.branches.items():
            value = branch.loss(output.branches[name], target)
            if value is not None:
                parts[name] = value["loss"]
        if not parts:
            return None
        return {"loss": torch.stack(list(parts.values())).sum(), **{f"loss/{name}": v for name, v in parts.items()}}

    def columns(self, target: BaseTarget) -> list[str]:
        """Prediction columns: the heads' when ``fusion`` fuses head outputs, otherwise ``fusion``'s own."""
        if self.fusion.input_space == "output":
            return self._reference_head.columns(target)
        return self.fusion.columns(target)

    def stages_for(self, modality: str, branch: str | None = None) -> StageList:
        """The stage list for ``modality``; pass ``branch`` when more than one branch reads it."""
        matches = [
            (name, b)
            for name, b in self.branches.items()
            if (branch is None or name == branch) and modality in b.modalities
        ]
        if len(matches) != 1:
            candidates = [name for name, b in self.branches.items() if modality in b.modalities]
            raise ValueError(f"{len(matches)} branches match modality {modality!r}; pass branch= one of {candidates}")
        return matches[0][1].stages_for(modality)
