"""Models: per-modality stage lists wired into unimodal, early, intermediate and late fusion.

Execution rule shared by every model: stages run on the patients that have the modality, their outputs are
scattered back into the batch with NaN for absent patients, fusion methods select defined rows (never multiply
by a mask), and heads run on defined rows only. This keeps gradients finite when modalities are missing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from kalecancer.loaddata.dataset import MultimodalDataset
from kalecancer.loaddata.modalities import Modality
from kalecancer.loaddata.targets import TargetInfo
from kalecancer.model.incontext import InContextModule


@dataclass
class ModelOutput:
    """Raw head output (what the loss sees), prediction, and which patients have a prediction."""

    output: Tensor
    prediction: Tensor
    defined: Tensor
    branches: dict[str, ModelOutput] = field(default_factory=dict)


def _declared(stage: nn.Module, names: tuple[str, str]) -> int | None:
    for name in names:
        value = getattr(stage, name, None)
        if isinstance(value, int):
            return value
    return None


def _scatter(rows: Tensor, mask: Tensor) -> Tensor:
    if rows.shape[0] == mask.shape[0]:
        return rows
    full = rows.new_full((mask.shape[0], *rows.shape[1:]), float("nan"))
    return full.index_put((mask.nonzero().squeeze(1),), rows)


def _selected_ids(ids: Sequence[str], mask: Tensor) -> list[str]:
    return [pid for pid, keep in zip(ids, mask.tolist(), strict=True) if keep]


def _vectors(z: Any, name: str) -> Tensor:
    if not isinstance(z, Tensor) or z.ndim != 2:
        shape = tuple(z.shape) if isinstance(z, Tensor) else type(z).__name__
        raise TypeError(f"encoding[{name!r}] must end with (n, d) vectors, got {shape}")
    return z


def _describe(x: Any) -> str:
    if isinstance(x, Tensor):
        return f"a tensor of shape {tuple(x.shape)}"
    if isinstance(x, list) and all(isinstance(item, Tensor) for item in x):
        shapes = ", ".join(str(tuple(item.shape)) for item in x[:3]) + (", ..." if len(x) > 3 else "")
        return f"a list of {len(x)} tensor{'' if len(x) == 1 else 's'} shaped {shapes}"
    return f"a {type(x).__name__}"


class StageList(nn.ModuleList):
    """Stages applied in order. A stage with ``needs_ids = True`` is called as ``stage(x, ids)``."""

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

    def fit(self, train: MultimodalDataset, ids: list[str], load: Callable[[list[str]], Any]) -> None:
        """Fit the first stage on the rows ``load(ids)`` if it is an ``InContextModule``."""
        if self and isinstance(self[0], InContextModule):
            assert train.target is not None, "an InContextModule stage needs a dataset with a target"
            self[0].fit(load(ids), train.target.tensors(ids), ids)

    def forward(self, x: Any, ids: list[str]) -> Any:
        for k, stage in enumerate(self):
            try:
                x = stage(x, ids) if getattr(stage, "needs_ids", False) else stage(x)
            except Exception as error:
                error.add_note(f"in encoding[{self.name!r}][{k}] {type(stage).__name__}, called on {_describe(x)}")
                raise
        return x


def _loader(source: Modality) -> Callable[[list[str]], Any]:
    return lambda ids: source.collate([source.load(pid) for pid in ids])


def _check_modalities(data: MultimodalDataset, names: Sequence[str]) -> None:
    missing = [name for name in names if name not in data.modalities]
    if missing:
        raise ValueError(f"the model uses modalities {missing} that the dataset lacks; it has {list(data.modalities)}")


def _check_presence(data: MultimodalDataset, names: Sequence[str], fusion: nn.Module, allow_undefined: bool) -> None:
    present = data.present[list(names)]
    has_all, has_any = present.all(axis=1), present.any(axis=1)
    if not getattr(fusion, "handles_missing", False):
        lacking = (has_any & ~has_all) if allow_undefined else ~has_all
        if lacking.any():
            counts = {name: int((~present[name] & lacking).sum()) for name in names}
            raise ValueError(
                f"{type(fusion).__name__} cannot combine patients missing a modality ({counts}); add the modalities "
                "to required_modalities or use a fusion method that handles missing modalities"
            )
    elif not allow_undefined and (~has_any).any():
        raise ValueError(f"{int((~has_any).sum())} patients have none of the modalities {list(names)}")


class _SingleHeadModel(nn.Module):
    head: nn.Module

    @property
    def decomposable_loss(self) -> bool:
        return bool(getattr(self.head, "decomposable_loss", False))

    def check(self, data: MultimodalDataset) -> None:
        """Raise if this dataset cannot be used with this model (modalities, target, missing modalities)."""
        self._check(data, allow_undefined=False)

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        raise NotImplementedError

    def _check_head_width(self, width: int | None) -> None:
        expected = _declared(self.head, ("in_dim", "in_features"))
        if width is not None and expected is not None and width != expected:
            raise ValueError(f"head {type(self.head).__name__} expects width {expected} but receives {width}")

    def _check_target(self, data: MultimodalDataset) -> None:
        if data.target is not None:
            self.head.check_target(data.target.info())

    def _finish(self, rows: Tensor | None, defined: Tensor) -> ModelOutput:
        if rows is None:
            output = torch.full((defined.shape[0], self.head.out_dim), float("nan"), device=defined.device)
        else:
            output = _scatter(self.head(rows), defined)
        return ModelOutput(output=output, prediction=self.head.predict(output), defined=defined)

    def loss(self, output: ModelOutput, target: Mapping[str, Tensor]) -> dict[str, Tensor] | None:
        defined = output.defined
        if not bool(defined.any()):
            return None
        if bool(defined.all()):
            value = self.head.loss(output.output, target)
        else:
            index = defined.nonzero().squeeze(1)
            value = self.head.loss(output.output[index], {key: t[index] for key, t in target.items()})
        return None if value is None else {"loss": value}

    def columns(self, info: TargetInfo) -> list[str]:
        return self.head.columns(info)


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
        return [self.modality]

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        _check_modalities(data, self.modalities)
        self._check_target(data)
        missing = int((~data.present[self.modality]).sum())
        if missing and not allow_undefined:
            raise ValueError(f"{missing} patients lack {self.modality!r}; add it to required_modalities")

    def fit(self, train: MultimodalDataset) -> None:
        self.encoding.fit(train, train.present_ids(self.modality), _loader(train.modalities[self.modality]))

    def forward(self, batch: dict) -> ModelOutput:
        present = batch["present"][self.modality]
        rows = None
        if bool(present.any()):
            z = self.encoding(batch["inputs"][self.modality], _selected_ids(batch["ids"], present))
            rows = _vectors(z, self.modality)
        return self._finish(rows, present)

    def stages_for(self, modality: str) -> StageList:
        if modality != self.modality:
            raise KeyError(f"this model uses {self.modality!r}, not {modality!r}")
        return self.encoding


class IntermediateFusion(_SingleHeadModel):
    """Each modality's stages, then a fusion method, then one head."""

    def __init__(self, encoding: dict[str, list[nn.Module]], fusion: nn.Module, head: nn.Module):
        super().__init__()
        if not encoding:
            raise ValueError("IntermediateFusion needs at least one modality in encoding")
        self.encoding = nn.ModuleDict({name: StageList(stages, name) for name, stages in encoding.items()})
        self.fusion = fusion
        self.head = head
        widths = {name: stages.output_width(None) for name, stages in self.encoding.items()}
        if None not in widths.values():
            self._check_head_width(fusion.output_dim(widths))

    @property
    def modalities(self) -> list[str]:
        return list(self.encoding)

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        _check_modalities(data, self.modalities)
        self._check_target(data)
        _check_presence(data, self.modalities, self.fusion, allow_undefined)

    def fit(self, train: MultimodalDataset) -> None:
        for name, stages in self.encoding.items():
            stages.fit(train, train.present_ids(name), _loader(train.modalities[name]))

    def forward(self, batch: dict) -> ModelOutput:
        present = {name: batch["present"][name] for name in self.encoding}
        masks = torch.stack(list(present.values()))
        defined = masks.any(dim=0) if getattr(self.fusion, "handles_missing", False) else masks.all(dim=0)
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
        if modality not in self.encoding:
            raise KeyError(f"this model uses {self.modalities}, not {modality!r}")
        return self.encoding[modality]


class EarlyFusion(_SingleHeadModel):
    """Fuse raw vector modalities first, then one stage list, then one head."""

    def __init__(self, modalities: list[str], fusion: nn.Module, encoding: list[nn.Module], head: nn.Module):
        super().__init__()
        if isinstance(modalities, str) or len(modalities) < 2:
            raise ValueError("EarlyFusion needs a list of at least two modalities; use Unimodal for one")
        self.modalities = list(modalities)
        self.fusion = fusion
        self.encoding = StageList(encoding, "fused")
        self.head = head
        self._check_head_width(self.encoding.output_width(None))

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        _check_modalities(data, self.modalities)
        self._check_target(data)
        _check_presence(data, self.modalities, self.fusion, allow_undefined)

    def _defined(self, present: Mapping[str, Tensor]) -> Tensor:
        masks = torch.stack([present[name] for name in self.modalities])
        return masks.any(dim=0) if getattr(self.fusion, "handles_missing", False) else masks.all(dim=0)

    def fit(self, train: MultimodalDataset) -> None:
        # the context is built from complete rows, fused exactly as forward fuses them
        ids = [pid for pid in train.ids if all(train.present.at[pid, name] for name in self.modalities)]

        def load(batch_ids: list[str]) -> Tensor:
            inputs = {name: _loader(train.modalities[name])(batch_ids) for name in self.modalities}
            everyone = torch.ones(len(batch_ids), dtype=torch.bool)
            return self.fusion(inputs, {name: everyone for name in self.modalities})

        self.encoding.fit(train, ids, load)

    def forward(self, batch: dict) -> ModelOutput:
        present = {name: batch["present"][name] for name in self.modalities}
        defined = self._defined(present)
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
        if modality != "fused":
            raise KeyError("EarlyFusion has one stage list, named 'fused', which runs on the fused input")
        return self.encoding


class LateFusion(nn.Module):
    """Branch models trained jointly (loss = unweighted sum of branch losses); ``combine`` builds the prediction."""

    def __init__(self, branches: dict[str, nn.Module], combine: nn.Module):
        super().__init__()
        if len(branches) < 2:
            raise ValueError("LateFusion needs at least two branches")
        for name, branch in branches.items():
            if not isinstance(branch, _SingleHeadModel):
                raise TypeError(
                    f"branches[{name!r}] must be Unimodal, EarlyFusion or IntermediateFusion, "
                    f"got {type(branch).__name__}"
                )
        heads = {name: branch.head for name, branch in branches.items()}
        combine.check_branches(heads, complete=True)
        if len({getattr(head, "out_dim", None) for head in heads.values()}) != 1:
            raise ValueError("LateFusion branch heads must produce the same output width")
        self.branches = nn.ModuleDict(branches)
        self.combine = combine

    @property
    def modalities(self) -> list[str]:
        return list(dict.fromkeys(name for branch in self.branches.values() for name in branch.modalities))

    @property
    def decomposable_loss(self) -> bool:
        return all(branch.decomposable_loss for branch in self.branches.values())

    @property
    def _reference_head(self) -> nn.Module:
        return next(iter(self.branches.values())).head

    def check(self, data: MultimodalDataset) -> None:
        for branch in self.branches.values():
            branch._check(data, allow_undefined=True)
        present = data.present[self.modalities]
        if (~present.any(axis=1)).any():
            raise ValueError(
                f"{int((~present.any(axis=1)).sum())} patients have none of the modalities {self.modalities}"
            )
        heads = {name: branch.head for name, branch in self.branches.items()}
        self.combine.check_branches(heads, complete=bool(present.to_numpy().all()))

    def fit(self, train: MultimodalDataset) -> None:
        for branch in self.branches.values():
            branch.fit(train)

    def forward(self, batch: dict) -> ModelOutput:
        outputs = {name: branch(batch) for name, branch in self.branches.items()}
        defined = {name: out.defined for name, out in outputs.items()}
        any_defined = torch.stack(list(defined.values())).any(dim=0)
        if self.combine.input_space == "output":
            output = self.combine({name: out.output for name, out in outputs.items()}, defined)
            prediction = self._reference_head.predict(output)
        else:
            output = self.combine({name: out.prediction for name, out in outputs.items()}, defined)
            prediction = output
        prediction = torch.where(any_defined.unsqueeze(-1), prediction, torch.full_like(prediction, float("nan")))
        return ModelOutput(output=output, prediction=prediction, defined=any_defined, branches=outputs)

    def loss(self, output: ModelOutput, target: Mapping[str, Tensor]) -> dict[str, Tensor] | None:
        parts = {}
        for name, branch in self.branches.items():
            value = branch.loss(output.branches[name], target)
            if value is not None:
                parts[name] = value["loss"]
        if not parts:
            return None
        return {"loss": torch.stack(list(parts.values())).sum(), **{f"loss/{name}": v for name, v in parts.items()}}

    def columns(self, info: TargetInfo) -> list[str]:
        if self.combine.input_space == "output":
            return self._reference_head.columns(info)
        return self.combine.columns(info)

    def stages_for(self, modality: str, branch: str | None = None) -> StageList:
        matches = [
            (name, b)
            for name, b in self.branches.items()
            if (branch is None or name == branch) and modality in b.modalities
        ]
        if len(matches) != 1:
            candidates = [name for name, b in self.branches.items() if modality in b.modalities]
            raise ValueError(f"{len(matches)} branches match modality {modality!r}; pass branch= one of {candidates}")
        return matches[0][1].stages_for(modality)
