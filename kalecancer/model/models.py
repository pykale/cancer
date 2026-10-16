"""Models: per-modality wiring for unimodal, early, intermediate and late fusion.

Execution rule shared by every model: encoders run on the patients that have the modality, their outputs are
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
from kalecancer.model.encode.encoder_list import EncoderList
from kalecancer.model.fusion import FusionContext, FusionMethod, Stage


@dataclass
class ModelOutput:
    """Raw head output (what the loss sees), prediction, and which patients have a prediction.

    In a ``LateFusion`` model, ``branches`` holds each branch's own output. It is empty for other models.
    """

    output: Tensor
    prediction: Tensor
    defined: Tensor
    branches: dict[str, ModelOutput] = field(default_factory=dict)


def _scatter(rows: Tensor, mask: Tensor) -> Tensor:
    """Put ``rows`` back where ``mask`` is ``True``, with NaN rows for everyone else."""
    if rows.shape[0] == mask.shape[0]:
        return rows
    full = rows.new_full((mask.shape[0], *rows.shape[1:]), float("nan"))
    return full.index_put((mask.nonzero().squeeze(1),), rows)


def _selected_ids(ids: Sequence[str], mask: Tensor) -> list[str]:
    """The ids where ``mask`` is ``True``, in batch order."""
    return [pid for pid, keep in zip(ids, mask.tolist(), strict=True) if keep]


def _loader(source: Modality) -> Callable[[list[str]], Any]:
    """A function that loads the given patients from ``source`` and collates them into one batch."""
    return lambda ids: source[ids]


def _fit_encoders(
    encoders: EncoderList, train: MultimodalDataset, ids: list[str], load: Callable[[list[str]], Any]
) -> None:
    """Fit ``encoders`` on the rows ``load(ids)``, loading them only if its first encoder is fitted."""
    if encoders.needs_fit:
        assert train.target is not None, "fitting an encoder needs a dataset with a target"
        encoders.fit(load(ids), train.target, ids)


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


def _width(x: Any) -> int | None:
    """The width of an ``(n, d)`` batch, ``None`` for any other input."""
    return x.shape[1] if isinstance(x, Tensor) and x.ndim == 2 else None


class _SingleHeadModel(nn.Module):
    """Base for the models that end in one head: ``Unimodal``, ``EarlyFusion`` and ``IntermediateFusion``."""

    head: nn.Module

    @property
    def decomposable_loss(self) -> bool:
        """Whether the head's loss is a mean over patients, so accumulating gradients reproduces a larger batch."""
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

    def dry_run(self, data: MultimodalDataset, batch: dict) -> None:
        """Run the model once on ``batch``, a few of ``data``'s training patients.

        It also checks the fusion method against the real widths of its inputs. ``Pipeline.fit`` calls this before
        training, in eval mode and without gradients, so an encoder that cannot take its input fails before the first
        optimizer step.
        """
        self._dry_run(data, batch, allow_undefined=False)

    def _dry_run(self, data: MultimodalDataset, batch: dict, allow_undefined: bool) -> None:
        """``dry_run``; ``allow_undefined`` is ``True`` for a ``LateFusion`` branch, as in ``_check``."""
        self(batch)

    def _check_target(self, data: MultimodalDataset) -> None:
        """Raise if the head cannot predict the dataset's target."""
        if data.target is not None:
            self.head.check_target(data.target)

    def _finish(self, rows: Tensor | None, defined: Tensor) -> ModelOutput:
        """Run the head on the rows of the defined patients and scatter its output back over the batch.

        ``rows`` is ``None`` when no patient is defined. The other patients get NaN.
        """
        if rows is None:
            output = torch.full((defined.shape[0], self.head.out_dim), float("nan"), device=defined.device)
        else:
            try:
                output = _scatter(self.head(rows), defined)
            except Exception as error:
                error.add_note(f"in head {type(self.head).__name__}, called on a tensor of shape {tuple(rows.shape)}")
                raise
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
    """One modality's list of encoders followed by a head.

    On its own it needs every patient to have the modality. As a ``LateFusion`` branch, it predicts only for the
    patients who do.

    Args:
        modality: The name of the modality in the dataset.
        encoders: Encoders run in order on the modality. The last one must output an ``(n, d)`` embedding.
        head: The head, e.g. ``CoxHead``. Its ``in_dim`` is the width of the embedding.
    """

    def __init__(self, modality: str, encoders: list[nn.Module], head: nn.Module):
        super().__init__()
        self.modality = modality
        self.encoders = EncoderList(encoders, modality)
        self.head = head

    @property
    def modalities(self) -> list[str]:
        """The one modality this model reads."""
        return [self.modality]

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        """Raise if the dataset lacks the modality or its target does not suit the head.

        Used on its own, it also raises if some patients lack the modality.
        """
        _check_modalities(data, self.modalities)
        self._check_target(data)
        missing = int((~data.present[self.modality]).sum())
        if missing and not allow_undefined:
            raise ValueError(f"{missing} patients lack {self.modality!r}; add it to required_modalities")

    def _defined_for(self, data: MultimodalDataset) -> pd.Series:
        """The patients who have the modality."""
        return data.present[self.modality]

    def fit(self, train: MultimodalDataset) -> None:
        """Fit the first encoder, if it is a ``FittedEncoder``, on the training patients who have the modality."""
        _fit_encoders(self.encoders, train, train.present_ids(self.modality), _loader(train.modalities[self.modality]))

    def forward(self, batch: dict) -> ModelOutput:
        """Encode the patients who have the modality and run the head on them."""
        present = batch["present"][self.modality]
        z = None
        if bool(present.any()):
            z = self.encoders(batch["inputs"][self.modality], _selected_ids(batch["ids"], present))
        return self._finish(z, present)

    def encoders_for(self, modality: str) -> EncoderList:
        """The encoder list, if ``modality`` is this model's modality."""
        if modality != self.modality:
            raise KeyError(f"this model uses {self.modality!r}, not {modality!r}")
        return self.encoders


class IntermediateFusion(_SingleHeadModel):
    """Each modality's encoders, then a fusion method, then one head.

    Args:
        encoders: A list of encoders for each modality, by modality name. Each list must end in an ``(n, d)``
            embedding.
        fusion: How the embeddings are combined, e.g. ``Concat`` or ``MaskedMean``.
        head: The head, e.g. ``CoxHead``. Its ``in_dim`` is the width of the fused embedding: the sum of the
            embedding widths for ``Concat``, or their shared width for ``MaskedMean``.
    """

    def __init__(self, encoders: dict[str, list[nn.Module]], fusion: FusionMethod, head: nn.Module):
        super().__init__()
        if not encoders:
            raise ValueError("IntermediateFusion needs at least one modality in encoders")
        _require_fusion_method(fusion, "fusion")
        self.encoders = nn.ModuleDict({name: EncoderList(listed, name) for name, listed in encoders.items()})
        self.fusion = fusion
        self.head = head
        # the embedding widths are unknown until the dry run, so only rules that need no width run here
        fusion.check(FusionContext.without_data("intermediate", dict.fromkeys(self.encoders)))

    @property
    def modalities(self) -> list[str]:
        """The modalities in ``encoders``, in order."""
        return list(self.encoders)

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        """Raise if the dataset does not suit this model.

        It raises if the dataset lacks a modality, its target does not suit the head, or the fusion method cannot be
        fitted on it.
        """
        _check_modalities(data, self.modalities)
        self._check_target(data)
        _check_fusion(self.fusion, "intermediate", data, dict.fromkeys(self.modalities), allow_undefined)

    def _dry_run(self, data: MultimodalDataset, batch: dict, allow_undefined: bool) -> None:
        """Run each encoder list, check the fusion method against the embedding widths, then run the whole model."""
        widths: dict[str, int | None] = dict.fromkeys(self.modalities)
        for name, encoder_list in self.encoders.items():
            present = batch["present"][name]
            if bool(present.any()):
                widths[name] = _width(encoder_list(batch["inputs"][name], _selected_ids(batch["ids"], present)))
        _check_fusion(self.fusion, "intermediate", data, widths, allow_undefined)
        self(batch)

    def _defined_for(self, data: MultimodalDataset) -> pd.Series:
        """The patients the fusion method can combine."""
        return self.fusion.defined_rows(data.present[self.modalities])

    def fit(self, train: MultimodalDataset) -> None:
        """Fit each ``FittedEncoder`` on the training patients who have its modality."""
        for name, encoder_list in self.encoders.items():
            _fit_encoders(encoder_list, train, train.present_ids(name), _loader(train.modalities[name]))

    def forward(self, batch: dict) -> ModelOutput:
        """Encode each modality on the patients who have it, fuse the defined patients, and run the head."""
        present = {name: batch["present"][name] for name in self.encoders}
        defined = self.fusion.defined(_presence(present))
        if not bool(defined.any()):
            return self._finish(None, defined)
        z = {}
        for name, encoder_list in self.encoders.items():
            if bool(present[name].any()):
                rows = encoder_list(batch["inputs"][name], _selected_ids(batch["ids"], present[name]))
                z[name] = _scatter(rows, present[name])
        index = defined.nonzero().squeeze(1)
        fused = self.fusion({name: v[index] for name, v in z.items()}, {name: m[index] for name, m in present.items()})
        return self._finish(fused, defined)

    def encoders_for(self, modality: str) -> EncoderList:
        """The encoder list for ``modality``."""
        if modality not in self.encoders:
            raise KeyError(f"this model uses {self.modalities}, not {modality!r}")
        return self.encoders[modality]


class EarlyFusion(_SingleHeadModel):
    """Fuse raw vector modalities first, then one encoder list, then one head.

    Args:
        modalities: At least two modality names, all ``FixedShapeModality`` (e.g. ``Tabular``), in the order they
            are fused.
        fusion: How the raw inputs are combined, e.g. ``Concat``.
        encoders: Encoders run in order on the fused input. The last one must output an ``(n, d)`` embedding.
        head: The head, e.g. ``CoxHead``. Its ``in_dim`` is the width of the embedding.
    """

    def __init__(self, modalities: list[str], fusion: FusionMethod, encoders: list[nn.Module], head: nn.Module):
        super().__init__()
        if isinstance(modalities, str) or len(modalities) < 2:
            raise ValueError("EarlyFusion needs a list of at least two modalities; use Unimodal for one")
        _require_fusion_method(fusion, "fusion")
        self.modalities = list(modalities)
        self.fusion = fusion
        self.encoders = EncoderList(encoders, "fused")
        self.head = head
        # the raw widths depend on the transforms, which are fitted later, so only rules that need no width run here
        fusion.check(FusionContext.without_data("early", dict.fromkeys(self.modalities)))

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        """Raise if the dataset does not suit this model.

        It raises if the dataset lacks a modality, a modality is not a ``FixedShapeModality``, its target does not
        suit the head, or the fusion method cannot be fitted on it.
        """
        _check_modalities(data, self.modalities)
        bags = [name for name in self.modalities if not isinstance(data.modalities[name], FixedShapeModality)]
        if bags:
            raise TypeError(
                f"EarlyFusion fuses raw inputs, so its modalities must be FixedShapeModality. {bags} are not"
            )
        self._check_target(data)
        _check_fusion(self.fusion, "early", data, dict.fromkeys(self.modalities), allow_undefined)

    def _dry_run(self, data: MultimodalDataset, batch: dict, allow_undefined: bool) -> None:
        """Check the fusion method against the widths of the transformed inputs, then run the whole model."""
        widths = {name: _width(batch["inputs"].get(name)) for name in self.modalities}
        _check_fusion(self.fusion, "early", data, widths, allow_undefined)
        self(batch)

    def _defined_for(self, data: MultimodalDataset) -> pd.Series:
        """The patients the fusion method can combine."""
        return self.fusion.defined_rows(data.present[self.modalities])

    def fit(self, train: MultimodalDataset) -> None:
        """Fit the first encoder, if it is a ``FittedEncoder``.

        It is fitted on the fused input of the training patients who have every modality.
        """
        # the context is built from complete rows, fused exactly as forward fuses them
        ids = [pid for pid in train.ids if all(train.present.at[pid, name] for name in self.modalities)]

        def load(batch_ids: list[str]) -> Tensor:
            inputs = {name: _loader(train.modalities[name])(batch_ids) for name in self.modalities}
            everyone = torch.ones(len(batch_ids), dtype=torch.bool)
            return self.fusion(inputs, {name: everyone for name in self.modalities})

        _fit_encoders(self.encoders, train, ids, load)

    def forward(self, batch: dict) -> ModelOutput:
        """Fuse the raw inputs of the defined patients, run the encoder list on the result, then the head."""
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
        rows = self.encoders(fused, _selected_ids(batch["ids"], defined))
        return self._finish(rows, defined)

    def encoders_for(self, modality: str) -> EncoderList:
        """The one encoder list, named ``"fused"``."""
        if modality != "fused":
            raise KeyError("EarlyFusion has one encoder list, named 'fused', which runs on the fused input")
        return self.encoders


class LateFusion(nn.Module):
    """Branch models trained together, whose outputs ``fusion`` combines into one prediction.

    The loss is the unweighted sum of the branch losses. Each branch predicts only for the patients who have its
    modalities.

    Args:
        branches: At least two models, by name. Each is a ``Unimodal``, ``EarlyFusion`` or ``IntermediateFusion``
            with its own head.
        fusion: How the branch outputs are combined, e.g. ``MaskedMean`` or ``MajorityVote``.
    """

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
        """Whether every branch loss is a mean over patients, so accumulating gradients reproduces a larger batch."""
        return all(branch.decomposable_loss for branch in self.branches.values())

    @property
    def _reference_head(self) -> nn.Module:
        """The first branch's head, which turns a fused ``"output"`` into a prediction."""
        return next(iter(self.branches.values())).head

    def _fusion_context(self, data: MultimodalDataset | None) -> FusionContext:
        """The context for ``fusion``.

        It holds each branch head's width and kind and, given ``data``, which patients each branch predicts for.
        """
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
        """Call ``fit`` on each branch, which fits its first encoder if it is a ``FittedEncoder``."""
        for branch in self.branches.values():
            branch.fit(train)

    def dry_run(self, data: MultimodalDataset, batch: dict) -> None:
        """Dry-run each branch, as ``_SingleHeadModel.dry_run`` does, then run the whole model once on ``batch``."""
        for name, branch in self.branches.items():
            try:
                branch._dry_run(data, batch, allow_undefined=True)
            except Exception as error:
                error.add_note(f"in branch {name!r} ({type(branch).__name__})")
                raise
        self(batch)

    def forward(self, batch: dict) -> ModelOutput:
        """Run every branch, then fuse the branch outputs of the patients ``fusion`` defines."""
        outputs = {}
        for name, branch in self.branches.items():
            try:
                outputs[name] = branch(batch)
            except Exception as error:
                error.add_note(f"in branch {name!r} ({type(branch).__name__})")
                raise
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

    def encoders_for(self, modality: str, branch: str | None = None) -> EncoderList:
        """The encoder list for ``modality``; pass ``branch`` when more than one branch reads it."""
        matches = [
            (name, b)
            for name, b in self.branches.items()
            if (branch is None or name == branch) and modality in b.modalities
        ]
        if len(matches) != 1:
            candidates = [name for name, b in self.branches.items() if modality in b.modalities]
            raise ValueError(f"{len(matches)} branches match modality {modality!r}; pass branch= one of {candidates}")
        return matches[0][1].encoders_for(modality)
