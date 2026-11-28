"""Models: per-modality wiring for unimodal, early, intermediate and late fusion.

Execution rule shared by every model: encoders run on the patients that have the modality, their outputs are
scattered back into the batch with NaN for absent patients, fusion methods select defined rows (never multiply
by a mask), and heads run on defined rows only. This keeps gradients finite when modalities are missing.

``Unimodal``, ``EarlyFusion`` and ``IntermediateFusion`` end in one endpoint, which turns their embedding into a typed
output. ``LateFusion`` combines the typed outputs of such models, and an optional endpoint of its own takes the fused
output.

Models never test for a particular fusion method, only for its interface (``TensorFusion`` or ``OutputFusion``). They
ask the method's ``defined`` which patients it combines, and check themselves that it covers every patient they must
predict for; the method's own rules run through its hooks (``output_type``, ``check_branches``, ``check_widths``).
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, Literal

import pandas as pd
import torch
from torch import Tensor, nn

from kalecancer.loaddata.dataset import MultimodalDataset
from kalecancer.loaddata.modalities import FixedShapeModality, Modality
from kalecancer.loaddata.targets import BaseTarget
from kalecancer.model.encode.encoder_list import EncoderList
from kalecancer.model.endpoints import BaseEndpoint
from kalecancer.model.fusion import FusionMethod, OutputFusion, TensorFusion
from kalecancer.model.outputs import BaseOutput

Stage = Literal["early", "intermediate", "late"]


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
    return lambda ids: source.tensors(ids)


def _fit_encoders(
    encoders: EncoderList, train: MultimodalDataset, ids: list[str], load: Callable[[list[str]], Any]
) -> None:
    """Fit ``encoders`` on the rows ``load(ids)``, loading them only if its first encoder is fitted.

    The rows are loaded without gradients: they are the encoder's fitting data, not part of training.
    """
    if not encoders.needs_fit:
        return
    assert train.target is not None, "fitting an encoder needs a dataset with a target"
    with torch.no_grad():
        x = load(ids)
    encoders.fit(x, train.target, ids)


def _fusion_stages(fusion: FusionMethod) -> str:
    """The stages ``fusion`` can be used in, from the interfaces it implements, for error messages."""
    stages = (["early", "intermediate"] if isinstance(fusion, TensorFusion) else []) + (
        ["late"] if isinstance(fusion, OutputFusion) else []
    )
    return " or ".join(stages)


def _require_interface(fusion: Any, stage: Stage) -> None:
    """Raise unless ``fusion`` has the interface ``stage`` needs: ``OutputFusion`` in late fusion, else ``TensorFusion``."""
    if not isinstance(fusion, FusionMethod):
        raise TypeError(
            f"fusion must be a FusionMethod, such as Concat, MaskedMean or MajorityVote, got {type(fusion).__name__}"
        )
    kind = OutputFusion if stage == "late" else TensorFusion
    if not isinstance(fusion, kind):
        raise TypeError(
            f"{type(fusion).__name__} is for {_fusion_stages(fusion)} fusion, not {stage} fusion: {stage} fusion takes "
            f"a fusion method that subclasses {kind.__name__}"
        )


def _require_endpoint(endpoint: Any) -> None:
    """Raise if ``endpoint`` is not a ``BaseEndpoint``."""
    if not isinstance(endpoint, BaseEndpoint):
        raise TypeError(
            f"endpoint must be a BaseEndpoint, such as CoxEndpoint or ClassificationEndpoint, got {type(endpoint).__name__}"
        )


def _presence(present: Mapping[str, Tensor]) -> Tensor:
    """Stack per-input ``(n,)`` masks into the ``(n, n_inputs)`` matrix that ``FusionMethod.defined`` takes."""
    return torch.stack(list(present.values()), dim=1)


def _check_presence(fusion: FusionMethod, stage: Stage, present: pd.DataFrame, required: pd.Series) -> None:
    """Raise unless ``fusion`` defines every ``required`` patient.

    Args:
        fusion: The model's fusion method, which says through ``defined`` which patients it combines.
        stage: The model's stage, which names the inputs in the error message.
        present: One row per patient and one bool column per input: modalities, or in late fusion branches.
        required: The patients the model must predict for, indexed like ``present``.
    """
    lacking = required & ~fusion.defined_rows(present)
    if lacking.any():
        counts = {col: int((lacking & ~present[col]).sum()) for col in present.columns}
        inputs, advice = (
            ("branches", "each patient needs the modalities of at least one branch")
            if stage == "late"
            else (
                "modalities",
                "add the modalities to required_modalities, or use a fusion method that accepts patients who lack some",
            )
        )
        raise ValueError(
            f"{type(fusion).__name__} cannot combine {int(lacking.sum())} patients "
            f"(how many of them lack each of the {inputs}: {counts}); {advice}"
        )


def _check_fusion(
    fusion: FusionMethod, stage: Stage, data: MultimodalDataset, modalities: list[str], allow_undefined: bool
) -> None:
    """Raise unless ``fusion`` combines every patient of ``data`` the model must predict for."""
    present = data.present[modalities]
    # a LateFusion branch need not cover patients with none of its modalities: another branch predicts for them
    required = present.any(axis=1) if allow_undefined else pd.Series(True, index=present.index)
    _check_presence(fusion, stage, present, required)


def _width(x: Any) -> int | None:
    """The width of an ``(n, d)`` batch, ``None`` for any other input."""
    return x.shape[1] if isinstance(x, Tensor) and x.ndim == 2 else None


def _known(widths: Mapping[str, int | None]) -> dict[str, int]:
    """The widths that are known."""
    return {name: width for name, width in widths.items() if width is not None}


def _defined_rows(output: BaseOutput, target: Mapping[str, Tensor]) -> tuple[BaseOutput, Mapping[str, Tensor]] | None:
    """``output`` and ``target`` for the patients ``output`` defines; ``None`` when it defines none."""
    defined = output.defined
    if not bool(defined.any()):
        return None
    if not bool(defined.all()):
        return output.select(defined), {key: t[defined] for key, t in target.items()}
    return output, target


def _endpoint_loss(endpoint: BaseEndpoint, output: BaseOutput, target: Mapping[str, Tensor]) -> Tensor | None:
    """``endpoint``'s loss over the patients ``output`` defines; ``None`` when it defines none or there is no signal."""
    rows = _defined_rows(output, target)
    return None if rows is None else endpoint.loss(*rows)


class BaseModel(nn.Module, ABC):
    """What ``Pipeline`` needs from a model: its modalities, its output type, and how to check, fit, run and train it.

    ``Pipeline.fit`` calls, in order, ``check`` (when ``fit`` starts), ``fit`` (fitted encoders and endpoints, on the
    training patients), ``dry_run`` (a few training patients, before training), then ``forward`` and ``loss`` on each
    batch.
    """

    @property
    @abstractmethod
    def modalities(self) -> list[str]:
        """The modalities this model reads."""

    @property
    @abstractmethod
    def output_type(self) -> type[BaseOutput]:
        """The type of output ``forward`` returns."""

    @property
    @abstractmethod
    def decomposable_loss(self) -> bool:
        """Whether the loss is a mean over patients, so accumulating gradients reproduces a larger batch."""

    @abstractmethod
    def check(self, data: MultimodalDataset) -> None:
        """Raise if this dataset cannot be used with this model (modalities, target, missing modalities)."""

    @abstractmethod
    def fit(self, train: MultimodalDataset) -> None:
        """Fit what is learned from the training patients before training: endpoints and fitted encoders."""

    @abstractmethod
    def dry_run(self, data: MultimodalDataset, batch: dict) -> None:
        """Run the model and its loss once on ``batch``, a few of ``data``'s training patients.

        ``Pipeline.fit`` calls this before training, in eval mode and without gradients, so an encoder that cannot
        take its input, or a loss that cannot read the target, fails before the first optimizer step.
        """

    @abstractmethod
    def forward(self, batch: dict) -> BaseOutput:
        """The output for every patient in ``batch``, NaN where the model does not predict."""

    @abstractmethod
    def fit_state(self, output: BaseOutput, target: BaseTarget) -> dict[str, int]:
        """Fit the endpoints' state on ``output``, the training patients' output, once training ends.

        Returns:
            The counts the endpoint of the model's own output reports, such as ``baseline_events``.
        """

    @abstractmethod
    def loss(self, output: BaseOutput, target: Mapping[str, Tensor]) -> dict[str, Tensor] | None:
        """The loss under ``"loss"``, plus any parts to report; ``None`` when the batch carries no signal."""

    @abstractmethod
    def encoders_for(self, modality: str) -> EncoderList:
        """The encoder list for ``modality``."""


class SingleEndpointModel(BaseModel):
    """Base for the models that end in one endpoint: ``Unimodal``, ``EarlyFusion`` and ``IntermediateFusion``."""

    endpoint: BaseEndpoint

    def __init__(self, endpoint: BaseEndpoint):
        super().__init__()
        _require_endpoint(endpoint)
        self.endpoint = endpoint

    @property
    def output_type(self) -> type[BaseOutput]:
        """The endpoint's output type."""
        return self.endpoint.output_type

    @property
    def decomposable_loss(self) -> bool:
        """Whether the endpoint's loss is a mean over patients."""
        return self.endpoint.decomposable_loss

    def check(self, data: MultimodalDataset) -> None:
        """Raise if this dataset cannot be used with this model (modalities, target, missing modalities)."""
        # check modalities
        missing = [modality for modality in self.modalities if modality not in data.modalities]
        if missing:
            raise ValueError(
                f"the model uses modalities {missing} that the dataset lacks; it has {list(data.modalities)}"
            )

        # check target
        if data.target is not None:
            self.endpoint.check_target(data.target)

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

    def fit(self, train: MultimodalDataset) -> None:
        """Bind the endpoint to the training target, then fit any ``FittedEncoder`` on the training patients."""
        assert train.target is not None, "fit needs a dataset with a target"
        self.endpoint.bind(train.target)
        self._fit_encoders(train)

    def _fit_encoders(self, train: MultimodalDataset) -> None:
        raise NotImplementedError

    def dry_run(self, data: MultimodalDataset, batch: dict) -> None:
        """Run the model and its loss once on ``batch``.

        It also checks the fusion method against the real widths of its inputs.
        """
        self.loss(self._dry_run(data, batch, allow_undefined=False), batch["target"])

    def fit_state(self, output: BaseOutput, target: BaseTarget) -> dict[str, int]:
        """Fit the endpoint's state on the training patients' output."""
        return self.endpoint.fit_state(output, target)

    def forward(self, batch: dict) -> BaseOutput:
        """Embed the defined patients, run the endpoint's head on them, and build the output for the whole batch."""
        rows, defined = self._embed(batch)
        return self._finish(None if rows is None else self._head(rows), defined, batch["ids"])

    def _embed(self, batch: dict) -> tuple[Tensor | None, Tensor]:
        """The embedding of the defined patients, ``None`` when no patient is defined, and which patients are."""
        raise NotImplementedError

    def _dry_run(self, data: MultimodalDataset, batch: dict, allow_undefined: bool) -> BaseOutput:
        """``forward`` with the checks of ``dry_run``; ``allow_undefined`` is ``True`` for a ``LateFusion`` branch.

        The fusion method is checked against the real widths of its inputs, and the head's output against the width
        the endpoint declares (``out_dim``). The head cannot change width after this, so ``forward`` does not check it.
        """
        self._check_widths(batch)
        rows, defined = self._embed(batch)
        raw = None
        if rows is not None:
            raw = self._head(rows)
            expected = self.endpoint.out_dim
            if raw.ndim != 2 or raw.shape[1] != expected:
                name = type(self.endpoint).__name__
                raise ValueError(
                    f"the head of {name} ({type(self.endpoint.head).__name__}) outputs a tensor of shape "
                    f"{tuple(raw.shape)}, but {name} expects ({len(rows)}, {expected})"
                )
        return self._finish(raw, defined, batch["ids"])

    def _check_widths(self, batch: dict) -> None:
        """Check the fusion method, if any, against the real widths of its inputs in ``batch``. Nothing by default."""

    def _head(self, rows: Tensor) -> Tensor:
        """The endpoint's head on the embedding of the defined patients, with a note naming it on error."""
        try:
            return self.endpoint(rows)
        except Exception as error:
            error.add_note(
                f"in the head of {type(self.endpoint).__name__} ({type(self.endpoint.head).__name__}), "
                f"called on a tensor of shape {tuple(rows.shape)}"
            )
            raise

    def _finish(self, raw: Tensor | None, defined: Tensor, ids: list[str]) -> BaseOutput:
        """Build the output for the whole batch from the head's output on the defined patients.

        ``raw`` is ``None`` when no patient is defined. The other patients get NaN rows of the endpoint's ``out_dim``.
        """
        if raw is None:
            full = torch.full((defined.shape[0], self.endpoint.out_dim), float("nan"), device=defined.device)
        else:
            full = _scatter(raw, defined)
        return self.endpoint.to_output(full, list(ids), defined)

    def loss(self, output: BaseOutput, target: Mapping[str, Tensor]) -> dict[str, Tensor] | None:
        """The endpoint's loss over the defined patients, or ``None`` when none is defined or the batch has no signal."""
        value = _endpoint_loss(self.endpoint, output, target)
        return None if value is None else {"loss": value}


class Unimodal(SingleEndpointModel):
    """One modality's list of encoders followed by an endpoint.

    On its own it needs every patient to have the modality. As a ``LateFusion`` branch, it predicts only for the
    patients who do.

    Args:
        modality: The name of the modality in the dataset.
        encoders: Encoders run in order on the modality. The last one must output an ``(n, d)`` embedding.
        endpoint: The endpoint, e.g. ``CoxEndpoint(nn.Linear(d, 1, bias=False), ties="efron")``. Its head takes the
            embedding.
    """

    def __init__(self, modality: str, encoders: list[nn.Module], endpoint: BaseEndpoint):
        super().__init__(endpoint)
        self.modality = modality
        self.encoders = EncoderList(encoders, modality)

    @property
    def modalities(self) -> list[str]:
        """The one modality this model reads."""
        return [self.modality]

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        """Raise if the dataset lacks the modality or its target does not suit the endpoint.

        Used on its own, it also raises if some patients lack the modality.
        """
        missing = int((~data.present[self.modality]).sum())
        if missing and not allow_undefined:
            raise ValueError(f"{missing} patients lack {self.modality!r}; add it to required_modalities")

    def _defined_for(self, data: MultimodalDataset) -> pd.Series:
        """The patients who have the modality."""
        return data.present[self.modality]

    def _fit_encoders(self, train: MultimodalDataset) -> None:
        """Fit the first encoder, if it is a ``FittedEncoder``, on the training patients who have the modality."""
        _fit_encoders(self.encoders, train, train.present_ids(self.modality), _loader(train.modalities[self.modality]))

    def _embed(self, batch: dict) -> tuple[Tensor | None, Tensor]:
        """Encode the patients who have the modality."""
        present = batch["present"][self.modality]
        z = None
        if bool(present.any()):
            z = self.encoders(batch["inputs"][self.modality], _selected_ids(batch["ids"], present))
        return z, present

    def encoders_for(self, modality: str) -> EncoderList:
        """The encoder list, if ``modality`` is this model's modality."""
        if modality != self.modality:
            raise KeyError(f"this model uses {self.modality!r}, not {modality!r}")
        return self.encoders


class IntermediateFusion(SingleEndpointModel):
    """Each modality's encoders, then a fusion method, then one endpoint.

    Args:
        encoders: A list of encoders for each modality, by modality name. Each list must end in an ``(n, d)``
            embedding.
        fusion: How the embeddings are combined, e.g. ``Concat`` or ``MaskedMean``.
        endpoint: The endpoint, e.g. ``CoxEndpoint(nn.Linear(d, 1, bias=False), ties="efron")``. Its head takes the fused
            embedding: the sum of the embedding widths for ``Concat``, or their shared width for ``MaskedMean``.
    """

    stage: ClassVar[Stage] = "intermediate"

    def __init__(self, encoders: dict[str, list[nn.Module]], fusion: TensorFusion, endpoint: BaseEndpoint):
        super().__init__(endpoint)
        if not encoders:
            raise ValueError("IntermediateFusion needs at least one modality in encoders")
        self.encoders = nn.ModuleDict({name: EncoderList(listed, name) for name, listed in encoders.items()})
        _require_interface(fusion, self.stage)
        self.fusion = fusion

    @property
    def modalities(self) -> list[str]:
        """The modalities in ``encoders``, in order."""
        return list(self.encoders)

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        """Raise if the dataset does not suit this model.

        It raises if the dataset lacks a modality, its target does not suit the endpoint, or the fusion method cannot
        be fitted on it.
        """
        _check_fusion(self.fusion, self.stage, data, self.modalities, allow_undefined)

    def _check_widths(self, batch: dict) -> None:
        """Run each encoder list and check the fusion method against the embedding widths."""
        widths: dict[str, int | None] = {}
        for name, encoder_list in self.encoders.items():
            present = batch["present"][name]
            if bool(present.any()):
                widths[name] = _width(encoder_list(batch["inputs"][name], _selected_ids(batch["ids"], present)))
        self.fusion.check_widths(_known(widths))

    def _defined_for(self, data: MultimodalDataset) -> pd.Series:
        """The patients the fusion method can combine."""
        return self.fusion.defined_rows(data.present[self.modalities])

    def _fit_encoders(self, train: MultimodalDataset) -> None:
        """Fit each ``FittedEncoder`` on the training patients who have its modality."""
        for name, encoder_list in self.encoders.items():
            _fit_encoders(encoder_list, train, train.present_ids(name), _loader(train.modalities[name]))

    def _embed(self, batch: dict) -> tuple[Tensor | None, Tensor]:
        """Encode each modality on the patients who have it, and fuse the defined patients."""
        present = {name: batch["present"][name] for name in self.encoders}
        defined = self.fusion.defined(_presence(present))
        if not bool(defined.any()):
            return None, defined
        z = {}
        for name, encoder_list in self.encoders.items():
            if bool(present[name].any()):
                rows = encoder_list(batch["inputs"][name], _selected_ids(batch["ids"], present[name]))
                z[name] = _scatter(rows, present[name])
        index = defined.nonzero().squeeze(1)
        fused = self.fusion({name: v[index] for name, v in z.items()}, {name: m[index] for name, m in present.items()})
        return fused, defined

    def encoders_for(self, modality: str) -> EncoderList:
        """The encoder list for ``modality``."""
        if modality not in self.encoders:
            raise KeyError(f"this model uses {self.modalities}, not {modality!r}")
        return self.encoders[modality]


class EarlyFusion(SingleEndpointModel):
    """Fuse raw vector modalities first, then one encoder list, then one endpoint.

    Args:
        modalities: At least two modality names, all ``FixedShapeModality`` (e.g. ``Tabular``), in the order they
            are fused.
        fusion: How the raw inputs are combined, e.g. ``Concat``.
        encoders: Encoders run in order on the fused input. The last one must output an ``(n, d)`` embedding.
        endpoint: The endpoint, e.g. ``CoxEndpoint(nn.Linear(d, 1, bias=False), ties="efron")``. Its head takes the
            embedding.
    """

    stage: ClassVar[Stage] = "early"

    def __init__(self, modalities: list[str], fusion: TensorFusion, encoders: list[nn.Module], endpoint: BaseEndpoint):
        super().__init__(endpoint)
        if isinstance(modalities, str) or len(modalities) < 2:
            raise ValueError("EarlyFusion needs a list of at least two modalities; use Unimodal for one")
        self._modalities = list(modalities)
        _require_interface(fusion, self.stage)
        self.fusion = fusion
        self.encoders = EncoderList(encoders, "fused")

    @property
    def modalities(self) -> list[str]:
        """The modalities that are fused, in order."""
        return self._modalities

    def _check(self, data: MultimodalDataset, allow_undefined: bool) -> None:
        """Raise if the dataset does not suit this model.

        It raises if the dataset lacks a modality, a modality is not a ``FixedShapeModality``, its target does not
        suit the endpoint, or the fusion method cannot be fitted on it.
        """
        bags = [name for name in self.modalities if not isinstance(data.modalities[name], FixedShapeModality)]
        if bags:
            raise TypeError(
                f"EarlyFusion fuses raw inputs, so its modalities must be FixedShapeModality. {bags} are not"
            )
        _check_fusion(self.fusion, self.stage, data, self.modalities, allow_undefined)

    def _check_widths(self, batch: dict) -> None:
        """Check the fusion method against the widths of the transformed inputs."""
        self.fusion.check_widths(_known({name: _width(batch["inputs"].get(name)) for name in self.modalities}))

    def _defined_for(self, data: MultimodalDataset) -> pd.Series:
        """The patients the fusion method can combine."""
        return self.fusion.defined_rows(data.present[self.modalities])

    def _fit_encoders(self, train: MultimodalDataset) -> None:
        """Fit the first encoder, if it is a ``FittedEncoder``.

        It is fitted on the fused input of the training patients who have every modality. So the fusion method must
        have no trainable parameters: training would change the fused input after the encoder was fitted on it.

        Raises:
            ValueError: If the first encoder is fitted and the fusion method has trainable parameters.
        """
        if self.encoders.needs_fit and any(p.requires_grad for p in self.fusion.parameters()):
            raise ValueError(
                f"{type(self.encoders[0]).__name__} is fitted on the fused input before training, so the fusion "
                f"method in front of it must not train, but {type(self.fusion).__name__} has trainable parameters. "
                "Freeze them (requires_grad_(False)), or use IntermediateFusion, whose encoders see each modality's "
                "own input"
            )
        # the context is built from complete rows, fused exactly as forward fuses them
        ids = [pid for pid in train.ids if all(train.present.at[pid, name] for name in self.modalities)]

        def load(batch_ids: list[str]) -> Tensor:
            inputs = {name: _loader(train.modalities[name])(batch_ids) for name in self.modalities}
            everyone = torch.ones(len(batch_ids), dtype=torch.bool)
            return self.fusion(inputs, {name: everyone for name in self.modalities})

        _fit_encoders(self.encoders, train, ids, load)

    def _embed(self, batch: dict) -> tuple[Tensor | None, Tensor]:
        """Fuse the raw inputs of the defined patients, and run the encoder list on the result."""
        present = {name: batch["present"][name] for name in self.modalities}
        defined = self.fusion.defined(_presence(present))
        if not bool(defined.any()):
            return None, defined
        index = defined.nonzero().squeeze(1)
        inputs = {
            name: _scatter(batch["inputs"][name], present[name])[index]
            for name in self.modalities
            if bool(present[name].any())
        }
        fused = self.fusion(inputs, {name: present[name][index] for name in self.modalities})
        return self.encoders(fused, _selected_ids(batch["ids"], defined)), defined

    def encoders_for(self, modality: str) -> EncoderList:
        """The one encoder list, named ``"fused"``."""
        if modality != "fused":
            raise KeyError("EarlyFusion has one encoder list, named 'fused', which runs on the fused input")
        return self.encoders


class LateFusion(BaseModel):
    """Branch models trained together, whose outputs ``fusion`` combines into one output.

    Each branch predicts only for the patients who have its modalities. ``fusion`` combines the branches' typed
    outputs, and ``endpoint``, when given, takes the result: it attaches the fused output's own state, such as a
    Breslow baseline fitted on the fused log-hazards, and its loss trains the fused output.

    The total loss is::

        fused loss + branch_weight × (sum of the branch losses) + the fusion method's regulariser

    The fused loss scores the fused prediction and comes from ``endpoint``; there is none when ``endpoint`` is
    ``None``. The branch losses score each branch's own prediction, which keeps the branches useful on their own for
    ``branch=`` predictions. The regulariser is a penalty the fusion method may put on itself (``OutputFusion.loss``),
    weighted by the method; most methods add none.

    Args:
        branches: At least two models, by name. Each is a ``Unimodal``, ``EarlyFusion`` or ``IntermediateFusion``
            with its own endpoint.
        fusion: How the branch outputs are combined, e.g. ``MaskedMean`` or ``MajorityVote``.
        endpoint: The endpoint of the fused output, with no head, e.g. ``CoxEndpoint(None, ties="efron")``. It must
            accept what ``fusion`` outputs. ``None`` to use the fusion method's output as it is, with no fused loss
            and no fused state: for ``MajorityVote``, whose scores are not differentiable, or to train the branches
            only. Over Cox branches, ``MaskedMean`` without an endpoint gives a ``RiskScore`` with no survival curve.
        branch_weight: The weight of the branch losses against the fused loss, >= 0.
    """

    stage: ClassVar[Stage] = "late"

    def __init__(
        self,
        branches: dict[str, SingleEndpointModel],
        fusion: OutputFusion,
        endpoint: BaseEndpoint | None,
        branch_weight: float = 1.0,
    ):
        super().__init__()
        if len(branches) < 2:
            raise ValueError("LateFusion needs at least two branches")
        if reserved := sorted({"fused", "regulariser"} & set(branches)):
            raise ValueError(
                f"'fused' and 'regulariser' name the fused loss and the fusion method's regulariser in the loss report, "
                f"so a branch cannot be called {reserved[0]!r}"
            )
        for name, branch in branches.items():
            if not isinstance(branch, SingleEndpointModel):
                raise TypeError(
                    f"branches[{name!r}] must be Unimodal, EarlyFusion or IntermediateFusion, "
                    f"got {type(branch).__name__}"
                )
        if branch_weight < 0:
            raise ValueError(f"branch_weight must be >= 0, got {branch_weight}")
        self.branches = nn.ModuleDict(branches)
        types = {name: branch.output_type for name, branch in self.branches.items()}
        _require_interface(fusion, self.stage)
        self.fusion = fusion
        self.branch_weight = branch_weight
        fused = fusion.output_type(types)
        if endpoint is not None:
            _require_endpoint(endpoint)
            name = type(endpoint).__name__
            if endpoint.head is not None:
                raise ValueError(
                    f"the fused endpoint takes the output of {type(fusion).__name__}: give {name} head=None"
                )
            if not issubclass(fused, endpoint.accepts):
                raise TypeError(
                    f"{name} takes a {endpoint.accepts.__name__}, but {type(fusion).__name__} outputs a {fused.__name__} "
                    "from these branches"
                )
        self.endpoint = endpoint
        self._fused_type = fused

    @property
    def modalities(self) -> list[str]:
        """Every modality any branch reads, in order of first use."""
        return list(dict.fromkeys(name for branch in self.branches.values() for name in branch.modalities))

    @property
    def output_type(self) -> type[BaseOutput]:
        """The endpoint's output type, or without one, what ``fusion`` makes from the branch outputs."""
        return self._fused_type if self.endpoint is None else self.endpoint.output_type

    @property
    def decomposable_loss(self) -> bool:
        """Whether every loss, branch, fused and the fusion method's, is a mean over patients, so accumulating works."""
        branches = all(branch.decomposable_loss for branch in self.branches.values())
        fused = self.endpoint is None or self.endpoint.decomposable_loss
        return branches and fused and self.fusion.decomposable_loss

    def check(self, data: MultimodalDataset) -> None:
        """Raise if a branch or the fused endpoint cannot be used with ``data``, or ``fusion`` cannot fuse the branches.

        Every patient must have a branch that predicts for them, and ``fusion.check_branches`` must accept which
        branches each patient has.
        """
        for branch in self.branches.values():
            branch._check(data, allow_undefined=True)
        if self.endpoint is not None and data.target is not None:
            self.endpoint.check_target(data.target)
        # the inputs of late fusion are branches: a patient has one when that branch predicts for them
        covered = pd.DataFrame({name: branch._defined_for(data) for name, branch in self.branches.items()})
        _check_presence(self.fusion, self.stage, covered, pd.Series(True, index=covered.index))
        self.fusion.check_branches(covered, {name: branch.output_type for name, branch in self.branches.items()})

        # every branch must learn from at least one patient: a branch that predicts for nobody fits nothing,
        # yet later fails (e.g. a Breslow baseline fit on zero patients) with an unclear error, so fail early here
        uncovered = [name for name, defined in covered.items() if not bool(defined.any())]
        if uncovered:
            raise ValueError(
                f"the following branches predict for none of the patients: {uncovered}; "
                "remove them or add patients with the missing modality"
            )

    def fit(self, train: MultimodalDataset) -> None:
        """Call ``fit`` on each branch, which binds its endpoint and fits its first encoder if it is fitted.

        The fused endpoint is bound too.
        """
        for branch in self.branches.values():
            branch.fit(train)
        if self.endpoint is not None:
            assert train.target is not None, "fit needs a dataset with a target"
            self.endpoint.bind(train.target)

    def dry_run(self, data: MultimodalDataset, batch: dict) -> None:
        """Dry-run each branch, as ``SingleEndpointModel.dry_run`` does, then run the whole model and its loss."""
        for name, branch in self.branches.items():
            try:
                branch._dry_run(data, batch, allow_undefined=True)
            except Exception as error:
                error.add_note(f"in branch {name!r} ({type(branch).__name__})")
                raise
        self.loss(self(batch), batch["target"])

    def forward(self, batch: dict) -> BaseOutput:
        """Run every branch, fuse the branch outputs of the patients ``fusion`` defines, and pass them to the endpoint."""
        outputs = {}
        for name, branch in self.branches.items():
            try:
                outputs[name] = branch(batch)
            except Exception as error:
                error.add_note(f"in branch {name!r} ({type(branch).__name__})")
                raise
        present = {name: out.defined for name, out in outputs.items()}
        defined = self.fusion.defined(_presence(present))
        index = defined.nonzero().squeeze(1)
        combined = self.fusion(
            {name: out.select(index) for name, out in outputs.items()}, {n: m[index] for n, m in present.items()}
        )
        fused = combined.scatter(defined, list(batch["ids"]))
        if self.endpoint is not None:
            fused = self.endpoint.adopt(fused)
        return dataclasses.replace(fused, branches=outputs)

    def fit_state(self, output: BaseOutput, target: BaseTarget) -> dict[str, int]:
        """Fit each branch's endpoint on its own output, and the fused endpoint on the fused output."""
        for name, branch in self.branches.items():
            branch.fit_state(output.branches[name], target)
        return {} if self.endpoint is None else self.endpoint.fit_state(output, target)

    def loss(self, output: BaseOutput, target: Mapping[str, Tensor]) -> dict[str, Tensor] | None:
        """The total loss, ``fused + branch_weight × branches + regulariser``, under ``loss``, and each part.

        The parts are reported as ``loss/fused``, ``loss/<branch>`` and ``loss/regulariser``. Losses that have no signal
        in the batch are left out, and the regulariser is added only when some other loss has a signal.
        """
        parts = {}
        if self.endpoint is not None:
            value = _endpoint_loss(self.endpoint, output, target)
            if value is not None:
                parts["fused"] = value
        for name, branch in self.branches.items():
            branch_loss = branch.loss(output.branches[name], target)
            if branch_loss is not None:
                parts[name] = branch_loss["loss"]
        if not parts:
            return None
        rows = _defined_rows(output, target)
        term = None if rows is None else self.fusion.loss(*rows)
        if term is not None:
            parts["regulariser"] = term
        unweighted = ("fused", "regulariser")
        total = sum(value if name in unweighted else self.branch_weight * value for name, value in parts.items())
        return {"loss": total, **{f"loss/{name}": value for name, value in parts.items()}}

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
