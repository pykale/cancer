"""Pipeline: train one model end to end with Lightning, then predict, evaluate and encode with it.

Transforms and fold-local state are fitted on the training patients only. ``run_encoders`` runs a modality's
encoders for interpretation tools.
"""

from __future__ import annotations

import copy
import tempfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import lightning.pytorch as L
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import BaseShuffleSplit
from torch import nn
from torch.utils.data import DataLoader

from kalecancer.evaluate.metrics import Metric
from kalecancer.loaddata.dataset import MultimodalDataset
from kalecancer.model.encode.encoder_list import EncoderList
from kalecancer.model.models import BaseModel, EarlyFusion, IntermediateFusion, LateFusion, Unimodal
from kalecancer.model.outputs import BaseOutput, ClassProbabilities, SurvivalFunction
from kalecancer.pipeline.training import (
    _EarlyStoppingWithRestore,
    _InferenceModule,
    _lightning_session,
    _TrainingModule,
)
from kalecancer.prepdata.transforms import _check_no_dropped_columns

_PRECISIONS = ("32-true", "16-mixed", "bf16-mixed")


_MANAGED_TRAINER_KEYS = frozenset(
    {
        "accelerator",
        "callbacks",
        "check_val_every_n_epoch",
        "default_root_dir",
        "devices",
        "enable_checkpointing",
        "enable_model_summary",
        "enable_progress_bar",
        "logger",
        "max_epochs",
        "max_steps",
        "num_sanity_val_steps",
        "precision",
        "val_check_interval",
    }
)


_MODELS = (Unimodal, IntermediateFusion, EarlyFusion, LateFusion)


_TRAINED = "_kalecancer_trained"


class NonFiniteInputWarning(UserWarning):
    """Some patients' inputs had NaN or infinite values, and the encoders turned them into finite embeddings.

    This is expected when an encoder handles missing values itself; silence it with
    ``warnings.filterwarnings("ignore", category=NonFiniteInputWarning)``. Otherwise an encoder swallowed the values,
    and they should be imputed in the modality's transform.
    """


def _warn_non_finite(found: Mapping[str, set[str]], data: MultimodalDataset, stacklevel: int) -> None:
    """Warn once for each modality whose input had NaN or infinite values for some of ``data``'s patients."""
    for name, ids in found.items():
        bad = sorted(ids)
        warnings.warn(
            f"{len(bad)} of {len(data.present_ids(name))} patients have NaN or infinite values in {name!r} "
            f"(e.g. {bad[:5]}); the encoders turned them into finite embeddings. If they are not meant to handle "
            "missing values, impute them in the modality's transform",
            NonFiniteInputWarning,
            stacklevel=stacklevel + 1,
        )


class EarlyStopping:
    """Stop training when the validation score stops improving.

    Args:
        metric: A metric computed on the whole validation set after each epoch, or ``"loss"`` for the model's loss
            on the whole validation set (for Cox, all validation patients form one risk set).
        patience: Epochs without improvement before stopping.
        restore_best: Put back the weights of the best epoch when training ends.
        min_delta: Smallest change that counts as an improvement.
    """

    def __init__(self, metric: Metric | Literal["loss"], patience: int, restore_best: bool, min_delta: float = 0.0):
        if metric != "loss" and not isinstance(metric, Metric):
            raise TypeError(f"metric must be a kalecancer Metric or 'loss', got {metric!r}")
        if metric != "loss" and issubclass(metric.requires, SurvivalFunction):
            raise ValueError(
                f"{metric!r} cannot be monitored: it needs survival functions, and the state that gives them, such as "
                "a Breslow baseline, is fitted only once training ends"
            )
        if patience < 1 or min_delta < 0:
            raise ValueError(f"patience must be >= 1 and min_delta >= 0, got {patience} and {min_delta}")
        self.metric = metric
        self.patience = patience
        self.restore_best = restore_best
        self.min_delta = min_delta

    @property
    def mode(self) -> str:
        """``"max"`` if a higher score is better, otherwise ``"min"``."""
        if self.metric == "loss":
            return "min"
        return "max" if self.metric.higher_is_better else "min"


class Pipeline(BaseEstimator):
    """Transforms, a model and its training settings: an sklearn-style estimator over a ``MultimodalDataset``.

    ``fit`` never changes ``model``: it trains a copy, ``model_``, so ``sklearn.base.clone`` gives an independent,
    unfitted pipeline for each cross-validation fold.

    Args:
        model: A ``Unimodal``, ``EarlyFusion``, ``IntermediateFusion`` or ``LateFusion`` model.
        transforms: sklearn transformers per table modality, fitted on the training patients only.
        optimizer: Called with parameter groups, e.g. ``functools.partial(torch.optim.AdamW, lr=1e-4)``.
        batch_size: Patients per batch. For a Cox endpoint the risk set of the loss is the batch.
        drop_last: Drop the last, smaller training batch.
        max_epochs: Upper bound on training epochs.
        validation: An sklearn splitter with ``n_splits=1`` that carves validation patients out of the training
            patients; it receives the target's strata (events or labels) as ``y``. ``None`` trains on everyone.
        early_stopping: Stop on a validation score; needs ``validation``.
        accelerator: Lightning accelerator (``"cpu"``, ``"gpu"``, ``"auto"``).
        precision: ``"32-true"``, ``"16-mixed"`` or ``"bf16-mixed"``.
        random_state: Seeds what ``fit`` does: the validation split, shuffling, dropout and sampling during training,
            and the parameters of lazy modules when the dry run creates them. ``fit`` always starts from the weights
            the model was built with, so to vary them, seed before building the model. ``None`` does not seed.
        param_groups: Optimizer options for parameters under a name prefix,
            e.g. ``{"encoders.clinical": {"lr": 1e-5}}``.
        num_workers: DataLoader workers.
        trainer_kwargs: Further ``lightning.pytorch.Trainer`` arguments (not the ones this class manages).

    Attributes:
        model_: The trained copy of ``model``, on the CPU in eval mode.
        transforms_: The fitted transforms, by modality.
        train_ids_: The patients the model was trained on.
        val_ids_: The validation patients. Empty without ``validation``.
        target_: The target of the dataset given to ``fit``.
        train_target_: The target of the training patients only. Metrics such as ``UnoC`` estimate censoring
            from it.
        history_: One row per epoch, indexed by ``epoch``, with ``train_loss``, ``val_loss`` (with validation),
            ``val_metric`` (when early stopping on a metric), ``optimizer_steps`` (a running total) and
            ``skipped_batches``.
        fit_report_: A summary of the fit: ``n_train``, ``n_val``, the target's counts (e.g. ``events``),
            ``epochs_run``, ``best_epoch`` and ``best_score`` (``None`` without early stopping), the total
            ``optimizer_steps`` and ``skipped_batches``, and ``non_finite_inputs`` (for each modality, how many
            patients had NaN or infinite values in their input), and what the model's endpoint reports when it
            fits its state, such as ``baseline_events`` (the events a Cox baseline was fitted on).

    State that endpoints fit once training ends, such as a Cox model's Breslow baseline, lives on the endpoints of
    ``model_``, e.g. ``model_.endpoint.baseline_``.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        transforms: dict[str, TransformerMixin],
        optimizer: Callable[[list[dict]], torch.optim.Optimizer],
        batch_size: int,
        drop_last: bool,
        max_epochs: int,
        validation: BaseShuffleSplit | None,
        early_stopping: EarlyStopping | None,
        accelerator: str,
        precision: Literal["32-true", "16-mixed", "bf16-mixed"],
        random_state: int | None,
        param_groups: dict[str, dict[str, float]] | None = None,
        num_workers: int = 0,
        trainer_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        self.transforms = transforms
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.max_epochs = max_epochs
        self.validation = validation
        self.early_stopping = early_stopping
        self.accelerator = accelerator
        self.precision = precision
        self.random_state = random_state
        self.param_groups = param_groups
        self.num_workers = num_workers
        self.trainer_kwargs = trainer_kwargs

    # ---------------------------------------------------------------------------------------------------- fit

    def fit(self, data: MultimodalDataset) -> Pipeline:
        """Train a copy of ``model`` on ``data``.

        In order, it splits off the validation patients (when ``validation`` is set), fits the transforms and any
        ``FittedEncoder`` on the training patients, runs the model once on a few of them so that errors show up
        before training starts (the dry run), and trains with Lightning. The endpoints then fit their state, such as a
        Cox model's Breslow baseline, on the outputs that ``predict`` gives the training patients, so that it matches
        the final (restored) weights. The results are stored in the attributes that end in ``_``.

        It warns with a ``UserWarning`` if some training batches were skipped because their loss had no signal, and
        with a ``NonFiniteInputWarning`` if some inputs had NaN or infinite values.

        Args:
            data: The patients to train on. It must have a target.

        Returns:
            The pipeline itself.
        """
        self._check_arguments(data)
        assert data.target is not None, "_check_arguments requires a target"
        with _lightning_session():
            if self.random_state is not None:
                L.seed_everything(self.random_state, workers=True, verbose=False)
            train_ids, val_ids = self._split_validation(data)
            transforms = self._fit_transforms(data, train_ids)
            view = data.with_transforms(transforms)
            model = copy.deepcopy(self.model)
            train_view = view.subset(train_ids)
            model.fit(train_view)
            _dry_run(model, train_view)

            loader = self._loader(train_view, shuffle=True)
            if len(loader) == 0:
                raise ValueError(
                    f"{len(train_ids)} training patients make no batch of batch_size={self.batch_size} "
                    f"with drop_last={self.drop_last}"
                )
            stopper = None
            callbacks = []
            if self.early_stopping is not None:
                stopper = _EarlyStoppingWithRestore(self.early_stopping)
                callbacks.append(stopper)
            module = _TrainingModule(
                model=model,
                optimizer=self.optimizer,
                param_groups=self.param_groups,
                monitor=self.early_stopping.metric if self.early_stopping else None,
                validation_target=data.target if val_ids else None,
                train_target=data.target.subset(train_ids),
            )
            with tempfile.TemporaryDirectory() as root:
                trainer = self._trainer(callbacks, root)
                validation_loader = self._loader(view.subset(val_ids), shuffle=False) if val_ids else None
                trainer.fit(module, loader, validation_loader)
            if stopper is not None and stopper.best_state is not None:
                assert self.early_stopping is not None, "stopper is only created when early_stopping is set"
                if self.early_stopping.restore_best:
                    model.load_state_dict(stopper.best_state)

        model.eval().cpu()
        for submodule in model.modules():
            setattr(submodule, _TRAINED, True)
        self.model_ = model
        self.transforms_ = transforms
        self.train_ids_ = train_ids
        self.val_ids_ = val_ids
        self.target_ = data.target
        self.train_target_ = data.target.subset(train_ids)
        state_report = self._fit_state(data)
        self.history_ = pd.DataFrame(module.history).set_index("epoch") if module.history else pd.DataFrame()
        skipped = int(sum(row["skipped_batches"] for row in module.history))
        self.fit_report_ = {
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            **data.target.counts(train_ids),
            "epochs_run": len(module.history),
            "best_epoch": stopper.best_epoch if stopper else None,
            "best_score": stopper.best_value if stopper else None,
            "optimizer_steps": module.optimizer_steps,
            "skipped_batches": skipped,
            "non_finite_inputs": {name: len(ids) for name, ids in module.non_finite.items()},
            **state_report,
        }
        if skipped:
            warnings.warn(
                f"{skipped} training batches were skipped because their loss had no signal (for Cox: no event with "
                "another patient at risk); consider a larger batch_size",
                UserWarning,
                stacklevel=2,
            )
        _warn_non_finite(module.non_finite, data, stacklevel=2)
        return self

    def _check_arguments(self, data: MultimodalDataset) -> None:
        if not isinstance(data, MultimodalDataset):
            raise TypeError(f"fit takes a MultimodalDataset, got {type(data).__name__}")
        if data.target is None:
            raise ValueError("fit needs a dataset with a target")
        if not isinstance(self.model, _MODELS):
            raise TypeError(f"model must be one of {[m.__name__ for m in _MODELS]}, got {type(self.model).__name__}")
        if self.precision not in _PRECISIONS:
            raise ValueError(f"precision must be one of {_PRECISIONS}, got {self.precision!r}")
        if self.batch_size < 1 or self.max_epochs < 1:
            raise ValueError(f"batch_size and max_epochs must be positive, got {self.batch_size} and {self.max_epochs}")
        if self.early_stopping is not None and self.validation is None:
            raise ValueError("early_stopping needs validation patients: set validation")
        if self.early_stopping is not None and isinstance(self.early_stopping.metric, Metric):
            self.early_stopping.metric.check_output_type(self.model.output_type)
        if self.validation is not None and getattr(self.validation, "n_splits", None) != 1:
            raise ValueError(
                "validation must be an sklearn splitter with n_splits=1, "
                "e.g. StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)"
            )
        trainer_kwargs = self.trainer_kwargs or {}
        if clash := sorted(_MANAGED_TRAINER_KEYS & set(trainer_kwargs)):
            raise ValueError(f"trainer_kwargs cannot set {clash}: they are Pipeline arguments or managed by it")
        complete = bool(data.present[self.model.modalities].to_numpy().all()) if self.model.modalities else True
        if trainer_kwargs.get("accumulate_grad_batches", 1) != 1 and not (self.model.decomposable_loss and complete):
            raise ValueError(
                "accumulate_grad_batches does not reproduce a larger batch for this model: the loss is not a mean over "
                "patients (a Cox risk set, a fusion method's own term, or branch losses averaged over patients with a "
                "modality). "
                "Increase batch_size instead; memory-bound encoders should process their inputs in chunks"
            )
        if unknown := sorted(set(self.transforms) - set(data.modalities)):
            raise ValueError(f"transforms name modalities {unknown} that the dataset lacks")
        trained = [name for name, module in self.model.named_modules() if getattr(module, _TRAINED, False)]
        if trained:
            raise ValueError(
                f"model contains modules from a fitted pipeline ({trained[:3]}); build a fresh model, or every fold "
                "would start from weights trained on other patients"
            )
        self.model.check(data)

    def _fit_state(self, data: MultimodalDataset) -> dict[str, int]:
        """Fit the endpoints' state on the training patients' outputs, and return what they report.

        The outputs come through ``predict``'s own path, so fold-local state such as TabICL's in-context embedding of
        training rows is the same as at prediction. Validation patients are left out: they chose the model, they did
        not fit it.
        """
        assert data.target is not None, "fit checks that data has a target"
        # fit has already warned about non-finite inputs of these patients
        output = self._output(data.subset(self.train_ids_), None, stacklevel=None)
        return self.model_.fit_state(output, data.target)

    def _split_validation(self, data: MultimodalDataset) -> tuple[list[str], list[str]]:
        if self.validation is None:
            return list(data.ids), []
        assert data.target is not None, "fit checks that data has a target before calling _split_validation"
        ids = np.array(data.ids, dtype=object)
        train_index, val_index = next(self.validation.split(ids, data.target.strata(data.ids)))
        return sorted(ids[train_index].tolist()), sorted(ids[val_index].tolist())

    def _fit_transforms(self, data: MultimodalDataset, train_ids: list[str]) -> dict[str, TransformerMixin]:
        train = data.subset(train_ids)
        fitted = {}
        for name, transform in self.transforms.items():
            source = data.modalities[name]
            rows = source.transform_input(train.present_ids(name))
            estimator = clone(transform).fit(rows)
            _check_no_dropped_columns(estimator, name)
            fitted[name] = estimator
        return fitted

    def _loader(self, data: MultimodalDataset, shuffle: bool) -> DataLoader:
        generator = None
        if shuffle:
            # A dedicated generator keeps the order independent of worker settings and of other RNG consumers.
            generator = torch.Generator()
            if self.random_state is not None:
                generator.manual_seed(self.random_state)
        torch_data = data.as_torch()
        return DataLoader(
            torch_data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,
            drop_last=shuffle and self.drop_last,
            collate_fn=torch_data.collate,
            num_workers=self.num_workers,
        )

    def _trainer(self, callbacks: list, root: str, max_epochs: int | None = None) -> L.Trainer:
        return L.Trainer(
            accelerator=self.accelerator,
            devices=1,
            precision=self.precision,
            max_epochs=self.max_epochs if max_epochs is None else max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            default_root_dir=root,
            callbacks=callbacks,
            **(self.trainer_kwargs or {}),
        )

    # ------------------------------------------------------------------------------------------------ predict

    def _require_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("this Pipeline is not fitted; call fit first")

    def transformed(self, data: MultimodalDataset) -> MultimodalDataset:
        """The dataset as the fitted model sees it, with the fitted transforms applied.

        Raises if the fitted model cannot be used with ``data``, for example when a modality it reads is missing.
        """
        self._require_fitted()
        if not isinstance(data, MultimodalDataset):
            raise TypeError(f"expected a MultimodalDataset, got {type(data).__name__}")
        self.model_.check(data)
        return data.with_transforms(self.transforms_)

    def _infer(
        self, data: MultimodalDataset, step: Callable[[nn.Module, dict], Any], stacklevel: int | None = 3
    ) -> list:
        """Run ``step`` on each batch, and warn about non-finite inputs at ``stacklevel`` (``None`` does not warn)."""
        module = _InferenceModule(self.model_, step)
        with _lightning_session(), tempfile.TemporaryDirectory() as root:
            results = self._trainer([], root, max_epochs=1).predict(module, self._loader(data, shuffle=False))
        self.model_.eval().cpu()
        # _infer serves predict and run_encoders, so encode and attention export warn too
        if stacklevel is not None:
            _warn_non_finite(module.non_finite, data, stacklevel=stacklevel)
        return results

    def predict(self, data: MultimodalDataset, branch: str | None = None) -> pd.DataFrame:
        """Predictions indexed by patient id, with NaN for patients without a prediction.

        Survival curves come from ``predict_survival_function``, and class probabilities with the classes as columns
        from ``predict_proba``.

        Args:
            data: The patients to predict.
            branch: One branch of a ``LateFusion`` model to predict with, instead of the fused model.
        """
        self._require_fitted()
        return self._output(data, branch, stacklevel=3).frame()

    def predict_output(self, data: MultimodalDataset, branch: str | None = None) -> BaseOutput:
        """The model's typed output for ``data``, which metrics score, e.g. for ``bootstrap_ci``.

        Args:
            data: The patients to predict.
            branch: One branch of a ``LateFusion`` model to predict with, instead of the fused model.
        """
        self._require_fitted()
        return self._output(data, branch, stacklevel=3)

    def predict_proba(self, data: MultimodalDataset, branch: str | None = None) -> pd.DataFrame:
        """Class probabilities indexed by patient id, one column per class in the target's order.

        The columns are the classes themselves, so each keeps its own type. Patients without a prediction get NaN.
        Raises unless the model outputs class probabilities (``MajorityVote`` outputs scores that are not).
        """
        self._require_fitted()
        self._require_output(ClassProbabilities, branch, "predict_proba")
        output = self._output(data, branch, stacklevel=3)
        assert isinstance(output, ClassProbabilities), "_require_output checked the output type"
        return pd.DataFrame(
            output.probabilities.numpy(),
            index=pd.Index(output.ids, name="id"),
            columns=pd.Index(list(output.classes), name="class"),
        )

    def predict_survival_function(
        self, data: MultimodalDataset, times: Sequence[float], branch: str | None = None
    ) -> pd.DataFrame:
        """Survival probabilities S(t) indexed by patient id, one column per time.

        Patients without a prediction get NaN. Raises unless the model outputs survival functions.

        Args:
            data: The patients to predict.
            times: Increasing times >= 0.
            branch: One branch of a ``LateFusion`` model to predict with, instead of the fused model. Each branch
                has its own fitted state, such as its own Breslow baseline.
        """
        self._require_fitted()
        self._require_output(SurvivalFunction, branch, "predict_survival_function")
        checked = _checked_times(times, "times")
        return self._survival_frame(self._output(data, branch, stacklevel=3), checked)

    def _survival_frame(self, output: BaseOutput, times: list[float]) -> pd.DataFrame:
        assert isinstance(output, SurvivalFunction), "_require_output checked the output type"
        return pd.DataFrame(
            output.survival(times).numpy(), index=pd.Index(output.ids, name="id"), columns=pd.Index(times, name="time")
        )

    def _require_output(self, required: type, branch: str | None, what: str) -> None:
        """Raise unless the model, or ``branch`` of it, outputs ``required``."""
        produced = self._predictor(branch).output_type
        if not issubclass(produced, required):
            raise ValueError(
                f"{what} needs a model that outputs {required.__name__}; this one outputs {produced.__name__}"
            )

    def _predictor(self, branch: str | None) -> BaseModel:
        """The fitted model, or its ``branch``."""
        if branch is None:
            assert isinstance(self.model_, BaseModel), "fit checks the model type"
            return self.model_
        if not isinstance(self.model_, LateFusion) or branch not in self.model_.branches:
            names = list(self.model_.branches) if isinstance(self.model_, LateFusion) else []
            raise ValueError(f"branch={branch!r} needs a LateFusion model with that branch; branches: {names}")
        return self.model_.branches[branch]

    def _output(self, data: MultimodalDataset, branch: str | None, stacklevel: int | None) -> BaseOutput:
        """The output of the fitted model, or of ``branch``, for ``data``."""
        view = self.transformed(data)
        self._predictor(branch)
        step = lambda model, batch: model(batch).detach_cpu()  # noqa: E731
        output = BaseOutput.cat(self._infer(view, step, None if stacklevel is None else stacklevel + 1))
        output = output if branch is None else output.branches[branch]
        if len(set(output.ids)) != len(output.ids):
            raise AssertionError("duplicate patient ids in predictions")
        return output

    def evaluate(
        self,
        data: MultimodalDataset,
        metrics: Mapping[str, Metric],
        branch: str | None = None,
        allow_seen: bool = False,
    ) -> pd.Series:
        """Score the output for ``data``. Patients used in fit (training or validation) raise unless ``allow_seen``.

        Every metric is checked against the model's output type before predicting. IPCW metrics estimate censoring
        from ``train_target_``.

        Args:
            data: The patients to score. It must have a target.
            metrics: The metrics, by the name to report them under.
            branch: One branch of a ``LateFusion`` model to score, instead of the fused model.
            allow_seen: Score patients used in fit too.
        """
        self._require_fitted()
        if data.target is None:
            raise ValueError("evaluate needs a dataset with a target")
        if not metrics:
            raise ValueError("metrics is empty")
        seen = sorted(set(data.ids) & (set(self.train_ids_) | set(self.val_ids_)))
        if seen and not allow_seen:
            raise ValueError(
                f"{len(seen)} patients were used to fit this pipeline (e.g. {seen[:5]}); evaluate on held-out "
                "patients, or pass allow_seen=True"
            )
        output_type = self._predictor(branch).output_type
        for metric in metrics.values():
            metric.check_output_type(output_type)
        output = self._output(data, branch, stacklevel=3)
        return pd.Series(
            {name: metric(output, data.target, self.train_target_) for name, metric in metrics.items()}, name="score"
        )

    def _single_modality(self, data: MultimodalDataset, modality: str, branch: str | None) -> tuple:
        if modality not in data.modalities:
            raise ValueError(f"the dataset has no modality {modality!r}")
        view = self.transformed(data)
        encoders = (
            self.model_.encoders_for(modality, branch) if branch is not None else self.model_.encoders_for(modality)
        )
        present = view.present_ids(modality)
        if not present:
            raise ValueError(f"no patient in this dataset has {modality!r}")
        single = MultimodalDataset({modality: view.modalities[modality]}, target=None, required_modalities=[modality])
        return single.subset(present), encoders

    def run_encoders(
        self,
        data: MultimodalDataset,
        modality: str,
        fn: Callable[[EncoderList, Any, list[str]], Any],
        branch: str | None = None,
    ) -> list[tuple[list[str], Any]]:
        """Call ``fn(encoders, inputs, ids)`` on each batch of the patients in ``data`` who have ``modality``.

        The fitted transforms are applied and the model is in eval mode. ``encoders`` is the fitted encoder list for
        ``modality``. Returns ``(ids, result)`` for each batch. Interpretation tools in ``kalecancer.interpret``
        build on this. ``EarlyFusion`` models are not supported yet, because their encoders run on the fused input.
        """
        single, encoders = self._single_modality(data, modality, branch)

        def step(model: nn.Module, batch: dict) -> tuple[list[str], Any]:
            return batch["ids"], fn(encoders, batch["inputs"][modality], batch["ids"])

        return self._infer(single, step)

    def encode(self, data: MultimodalDataset, modality: str, branch: str | None = None) -> pd.DataFrame:
        """The embeddings of ``modality``: the output of its encoder list, indexed by patient id.

        ``EarlyFusion`` models are not supported yet, because their encoders run on the fused input.
        """
        # forward checks the embedding, so encode needs no check of its own
        results = self.run_encoders(data, modality, lambda encoders, x, ids: encoders(x, ids).float().cpu(), branch)
        ids = [pid for batch_ids, _ in results for pid in batch_ids]
        values = torch.cat([z for _, z in results]).numpy()
        return pd.DataFrame(
            values, index=pd.Index(ids, name="id"), columns=[f"{modality}_{k}" for k in range(values.shape[1])]
        )


def _checked_times(times: Sequence[float], name: str) -> list[float]:
    """``times`` as a list, after checking that they are finite, >= 0 and strictly increasing."""
    array = np.asarray(times, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be finite times >= 0, got {times!r}")
    if (np.diff(array) <= 0).any():
        raise ValueError(f"{name} must be strictly increasing, got {times!r}")
    return array.tolist()


def _dry_run_ids(train: MultimodalDataset, modalities: list[str]) -> list[str]:
    """Up to two training patients who have all of ``modalities``, and up to two who have each one, in id order."""
    present = train.present[modalities]
    chosen: list[str] = []
    for mask in [present.all(axis=1), *(present[name] for name in modalities)]:
        have = sum(bool(mask[pid]) for pid in chosen)
        chosen += [pid for pid in sorted(mask.index[mask]) if pid not in chosen][: max(0, 2 - have)]
    return sorted(chosen)


def _dry_run(model: nn.Module, train: MultimodalDataset) -> None:
    """Run ``model`` once on a few training patients before training, in eval mode and without gradients.

    An encoder that cannot take its input then fails before the first optimizer step, and the fusion method is
    checked against the real widths. Validation patients are never used.
    """
    sample = train.subset(_dry_run_ids(train, model.modalities)).as_torch()
    batch = sample.collate([sample[k] for k in range(len(sample))])
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model.dry_run(train, batch)
    finally:
        model.train(was_training)
