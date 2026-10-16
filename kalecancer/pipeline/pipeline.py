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

from kalecancer.evaluate.metrics import EvalContext, IntegratedBrier, Metric
from kalecancer.loaddata.dataset import MultimodalDataset
from kalecancer.loaddata.targets import Classification
from kalecancer.model.baseline import BreslowBaseline
from kalecancer.model.encode.encoder_list import EncoderList
from kalecancer.model.models import EarlyFusion, IntermediateFusion, LateFusion, Unimodal
from kalecancer.pipeline.training import (
    _concatenate,
    _EarlyStoppingWithRestore,
    _InferenceModule,
    _lightning_session,
    _to_cpu,
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
        if isinstance(metric, IntegratedBrier):
            raise ValueError(
                "IntegratedBrier cannot be monitored: it needs survival curves, and the baseline hazard that gives them "
                "is fitted only once training ends"
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
        batch_size: Patients per batch. For Cox heads the risk set of the loss is the batch.
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
        train_target_: The target frame of the training patients. Metrics such as ``UnoC`` estimate censoring
            from it.
        baseline_: For a model that predicts a ``log_hazard``, the ``BreslowBaseline`` fitted on the log-hazards
            of every patient used in fit (training and validation) once training ends; it gives the survival columns.
            ``None`` for other models.
        history_: One row per epoch, indexed by ``epoch``, with ``train_loss``, ``val_loss`` (with validation),
            ``val_metric`` (when early stopping on a metric), ``optimizer_steps`` (a running total) and
            ``skipped_batches``.
        fit_report_: A summary of the fit: ``n_train``, ``n_val``, the target's counts (e.g. ``events``),
            ``epochs_run``, ``best_epoch`` and ``best_score`` (``None`` without early stopping), the total
            ``optimizer_steps`` and ``skipped_batches``, and ``non_finite_inputs`` (for each modality, how many
            patients had NaN or infinite values in their input), and ``baseline_events`` (the events the baseline
            was fitted on) when there is a baseline.
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
        before training starts (the dry run), and trains with Lightning. For a model that predicts a ``log_hazard``,
        it then fits the Breslow baseline hazard on the log-hazards that ``predict`` gives the training and validation
        patients, so that the baseline matches the final (restored) weights and the fused prediction. The results are stored in the attributes
        that end in ``_``.

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
                validation_target=data.target.frame.loc[val_ids] if val_ids else None,
                context=EvalContext(
                    train_target=data.target.frame.loc[train_ids],
                    classes=data.target.classes if isinstance(data.target, Classification) else None,
                ),
                columns=model.columns(data.target),
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
        self.train_target_ = data.target.frame.loc[train_ids]
        self.baseline_, baseline_events = self._fit_baseline(data)
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
            **({"baseline_events": baseline_events} if self.baseline_ is not None else {}),
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
                "patients (a Cox risk set, or branch losses averaged over patients with a modality). "
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

    def _fit_baseline(self, data: MultimodalDataset) -> tuple[BreslowBaseline | None, int]:
        """The Breslow baseline of a ``log_hazard`` model, and its number of events; ``(None, 0)`` for other models.

        It is fitted on the fused prediction for every patient used in fit, through ``predict``'s own path, so
        fold-local state such as TabICL's in-context embedding of training rows is the same as at prediction.
        """
        assert data.target is not None, "fit checks that data has a target"
        if self.model_.columns(self.target_) != ["log_hazard"]:
            return None, 0
        # fit has already warned about non-finite inputs of these patients
        log_hazard = self._predict(data.subset(self.train_ids_ + self.val_ids_), None, stacklevel=None)["log_hazard"]
        log_hazard = log_hazard[log_hazard.notna()]
        outcome = data.target.frame.loc[log_hazard.index]
        baseline = BreslowBaseline().fit(log_hazard.to_numpy(), outcome["time"].to_numpy(), outcome["event"].to_numpy())
        return baseline, int(outcome["event"].sum())

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
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=generator,
            drop_last=shuffle and self.drop_last,
            collate_fn=data.collate,
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

    def predict(
        self, data: MultimodalDataset, branch: str | None = None, survival_times: Sequence[float] | None = None
    ) -> pd.DataFrame:
        """Predictions indexed by patient id, with NaN for patients without a prediction.

        Args:
            data: The patients to predict.
            branch: One branch of a ``LateFusion`` model to predict with, instead of the fused model.
            survival_times: Increasing times >= 0 at which to add the survival probability of a ``log_hazard``
                prediction, from ``baseline_``, as ``survival[<time>]`` columns (e.g. for ``IntegratedBrier``). Not
                with ``branch``, because the baseline belongs to the fused log-hazard.
        """
        times = self._check_survival_times(survival_times, branch)
        frame = self._predict(data, branch, stacklevel=3)
        if times is None:
            return frame
        assert self.baseline_ is not None, "_check_survival_times requires a baseline"
        survival = pd.DataFrame(
            self.baseline_.survival(frame["log_hazard"].to_numpy(), times),
            index=frame.index,
            columns=[f"survival[{t!r}]" for t in times],
        )
        return pd.concat([frame, survival], axis=1)

    def _check_survival_times(self, survival_times: Sequence[float] | None, branch: str | None) -> list[float] | None:
        self._require_fitted()
        if survival_times is None:
            return None
        if self.baseline_ is None:
            raise ValueError(
                f"survival_times needs a model that predicts a log_hazard (a CoxHead); this one predicts "
                f"{self.model_.columns(self.target_)}"
            )
        if branch is not None:
            raise ValueError("survival_times cannot be used with branch: the baseline belongs to the fused log-hazard")
        times = np.asarray(survival_times, dtype=np.float64)
        if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (times < 0).any():
            raise ValueError(f"survival_times must be finite times >= 0, got {survival_times!r}")
        if (np.diff(times) <= 0).any():
            raise ValueError(f"survival_times must be strictly increasing, got {survival_times!r}")
        return times.tolist()

    def _predict(self, data: MultimodalDataset, branch: str | None, stacklevel: int | None) -> pd.DataFrame:
        view = self.transformed(data)
        if branch is not None and (not isinstance(self.model_, LateFusion) or branch not in self.model_.branches):
            names = list(self.model_.branches) if isinstance(self.model_, LateFusion) else []
            raise ValueError(f"branch={branch!r} needs a LateFusion model with that branch; branches: {names}")
        step = lambda model, batch: (batch["ids"], _to_cpu(model(batch)))  # noqa: E731
        results = self._infer(view, step, None if stacklevel is None else stacklevel + 1)
        ids = [pid for batch_ids, _ in results for pid in batch_ids]
        output = _concatenate([out for _, out in results])
        if branch is None:
            values, columns = output.prediction, self.model_.columns(self.target_)
        else:
            values, columns = (
                output.branches[branch].prediction,
                self.model_.branches[branch].columns(self.target_),
            )
        frame = pd.DataFrame(values.numpy(), index=pd.Index(ids, name="id"), columns=columns)
        if not frame.index.is_unique:
            raise AssertionError("duplicate patient ids in predictions")
        return frame

    def evaluate(
        self,
        data: MultimodalDataset,
        metrics: Mapping[str, Metric],
        branch: str | None = None,
        allow_seen: bool = False,
        survival_times: Sequence[float] | None = None,
    ) -> pd.Series:
        """Score predictions for ``data``. Patients used in fit (training or validation) raise unless ``allow_seen``.

        ``branch`` and ``survival_times`` are passed to ``predict``.
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
        prediction = self.predict(data, branch=branch, survival_times=survival_times)
        eval_target = data.target.frame.loc[prediction.index]
        context = EvalContext(
            train_target=self.train_target_,
            classes=self.target_.classes if isinstance(self.target_, Classification) else None,
        )
        return pd.Series(
            {name: metric(prediction, eval_target, context) for name, metric in metrics.items()}, name="score"
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
    sample = train.subset(_dry_run_ids(train, model.modalities))
    batch = sample.collate([sample[k] for k in range(len(sample))])
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model.dry_run(train, batch)
    finally:
        model.train(was_training)
