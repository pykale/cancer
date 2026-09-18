"""Pipeline: fit transforms and fold-local state on training patients, train one model end to end with Lightning,
then predict, evaluate, encode and export attention."""

from __future__ import annotations

import copy
import tempfile
import warnings
from collections.abc import Callable, Mapping
from typing import Any, Literal

import lightning.pytorch as L
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import BaseShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from kalecancer.evaluate.metrics import EvalContext, Metric
from kalecancer.interpret.attention import attention as _attention
from kalecancer.loaddata.dataset import MultimodalDataset
from kalecancer.model.models import EarlyFusion, IntermediateFusion, LateFusion, Unimodal
from kalecancer.pipeline.training import (
    _concatenate,
    _EarlyStoppingWithRestore,
    _InferenceModule,
    _lightning_session,
    _run_stages,
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
        if patience < 1 or min_delta < 0:
            raise ValueError(f"patience must be >= 1 and min_delta >= 0, got {patience} and {min_delta}")
        self.metric = metric
        self.patience = patience
        self.restore_best = restore_best
        self.min_delta = min_delta

    @property
    def mode(self) -> str:
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
        random_state: Seeds shuffling and re-initialises every parameter not marked ``keep_weights``. ``None``
            keeps the weights the model was built with and does not seed.
        param_groups: Optimizer options for parameters under a name prefix,
            e.g. ``{"encoding.clinical": {"lr": 1e-5}}``.
        num_workers: DataLoader workers.
        trainer_kwargs: Further ``lightning.pytorch.Trainer`` arguments (not the ones this class manages).
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
        self._check_arguments(data)
        assert data.target is not None, "_check_arguments requires a target"
        with _lightning_session():
            if self.random_state is not None:
                L.seed_everything(self.random_state, workers=True, verbose=False)
            train_ids, val_ids = self._split_validation(data)
            transforms = self._fit_transforms(data, train_ids)
            view = data.with_transforms(transforms)
            view.check_inputs()
            model = copy.deepcopy(self.model)
            if self.random_state is not None:
                _reinitialise(model)
            model.fit(view.subset(train_ids))

            info = data.target.info()
            loader = self._loader(view.subset(train_ids), shuffle=True)
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
                context=EvalContext(train_target=data.target.frame.loc[train_ids], classes=info.classes),
                columns=model.columns(info),
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
        self.target_info_ = info
        self.train_target_ = data.target.frame.loc[train_ids]
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
        }
        if skipped:
            warnings.warn(
                f"{skipped} training batches were skipped because their loss had no signal (for Cox: no event with "
                "another patient at risk); consider a larger batch_size",
                UserWarning,
                stacklevel=2,
            )
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
                "accumulate_grad_batches does not reproduce a larger batch for this model: the loss is not a sum over "
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
            if not hasattr(source, "transform_input"):
                raise TypeError(f"modality {name!r} ({type(source).__name__}) does not accept transforms")
            rows = source.transform_input(train.present_ids(name))
            estimator = clone(transform).fit(rows)
            _check_no_dropped_columns(estimator, name)
            fitted[name] = estimator
        return fitted

    def _loader(self, data: MultimodalDataset, shuffle: bool) -> DataLoader:
        sampler = None
        if shuffle:
            # A dedicated generator keeps the order independent of worker settings and of other RNG consumers.
            generator = torch.Generator()
            if self.random_state is not None:
                generator.manual_seed(self.random_state)
            sampler = RandomSampler(data, generator=generator)
        return DataLoader(
            data,
            batch_size=self.batch_size,
            sampler=sampler,
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
        """The dataset as the fitted model sees it: fitted transforms applied and inputs checked."""
        self._require_fitted()
        if not isinstance(data, MultimodalDataset):
            raise TypeError(f"expected a MultimodalDataset, got {type(data).__name__}")
        self.model_.check(data)
        view = data.with_transforms(self.transforms_)
        view.check_inputs()
        return view

    def _infer(self, data: MultimodalDataset, step: Callable[[nn.Module, dict], Any]) -> list:
        module = _InferenceModule(self.model_, step)
        with _lightning_session(), tempfile.TemporaryDirectory() as root:
            results = self._trainer([], root, max_epochs=1).predict(module, self._loader(data, shuffle=False))
        self.model_.eval().cpu()
        return results

    def predict(self, data: MultimodalDataset, branch: str | None = None) -> pd.DataFrame:
        """Predictions indexed by patient id; NaN for patients without a prediction. ``branch`` selects one
        branch of a ``LateFusion`` model."""
        view = self.transformed(data)
        if branch is not None and (not isinstance(self.model_, LateFusion) or branch not in self.model_.branches):
            names = list(self.model_.branches) if isinstance(self.model_, LateFusion) else []
            raise ValueError(f"branch={branch!r} needs a LateFusion model with that branch; branches: {names}")
        results = self._infer(view, lambda model, batch: (batch["ids"], _to_cpu(model(batch))))
        ids = [pid for batch_ids, _ in results for pid in batch_ids]
        output = _concatenate([out for _, out in results])
        if branch is None:
            values, columns = output.prediction, self.model_.columns(self.target_info_)
        else:
            values, columns = (
                output.branches[branch].prediction,
                self.model_.branches[branch].columns(self.target_info_),
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
    ) -> pd.Series:
        """Score predictions for ``data``. Patients used in fit (training or validation) raise unless ``allow_seen``."""
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
        prediction = self.predict(data, branch=branch)
        target = data.target.frame.loc[prediction.index]
        context = EvalContext(train_target=self.train_target_, classes=self.target_info_.classes)
        return pd.Series({name: metric(prediction, target, context) for name, metric in metrics.items()}, name="score")

    def _single_modality(self, data: MultimodalDataset, modality: str, branch: str | None) -> tuple:
        if modality not in data.modalities:
            raise ValueError(f"the dataset has no modality {modality!r}")
        view = self.transformed(data)
        stages = self.model_.stages_for(modality, branch) if branch is not None else self.model_.stages_for(modality)
        present = view.present_ids(modality)
        if not present:
            raise ValueError(f"no patient in this dataset has {modality!r}")
        single = MultimodalDataset({modality: view.modalities[modality]}, target=None, required_modalities=[modality])
        return single.subset(present), stages

    def encode(self, data: MultimodalDataset, modality: str, branch: str | None = None) -> pd.DataFrame:
        """The output of ``modality``'s stage list (the fusion input), indexed by patient id."""
        single, stages = self._single_modality(data, modality, branch)
        results = self._infer(single, lambda model, batch: (batch["ids"], _run_stages(stages, batch, modality)))
        ids = [pid for batch_ids, _ in results for pid in batch_ids]
        values = torch.cat([z for _, z in results]).numpy()
        return pd.DataFrame(
            values, index=pd.Index(ids, name="id"), columns=[f"{modality}_{k}" for k in range(values.shape[1])]
        )

    def attention(self, data: MultimodalDataset, modality: str, branch: str | None = None) -> pd.DataFrame:
        """Per-instance attention weights joined with the modality's coordinates; see ``kalecancer.interpret``."""
        return _attention(self, data, modality, branch)


def _reinitialise(model: nn.Module) -> None:
    """Re-draw parameters under the current seed, except inside modules marked ``keep_weights``."""

    def visit(module: nn.Module, path: str) -> None:
        if getattr(module, "keep_weights", False):
            return
        if any(True for _ in module.parameters(recurse=False)):
            if not callable(getattr(module, "reset_parameters", None)):
                raise TypeError(
                    f"{path or 'model'} ({type(module).__name__}) has parameters but no reset_parameters(), so "
                    "random_state cannot initialise it; add reset_parameters or set keep_weights = True"
                )
            module.reset_parameters()
        for name, child in module.named_children():
            visit(child, f"{path}.{name}" if path else name)

    visit(model, "")
