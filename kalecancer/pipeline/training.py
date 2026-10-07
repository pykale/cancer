"""Lightning machinery behind Pipeline: training and inference modules, early stopping that restores the best
epoch, and a session that keeps Lightning quiet and undoes its global state."""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import warnings
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import lightning.pytorch as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping as _LightningEarlyStopping
from torch import Tensor, nn

from kalecancer.evaluate.metrics import EvalContext, Metric
from kalecancer.model.models import ModelOutput, StageList

if TYPE_CHECKING:
    from kalecancer.pipeline.pipeline import EarlyStopping


def _parameter_groups(model: nn.Module, overrides: Mapping[str, Mapping[str, float]] | None) -> list[dict]:
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named:
        raise ValueError("the model has no trainable parameters")
    groups, claimed = [], set()
    for prefix, options in (overrides or {}).items():
        params = [p for name, p in named if name == prefix or name.startswith(prefix + ".")]
        if not params:
            raise ValueError(
                f"param_groups[{prefix!r}] matches no trainable parameter; names look like {named[0][0]!r}"
            )
        if any(id(p) in claimed for p in params):
            raise ValueError(f"param_groups[{prefix!r}] overlaps another prefix")
        claimed |= {id(p) for p in params}
        groups.append({"params": params, **options})
    rest = [p for _, p in named if id(p) not in claimed]
    return ([{"params": rest}] if rest else []) + groups


def _run_stages(stages: StageList, x: Any, ids: list[str], modality: str) -> Tensor:
    z = stages(x, ids)
    if not isinstance(z, Tensor) or z.ndim != 2:
        raise TypeError(f"encoding[{modality!r}] must end with (n, d) vectors")
    return z.float().cpu()


def _to_cpu(output: ModelOutput) -> ModelOutput:
    return ModelOutput(
        output=output.output.detach().float().cpu(),
        prediction=output.prediction.detach().float().cpu(),
        defined=output.defined.detach().cpu(),
        branches={name: _to_cpu(branch) for name, branch in output.branches.items()},
    )


def _concatenate(outputs: list[ModelOutput]) -> ModelOutput:
    return ModelOutput(
        output=torch.cat([o.output for o in outputs]),
        prediction=torch.cat([o.prediction for o in outputs]),
        defined=torch.cat([o.defined for o in outputs]),
        branches={name: _concatenate([o.branches[name] for o in outputs]) for name in outputs[0].branches},
    )


class _TrainingModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Callable,
        param_groups: Mapping | None,
        monitor: Metric | str | None,
        validation_target: pd.DataFrame | None,
        context: EvalContext,
        columns: list[str],
    ):
        super().__init__()
        self.model = model
        self.optimizer_factory = optimizer
        self.param_groups = param_groups
        self.monitor = monitor
        self.validation_target = validation_target
        self.context = context
        self.columns = columns
        self.history: list[dict] = []
        self.optimizer_steps = 0
        self._has_loss = False
        self._train_losses: list[float] = []
        self._skipped = 0
        self._validation: list[tuple] = []
        self._last_validation: dict = {}

    def training_step(self, batch: dict, batch_index: int) -> Tensor | None:
        losses = self.model.loss(self.model(batch), batch["target"])
        self._has_loss = losses is not None
        if losses is None:
            self._skipped += 1
            return None
        self._train_losses.append(float(losses["loss"].detach()))
        return losses["loss"]

    def optimizer_step(self, epoch: int, batch_idx: int, optimizer: Any, optimizer_closure: Any = None) -> None:
        optimizer.step(closure=optimizer_closure)
        # Lightning calls the optimizer even when training_step returned None; count only steps that had a loss.
        if self._has_loss:
            self.optimizer_steps += 1

    def validation_step(self, batch: dict, batch_index: int) -> None:
        target = {key: value.detach().cpu() for key, value in batch["target"].items()}
        self._validation.append((batch["ids"], _to_cpu(self.model(batch)), target))

    def on_validation_epoch_end(self) -> None:
        ids = [pid for batch_ids, _, _ in self._validation for pid in batch_ids]
        output = _concatenate([out for _, out, _ in self._validation])
        target = {key: torch.cat([t[key] for _, _, t in self._validation]) for key in self._validation[0][2]}
        self._validation.clear()
        losses = self.model.loss(output, target)
        row = {"val_loss": float(losses["loss"]) if losses is not None else float("nan")}
        if isinstance(self.monitor, Metric):
            assert self.validation_target is not None, "a Metric monitor needs a validation target"
            prediction = pd.DataFrame(output.prediction.numpy(), index=pd.Index(ids, name="id"), columns=self.columns)
            row["val_metric"] = float(self.monitor(prediction, self.validation_target.loc[ids], self.context))
        if self.monitor is not None:
            self.log("monitor", row["val_loss"] if self.monitor == "loss" else row["val_metric"])
        self._last_validation = row

    def on_train_epoch_end(self) -> None:
        losses = self._train_losses
        self.history.append(
            {
                "epoch": self.current_epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                **self._last_validation,
                "optimizer_steps": self.optimizer_steps,
                "skipped_batches": self._skipped,
            }
        )
        self._train_losses, self._skipped, self._last_validation = [], 0, {}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer_factory(_parameter_groups(self.model, self.param_groups))


class _InferenceModule(L.LightningModule):
    def __init__(self, model: nn.Module, step: Callable[[nn.Module, dict], Any]):
        super().__init__()
        self.model = model
        self.step = step

    def predict_step(self, batch: dict, batch_index: int) -> Any:
        return self.step(self.model, batch)


class _EarlyStoppingWithRestore(_LightningEarlyStopping):
    """Lightning's early stopping, which also snapshots the weights whenever its own rule records an improvement,
    so stopping and restoring agree on which epoch was best."""

    def __init__(self, settings: EarlyStopping):
        super().__init__(
            monitor="monitor",
            mode=settings.mode,
            patience=settings.patience,
            min_delta=settings.min_delta,
            check_finite=True,
            verbose=False,
        )
        self.keep_state = settings.restore_best
        self.best_state: dict[str, Tensor] | None = None
        self.best_epoch: int | None = None
        self.best_value: float | None = None

    def _run_early_stopping_check(self, trainer: L.Trainer) -> None:
        # compared as floats: Lightning moves best_score to the metric's device during the check
        previous = float(self.best_score)
        super()._run_early_stopping_check(trainer)
        current = trainer.callback_metrics.get(self.monitor)
        if current is None or not torch.isfinite(current).item() or float(self.best_score) == previous:
            return
        self.best_epoch = trainer.current_epoch
        self.best_value = float(self.best_score)
        if self.keep_state:
            state = trainer.lightning_module.model.state_dict()
            # fitted stages may keep non-tensor extra state (e.g. TabICL's context ids)
            self.best_state = {
                key: value.detach().to("cpu", copy=True) if isinstance(value, Tensor) else copy.deepcopy(value)
                for key, value in state.items()
            }


_LIGHTNING_LOGGERS = ("lightning", "lightning.pytorch", "lightning.fabric")


_LIGHTNING_ENVIRONMENT = ("PL_GLOBAL_SEED", "PL_SEED_WORKERS", "CUBLAS_WORKSPACE_CONFIG")


@contextlib.contextmanager
def _lightning_session() -> Iterator[None]:
    """Quiet Lightning and undo the global state it leaves behind (loggers, deterministic flags, environment)."""
    levels = {name: logging.getLogger(name).level for name in _LIGHTNING_LOGGERS}
    environment = {key: os.environ.get(key) for key in _LIGHTNING_ENVIRONMENT}
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    for name in _LIGHTNING_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not have many workers.*")
            warnings.filterwarnings("ignore", message=".*LeafSpec.*")
            # accelerator is always an explicit Pipeline argument, so this hint is noise
            warnings.filterwarnings("ignore", message=".*GPU available but not used.*")
            # _TrainingModule always defines validation_step; with validation=None there is simply no loader
            warnings.filterwarnings("ignore", message=".*no `val_dataloader`.*")
            # a skipped Cox batch returns None on purpose; fit reports the count once
            warnings.filterwarnings("ignore", message=".*`training_step` returned `None`.*")
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
