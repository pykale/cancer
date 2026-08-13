"""Full-batch training loop for Cox survival heads.

Boundary rules: this module imports only ``torch`` (and stdlib).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .cox import neg_partial_log_likelihood


def _call_model(model: nn.Module, inputs: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(inputs, Mapping):
        return model(**inputs)
    return model(inputs)


def fit_survival_model(
    model: nn.Module,
    inputs: torch.Tensor | Mapping[str, torch.Tensor],
    times: torch.Tensor,
    events: torch.Tensor,
    *,
    val_inputs: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    val_times: torch.Tensor | None = None,
    val_events: torch.Tensor | None = None,
    max_epochs: int = 500,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    patience: int = 50,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Fit any log-hazard model with full-batch Adam on the Cox loss.

    Training is full-batch, not mini-batch: a Cox partial-likelihood risk
    set is defined relative to everyone else in the same batch, so a
    mini-batch estimates the likelihood of a *different, batch-local*
    problem and biases the gradient. There is no such thing as an
    unbiased mini-batch for this loss.

    ``model`` can be anything whose ``forward`` returns a ``(B, 1)`` or
    ``(B,)`` log-hazard -- a tabular-only model, an imaging-only model, or
    a fused multimodal model all fit through this same function. ``inputs``
    is passed positionally (``model(inputs)``) unless it is a ``Mapping``,
    in which case it is unpacked as keyword arguments (``model(**inputs)``),
    which is what lets a multi-branch model (e.g. one argument per
    modality) share this trainer with a single-tensor model.

    Args:
        model: Any ``nn.Module`` producing a log-hazard.
        inputs: Model input; a tensor, or a ``Mapping`` of keyword arguments.
        times: Training observed times, shape ``(B,)``.
        events: Training event indicators, shape ``(B,)``.
        val_inputs: Validation input, same convention as ``inputs``. If
            given, ``val_times`` and ``val_events`` must be given too.
        val_times: Validation observed times, shape ``(B_val,)``.
        val_events: Validation event indicators, shape ``(B_val,)``.
        max_epochs: Maximum number of full-batch training epochs.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        patience: Number of epochs without validation-loss improvement
            before stopping early. Only used when validation data is given.
        seed: Seed for ``torch.manual_seed``, set once before training starts.
            This controls training-time stochasticity only (e.g. any
            randomness inside ``model.forward``); it runs *after* ``model``
            is constructed, so it cannot control weight initialisation.
            Callers who need reproducible initialisation must seed before
            constructing ``model`` themselves, as ``tests/test_trainer.py`` does.
        verbose: If ``True``, print train (and validation) loss each epoch.

    Returns:
        A history dict with keys ``"train_loss"`` (list of per-epoch
        training loss), ``"val_loss"`` (list of per-epoch validation loss;
        present only when validation data was given), ``"best_epoch"``
        (the epoch, 0-indexed, with the lowest monitored loss -- validation
        loss if given, else training loss), and ``"epochs_run"`` (the
        number of epochs actually executed).

    Raises:
        ValueError: If only some of ``val_inputs``, ``val_times``,
            ``val_events`` are given.
    """
    has_validation = val_inputs is not None or val_times is not None or val_events is not None
    if has_validation and (val_inputs is None or val_times is None or val_events is None):
        raise ValueError("val_inputs, val_times and val_events must all be given together, or none of them")

    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_monitored_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        log_hazard = _call_model(model, inputs)
        train_loss = neg_partial_log_likelihood(log_hazard, times, events)
        train_loss.backward()
        optimizer.step()
        train_loss_value = train_loss.item()
        train_loss_history.append(train_loss_value)

        monitored_loss = train_loss_value
        if has_validation:
            model.eval()
            with torch.no_grad():
                val_log_hazard = _call_model(model, val_inputs)
                val_loss_value = neg_partial_log_likelihood(val_log_hazard, val_times, val_events).item()
            val_loss_history.append(val_loss_value)
            monitored_loss = val_loss_value

        if verbose:
            message = f"epoch {epoch}: train_loss={train_loss_value:.4f}"
            if has_validation:
                message += f" val_loss={val_loss_value:.4f}"
            print(message)  # noqa: T201

        if monitored_loss < best_monitored_loss:
            best_monitored_loss = monitored_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            if has_validation:
                best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if has_validation and epochs_without_improvement >= patience:
            break

    if has_validation and best_state is not None:
        model.load_state_dict(best_state)

    history: dict = {
        "train_loss": train_loss_history,
        "best_epoch": best_epoch,
        "epochs_run": len(train_loss_history),
    }
    if has_validation:
        history["val_loss"] = val_loss_history
    return history
