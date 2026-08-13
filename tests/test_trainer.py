"""Tests for ``kalecancer.survival.trainer``."""

from __future__ import annotations

import torch
from torch import nn

from kalecancer.survival.cox import CoxHead
from kalecancer.survival.metrics import concordance_index
from kalecancer.survival.synthetic import make_synthetic_survival
from kalecancer.survival.trainer import fit_survival_model


class _TwoBranchModel(nn.Module):
    """Toy multi-input model: two linear branches concatenated into a CoxHead."""

    def __init__(self, in_a: int, in_b: int, hidden: int) -> None:
        super().__init__()
        self.branch_a = nn.Linear(in_a, hidden)
        self.branch_b = nn.Linear(in_b, hidden)
        self.head = CoxHead(in_features=hidden * 2)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.branch_a(a), self.branch_b(b)], dim=-1)
        return self.head(z)


def test_trains_cox_head_and_improves_loss() -> None:
    data = make_synthetic_survival(n_samples=512, n_features=16, seed=7)

    torch.manual_seed(0)
    model = CoxHead(in_features=16)
    history = fit_survival_model(model, data.embeddings, data.times, data.events, max_epochs=300, lr=1e-2, seed=0)

    assert history["train_loss"][-1] < history["train_loss"][0]

    with torch.no_grad():
        risk = model(data.embeddings).squeeze(-1).numpy()
    c_index = concordance_index(risk, data.times.numpy(), data.events.numpy())
    assert c_index > 0.65


def test_works_with_mapping_input() -> None:
    # Proves the trainer is model-agnostic: it never assumes a bare tensor.
    data = make_synthetic_survival(n_samples=256, n_features=10, seed=3)
    branch_a = data.embeddings[:, :6]
    branch_b = data.embeddings[:, 6:]

    torch.manual_seed(0)
    model = _TwoBranchModel(in_a=6, in_b=4, hidden=8)
    history = fit_survival_model(
        model,
        {"a": branch_a, "b": branch_b},
        data.times,
        data.events,
        max_epochs=100,
        lr=1e-2,
        seed=0,
    )

    assert history["train_loss"][-1] < history["train_loss"][0]


def test_early_stopping_triggers_and_tracks_best_epoch() -> None:
    data = make_synthetic_survival(n_samples=400, n_features=16, seed=7)
    train_embeddings, val_embeddings = data.embeddings[:300], data.embeddings[300:]
    train_times, val_times = data.times[:300], data.times[300:]
    train_events, val_events = data.events[:300], data.events[300:]

    torch.manual_seed(0)
    model = CoxHead(in_features=16)
    history = fit_survival_model(
        model,
        train_embeddings,
        train_times,
        train_events,
        val_inputs=val_embeddings,
        val_times=val_times,
        val_events=val_events,
        max_epochs=2000,
        lr=5e-2,
        patience=10,
        seed=0,
    )

    assert history["epochs_run"] < 2000
    assert history["best_epoch"] < history["epochs_run"]
    assert len(history["val_loss"]) == history["epochs_run"]


def test_same_seed_gives_identical_final_loss() -> None:
    data = make_synthetic_survival(n_samples=256, n_features=10, seed=4)

    def run() -> float:
        torch.manual_seed(123)
        model = CoxHead(in_features=10)
        history = fit_survival_model(
            model, data.embeddings, data.times, data.events, max_epochs=50, lr=1e-2, seed=123
        )
        return history["train_loss"][-1]

    assert run() == run()
