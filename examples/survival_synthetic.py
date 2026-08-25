"""Integration reference for the kalecancer predict + evaluate stages.

Runs predict + evaluate end to end on synthetic data, no real data or
external files. Any encoder or fusion block emitting a fixed-width (B, D)
embedding -- tabular, imaging, or fused -- plugs into exactly this
pipeline: CoxHead, the Cox loss, fit_survival_model, and every
kalecancer.evaluate metric are agnostic to the embedding's source. Step 7
previews the clinical-only / imaging-only / fused comparison table's shape.

Run: python examples/survival_synthetic.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from kalecancer.evaluate import (
    compare_models,
    cross_validate_survival,
    integrated_brier,
    patient_stratified_splits,
    time_dependent_auc,
)
from kalecancer.survival import (
    CoxHead,
    breslow_baseline_hazard,
    concordance_index,
    fit_survival_model,
    make_synthetic_survival,
    predict_survival_function,
)

EVAL_YEARS = (1, 3, 5)
EVAL_DAYS = np.array(EVAL_YEARS) * 365.25
CV_KWARGS: dict[str, Any] = {
    "eval_years": EVAL_YEARS,
    "n_splits": 5,
    "seed": 0,
    "max_epochs": 150,
    "lr": 1e-2,
    "patience": 20,
}


class TabularEncoder(nn.Module):
    """Toy tabular encoder: emits a fixed-width (B, 64) embedding."""

    def __init__(self, in_features: int, out_features: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SurvivalModel(nn.Module):
    """Encoder -> CoxHead: the (B, D) -> (B, 1) contract every modality/fusion block fits."""

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.encoder = TabularEncoder(in_features)
        self.head = CoxHead(in_features=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def main() -> None:
    print("=== 1. Synthetic cohort ===")
    data = make_synthetic_survival(n_samples=600, n_features=12, seed=0)
    n_features = data.embeddings.shape[1]
    print(f"{data.times.shape[0]} patients, {int(data.events.sum())} events, {n_features} features")
    times_np, events_np = data.times.numpy(), data.events.numpy()
    train_idx, test_idx = patient_stratified_splits(times_np, events_np, n_splits=5, seed=0)[0]
    train_idx_t, test_idx_t = torch.from_numpy(train_idx).long(), torch.from_numpy(test_idx).long()

    print("\n=== 2-3. Fit TabularEncoder -> CoxHead ===")
    torch.manual_seed(0)
    model = SurvivalModel(in_features=n_features)
    history = fit_survival_model(
        model, data.embeddings[train_idx_t], data.times[train_idx_t], data.events[train_idx_t],
        val_inputs=data.embeddings[test_idx_t], val_times=data.times[test_idx_t], val_events=data.events[test_idx_t],
        max_epochs=200, lr=1e-2, patience=20, seed=0,
    )
    print(f"train loss: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f} ({history['epochs_run']} epochs)")

    print("\n=== 4. Harrell C-index (held-out) ===")
    model.eval()
    with torch.no_grad():
        train_log_hazard = model(data.embeddings[train_idx_t]).squeeze(-1).numpy()
        test_log_hazard = model(data.embeddings[test_idx_t]).squeeze(-1).numpy()
    train_times, train_events = times_np[train_idx], events_np[train_idx]
    test_times, test_events = times_np[test_idx], events_np[test_idx]
    print(f"C-index: {concordance_index(test_log_hazard, test_times, test_events):.3f}")

    print("\n=== 5. Baseline hazard + calibrated metrics ===")
    event_times, cum_hazard = breslow_baseline_hazard(train_log_hazard, train_times, train_events)
    survival_probs = predict_survival_function(test_log_hazard, event_times, cum_hazard, EVAL_DAYS)
    ibs = integrated_brier(train_times, train_events, test_times, test_events, survival_probs, EVAL_DAYS)
    auc_per_time, mean_auc = time_dependent_auc(train_times, train_events, test_times, test_events, test_log_hazard, EVAL_DAYS)
    print(f"integrated Brier score: {ibs:.4f}")
    for year, auc in zip(EVAL_YEARS, auc_per_time, strict=True):
        print(f"  td-AUC @ {year}y: {auc:.3f}")
    print(f"  mean td-AUC: {mean_auc:.3f}")

    print("\n=== 6. 5-fold cross-validation ===")
    cv = cross_validate_survival(lambda: SurvivalModel(n_features), data.embeddings, data.times, data.events, **CV_KWARGS)
    for fold in cv["folds"]:
        print(f"  fold {fold['fold']}: C-index={fold['c_index']:.3f}  IBS={fold['integrated_brier']:.4f}  log-rank p={fold['log_rank_p_value']:.2e}")
    agg = cv["aggregated"]
    print(f"  aggregate C-index: {agg['c_index']['mean']:.3f} +/- {agg['c_index']['std']:.3f}")

    print("\n=== 7. compare_models: real-signal vs random-input ===")
    torch.manual_seed(1)
    random_inputs = torch.randn_like(data.embeddings)
    factories = {"real_signal": lambda: SurvivalModel(n_features), "random_input": lambda: SurvivalModel(n_features)}
    inputs_per_model = {"real_signal": data.embeddings, "random_input": random_inputs}
    table = compare_models(factories, inputs_per_model, data.times, data.events, **CV_KWARGS)
    for row in table:
        print(f"  {row['model']:>12}: C-index={row['c_index']:.3f} +/- {row['c_index_std']:.3f}  IBS={row['integrated_brier']:.4f}  td-AUC={row['td_auc_mean']:.3f}")


if __name__ == "__main__":
    main()
