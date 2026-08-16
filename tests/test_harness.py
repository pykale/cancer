"""Tests for ``kalecancer.evaluate.harness``."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from kalecancer.evaluate.harness import (
    _index_inputs,
    bootstrap_ci,
    compare_models,
    cross_validate_survival,
    patient_stratified_splits,
)
from kalecancer.survival.cox import CoxHead
from kalecancer.survival.metrics import concordance_index
from kalecancer.survival.synthetic import make_synthetic_survival


class _BagModel(nn.Module):
    """Toy multimodal model: tabular branch + mean-pooled variable-length bag branch."""

    def __init__(self, tabular_dim: int, bag_feature_dim: int, hidden: int) -> None:
        super().__init__()
        self.tabular_branch = nn.Linear(tabular_dim, hidden)
        self.bag_branch = nn.Linear(bag_feature_dim, hidden)
        self.head = CoxHead(in_features=hidden * 2)

    def forward(self, tabular: torch.Tensor, bags: list[torch.Tensor]) -> torch.Tensor:
        tabular_z = self.tabular_branch(tabular)
        pooled = torch.stack([self.bag_branch(bag).mean(dim=0) for bag in bags])
        z = torch.cat([tabular_z, pooled], dim=-1)
        return self.head(z)


def test_every_fold_has_at_least_one_event() -> None:
    data = make_synthetic_survival(n_samples=300, n_features=8, seed=11)
    times, events = data.times.numpy(), data.events.numpy()

    splits = patient_stratified_splits(times, events, n_splits=5, seed=0)

    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert events[train_idx].sum() > 0
        assert events[test_idx].sum() > 0


def test_same_seed_reproduces_splits_and_cv_results() -> None:
    data = make_synthetic_survival(n_samples=400, n_features=8, seed=11)
    times, events = data.times.numpy(), data.events.numpy()

    splits_a = patient_stratified_splits(times, events, n_splits=5, seed=0)
    splits_b = patient_stratified_splits(times, events, n_splits=5, seed=0)
    assert all(
        np.array_equal(a_train, b_train) and np.array_equal(a_test, b_test)
        for (a_train, a_test), (b_train, b_test) in zip(splits_a, splits_b, strict=True)
    )

    def run() -> dict:
        return cross_validate_survival(
            lambda: CoxHead(in_features=8),
            data.embeddings,
            data.times,
            data.events,
            n_splits=5,
            seed=0,
            max_epochs=80,
        )

    result_a, result_b = run(), run()
    assert [fold["c_index"] for fold in result_a["folds"]] == [fold["c_index"] for fold in result_b["folds"]]
    assert result_a["aggregated"] == result_b["aggregated"]

    fold_c_indices = [fold["c_index"] for fold in result_a["folds"]]
    # The aggregation itself, not just its reproducibility: aggregated
    # c_index must be the plain across-fold mean of the per-fold values.
    assert result_a["aggregated"]["c_index"]["mean"] == np.mean(fold_c_indices)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    data = make_synthetic_survival(n_samples=300, n_features=8, seed=11)
    risk, times, events = data.true_risk.numpy(), data.times.numpy(), data.events.numpy()

    point, lower, upper = bootstrap_ci(concordance_index, risk, times, events, n_boot=1000, seed=0)

    assert lower <= point <= upper


def test_bootstrap_ci_width_stabilises_as_n_boot_grows() -> None:
    # NOTE on what "narrows" actually means for a percentile bootstrap:
    # increasing n_boot (the number of RESAMPLES) does NOT shrink the CI --
    # with few resamples, np.percentile's 2.5th/97.5th estimates are pinned
    # near the min/max of a small draw, which *underestimates* the true
    # tail extent. As n_boot grows the estimate widens and then converges
    # (plateaus) to the interval's true asymptotic width; it does not keep
    # narrowing. So the property we can actually assert here is
    # *stability*: two large-n_boot runs (different resampling seeds)
    # should agree closely, having both converged to ~the same width.
    data = make_synthetic_survival(n_samples=300, n_features=8, seed=11)
    risk, times, events = data.true_risk.numpy(), data.times.numpy(), data.events.numpy()

    _, lower_1000, upper_1000 = bootstrap_ci(concordance_index, risk, times, events, n_boot=1000, seed=0)
    _, lower_3000, upper_3000 = bootstrap_ci(concordance_index, risk, times, events, n_boot=3000, seed=1)

    width_1000 = upper_1000 - lower_1000
    width_3000 = upper_3000 - lower_3000
    assert abs(width_1000 - width_3000) < 0.03


def test_bootstrap_ci_narrows_with_more_patients() -> None:
    # This IS the genuine narrowing statement: CI width tracks the
    # underlying SAMPLE size (more patients -> less sampling uncertainty
    # about the population metric), not the bootstrap resample count.
    small = make_synthetic_survival(n_samples=200, n_features=8, seed=11)
    large = make_synthetic_survival(n_samples=1000, n_features=8, seed=11)

    _, lower_small, upper_small = bootstrap_ci(
        concordance_index, small.true_risk.numpy(), small.times.numpy(), small.events.numpy(), n_boot=1000, seed=0
    )
    _, lower_large, upper_large = bootstrap_ci(
        concordance_index, large.true_risk.numpy(), large.times.numpy(), large.events.numpy(), n_boot=1000, seed=0
    )

    assert (upper_large - lower_large) < (upper_small - lower_small)


def test_compare_models_ranks_true_risk_above_random_risk() -> None:
    data = make_synthetic_survival(n_samples=800, n_features=8, seed=11)
    torch.manual_seed(99)
    random_inputs = torch.randn_like(data.embeddings)

    table = compare_models(
        {"true_risk": lambda: CoxHead(in_features=8), "random_risk": lambda: CoxHead(in_features=8)},
        {"true_risk": data.embeddings, "random_risk": random_inputs},
        data.times,
        data.events,
        n_splits=5,
        seed=0,
        max_epochs=150,
    )

    by_name = {row["model"]: row for row in table}
    assert by_name["true_risk"]["c_index"] > by_name["random_risk"]["c_index"]
    assert by_name["true_risk"]["c_index"] > 0.65
    assert "c_index_std" in by_name["true_risk"]
    assert by_name["true_risk"]["c_index_std"] >= 0
    assert "c_index_ci_lower" not in by_name["true_risk"]


def test_index_inputs_handles_variable_length_bag_lists() -> None:
    tabular = torch.randn(5, 4)
    bag_lengths = [10, 3, 7, 1, 20]
    bags = [torch.randn(n, 2) for n in bag_lengths]

    idx = torch.tensor([4, 1, 0])
    indexed = _index_inputs({"tabular": tabular, "bags": bags}, idx)

    assert torch.equal(indexed["tabular"], tabular[idx])
    assert isinstance(indexed["bags"], list)
    assert len(indexed["bags"]) == len(idx)
    for out_bag, original_i in zip(indexed["bags"], idx.tolist(), strict=True):
        assert torch.equal(out_bag, bags[original_i])
        assert out_bag.shape == bags[original_i].shape


def test_cross_validate_survival_with_variable_length_bags() -> None:
    torch.manual_seed(0)
    data = make_synthetic_survival(n_samples=300, n_features=6, seed=5)
    bag_feature_dim = 4
    # Variable-length per-patient bags (HANCOCK-style WSI tile bags), each
    # with a different number of tiles -- cannot be stacked into one tensor.
    bags = [torch.randn(int(torch.randint(5, 50, (1,)).item()), bag_feature_dim) for _ in range(300)]

    result = cross_validate_survival(
        lambda: _BagModel(tabular_dim=6, bag_feature_dim=bag_feature_dim, hidden=8),
        {"tabular": data.embeddings, "bags": bags},
        data.times,
        data.events,
        eval_years=(1, 2),
        n_splits=3,
        seed=0,
        max_epochs=30,
        lr=1e-2,
    )

    assert len(result["folds"]) == 3
    for fold in result["folds"]:
        assert isinstance(fold["c_index"], float)
        assert 0.0 <= fold["c_index"] <= 1.0
