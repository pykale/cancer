"""Cross-validation harness for survival models.

Splits are stratified by event status (:func:`patient_stratified_splits`)
because a fold with zero events breaks the Cox partial-likelihood loss --
every fold, train and test, needs at least one event. Any ``eval_years``
passed to :func:`cross_validate_survival` / :func:`compare_models` are
converted to days (``year * 365.25``) before being handed to the
day-scaled time-dependent metrics in
:mod:`kalecancer.evaluate.survival_metrics`.

Follow-up length varies fold to fold, and ``sksurv`` raises ``ValueError``
if an eval time is not strictly inside a fold's test-set follow-up range.
HANCOCK's median follow-up (~1,211 days) is shorter than a naive 5-year
(1,826-day) evaluation point, so the 5-year point routinely falls outside
some fold's range even though the full-cohort range covers it fine. Each
fold therefore clamps ``eval_years`` down to whichever years actually fall
strictly inside *that fold's* ``(test_times.min(), test_times.max())``,
records which ones survived as ``"eval_years_used"`` in the fold result,
and raises a clear, fold-identifying error if none do.

For each fold, the baseline hazard (:func:`~kalecancer.survival.baseline.breslow_baseline_hazard`)
is fitted on the TRAIN fold only and applied to the TEST fold -- fitting
it on the test fold itself would leak test-set event times into the
"absolute risk" it produces.

Cox risk scores have no absolute scale: they are comparable only within
the one fitted model that produced them. Each fold trains a *separate*
model, so the aggregated C-index is the per-fold mean and across-fold
std (like every other aggregated metric here), not a bootstrap CI over
scores pooled across folds -- pooling would rank one fold's model's
scores against another's, which is not a meaningful comparison.
``out_of_fold_risk`` is still returned for inspection (e.g. plotting),
but for the same reason it must never be ranked or fed into a metric
across its full length; :func:`bootstrap_ci` is provided for computing a
proper confidence interval on scores that do share one model's scale
(e.g. within a single fold, or for one final production model).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from kalecancer.survival.baseline import breslow_baseline_hazard, predict_survival_function
from kalecancer.survival.metrics import concordance_index
from kalecancer.survival.trainer import fit_survival_model

from .survival_metrics import integrated_brier, kaplan_meier_groups, time_dependent_auc

_DAYS_PER_YEAR = 365.25


def patient_stratified_splits(
    times: np.ndarray,
    events: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold splits stratified by event status, one row per patient.

    A fold with no events at all breaks the Cox partial-likelihood loss
    (:func:`~kalecancer.survival.cox.neg_partial_log_likelihood` raises),
    so folds are stratified on ``events`` (not plain ``KFold``) to keep
    events spread proportionally across folds.

    Args:
        times: Observed times, shape ``(N,)``; used only to validate that
            ``events`` describes the same cohort.
        events: Event indicators, shape ``(N,)``.
        n_splits: Number of folds.
        seed: Seed controlling the fold assignment shuffle.

    Returns:
        A list of ``(train_idx, test_idx)`` index-array pairs, one per fold.

    Raises:
        ValueError: If ``times`` and ``events`` have different lengths, or
            there are fewer observed events than folds.
    """
    times = np.asarray(times)
    events = np.asarray(events, dtype=bool)
    if times.shape[0] != events.shape[0]:
        raise ValueError(f"times and events must share length, got {times.shape[0]} and {events.shape[0]}")

    n_events = int(events.sum())
    if n_events < n_splits:
        raise ValueError(
            f"n_splits={n_splits} exceeds the number of observed events ({n_events}); "
            "every fold requires at least one event"
        )

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(events.shape[0]), events))


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Patient-level bootstrap confidence interval for a metric.

    All arrays must already share one comparable scale (e.g. risk scores
    from a single fitted model) -- do not use this across risk scores
    pooled from several different models/folds.

    Args:
        metric_fn: Callable taking the same positional arrays as
            ``*arrays`` and returning a scalar metric.
        *arrays: Arrays to resample together, one row per patient (e.g.
            ``risk, times, events``).
        n_boot: Number of bootstrap resamples.
        alpha: Significance level; the returned interval covers
            ``1 - alpha``.
        seed: Seed for the resampling.

    Returns:
        A tuple ``(point_estimate, lower, upper)``, where
        ``point_estimate`` is ``metric_fn`` evaluated on the original
        (non-resampled) data.

    Raises:
        ValueError: If the arrays do not share the same first-dimension length.
    """
    arrays = tuple(np.asarray(a) for a in arrays)
    n = arrays[0].shape[0]
    if any(a.shape[0] != n for a in arrays):
        raise ValueError("all arrays passed to bootstrap_ci must share the same first-dimension length")

    point_estimate = float(metric_fn(*arrays))

    rng = np.random.default_rng(seed)
    boot_values = np.empty(n_boot)
    for i in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        boot_values[i] = metric_fn(*(a[sample_idx] for a in arrays))

    lower, upper = np.percentile(boot_values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point_estimate, float(lower), float(upper)


def _call_model(model: nn.Module, inputs: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(inputs, Mapping):
        return model(**inputs)
    return model(inputs)


def _index_inputs(
    inputs: torch.Tensor | Mapping[str, torch.Tensor], idx: torch.Tensor
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(inputs, Mapping):
        return {key: value[idx] for key, value in inputs.items()}
    return inputs[idx]


def cross_validate_survival(
    model_factory: Callable[[], nn.Module],
    inputs: torch.Tensor | Mapping[str, torch.Tensor],
    times: torch.Tensor,
    events: torch.Tensor,
    *,
    eval_years: tuple[float, ...] = (1, 3, 5),
    n_splits: int = 5,
    seed: int = 0,
    max_epochs: int = 500,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    patience: int = 50,
) -> dict:
    """Stratified K-fold cross-validation for a survival model.

    Each fold fits a fresh model (via ``model_factory()``) on the train
    split with :func:`~kalecancer.survival.trainer.fit_survival_model`,
    then evaluates it on the held-out test split: Harrell C-index, the
    time-dependent AUC at whichever of ``eval_years`` fall strictly inside
    that fold's test follow-up range (converted to days), integrated
    Brier score, and the log-rank p-value from a 2-group KM split of the
    test fold by predicted risk. The Breslow baseline hazard used for the
    Brier score is fit on the train fold only.

    Args:
        model_factory: Zero-argument callable returning a fresh, untrained
            ``nn.Module`` (called once per fold).
        inputs: Model input for the full cohort; a tensor, or a
            ``Mapping`` of tensors (one key per modality/branch), matching
            :func:`~kalecancer.survival.trainer.fit_survival_model`'s convention.
        times: Observed times for the full cohort, shape ``(N,)``.
        events: Event indicators for the full cohort, shape ``(N,)``.
        eval_years: Candidate years for time-dependent AUC / Brier score;
            each fold uses whichever of these fall strictly inside its own
            test-set follow-up range (see module docstring).
        n_splits: Number of cross-validation folds.
        seed: Seed controlling fold assignment and per-fold model initialisation.
        max_epochs: Passed through to ``fit_survival_model``.
        lr: Passed through to ``fit_survival_model``.
        weight_decay: Passed through to ``fit_survival_model``.
        patience: Passed through to ``fit_survival_model``.

    Returns:
        A dict with keys ``"folds"`` (a list of one dict per fold, keys
        ``"fold"``, ``"c_index"``, ``"eval_years_used"`` (the subset of
        ``eval_years`` that survived clamping for this fold), ``"td_auc"``
        (dict keyed by those years), ``"td_auc_mean"``,
        ``"integrated_brier"``, ``"log_rank_p_value"``), ``"aggregated"``
        (dict keyed by the same metric names, each a ``{"mean", "std"}``
        across folds), and ``"out_of_fold_risk"`` (each patient's risk
        score from the fold where it was held out -- for inspection only,
        see module docstring for why it must not be ranked across folds).

    Raises:
        ValueError: If, for some fold, none of ``eval_years`` fall
            strictly inside that fold's test-set follow-up range.
    """
    times_np = times.numpy()
    events_np = events.numpy()
    eval_years_arr = np.asarray(eval_years, dtype=np.float64)
    eval_days_all = eval_years_arr * _DAYS_PER_YEAR

    splits = patient_stratified_splits(times_np, events_np, n_splits=n_splits, seed=seed)

    fold_results = []
    out_of_fold_risk = np.full(times_np.shape[0], np.nan)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_idx_t = torch.from_numpy(train_idx).long()
        test_idx_t = torch.from_numpy(test_idx).long()

        # Reseed before constructing the model so that, regardless of how
        # much of the global RNG stream earlier folds consumed, the same
        # top-level `seed` always reproduces the same per-fold init.
        torch.manual_seed(seed + fold_idx)
        model = model_factory()

        fit_survival_model(
            model,
            _index_inputs(inputs, train_idx_t),
            times[train_idx_t],
            events[train_idx_t],
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            seed=seed + fold_idx,
        )

        model.eval()
        with torch.no_grad():
            train_log_hazard = _call_model(model, _index_inputs(inputs, train_idx_t)).squeeze(-1).numpy()
            test_log_hazard = _call_model(model, _index_inputs(inputs, test_idx_t)).squeeze(-1).numpy()

        out_of_fold_risk[test_idx] = test_log_hazard

        train_times, train_events = times_np[train_idx], events_np[train_idx]
        test_times, test_events = times_np[test_idx], events_np[test_idx]

        c_index = concordance_index(test_log_hazard, test_times, test_events)

        test_time_lo, test_time_hi = float(test_times.min()), float(test_times.max())
        in_range = (eval_days_all > test_time_lo) & (eval_days_all < test_time_hi)
        if not np.any(in_range):
            raise ValueError(
                f"fold {fold_idx}: none of eval_years={tuple(eval_years)} fall strictly inside "
                f"this fold's test-set follow-up range ({test_time_lo:.1f}, {test_time_hi:.1f}) days; "
                "pass a coarser eval_years or increase n_splits"
            )
        eval_years_used = tuple(eval_years_arr[in_range].tolist())
        eval_days_fold = eval_days_all[in_range]

        event_times, cumulative_baseline_hazard = breslow_baseline_hazard(train_log_hazard, train_times, train_events)
        survival_probs = predict_survival_function(
            test_log_hazard, event_times, cumulative_baseline_hazard, eval_days_fold
        )
        ibs = integrated_brier(train_times, train_events, test_times, test_events, survival_probs, eval_days_fold)

        td_auc_per_time, td_auc_mean = time_dependent_auc(
            train_times, train_events, test_times, test_events, test_log_hazard, eval_days_fold
        )

        km_result = kaplan_meier_groups(test_log_hazard, test_times, test_events, n_groups=2)

        fold_results.append(
            {
                "fold": fold_idx,
                "c_index": c_index,
                "eval_years_used": eval_years_used,
                "td_auc": dict(zip(eval_years_used, td_auc_per_time.tolist(), strict=True)),
                "td_auc_mean": td_auc_mean,
                "integrated_brier": ibs,
                "log_rank_p_value": km_result["log_rank_p_value"],
            }
        )

    def _mean_std(key: str) -> dict:
        values = [fold[key] for fold in fold_results]
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    aggregated = {
        "c_index": _mean_std("c_index"),
        "td_auc_mean": _mean_std("td_auc_mean"),
        "integrated_brier": _mean_std("integrated_brier"),
        "log_rank_p_value": _mean_std("log_rank_p_value"),
    }

    return {"folds": fold_results, "aggregated": aggregated, "out_of_fold_risk": out_of_fold_risk}


def compare_models(
    named_factories: Mapping[str, Callable[[], nn.Module]],
    inputs_per_model: Mapping[str, torch.Tensor | Mapping[str, torch.Tensor]],
    times: torch.Tensor,
    events: torch.Tensor,
    *,
    eval_years: tuple[float, ...] = (1, 3, 5),
    n_splits: int = 5,
    seed: int = 0,
    max_epochs: int = 500,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    patience: int = 50,
) -> list[dict]:
    """Cross-validate several models on the same cohort and tabulate the results.

    Runs :func:`cross_validate_survival` once per named model (e.g.
    ``"clinical_only"``, ``"imaging_only"``, ``"fused"``), all against the
    same ``times``/``events`` labels and the same fold assignment, so
    baselines are reported side by side rather than in separate runs.

    Args:
        named_factories: Mapping of model name to a zero-argument callable
            returning a fresh ``nn.Module``.
        inputs_per_model: Mapping of model name to that model's input
            (tensor or ``Mapping``), same keys as ``named_factories``.
        times: Observed times for the full cohort, shape ``(N,)``.
        events: Event indicators for the full cohort, shape ``(N,)``.
        eval_years: Passed through to ``cross_validate_survival``.
        n_splits: Passed through to ``cross_validate_survival``.
        seed: Passed through to ``cross_validate_survival`` (same for every model).
        max_epochs: Passed through to ``cross_validate_survival``.
        lr: Passed through to ``cross_validate_survival``.
        weight_decay: Passed through to ``cross_validate_survival``.
        patience: Passed through to ``cross_validate_survival``.

    Returns:
        A list of one dict per model (DataFrame-ready via
        ``pandas.DataFrame(result)``), with keys ``"model"``, ``"c_index"``
        (across-fold mean), ``"c_index_std"``, ``"td_auc_mean"``,
        ``"integrated_brier"``, ``"log_rank_p_value"``.
    """
    rows = []
    for name, factory in named_factories.items():
        result = cross_validate_survival(
            factory,
            inputs_per_model[name],
            times,
            events,
            eval_years=eval_years,
            n_splits=n_splits,
            seed=seed,
            max_epochs=max_epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
        )
        aggregated = result["aggregated"]
        rows.append(
            {
                "model": name,
                "c_index": aggregated["c_index"]["mean"],
                "c_index_std": aggregated["c_index"]["std"],
                "td_auc_mean": aggregated["td_auc_mean"]["mean"],
                "integrated_brier": aggregated["integrated_brier"]["mean"],
                "log_rank_p_value": aggregated["log_rank_p_value"]["mean"],
            }
        )
    return rows
