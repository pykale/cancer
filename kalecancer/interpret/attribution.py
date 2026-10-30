"""Attribution: how much each clinical feature matters to a fitted pipeline's score.

Both functions perturb one ``Tabular`` modality's raw, untransformed columns and re-score with
``Pipeline.evaluate`` -- the perturbed values flow through the pipeline's fitted transforms exactly as a real
value would, so nothing here needs to know what those transforms are. Perturbing a transformed or embedded
representation instead would silently go around that, and could produce values the transforms were never fitted
to see.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from kalecancer.evaluate.metrics import Metric
from kalecancer.loaddata.dataset import MultimodalDataset
from kalecancer.loaddata.modalities import Tabular
from kalecancer.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _resolve_tabular_modality(data: MultimodalDataset, modality: str | None) -> tuple[str, Tabular]:
    """The name and ``Tabular`` instance to perturb: ``modality`` if given, else ``data``'s only one.

    Raises:
        ValueError: If ``modality`` names a modality ``data`` does not have, or if ``modality`` is ``None`` and
            ``data`` does not have exactly one ``Tabular`` modality to discover.
        TypeError: If ``modality`` names a modality that is not ``Tabular``.
    """
    if modality is not None:
        if modality not in data.modalities:
            raise ValueError(f"modality {modality!r} is not in data.modalities: {sorted(data.modalities)}")
        found = data.modalities[modality]
        if not isinstance(found, Tabular):
            raise TypeError(f"modality {modality!r} is a {type(found).__name__}, not a Tabular modality")
        return modality, found
    tabular = {name: source for name, source in data.modalities.items() if isinstance(source, Tabular)}
    if len(tabular) != 1:
        raise ValueError(
            f"modality=None needs exactly one Tabular modality to discover; found {sorted(tabular)} of "
            f"{sorted(data.modalities)}"
        )
    name, source = next(iter(tabular.items()))
    return name, source


def _tabicl_encoder_class() -> type | None:
    """``TabICLEncoder``, or ``None`` if the optional ``tabicl`` extra is not installed.

    Imported lazily, and only here: a top-level import would force every caller of this module -- including one
    that never touches a TabICLEncoder -- to have the optional ``tabicl`` extra installed, the same reason
    ``kalecancer.model``'s own ``__getattr__`` defers it.
    """
    try:
        from kalecancer.model.encode.tabicl import TabICLEncoder
    except ImportError:
        return None
    return TabICLEncoder


def _require_no_tabicl(pipeline: Pipeline, name: str) -> None:
    """Raise if ``name``'s fitted encoder is a ``TabICLEncoder``: it cannot score a perturbed row.

    ``TabICLEncoder.forward`` (``kalecancer/model/encode/tabicl.py``) recognises a context (training) row by its
    patient id and raises if that row's values differ from what it saw at fit time -- exactly what every
    perturbed row :func:`_perturbed_view` builds does, by construction, since it keeps the patient's own id.
    Giving the perturbed row a FRESH id instead would dodge that check, but it would also drop the row out of
    its own context-fold exclusion (see the class docstring): the encoder would then embed it against the WHOLE
    context, including its own true label, leaking the patient's own outcome into its own prediction through the
    rest of the context -- a worse bug than the crash it would fix. A genuine counterfactual query -- scored
    with the fold exclusion kept but the value check skipped -- is not something ``forward`` exposes as a public
    path; it would need the encoder's own ``_embed``, ``context_fold`` and ``_position`` reimplemented here. This
    module does neither: it names the limitation up front instead, the same way
    :mod:`kalecancer.interpret.modality` rejects ``Concat``/``LateFusion`` for ablation rather than silently
    producing a wrong answer.

    Checked against the ACTUAL fitted encoder for ``name`` (:meth:`Pipeline.model_.encoders_for`), not the
    modality's type, so an ordinary tabular encoder (e.g. ``nn.Linear``) on the same kind of modality is
    unaffected; and only once the optional ``tabicl`` package is confirmed installed
    (:func:`_tabicl_encoder_class`) -- without it, no ``TabICLEncoder`` could have been built in the first place.

    This raises unconditionally, not only when the request would actually perturb a context row with
    ``allow_seen=True``: the crash is reachable only then (an un-seen patient is never a context id, so its
    perturbed row is never compared against one), but telling the two apart here would mean checking every
    perturbed patient against the encoder's own ``context_ids`` and re-deriving when that is and is not safe --
    exactly the kind of bespoke, encoder-specific reasoning :func:`_perturbed_view` is written not to need. The
    narrower, already-correct case (attributing a TabICL model on data that is never seen) is blocked too, as the
    price of a check this simple.
    """
    tabicl_encoder = _tabicl_encoder_class()
    if tabicl_encoder is None:
        return
    try:
        encoders = cast(Any, pipeline.model_).encoders_for(name)
    except KeyError:
        return
    if any(isinstance(encoder, tabicl_encoder) for encoder in encoders):
        raise ValueError(
            f"{name!r} is encoded by a TabICLEncoder, which permutation_importance, feature_shapley, and "
            "patient_report cannot perturb: TabICLEncoder.forward recognises a training row by its patient id "
            "and raises if the row's values differ from what it saw at fit time, which every perturbed row here "
            "does by construction. Giving the perturbed row a fresh id would dodge that check, but it would "
            "also drop the row out of its own context-fold exclusion, letting the rest of the context see the "
            "patient's own label when scoring it -- a leak worse than the crash it would avoid. There is no "
            "supported way to query a TabICLEncoder with changed values while keeping that exclusion."
        )


def _feature_names(frame: pd.DataFrame, feature_names: Sequence[str] | None) -> list[str]:
    """``feature_names`` if given (every one must be a column of ``frame``), else every column of ``frame``.

    Typed ``str`` to match every existing caller (including ``kalecancer.interpret.report``, outside this
    module's scope to retype), but at runtime this returns whatever ``frame.columns`` holds -- a table with
    pandas' default columns has plain integers, not strings -- and everything downstream (``_perturbed_view``'s
    column indexing in particular) works the same regardless of the label's actual type.
    """
    if feature_names is None:
        return list(frame.columns)
    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        raise ValueError(f"feature_names {missing} are not columns of this modality's frame: {list(frame.columns)}")
    return list(feature_names)


def _evaluate(
    pipeline: Pipeline,
    view: MultimodalDataset,
    metric: Metric,
    *,
    allow_seen: bool,
    survival_times: Sequence[float] | None,
) -> float:
    """Score ``metric`` on ``view``, forwarding ``survival_times``.

    ``Pipeline.evaluate`` already raises when a metric needs ``survival[<time>]`` columns (``IntegratedBrier``,
    ``CalibrationError``) that ``survival_times`` was never given to produce; its own message names
    ``survival_times``, but points at ``Pipeline.predict``/``evaluate``/``cross_validate`` -- not at this
    function's own ``survival_times`` argument, which a caller of ``permutation_importance``/``feature_shapley``
    is more likely to reach for. Rather than re-implementing that check, this only adds a pointer to it.
    """
    try:
        score = pipeline.evaluate(view, {"score": metric}, allow_seen=allow_seen, survival_times=survival_times)[
            "score"
        ]
    except ValueError as error:
        if survival_times is None and "'survival[<time>]'" in str(error):
            raise ValueError(
                f"{error}; permutation_importance and feature_shapley also accept survival_times directly"
            ) from error
        raise
    return float(score)


def _perturbed_view(
    data: MultimodalDataset, name: str, frame: pd.DataFrame, replacements: dict[str, pd.Series]
) -> MultimodalDataset:
    """Perturb one modality's columns in a dataset view, without widening who gets evaluated.

    ``data``, with ``name``'s raw frame (``frame``, already restricted to ``data.ids``) replaced column by column
    by ``replacements``; columns not in ``replacements`` keep ``frame``'s true values.

    Assigned by column indexing on a copy of ``frame``, not ``frame.assign(**replacements)``: ``assign`` unpacks
    ``replacements`` as keyword arguments, which only ever works for ``str`` column labels, while ``Tabular``'s
    frame (and ``_feature_names``, see its own note) accepts any label pandas allows -- a table with pandas'
    default integer columns would raise ``TypeError: keywords must be strings`` on the very first perturbation.
    (The type hint above stays ``str`` for the same reason ``_feature_names``'s does; the dict this actually
    receives can have non-``str`` keys too.) ``frame.copy()`` keeps exactly ``frame``'s own index (nothing
    widens it), so the leakage guard below is unaffected: it is a copy of the VALUES, never of which rows/ids
    are present.

    ``MultimodalDataset.subset`` only narrows ``.ids`` -- it never shrinks the modality or target objects
    underneath -- so rebuilding a dataset from ``data.modalities``/``data.target`` can silently widen who gets
    evaluated back to everyone those raw objects happen to cover (e.g. training patients ``data`` itself
    excludes), which would also quietly defeat the caller's ``allow_seen``. Building the new ``Tabular`` from
    ``frame`` (already just ``data.ids``) prevents that already, since every other modality and the target still
    require those same ids; the trailing ``.subset(data.ids)`` is a second, independent guarantee of the same
    thing, kept even though it is currently redundant, because it costs nothing and the alternative is a perturbed
    view that silently scores the wrong patients.
    """
    perturbed_frame = frame.copy()
    for column, values in replacements.items():
        perturbed_frame[column] = values
    perturbed = Tabular(perturbed_frame)
    view = MultimodalDataset({**data.modalities, name: perturbed}, data.target, data.required_modalities)
    return view.subset(data.ids)


def permutation_importance(
    pipeline: Pipeline,
    data: MultimodalDataset,
    metric: Metric,
    *,
    modality: str | None = None,
    feature_names: Sequence[str] | None = None,
    n_repeats: int = 10,
    seed: int = 0,
    allow_seen: bool = False,
    survival_times: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Shuffle each clinical feature in turn, among the evaluated patients, and measure the metric drop.

    ``metric.higher_is_better`` decides the sign: the reported drop is positive whenever removing a feature's
    information makes the model worse, whichever direction ``metric`` itself points in. For a metric like
    ``IntegratedBrier`` or ``CalibrationError`` (``higher_is_better = False``, lower is better), shuffling an
    informative feature INCREASES the raw score; without flipping that to a positive drop here, an informative
    feature would misleadingly look unimportant (or a noise feature falsely important) for exactly the metrics
    most sensitive to miscalibration.

    Args:
        pipeline: A fitted ``Pipeline``.
        data: Patients to evaluate on.
        metric: Scored before and after each shuffle.
        modality: Name of the ``Tabular`` modality to permute. ``None`` discovers it, and requires there to be
            exactly one such modality in ``data``.
        feature_names: Columns of that modality's frame to test. ``None`` means every column.
        n_repeats: Independent shuffles per feature; more repeats shrink ``std`` without materially changing
            ``mean``.
        seed: Seed for the shuffles. A different seed changes every value but not which features rank highest.
        allow_seen: ``Pipeline.evaluate`` refuses patients used to fit ``pipeline`` unless this is set. Default
            ``False``, explicit here as in ``evaluate`` itself: silently defaulting to ``True`` would make it easy
            to read importance off of patients the model already memorised, overstating every feature at once.
        survival_times: Forwarded to ``Pipeline.evaluate``. Required for a metric that reads
            ``survival[<time>]`` columns -- ``IntegratedBrier`` and ``CalibrationError`` -- since neither can
            otherwise be driven through this function; irrelevant to a ranking metric like ``HarrellC``.

    Returns:
        DataFrame indexed by feature name, columns ``mean`` and ``std`` (of the signed drop, over the
        ``n_repeats`` shuffles) and ``baseline`` (the unperturbed score, in ``metric``'s own units, repeated on
        every row). ``.attrs["n_excluded"]`` is how many of ``data``'s patients lack the ``Tabular`` modality
        (possible when it is not in ``required_modalities`` -- a ``MaskedMean``-style fusion can still predict
        such a patient from another modality): they are scored every time, same as everyone else, but never
        perturbed, since there is no value of theirs to shuffle.

    Raises:
        ValueError: Under the conditions of :func:`_resolve_tabular_modality` and :func:`_feature_names`, if
            ``metric`` needs ``survival_times`` and none was given, or if ``name``'s fitted encoder is a
            ``TabICLEncoder`` (see :func:`_require_no_tabicl`).
    """
    name, source = _resolve_tabular_modality(data, modality)
    _require_no_tabicl(pipeline, name)
    present_ids = data.present_ids(name)
    frame = source.transform_input(present_ids)
    features = _feature_names(frame, feature_names)
    sign = 1.0 if metric.higher_is_better else -1.0

    baseline = _evaluate(pipeline, data, metric, allow_seen=allow_seen, survival_times=survival_times)
    rng = np.random.default_rng(seed)
    rows: dict[str, dict[str, float]] = {}
    for feature in features:
        values = frame[feature].to_numpy()
        drops = np.empty(n_repeats)
        for repeat in range(n_repeats):
            shuffled = pd.Series(rng.permutation(values), index=frame.index)
            view = _perturbed_view(data, name, frame, {feature: shuffled})
            score = _evaluate(pipeline, view, metric, allow_seen=allow_seen, survival_times=survival_times)
            drops[repeat] = sign * (baseline - score)
        rows[feature] = {"mean": float(drops.mean()), "std": float(drops.std()), "baseline": baseline}
    result = pd.DataFrame.from_dict(rows, orient="index")[["mean", "std", "baseline"]]
    result.attrs["n_excluded"] = len(data.ids) - len(present_ids)
    return result


def feature_shapley(
    pipeline: Pipeline,
    data: MultimodalDataset,
    metric: Metric,
    *,
    modality: str | None = None,
    feature_names: Sequence[str] | None = None,
    n_samples: int = 50,
    seed: int = 0,
    allow_seen: bool = False,
    survival_times: Sequence[float] | None = None,
) -> pd.Series:
    """Sampling-based Shapley values of each clinical feature's contribution to ``metric``.

    These are ESTIMATES, from a Monte Carlo permutation-sampling scheme (Strumbelj & Kononenko), not exact
    Shapley values: exact computation needs every one of a feature's ``2 ** (d - 1)`` coalitions re-scored, which
    is infeasible past a handful of columns. Standard error of each feature's estimate shrinks as
    ``O(1 / sqrt(n_samples))`` -- doubling ``n_samples`` roughly halves the noise, at double the cost.

    Each of ``n_samples`` draws: every feature starts replaced by an independent permutation of its values across
    the evaluated patients (a "fully resampled background", the same shuffle :func:`permutation_importance`
    uses), and ``metric`` is scored. Features are then revealed back to their true values one at a time, in a
    freshly sampled random order, re-scoring after every reveal. A feature's marginal contribution on that draw is
    the signed change in score its reveal caused; averaged over every draw -- which also averages over every
    position a feature could occupy in the order -- this converges to the feature's Shapley value. The same
    ``metric.higher_is_better`` sign flip as :func:`permutation_importance` applies, so a positive value always
    means "this feature helps", regardless of which direction ``metric`` itself points in.

    This does not use ``shap`` or ``captum`` -- no runtime dependency on either is added. ``shap``'s
    ``KernelExplainer`` (or a model-specific explainer) computes the same quantity from fewer samples, via a
    weighted-regression formulation instead of plain Monte Carlo averaging, and would be the better long-term
    route if feature counts grow large enough that this estimator's cost becomes a problem; for the clinical
    tables this package targets -- a few dozen columns at most -- it is not yet worth the added dependency.

    Args:
        pipeline: A fitted ``Pipeline``.
        data: Patients to evaluate on.
        metric: Scored after every reveal.
        modality: Name of the ``Tabular`` modality to attribute. ``None`` discovers it, and requires there to be
            exactly one such modality in ``data``.
        feature_names: Columns of that modality's frame to attribute. ``None`` means every column.
        n_samples: Monte Carlo draws. Cost is ``n_samples * (len(feature_names) + 1)`` calls to
            ``pipeline.evaluate``.
        seed: Seed for the resampling and orderings. A different seed changes every value but not which feature
            ranks highest.
        allow_seen: As in :func:`permutation_importance`.
        survival_times: As in :func:`permutation_importance` -- required for ``IntegratedBrier`` or
            ``CalibrationError``.

    Returns:
        Series indexed by feature name, named ``"shapley"``. ``.attrs["n_excluded"]`` is how many of ``data``'s
        patients lack the ``Tabular`` modality, as in :func:`permutation_importance`: scored every time, never
        perturbed.

    Raises:
        ValueError: Under the conditions of :func:`_resolve_tabular_modality` and :func:`_feature_names`, if
            ``metric`` needs ``survival_times`` and none was given, or if ``name``'s fitted encoder is a
            ``TabICLEncoder`` (see :func:`_require_no_tabicl`).
    """
    name, source = _resolve_tabular_modality(data, modality)
    _require_no_tabicl(pipeline, name)
    present_ids = data.present_ids(name)
    frame = source.transform_input(present_ids)
    features = _feature_names(frame, feature_names)
    sign = 1.0 if metric.higher_is_better else -1.0

    rng = np.random.default_rng(seed)
    contributions = {feature: np.empty(n_samples) for feature in features}
    log_every = max(1, n_samples // 10)

    for sample in range(n_samples):
        background = {
            feature: pd.Series(rng.permutation(frame[feature].to_numpy()), index=frame.index) for feature in features
        }
        order = [features[i] for i in rng.permutation(len(features))]
        still_shuffled = dict(background)
        view = _perturbed_view(data, name, frame, still_shuffled)
        previous = _evaluate(pipeline, view, metric, allow_seen=allow_seen, survival_times=survival_times)
        for feature in order:
            del still_shuffled[feature]
            view = _perturbed_view(data, name, frame, still_shuffled)
            current = _evaluate(pipeline, view, metric, allow_seen=allow_seen, survival_times=survival_times)
            contributions[feature][sample] = sign * (current - previous)
            previous = current
        if (sample + 1) % log_every == 0 or sample + 1 == n_samples:
            logger.info("feature_shapley: %d/%d samples", sample + 1, n_samples)

    values = {feature: float(contributions[feature].mean()) for feature in features}
    for feature in features:
        standard_error = (
            float(contributions[feature].std(ddof=1) / np.sqrt(n_samples)) if n_samples > 1 else float("nan")
        )
        logger.info(
            "feature_shapley: %s = %.4g +/- %.4g (standard error, n_samples=%d)",
            feature,
            values[feature],
            standard_error,
            n_samples,
        )

    result = pd.Series(values, name="shapley")
    result.attrs["n_excluded"] = len(data.ids) - len(present_ids)
    return result
