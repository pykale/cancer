"""Per-patient interpretability reports: what the model predicted for one patient, and why.

**The distinction this module exists to keep.** :mod:`kalecancer.interpret.modality` and
:mod:`kalecancer.interpret.attribution` are COHORT-level: they ask whether a modality or feature helps the model
rank patients, answered as a change in a METRIC over a whole dataset. This module asks a different question --
did this modality or feature push THIS patient's own risk score up or down -- answered as a change in that
patient's own risk, with no metric involved. These are different numbers answering different questions:
``PatientReport.modality_contributions``/``feature_contributions`` below are risk deltas, computed with their own
Shapley-style enumeration/sampling over raw risk, never by calling :func:`kalecancer.interpret.modality.modality_shapley`
or :func:`kalecancer.interpret.attribution.feature_shapley` and relabelling their cohort-level numbers as if they
were about one patient.

**Why the expensive parts here are cheaper per call than the cohort-level ones.** A cohort-level Shapley value
needs the whole evaluated cohort in every coalition, because the METRIC is a property of the cohort. A
per-patient risk delta does not, because the quantity being explained -- one patient's own ``log_hazard`` -- is a
property of one row, PROVIDED every encoder's output for a row does not depend on which other rows share its
batch. That had to be verified, not assumed, for this codebase's encoders: ``TabICLEncoder.forward``/``_embed``
(``kalecancer/model/encode/tabicl.py``) embeds each query row against a CONTEXT fixed at ``fit`` time (the
training rows, split into folds only to exclude a training row's own fold from its own context) and its own
docstring states outright that "queries never attend to each other, so a patient's embedding does not depend on
the rest of its batch" -- the context a row is scored against depends on that row's own identity (whether it was
itself a training/context row, and if so which fold), never on who else is in the current prediction call. Every
other encoder in this package (``nn.Linear``, ``ABMIL``, and friends) is an ordinary per-row/per-bag computation
with no batch-level statistic either. So the one thing that genuinely needs the full cohort --
``risk_percentile``, a property of the cohort by definition -- is computed from one predict() call over ``data``;
every other, more expensive computation here (the ``2**k`` modality coalitions, the ``n_feature_samples * (d+1)``
feature-attribution samples) subsets to just the requested patients first.

Reuses :mod:`kalecancer.interpret.modality` and :mod:`kalecancer.interpret.attribution`'s private helpers --
``_absent``, ``_coalition_view``, ``_perturbed_view``, the fusion-ablatability check, the ``.subset(data.ids)``
restriction, ``_resolve_tabular_modality`` -- rather than duplicating them; only the Shapley SCORING differs
(raw per-patient risk, not a metric reduced over a cohort), so only that part is written fresh here.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from kalecancer.evaluate.metrics import _SURVIVAL_COLUMN
from kalecancer.interpret.attention import attention
from kalecancer.interpret.attribution import _feature_names, _perturbed_view, _resolve_tabular_modality
from kalecancer.interpret.modality import (
    _coalition_view,
    _defined_mask,
    _fusion_of,
    _model_modalities,
    _require_ablatable,
)
from kalecancer.loaddata.dataset import MultimodalDataset
from kalecancer.loaddata.modalities import BagModality
from kalecancer.model.encode.base_encoders import AttentionEncoder
from kalecancer.model.fusion import FusionMethod
from kalecancer.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _require_survival_model(pipeline: Pipeline) -> None:
    """Raise unless ``pipeline`` predicts a ``log_hazard`` -- ``risk`` and the survival curve both need one.

    ``baseline_`` is only ever set for a model with a Cox-style head (see ``Pipeline.fit``); it is already fitted
    by the time ``patient_report`` runs, so there is no baseline to fit here, only this to check.
    """
    if pipeline.baseline_ is None:
        raise ValueError(
            "patient_report needs a survival model that predicts a log_hazard (a CoxHead); this pipeline's "
            f"baseline_ is None, meaning it was fitted on a different kind of target ({type(pipeline.target_)})"
        )


def _require_known_patients(data: MultimodalDataset, patient_ids: Sequence[str]) -> None:
    """Raise naming any id in ``patient_ids`` that ``data`` does not have."""
    available = set(data.ids)
    unknown = [pid for pid in patient_ids if pid not in available]
    if unknown:
        raise ValueError(f"patient_ids {unknown} are not in data; it has {len(data.ids)} patients")


def _require_not_seen(pipeline: Pipeline, patient_ids: Sequence[str], allow_seen: bool) -> None:
    """Raise if any of ``patient_ids`` was used to fit ``pipeline``, unless ``allow_seen``.

    ``Pipeline.predict`` has no such guard (it needs no target, so it cannot tell seen from unseen on its own),
    unlike ``Pipeline.evaluate``; this is the same check, applied to the patients a report is actually explaining
    rather than to a cohort being scored.
    """
    seen = sorted(set(patient_ids) & (set(pipeline.train_ids_) | set(pipeline.val_ids_)))
    if seen and not allow_seen:
        raise ValueError(
            f"{len(seen)} of the requested patients were used to fit this pipeline (e.g. {seen[:5]}); explain "
            "held-out patients, or pass allow_seen=True"
        )


def _percentile_rank(value: float, population: pd.Series) -> float:
    """Percentile rank of ``value`` within ``population`` (0-100); a tie counts as half a step."""
    defined = population.dropna()
    below = float((defined < value).sum())
    tied = float((defined == value).sum())
    return 100.0 * (below + 0.5 * tied) / len(defined)


def _modality_risk_contributions(
    pipeline: Pipeline, requested: MultimodalDataset, fusion: FusionMethod, names: Sequence[str]
) -> pd.DataFrame:
    """Per-patient Shapley value (a risk delta, not a metric) of each modality in ``names``.

    Exactly :func:`kalecancer.interpret.modality.modality_shapley`'s coalition enumeration, weighting, and
    empty-coalition-by-extrapolation convention, applied to raw per-patient ``log_hazard`` instead of a metric --
    duplicated here (not imported) because that scoring, not the coalition bookkeeping, is the whole distinction
    this module exists to keep separate. Exact, not sampled: with ``len(names)`` typically 2-4, ``2**k`` is no
    more expensive here than for the cohort version.

    A patient who already lacks one of ``names`` naturally (possible when it is not in ``required_modalities``)
    can be left with none at all once a coalition ablates the rest -- ``fusion.defined`` is ``False`` for them
    under that one coalition specifically, not because the fusion method itself can never tolerate a missing
    modality (:func:`kalecancer.interpret.modality._require_ablatable` already checked that). Scoring only the
    patients that coalition still defines (:func:`kalecancer.interpret.modality._defined_mask`) and leaving the
    rest ``NaN`` for it -- rather than handing ``_coalition_view`` a patient it would have to drop entirely,
    which raises -- keeps that one coalition's missing information from crashing the whole computation.
    """
    k = len(names)
    present_subsets = [frozenset(s) for size in range(1, k + 1) for s in itertools.combinations(names, size)]
    ablated_sets = [frozenset(names) - present for present in present_subsets]

    scores: dict[frozenset[str], pd.Series] = {}
    for present, ablated in zip(present_subsets, ablated_sets, strict=True):
        defined = _defined_mask(requested, fusion, names, ablated)
        survivors = [pid for pid in requested.ids if defined.loc[pid]]
        predicted = (
            pipeline.predict(_coalition_view(requested.subset(survivors), ablated))["log_hazard"]
            if survivors
            else pd.Series(dtype=float)
        )
        scores[present] = predicted.reindex(requested.ids)

    full = frozenset(names)
    loo_drops = (scores[full] - scores[full - {name}] for name in names)
    scores[frozenset()] = scores[full] - sum(loo_drops)

    shapley = {}
    for name in names:
        others = [m for m in names if m != name]
        total = pd.Series(0.0, index=requested.ids)
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
            for combo in itertools.combinations(others, size):
                without = frozenset(combo)
                with_name = without | {name}
                total = total + weight * (scores[with_name] - scores[without])
        shapley[name] = total
    return pd.DataFrame(shapley, index=requested.ids)


def _feature_risk_contributions(
    pipeline: Pipeline,
    data: MultimodalDataset,
    requested: MultimodalDataset,
    name: str,
    features: Sequence[str],
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Per-patient Shapley-style risk delta (a sampling estimate) for each raw feature in ``features``.

    The same permutation-sampling procedure as :func:`kalecancer.interpret.attribution.feature_shapley` -- a
    random feature ordering, an independently-resampled "absent" background, revealed one column at a time -- but
    the per-draw contribution is a risk delta read off directly, never reduced through a metric. The background
    is resampled from the patients in ``data`` who HAVE ``name`` (the full, representative cohort; ``name`` need
    not be required, so not every cohort member has it), while every ``pipeline.predict`` call here runs only
    over ``requested`` (just the patients being reported on; see the module docstring on row-independence).

    A requested patient without ``name`` at all has no raw value of it to perturb, so their row is NaN for every
    feature -- never dropped (which would silently narrow who gets a row) and never zero (which would claim
    "measured, found no effect" about something never measured). ``patient_report`` records who this affected in
    ``metadata["patients_without_tabular_modality"]``.
    """
    source = data.modalities[name]
    pool = source.transform_input(data.present_ids(name))
    present = requested.present_ids(name)
    frame = source.transform_input(present)
    m = len(present)
    rng = np.random.default_rng(seed)
    contributions = {f: pd.Series(0.0, index=present) for f in features}
    log_every = max(1, n_samples // 10)

    for sample in range(n_samples if m else 0):
        background = {
            f: pd.Series(pool[f].to_numpy()[rng.integers(0, len(pool), size=m)], index=frame.index) for f in features
        }
        order = [features[i] for i in rng.permutation(len(features))]
        still_background = dict(background)
        view = _perturbed_view(requested, name, frame, still_background)
        previous = pipeline.predict(view)["log_hazard"].reindex(present)
        for f in order:
            del still_background[f]
            view = _perturbed_view(requested, name, frame, still_background)
            current = pipeline.predict(view)["log_hazard"].reindex(present)
            contributions[f] = contributions[f] + (current - previous)
            previous = current
        if (sample + 1) % log_every == 0 or sample + 1 == n_samples:
            logger.info("patient_report feature_contributions: %d/%d samples", sample + 1, n_samples)

    result = pd.DataFrame({f: values / n_samples for f, values in contributions.items()}, index=present)
    return result.reindex(requested.ids)


def _find_bag_modality(pipeline: Pipeline, data: MultimodalDataset) -> str | None:
    """The one modality the model reads that is a ``BagModality``, if there is exactly one."""
    bags = [name for name in _model_modalities(pipeline) if isinstance(data.modalities.get(name), BagModality)]
    return bags[0] if len(bags) == 1 else None


def _can_attend(pipeline: Pipeline, modality: str) -> bool:
    """Whether ``modality``'s encoder list has exactly one ``AttentionEncoder`` -- what :func:`attention` needs.

    Checked here, against the fitted encoder list itself (:meth:`Pipeline.model_.encoders_for`, the same lookup
    ``run_encoders`` uses), before ever calling :func:`attention` -- a mean-pooling (or any other non-attention)
    bag encoder is a known, nameable gap, not an exception to catch. ``attention`` itself raises for anything
    other than exactly one match; this mirrors that exact condition rather than a looser "has some attention
    encoder" guess, so the two can never disagree about what counts as attention-capable.
    """
    # Pipeline.model_ is typed as plain nn.Module (see kalecancer.interpret.modality._model_modalities for the
    # same pre-existing limitation), so .encoders_for resolves through nn.Module's own loosely-typed
    # __getattr__ rather than a real method signature; it is a genuine EncoderList at runtime regardless.
    encoders = cast(Any, pipeline.model_).encoders_for(modality)
    hits = [encoder for encoder in encoders if isinstance(encoder, AttentionEncoder)]
    return len(hits) == 1


def _top_patches(
    pipeline: Pipeline, requested: MultimodalDataset, bag_modality: str | None, top_k: int
) -> dict[str, pd.DataFrame | None]:
    """``top_k`` most-attended instances per patient with ``bag_modality``, ``None`` for a patient without it."""
    if bag_modality is None or not requested.present[bag_modality].any():
        return dict.fromkeys(requested.ids)
    weights = attention(pipeline, requested.subset(requested.present_ids(bag_modality)), bag_modality)
    result: dict[str, pd.DataFrame | None] = dict.fromkeys(requested.ids)
    for pid, rows in weights.groupby("patient_id"):
        top = rows.sort_values("attention", ascending=False).head(top_k).drop(columns="patient_id")
        result[pid] = top.reset_index(drop=True)
    return result


@dataclass(frozen=True)
class PatientReport:
    """One patient's interpretability bundle.

    Attributes:
        patient_id: Who this is.
        risk: This patient's predicted ``log_hazard``.
        risk_percentile: Where ``risk`` sits (0-100) within the FULL cohort in ``data``, not just the patients
            requested in the same call -- requesting more or fewer patients alongside this one never moves it.
        survival: ``time``/``survival`` columns, this patient's predicted survival probability at each of the
            requested ``survival_times``. ``None`` when ``survival_times`` was not given.
        modality_contributions: Risk delta per modality the model reads -- how much marking that modality absent
            (the same presence mechanism the model itself uses for a missing modality) would have changed THIS
            patient's own risk. Exact Shapley value over modality coalitions, NOT the cohort-level number
            :func:`kalecancer.interpret.modality.modality_shapley` reports.

            ``NaN`` for every modality when this patient already lacks one modality naturally (an optional one
            they do not have) and some coalition the exact Shapley weighting needs ablates the rest too, leaving
            them with none at all for that one coalition -- the same kind of "nothing to score" this module
            otherwise avoids by restricting ``feature_contributions`` to the patients who have the Tabular
            modality (see that field); here it shows up as ``NaN`` rather than a crash, because exact Shapley
            combines every coalition's value and one missing value makes the combination undefined too.

            This is a risk delta under the model's OWN fusion method (named in ``metadata["fusion"]``), not a
            clean measurement of "how much information this modality carries". For a mean-style fusion
            (``MaskedMean``), marking a modality absent changes how many inputs the mean divides by, which
            rescales the OTHER present modalities' effective weight in the average -- part of the delta
            attributed to the ablated modality is really this arithmetic rescaling, not its own information. A
            rank-based cohort metric like ``HarrellC`` is invariant to that rescaling; raw ``log_hazard``, which
            is what this reports, is not. Read these values as comparable to EACH OTHER for this one patient
            (which modality moved this patient's risk more), not as "this modality contributed X log-hazard of
            information" in isolation.
        feature_contributions: Risk delta per raw clinical feature, by the same logic, for the model's one
            Tabular modality. A sampling estimate (see ``metadata["n_feature_samples"]``). NOT the cohort-level
            number :func:`kalecancer.interpret.attribution.feature_shapley` reports.

            ``NaN`` for every feature when this patient has no row of the Tabular modality at all (possible when
            it is not in ``required_modalities``): there is no raw value of theirs to perturb, so there is
            nothing to estimate. ``NaN`` rather than dropping the patient (which would silently narrow who gets
            a row) or zero (which would claim "measured, found no effect" about something never measured); see
            ``metadata["patients_without_tabular_modality"]`` for who this affected.
        top_patches: The ``top_k`` most-attended patches for this patient's bag modality. ``None`` when this
            patient individually has no bag (their own ``present`` bit is ``False``), the model has no single,
            unambiguous bag-pooling modality, or that modality's encoder cannot produce attention weights at all
            (e.g. it pools by mean rather than attention) -- ``metadata["attention_unavailable_reason"]`` names
            which of the last two it is, and is ``None`` exactly when attention was available at all, so a
            patient-level ``None`` alongside a ``None`` reason means the first case (this one patient's own bag
            is missing).
        metadata: What was computed and how -- ``seed``, ``n_feature_samples``, ``modalities``,
            ``tabular_modality``, ``bag_modality``, ``cohort_size``, ``fusion`` (the fusion method
            ``modality_contributions`` was computed under -- the context needed to read it; see that field),
            ``patients_without_tabular_modality`` (which requested patients' ``feature_contributions`` are
            ``NaN`` for lacking the Tabular modality entirely; see that field), and
            ``attention_unavailable_reason`` (see ``top_patches``).
    """

    patient_id: str
    risk: float
    risk_percentile: float
    survival: pd.DataFrame | None
    modality_contributions: pd.Series
    feature_contributions: pd.Series
    top_patches: pd.DataFrame | None
    metadata: dict


def patient_report(
    pipeline: Pipeline,
    data: MultimodalDataset,
    patient_ids: Sequence[str],
    *,
    survival_times: Sequence[float] | None = None,
    top_k: int = 20,
    n_feature_samples: int = 50,
    seed: int = 0,
    allow_seen: bool = False,
) -> list[PatientReport]:
    """Build an interpretability bundle for each named patient.

    ``patient_ids`` is required, not optional, on purpose: a per-patient Shapley computation over every patient
    in ``data`` is not something anyone should start by accident.

    ``modality_contributions`` is a risk delta under the fitted model's OWN fusion method (recorded in
    ``metadata["fusion"]`` for exactly this reason), not a pure measurement of how much information a modality
    carries: for a mean-style fusion (``MaskedMean``), marking a modality absent changes how many inputs the mean
    divides by, which rescales the other present modalities' effective weight too. A rank-based cohort metric
    like ``HarrellC`` does not see this (it is invariant to the rescaling); raw ``log_hazard`` does. See
    ``PatientReport.modality_contributions`` for how to read the numbers given that.

    Args:
        pipeline: A fitted ``Pipeline`` whose model predicts a ``log_hazard``.
        data: The full cohort. ``risk_percentile`` is computed against all of it; the modality and feature
            contributions subset to ``patient_ids`` (see the module docstring).
        patient_ids: Which patients to build a report for. Every id must be in ``data``.
        survival_times: Horizons for the survival curve, forwarded to ``Pipeline.predict``. ``None`` skips it.
        top_k: Number of top-attended patches to keep per patient with a bag modality.
        n_feature_samples: Permutation-sampling budget for ``feature_contributions``; the same meaning as
            ``feature_shapley``'s ``n_samples``, cheaper here for the reason the module docstring gives.
        seed: Seeds the feature-contribution sampling; deterministic for a fixed seed.
        allow_seen: ``patient_ids`` used to fit ``pipeline`` raise unless this is set (see ``evaluate``'s own
            ``allow_seen``); it does not gate ``risk_percentile``, which is a property of ``data`` as a
            population, not a per-patient explanation.

    Returns:
        One ``PatientReport`` per id in ``patient_ids``, in that order.

    Raises:
        ValueError: If ``pipeline`` does not predict a ``log_hazard``, if any ``patient_ids`` entry is not in
            ``data``, if those patients were used to fit ``pipeline`` and ``allow_seen`` is not set, if the
            fitted model's fusion method cannot tolerate a missing modality (see ``modality_shapley``), if the
            fitted model is a ``LateFusion`` (not supported yet; see
            :func:`kalecancer.interpret.modality._require_ablatable`), or if ``data`` does not have exactly one
            ``Tabular`` modality.
    """
    _require_survival_model(pipeline)
    _require_known_patients(data, patient_ids)
    _require_not_seen(pipeline, patient_ids, allow_seen)

    model_modalities = _model_modalities(pipeline)
    fusion = _fusion_of(pipeline)
    _require_ablatable(fusion, model_modalities, model_modalities, pipeline.model_)
    tabular_name, tabular_source = _resolve_tabular_modality(data, None)
    features = _feature_names(tabular_source.transform_input(data.present_ids(tabular_name)), None)
    bag_modality = _find_bag_modality(pipeline, data)
    if bag_modality is None:
        attention_unavailable_reason: str | None = "no single, unambiguous bag-pooling modality found on the model"
    elif not _can_attend(pipeline, bag_modality):
        attention_unavailable_reason = (
            f"modality {bag_modality!r}'s encoder cannot produce attention weights (needs exactly one "
            "AttentionEncoder; e.g. a mean-pooling encoder has none)"
        )
    else:
        attention_unavailable_reason = None

    full_prediction = pipeline.predict(data, survival_times=survival_times)
    population = full_prediction["log_hazard"]

    requested = data.subset(list(patient_ids))
    modality_contributions = _modality_risk_contributions(pipeline, requested, fusion, model_modalities)
    feature_contributions = _feature_risk_contributions(
        pipeline, data, requested, tabular_name, features, n_feature_samples, seed
    )
    top_patches = _top_patches(
        pipeline, requested, bag_modality if attention_unavailable_reason is None else None, top_k
    )

    survival_columns = {float(m[1]): c for c in full_prediction.columns if (m := _SURVIVAL_COLUMN.fullmatch(str(c)))}
    times_sorted = sorted(survival_columns)

    present_tabular = set(requested.present_ids(tabular_name))
    patients_without_tabular = [pid for pid in patient_ids if pid not in present_tabular]

    metadata = {
        "seed": seed,
        "n_feature_samples": n_feature_samples,
        "modalities": model_modalities,
        "tabular_modality": tabular_name,
        "bag_modality": bag_modality,
        "attention_unavailable_reason": attention_unavailable_reason,
        "patients_without_tabular_modality": patients_without_tabular,
        "cohort_size": len(data.ids),
        # modality_contributions is a risk delta under THIS fusion method -- see its docstring for why a
        # mean-style fusion mixes rescaling into the delta, which is why a saved report needs to carry this.
        "fusion": type(fusion).__name__,
    }

    reports = []
    for pid in patient_ids:
        risk = float(population.loc[pid])
        survival = None
        if times_sorted:
            survival = pd.DataFrame(
                {
                    "time": times_sorted,
                    "survival": [float(full_prediction.loc[pid, survival_columns[t]]) for t in times_sorted],
                }
            )
        reports.append(
            PatientReport(
                patient_id=pid,
                risk=risk,
                risk_percentile=_percentile_rank(risk, population),
                survival=survival,
                modality_contributions=modality_contributions.loc[pid],
                feature_contributions=feature_contributions.loc[pid],
                top_patches=top_patches[pid],
                metadata=dict(metadata),
            )
        )
    return reports


def _json_safe(value: object) -> object:
    """Convert a numpy scalar or array ``json.dump`` cannot serialize on its own."""
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def save_reports(reports: Sequence[PatientReport], path: str | Path) -> None:
    """Write ``reports`` to JSON: readable, round-trippable, no pickle.

    Every ``pd.Series``/``pd.DataFrame`` field becomes a plain dict (column-oriented for a DataFrame, so
    ``pd.DataFrame(d)`` reconstructs it), and ``None`` stays ``None`` rather than an empty placeholder that would
    be ambiguous with a genuinely empty result.
    """
    payload = [
        {
            "patient_id": report.patient_id,
            "risk": report.risk,
            "risk_percentile": report.risk_percentile,
            "survival": None if report.survival is None else report.survival.to_dict(orient="list"),
            "modality_contributions": report.modality_contributions.to_dict(),
            "feature_contributions": report.feature_contributions.to_dict(),
            "top_patches": None if report.top_patches is None else report.top_patches.to_dict(orient="list"),
            "metadata": report.metadata,
        }
        for report in reports
    ]
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
