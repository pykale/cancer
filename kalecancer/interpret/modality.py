"""Modality attribution: how much each modality contributes to a fitted pipeline's score.

Reuses :mod:`kalecancer.interpret.attribution`'s conventions: the ``(pipeline, data, metric, ...)`` argument
order, ``allow_seen``, ``survival_times``, an ``_evaluate`` helper of the same shape, and the same
``metric.higher_is_better`` sign flip (positive always means "this modality helps").

A modality cannot be ablated by dropping it from the dataset: ``Pipeline.predict`` checks, via
``model.check()``/``_check_modalities``, that the dataset has every modality the model reads, and raises rather
than treating a missing key as "absent for everyone". Instead, both functions here mark a modality absent by
PRESENCE: the modality's key stays in ``data.modalities``, but its value becomes an empty copy of the same type
(one whose ``.ids`` covers nobody), dropped from ``required_modalities`` so the dataset does not exclude every
patient for lacking a "required" modality that is now never present. ``MultimodalDataset`` derives each
modality's presence purely from its ``.ids`` (see ``MultimodalDataset._set_ids``), and that presence bit is
exactly what ``IntermediateFusion.forward`` (and ``EarlyFusion``, ``LateFusion``) already read to treat a
modality as missing for a patient -- so this invents no second notion of "absent".

This only works with a fusion method that can still define a prediction when one modality is absent (e.g.
``MaskedMean``, ``MajorityVote``). ``Concat.defined`` is ``present.all(dim=1)``: with one modality absent for
everyone, nobody is defined and every prediction would be NaN. Both functions check this against the fitted
model's actual fusion method up front and raise, rather than silently returning a cohort of NaNs that looks like
a result.

Nor does it work with ``LateFusion`` yet: its outer fusion combines BRANCH predictions, each branch with its own
requirements (e.g. a ``Concat`` branch needing several modalities at once), not raw per-modality presence one
for one the way ``IntermediateFusion``/``EarlyFusion``'s fusion method does. Treating branch-structured
presence as if it were modality-structured -- which is what reusing the same up-front check would do -- can
call a coalition "defined" when every branch would actually fail to predict for it. Rather than that silent
mismatch, ``LateFusion`` is rejected explicitly, the same way ``Concat`` is: see :func:`_require_ablatable`.
"""

from __future__ import annotations

import copy
import itertools
import math
from collections.abc import Sequence
from typing import Any, cast

import pandas as pd
import torch

from kalecancer.evaluate.metrics import Metric
from kalecancer.loaddata.dataset import MultimodalDataset
from kalecancer.loaddata.modalities import Modality
from kalecancer.model import LateFusion
from kalecancer.model.fusion import FusionMethod
from kalecancer.pipeline.pipeline import Pipeline


def _model_modalities(pipeline: Pipeline) -> list[str]:
    """Every modality the fitted model reads, in the model's own order.

    ``Pipeline.model_`` is typed as plain ``nn.Module`` (every concrete model keeps its own ``.modalities``, with
    no shared base class declaring it), so ``.modalities`` resolves through ``nn.Module``'s own loosely-typed
    ``__getattr__`` rather than a real return type -- it is a genuine list of names at runtime regardless, hence
    the ``cast``.
    """
    return list(cast("list[str]", pipeline.model_.modalities))


def _fusion_of(pipeline: Pipeline) -> FusionMethod:
    """The fitted model's fusion method.

    Raises:
        ValueError: If the model has no fusion method at all -- it reads one modality (``Unimodal``), so there
            is nothing to ablate it against.
    """
    fusion = getattr(pipeline.model_, "fusion", None)
    if fusion is None:
        raise ValueError(
            f"{type(pipeline.model_).__name__} has no fusion method: it reads one modality "
            f"({_model_modalities(pipeline)}), so there is nothing to ablate"
        )
    return fusion


def _resolve_modalities(pipeline: Pipeline, modalities: Sequence[str] | None) -> list[str]:
    """``modalities``, checked against the fitted model, or every modality it reads."""
    available = _model_modalities(pipeline)
    if modalities is None:
        return available
    unknown = [name for name in modalities if name not in available]
    if unknown:
        raise ValueError(f"modalities {unknown} are not read by this model; it reads {available}")
    return list(modalities)


def _require_ablatable(
    fusion: FusionMethod, model_modalities: Sequence[str], names: Sequence[str], model: object | None = None
) -> None:
    """Raise unless ``fusion`` can still define a patient missing just one of ``names``, the rest present.

    Checked directly against ``fusion.defined`` on a synthetic probe (one row per name in ``names``, every
    modality present except that row's own), rather than against any particular class: a fusion method written
    the same way as ``Concat`` (it needs every input) is caught the same way, without this module needing to
    know its name.

    ``model``, when given, is also checked against ``LateFusion`` specifically: ``LateFusion``'s own ``fusion``
    combines BRANCH predictions (each branch's own ``_defined_for``, which can need several modalities at once,
    e.g. a ``Concat`` branch), not raw per-modality presence. Applying it directly to ``data.present`` the way
    :func:`_defined_mask` does -- correct for ``IntermediateFusion``/``EarlyFusion``, whose fusion method really
    does combine modalities one-for-one -- can call a coalition "defined" when every one of its branches would
    actually fail, and a genuinely branch-aware check would need to re-derive each branch's own requirements
    here, which this module does not yet do. ``model`` defaults to ``None`` only so a caller elsewhere in this
    package that does not pass it keeps its current behaviour rather than breaking; :func:`leave_one_out_ablation`
    and :func:`modality_shapley` -- the two functions this module actually supports LateFusion through -- always
    pass it.
    """
    if isinstance(model, LateFusion):
        raise ValueError(
            "leave_one_out_ablation and modality_shapley do not support LateFusion yet: its outer fusion "
            "combines branch predictions, not raw modality presence, so a coalition this module considers "
            "'defined' from modality-level presence alone may not be one any branch can actually predict for. "
            "Score each branch as its own model instead."
        )
    columns = {name: j for j, name in enumerate(model_modalities)}
    probe = torch.ones(len(names), len(model_modalities), dtype=torch.bool)
    for row, name in enumerate(names):
        probe[row, columns[name]] = False
    defined = fusion.defined(probe)
    if not bool(defined.all()):
        unablatable = [name for name, ok in zip(names, defined.tolist(), strict=True) if not ok]
        raise ValueError(
            f"{type(fusion).__name__} requires every modality to produce a prediction: with just {unablatable} "
            "absent (every other modality present), nobody would be defined. leave_one_out_ablation and "
            "modality_shapley need a fusion method that tolerates a missing modality, such as MaskedMean or "
            "MajorityVote."
        )


def _absent(modality: Modality) -> Modality:
    """A shallow copy of ``modality`` whose ``.ids`` covers nobody -- how ablation marks it absent.

    ``MultimodalDataset`` only ever asks a modality to load an id that is in its own ``.ids``
    (``MultimodalDataset.__getitem__``), so nothing else needs to change for loading data. Fitting a transform is
    a separate path (``MultimodalDataset.with_transforms``) and is not as forgiving: it still calls
    ``with_transform`` for this modality if ``Pipeline`` was given one, now asking it to transform the rows of
    whoever has it -- which, with ``.ids`` empty, is always nobody -- and most sklearn transforms (
    ``StandardScaler`` among them) refuse to run on zero rows. Since there is never anything to actually
    transform, ``with_transform`` is overridden on this one instance to skip straight to an empty result instead
    of handing a real transform an input it was never fitted to see.
    """
    empty = copy.copy(modality)
    empty.ids = modality.ids[:0]

    def _with_transform(fitted_transforms: Any, ids: Sequence[str]) -> Modality:
        """``with_transforms`` derives ``ids`` from ``.ids`` (empty here), so this is always empty too."""
        return empty

    # setattr, not plain assignment: this overrides with_transform on this one instance, which mypy's
    # method-assign check (rightly, in general) refuses for a plain attribute set.
    setattr(empty, "with_transform", _with_transform)  # noqa: B010
    return empty


def _defined_mask(
    data: MultimodalDataset, fusion: FusionMethod, model_modalities: Sequence[str], ablated: frozenset[str]
) -> pd.Series:
    """Which of ``data``'s patients ``fusion`` would define if exactly ``ablated`` were marked absent.

    Computed directly from ``data.present`` (as ``IntermediateFusion._defined_for`` does), with ``ablated``'s
    columns forced ``False`` -- this is cheap enough to call once per candidate coalition, without needing to
    build a dataset or run the pipeline just to find out who would be comparable.
    """
    present = data.present[list(model_modalities)].copy()
    for name in ablated:
        present[name] = False
    return fusion.defined_rows(present)


def _comparable_ids(
    data: MultimodalDataset,
    fusion: FusionMethod,
    model_modalities: Sequence[str],
    ablated_sets: Sequence[frozenset[str]],
) -> list[str]:
    """Patients ``fusion`` defines under every one of ``ablated_sets`` -- the cohort every coalition is compared on.

    Dropping a modality changes who ``fusion.defined`` covers (a patient naturally missing some OTHER modality
    may become defined or undefined depending on what is ablated), so different coalitions can otherwise score
    different patient sets, making the comparison meaningless. Fixing this cohort once, rather than letting each
    coalition use whoever it happens to define, is what makes the reported differences comparable.
    """
    comparable = pd.Series(True, index=data.present.index)
    for ablated in ablated_sets:
        comparable &= _defined_mask(data, fusion, model_modalities, ablated)
    return [pid for pid in data.ids if comparable.loc[pid]]


def _coalition_view(data: MultimodalDataset, ablated: frozenset[str]) -> MultimodalDataset:
    """``data``, with every modality in ``ablated`` replaced by an empty copy and dropped from required_modalities.

    Rebuilding a ``MultimodalDataset`` from ``data.modalities``/``data.target`` can silently widen the cohort
    back to everyone those raw objects cover (e.g. training patients ``data`` itself excludes), which would also
    defeat the caller's ``allow_seen`` -- the same trap
    :func:`kalecancer.interpret.attribution._perturbed_view` guards against. The trailing ``.subset(data.ids)``
    is the same guarantee here.
    """
    modalities = {name: _absent(source) if name in ablated else source for name, source in data.modalities.items()}
    required = [name for name in data.required_modalities if name not in ablated]
    view = MultimodalDataset(modalities, data.target, required)
    return view.subset(data.ids)


def _evaluate(
    pipeline: Pipeline,
    view: MultimodalDataset,
    metric: Metric,
    *,
    allow_seen: bool,
    survival_times: Sequence[float] | None,
) -> float:
    """Score ``metric`` on ``view``, forwarding ``survival_times``.

    Mirrors :func:`kalecancer.interpret.attribution._evaluate`: ``Pipeline.evaluate`` already raises when a
    metric needs ``survival[<time>]`` columns that ``survival_times`` was never given to produce, naming
    ``survival_times`` but pointing at ``Pipeline.predict``/``evaluate``/``cross_validate`` rather than at this
    function's own argument, which a caller is more likely to reach for -- so this only adds a pointer to it,
    without re-implementing the check.
    """
    try:
        score = pipeline.evaluate(view, {"score": metric}, allow_seen=allow_seen, survival_times=survival_times)[
            "score"
        ]
    except ValueError as error:
        if survival_times is None and "'survival[<time>]'" in str(error):
            raise ValueError(
                f"{error}; leave_one_out_ablation and modality_shapley also accept survival_times directly"
            ) from error
        raise
    return float(score)


def leave_one_out_ablation(
    pipeline: Pipeline,
    data: MultimodalDataset,
    metric: Metric,
    *,
    modalities: Sequence[str] | None = None,
    allow_seen: bool = False,
    survival_times: Sequence[float] | None = None,
) -> pd.Series:
    """Mark each modality absent in turn and report the change in the metric.

    A modality's "contribution" by this measure is its marginal contribution against one fixed coalition:
    everyone else present. It is cheap -- ``len(modalities) + 1`` calls to ``pipeline.evaluate`` -- but is not
    exact credit assignment the way :func:`modality_shapley` is; two modalities that are individually redundant
    with each other can each come back near zero here, even though removing both together would hurt a lot.

    ``metric.higher_is_better`` decides the sign, exactly as in
    :func:`kalecancer.interpret.attribution.permutation_importance`: the reported drop is positive whenever
    marking a modality absent makes the model worse, whichever direction ``metric`` itself points in.

    Args:
        pipeline: A fitted ``Pipeline``.
        data: Patients to evaluate on.
        metric: Scored with every modality present, and again with each one absent.
        modalities: Which modalities to ablate. ``None`` means every modality the fitted model reads.
        allow_seen: ``Pipeline.evaluate`` refuses patients used to fit ``pipeline`` unless this is set, exactly as
            in ``permutation_importance``.
        survival_times: Forwarded to ``Pipeline.evaluate``; required for a metric that reads
            ``survival[<time>]`` columns (``IntegratedBrier``, ``CalibrationError``).

    Returns:
        Series indexed by modality name, named ``"drop"``. Its ``.attrs["n_excluded"]`` is how many of
        ``data``'s patients were not comparable across every coalition scored (see
        :func:`_comparable_ids`) and so were left out of every score.

    Raises:
        ValueError: If the fitted model has no fusion method, if its fusion method cannot tolerate even one
            modality absent (e.g. ``Concat``; see the module docstring), if the fitted model is a ``LateFusion``
            (not supported yet; see :func:`_require_ablatable`), if ``modalities`` names one the model does not
            read, or if ``metric`` needs ``survival_times`` and none was given.
    """
    names = _resolve_modalities(pipeline, modalities)
    model_modalities = _model_modalities(pipeline)
    fusion = _fusion_of(pipeline)
    _require_ablatable(fusion, model_modalities, names, pipeline.model_)
    sign = 1.0 if metric.higher_is_better else -1.0

    coalitions = [frozenset(), *(frozenset({name}) for name in names)]
    comparable_ids = _comparable_ids(data, fusion, model_modalities, coalitions)
    base = data.subset(comparable_ids)

    baseline = _evaluate(
        pipeline, _coalition_view(base, frozenset()), metric, allow_seen=allow_seen, survival_times=survival_times
    )
    drops = {}
    for name in names:
        view = _coalition_view(base, frozenset({name}))
        score = _evaluate(pipeline, view, metric, allow_seen=allow_seen, survival_times=survival_times)
        drops[name] = sign * (baseline - score)

    result = pd.Series(drops, name="drop")
    result.attrs["n_excluded"] = len(data.ids) - len(comparable_ids)
    return result


def modality_shapley(
    pipeline: Pipeline,
    data: MultimodalDataset,
    metric: Metric,
    *,
    modalities: Sequence[str] | None = None,
    max_modalities: int = 4,
    allow_seen: bool = False,
    survival_times: Sequence[float] | None = None,
) -> pd.Series:
    """Exact Shapley value per modality over all ``2**k`` coalitions.

    Every nonempty coalition (every nonempty subset of ``modalities``) is scored exactly once, cached by its
    ``frozenset``, then combined with the exact Shapley weighting ``|S|! (k - |S| - 1)! / k!`` over subsets
    ``S`` of the other modalities -- the same weighting the pre-refactor version used.

    THE EMPTY COALITION ("v(empty)", every selected modality absent) is measured directly when it still means
    something, and only extrapolated when it cannot mean anything at all -- these are different situations:

    - ``modalities`` is a PROPER SUBSET of what the model reads (e.g. ``modalities=['clinical']`` on a model
      that also reads ``wsi``): with every selected modality absent, the unselected ones -- ``wsi`` here -- are
      still present, exactly as they are for every other coalition this function scores. That is a real,
      scorable background, not an edge case, so ``v(empty)`` is a genuine ``pipeline.evaluate`` call like any
      other coalition's, included in :func:`_comparable_ids`'s cohort the same way. This is the fix for a
      single selected modality on a multi-modality model in particular: with ``k = 1``, the "every other
      coalition" set this function scores is just ``v({that modality})``, and without a scorable background
      ``v(empty)`` there would be nothing to form the one Shapley value from at all.
    - ``modalities`` is ``None`` or covers every modality the model reads: there is no unselected modality left
      to fall back on, so ``fusion.defined`` is ``False`` for everyone (by the same mechanism
      :func:`_require_ablatable` checks up front) and there is no score ``pipeline.evaluate`` can produce --
      unlike the pre-refactor version, where an absent modality could still contribute a placeholder value and
      the empty coalition's prediction was merely *likely* non-finite, here it is architecturally guaranteed to
      be undefined.

      A first attempt defined ``v(empty) := 0`` in ``metric``'s own (sign-corrected) units, as an arbitrary but
      metric-agnostic convention. That turned out to be a bad choice, not just an ugly one: it shifts every
      modality's value by the same amount, ``(v(empty)_true - 0) / k``, where ``v(empty)_true`` is whatever a
      truly uninformative prediction scores on this metric -- 0.5 for a concordance-style metric's chance
      level, for instance, not 0 -- which hid an uninformative modality's value behind that shift rather than
      reporting it near zero.

      ``v(empty)`` is instead defined by linear extrapolation from the full coalition and each modality's
      leave-one-out drop, assuming no interaction between modalities' marginal contributions::

          v(empty) := v(full) - sum_i (v(full) - v(full minus modality i))
                    = v(full) - sum_i leave_one_out_drop(i)

      This needs no extra ``pipeline.evaluate`` calls: every ``v(full minus modality i)`` is already one of the
      ``2**k - 1`` coalitions this function scores anyway.

    Either way, the standard Shapley efficiency property holds exactly -- it is a property of the weighting
    formula, true for whatever value ``v(empty)`` turns out to be, measured or extrapolated:
    ``sum(shapley values) = v(full) - v(empty)``. Only what that total COMES OUT TO differs: with a scorable
    background it is the real gap between "every selected modality present" and "every selected modality
    absent, background present"; without one (the extrapolated case) it reduces to
    ``sum_i leave_one_out_drop(i)`` -- exactly what :func:`leave_one_out_ablation` would report, totalled over
    modalities. What Shapley adds over that cheaper function is splitting credit fairly when modalities
    interact (are redundant with, or complement, each other); in the extrapolated case it does not change the
    total credit assigned when interactions are, on net, absent.

    Args:
        pipeline: A fitted ``Pipeline``.
        data: Patients to evaluate on.
        metric: Scored once per nonempty coalition.
        modalities: Which modalities to value. ``None`` means every modality the fitted model reads.
        max_modalities: Refuses more than ``2**max_modalities`` coalitions. Cost is ``2**k - 1`` calls to
            ``pipeline.evaluate`` for ``k`` modalities.
        allow_seen: As in :func:`leave_one_out_ablation`.
        survival_times: As in :func:`leave_one_out_ablation`.

    Returns:
        Series indexed by modality name, named ``"shapley"``. ``.attrs["n_excluded"]`` is as in
        :func:`leave_one_out_ablation`; ``.attrs["n_coalitions"]`` is ``2**k``, the full game size including the
        empty coalition.

    Raises:
        ValueError: Under the conditions of :func:`leave_one_out_ablation`, or if ``len(modalities) >
            max_modalities``.
    """
    names = _resolve_modalities(pipeline, modalities)
    k = len(names)
    if k > max_modalities:
        raise ValueError(
            f"modality_shapley needs one pipeline.evaluate call per nonempty coalition (2**k - 1); {k} "
            f"modalities would need 2**{k} - 1 = {2**k - 1} of them, exceeding max_modalities={max_modalities}. "
            "Pass a smaller modalities subset, or raise max_modalities if that cost is acceptable."
        )
    model_modalities = _model_modalities(pipeline)
    fusion = _fusion_of(pipeline)
    _require_ablatable(fusion, model_modalities, names, pipeline.model_)
    sign = 1.0 if metric.higher_is_better else -1.0

    present_subsets = [frozenset(s) for size in range(1, k + 1) for s in itertools.combinations(names, size)]
    ablated_sets = [frozenset(names) - present for present in present_subsets]

    full = frozenset(names)
    # The "background" coalition: every selected modality absent. Scorable whenever an unselected modality (one
    # the model reads but modalities= did not name) stays present -- then it is a real coalition like any other,
    # not the architecturally-undefined case _require_ablatable guards against (that check is about every
    # MODEL modality, not just the selected ones).
    background_scorable = bool(_defined_mask(data, fusion, model_modalities, full).any())
    comparability_sets = [*ablated_sets, full] if background_scorable else ablated_sets
    comparable_ids = _comparable_ids(data, fusion, model_modalities, comparability_sets)
    base = data.subset(comparable_ids)

    scores: dict[frozenset[str], float] = {}
    for present, ablated in zip(present_subsets, ablated_sets, strict=True):
        view = _coalition_view(base, ablated)
        scores[present] = sign * _evaluate(pipeline, view, metric, allow_seen=allow_seen, survival_times=survival_times)

    if background_scorable:
        view = _coalition_view(base, full)
        scores[frozenset()] = sign * _evaluate(
            pipeline, view, metric, allow_seen=allow_seen, survival_times=survival_times
        )
    else:
        loo_drops = (scores[full] - scores[full - {name}] for name in names)
        scores[frozenset()] = scores[full] - sum(loo_drops)

    shapley = {}
    for name in names:
        others = [m for m in names if m != name]
        total = 0.0
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
            for combo in itertools.combinations(others, size):
                without = frozenset(combo)
                with_name = without | {name}
                total += weight * (scores[with_name] - scores[without])
        shapley[name] = total

    result = pd.Series(shapley, name="shapley")
    result.attrs["n_excluded"] = len(data.ids) - len(comparable_ids)
    result.attrs["n_coalitions"] = 2**k
    return result
