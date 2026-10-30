"""Calibration: whether a predicted survival probability of p means p of such patients actually survive.

Ranking (``HarrellC``, ``UnoC``), discrimination (``TimeDependentAUC``), separation (``LogRank``) and the combined
``IntegratedBrier`` score are all elsewhere in :mod:`kalecancer.evaluate`; none of them isolate calibration alone.
``observed`` below comes from a Kaplan-Meier estimate within each bin (reusing :mod:`kalecancer.evaluate.stratify`'s
machinery), not a naive event rate -- with a heavily censored cohort, an event rate would make every model look
badly calibrated for reasons that have nothing to do with the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kalecancer.evaluate.metrics import (
    _SURVIVAL_COLUMN,
    EvalContext,
    Metric,
    _aligned_values,
    _defined,
    _event_time,
    _quantile_groups,
)
from kalecancer.evaluate.stratify import _km_survival_at


def calibration_table(
    prediction: pd.DataFrame,
    target: pd.DataFrame,
    *,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Observed vs predicted survival per bin, at each ``survival[<time>]`` horizon.

    At each horizon, patients are binned separately into ``n_bins`` equal-sized quantile bins by their predicted
    survival at that horizon -- a patient's bin can differ horizon to horizon, since calibration asks "do patients
    predicted to survive to p at time t actually do so", not "is this patient always in the same bucket".

    Args:
        prediction: Predictions indexed by patient id, with one or more ``survival[<time>]`` columns, as
            ``Pipeline.predict`` and ``Pipeline.evaluate`` add when given ``survival_times``.
        target: The same patients' ``time``/``event`` columns.
        n_bins: Number of equal-sized predicted-survival bins, per horizon.

    Returns:
        One row per (time, bin), columns ``time``, ``bin`` (0 = lowest predicted survival through ``n_bins - 1`` =
        highest), ``n`` (patients in the bin), ``expected`` (mean predicted survival in the bin), ``observed`` (the
        bin's Kaplan-Meier survival estimate at that horizon, which handles censoring correctly where a naive event
        rate would not). ``observed`` is ``NaN`` -- not silently dropped -- for a bin where nobody is still at risk
        at the horizon: its last observation was a censoring before the horizon, so the curve has nothing to say
        past where its data ends.

    Raises:
        ValueError: If there is no ``survival[<time>]`` column, or if a horizon's predicted-survival values are too
            heavily tied to cut into ``n_bins`` distinct, non-empty bins.
    """
    columns = {float(m[1]): c for c in prediction.columns if (m := _SURVIVAL_COLUMN.fullmatch(str(c)))}
    if not columns:
        raise ValueError(
            f"calibration_table needs at least one 'survival[<time>]' column, got {list(prediction.columns)}; set "
            "survival_times in Pipeline.predict, evaluate or cross_validate"
        )
    times = sorted(columns)
    event, time = _event_time(target, "target")
    event_arr = event.numpy()
    time_arr = time.numpy().astype(np.float64)
    survival = _aligned_values(prediction, target, [columns[t] for t in times], np.float64)

    rows = []
    for column, horizon in enumerate(times):
        predicted = survival[:, column]
        try:
            codes = _quantile_groups(predicted, n_bins)
        except ValueError as error:
            raise ValueError(f"calibration_table at time={horizon}: {error}") from error
        for bin_id in range(n_bins):
            mask = codes == bin_id
            rows.append(
                {
                    "time": horizon,
                    "bin": bin_id,
                    "n": int(mask.sum()),
                    "expected": float(predicted[mask].mean()),
                    "observed": _km_survival_at(time_arr[mask], event_arr[mask], horizon),
                }
            )
    return pd.DataFrame(rows)


class CalibrationError(Metric):
    """Mean absolute calibration error, averaged over :func:`calibration_table`'s bins and horizons.

    ``mean(|observed - expected|)``, ignoring bins where ``observed`` is undefined.

    This is NOT the calibration slope. A slope's ideal value is 1.0, so neither ``higher_is_better = True`` nor
    ``False`` would be honest for it -- ``EarlyStopping``, which reads ``higher_is_better`` to pick a direction to
    improve, would chase the wrong end. This metric's ideal is 0 and larger is always worse, which is exactly what
    ``higher_is_better = False`` means. If the slope itself is wanted: it is one least-squares regression of
    ``observed`` on ``expected`` away from :func:`calibration_table`.

    This measures CALIBRATION only -- not ranking (``HarrellC``, ``UnoC``) or separation between risk groups
    (``LogRank``); a model can rank patients perfectly while being badly calibrated, or the reverse.

    Unlike ``IntegratedBrier``, a single ``survival[<time>]`` column is enough: ``IntegratedBrier`` needs at least
    two times because it integrates a Brier score *across* them, which needs an interval to integrate over. There is
    no such integration here -- "do patients predicted to survive to p at time t actually do so" is a complete
    question for one horizon alone, answered by averaging over that horizon's bins.

    This also does not need ``context.train_target``'s censoring distribution the way ``IntegratedBrier`` does:
    ``observed`` is a plain (non-IPCW) Kaplan-Meier estimate within each bin of the evaluated patients themselves,
    not a training-fitted censoring weight, so there is no censoring survival G(t) whose positivity needs checking.
    """

    higher_is_better = False

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins

    def __call__(self, prediction: pd.DataFrame, target: pd.DataFrame, context: EvalContext) -> float:
        """The mean absolute difference between observed and expected survival, ignoring undefined bins."""
        table = calibration_table(prediction, target, n_bins=self.n_bins)
        mean_absolute_error = (table["observed"] - table["expected"]).abs().mean()
        return _defined(
            float(mean_absolute_error), f"{self!r} is undefined: every bin's observed survival is undefined"
        )
