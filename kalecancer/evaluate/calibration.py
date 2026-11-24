"""Calibration: whether a predicted survival probability of p means p of such patients actually survive.

Ranking (``HarrellC``, ``UnoC``), discrimination (``TimeDependentAUC``), separation (``LogRank``) and the combined
``IntegratedBrier`` score are all elsewhere in :mod:`kalecancer.evaluate`; none of them isolate calibration alone.
``observed`` below comes from a Kaplan-Meier estimate within each bin (reusing :mod:`kalecancer.evaluate.stratify`'s
machinery), not a naive event rate -- with a heavily censored cohort, an event rate would make every model look
badly calibrated for reasons that have nothing to do with the model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from kalecancer.evaluate.metrics import Metric, _defined, _event_time, _outcomes, _quantile_groups
from kalecancer.evaluate.stratify import _km_survival_at
from kalecancer.loaddata.targets import BaseTarget, TimeToEvent
from kalecancer.model.outputs import BaseOutput, SurvivalFunction


def _check_times(times: Sequence[float], what: str) -> list[float]:
    """``times`` as floats, after checking there is at least one and they strictly increase."""
    checked = [float(t) for t in times]
    if not checked or any(b <= a for a, b in zip(checked, checked[1:], strict=False)):
        raise ValueError(f"{what} needs at least one time, strictly increasing, got {times!r}")
    return checked


def calibration_table(
    output: BaseOutput,
    target: BaseTarget,
    *,
    times: Sequence[float],
    n_bins: int = 5,
) -> pd.DataFrame:
    """Observed vs predicted survival per bin, at each of ``times``.

    At each horizon, patients are binned separately into ``n_bins`` equal-sized quantile bins by their predicted
    survival at that horizon -- a patient's bin can differ horizon to horizon, since calibration asks "do patients
    predicted to survive to p at time t actually do so", not "is this patient always in the same bucket".

    Args:
        output: An output with a survival function, such as a Cox model's ``CoxOutput``, as
            ``Pipeline.predict_output`` returns it. Every patient must be defined.
        target: A ``TimeToEvent`` target with the outcomes of at least the output's patients.
        times: The horizons, strictly increasing.
        n_bins: Number of equal-sized predicted-survival bins, per horizon.

    Returns:
        One row per (time, bin), columns ``time``, ``bin`` (0 = lowest predicted survival through ``n_bins - 1`` =
        highest), ``n`` (patients in the bin), ``expected`` (mean predicted survival in the bin), ``observed`` (the
        bin's Kaplan-Meier survival estimate at that horizon, which handles censoring correctly where a naive event
        rate would not). ``observed`` is ``NaN`` -- not silently dropped -- for a bin where nobody is still at risk
        at the horizon: its last observation was a censoring before the horizon, so the curve has nothing to say
        past where its data ends.

    Raises:
        TypeError: If ``output`` has no survival function, or ``target`` is not a ``TimeToEvent``.
        ValueError: If ``times`` is empty or not increasing, if a patient is undefined or has no outcome, or if a
            horizon's predicted-survival values are too heavily tied to cut into ``n_bins`` distinct, non-empty bins.
    """
    horizons = _check_times(times, "calibration_table")
    if not isinstance(output, SurvivalFunction):
        raise TypeError(f"calibration_table needs an output with a survival function, got {type(output).__name__}")
    output = output.detach_cpu()
    event, time = _event_time(_outcomes(output, target, TimeToEvent, "calibration_table"))
    event_arr = event.numpy()
    time_arr = time.numpy().astype(np.float64)
    assert isinstance(output, SurvivalFunction)
    survival = output.survival(horizons).double().numpy()

    rows = []
    for column, horizon in enumerate(horizons):
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

    Unlike ``IntegratedBrier``, a single time is enough: ``IntegratedBrier`` needs at least two times because it
    integrates a Brier score *across* them, which needs an interval to integrate over. There is no such integration
    here -- "do patients predicted to survive to p at time t actually do so" is a complete question for one horizon
    alone, answered by averaging over that horizon's bins.

    It also ignores ``train_target``, which ``IntegratedBrier`` reads for its censoring distribution: ``observed`` is
    a plain (non-IPCW) Kaplan-Meier estimate within each bin of the evaluated patients themselves, not a
    training-fitted censoring weight, so there is no censoring survival G(t) whose positivity needs checking.

    Args:
        times: The horizons, strictly increasing.
        n_bins: Number of equal-sized predicted-survival bins, per horizon.
    """

    higher_is_better = False
    requires = SurvivalFunction

    def __init__(self, times: Sequence[float], n_bins: int = 5):
        _check_times(times, "CalibrationError")
        self.times = list(times)
        self.n_bins = n_bins

    def score(self, output: BaseOutput, target: BaseTarget, train_target: BaseTarget) -> float:
        """The mean absolute difference between observed and expected survival, ignoring undefined bins."""
        table = calibration_table(output, target, times=self.times, n_bins=self.n_bins)
        mean_absolute_error = (table["observed"] - table["expected"]).abs().mean()
        return _defined(
            float(mean_absolute_error), f"{self!r} is undefined: every bin's observed survival is undefined"
        )
