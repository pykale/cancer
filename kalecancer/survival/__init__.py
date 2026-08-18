"""Time-to-event analysis: targets, Cox head, losses, and metrics.

Destined for PyKale core, so it must not import anything cancer-specific and does not
import ``kalecancer.loaddata`` at all -- ``SurvivalTarget`` satisfies the ``Target``
protocol structurally, so it travels without the data layer.
"""

from kalecancer.survival.survival_target import SurvivalTarget

__all__ = ["SurvivalTarget"]
