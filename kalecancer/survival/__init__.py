"""Time-to-event analysis: targets, Cox head, losses, and metrics.

This submodule is destined to be refactored into PyKale core later. It MUST NOT
import anything cancer-specific, and it deliberately does not import
``kalecancer.loaddata`` at all -- ``SurvivalTarget`` satisfies the ``Target``
protocol structurally, so it can travel without the data layer.
"""

from kalecancer.survival.survival_target import SurvivalTarget

__all__ = ["SurvivalTarget"]
