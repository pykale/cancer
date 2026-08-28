"""KaleCancer: cancer-domain machine learning for the PyKale ecosystem.

Organised as a verb-oriented pipeline (load → prep → model → evaluate → interpret),
with ``auto`` for high-level construction. Time-to-event support is not a stage of
its own: the head and loss are in ``model.predict``, the metrics in ``evaluate``, and
the supervision in ``loaddata``.
"""

from importlib import import_module
from pathlib import Path

__version__ = Path(__file__).with_name("_version.txt").read_text(encoding="utf-8").strip()

_SUBMODULES = frozenset(
    {
        "auto",
        "loaddata",
        "prepdata",
        "model",
        "pipeline",
        "evaluate",
        "interpret",
        "utils",
    }
)

__all__ = sorted(_SUBMODULES | {"__version__"})


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)
