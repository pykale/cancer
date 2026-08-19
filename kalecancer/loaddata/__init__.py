"""Datasets and records..

The layering here is three objects with three lifetimes:

* a :class:`~kalecancer.loaddata.base.Cohort` is an index, built once and read by
  every fold, never mutated;
* a :class:`~kalecancer.loaddata.protocols.Preprocessor` is fitted state belonging
  to exactly one fold;
* a :class:`~kalecancer.loaddata.view.CohortView` pairs a row subset with one
  preprocessor, and is the only ``torch.utils.data.Dataset`` in the package.

``CohortDataModule`` is deliberately *not* imported eagerly: it pulls in Lightning,
which costs about two seconds on top of torch, and loading data should not have to
pay for the training stack. ``from kalecancer.loaddata import CohortDataModule``
works and imports it on demand.
"""

from kalecancer.loaddata.base import Cohort, LeakageError, NotFittedError
from kalecancer.loaddata.protocols import Preprocessor, Target
from kalecancer.loaddata.sample import PatientBatch, PatientSample, collate_samples
from kalecancer.loaddata.tabular import TabularCohort
from kalecancer.loaddata.view import CohortView

__all__ = [
    "Cohort",
    "CohortDataModule",
    "CohortView",
    "LeakageError",
    "NotFittedError",
    "PatientBatch",
    "PatientSample",
    "Preprocessor",
    "TabularCohort",
    "Target",
    "collate_samples",
]


def __getattr__(name: str):
    if name == "CohortDataModule":
        from kalecancer.loaddata.module import CohortDataModule

        globals()[name] = CohortDataModule
        return CohortDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
