"""Loading data, by modality, plus the splitting that keeps it honest.

Four access APIs, none of which knows a dataset:

* :mod:`~kalecancer.loaddata.tabular` -- covariates from a flat table, with the
  fitted-per-fold discipline a scaler needs.
* :mod:`~kalecancer.loaddata.wsi_access` -- whole-slide patch features from HDF5 bags.
* :mod:`~kalecancer.loaddata.multimodal_access` -- any combination of the above, or of
  anything else, joined by patient identifier.
* :mod:`~kalecancer.loaddata.archive_access` -- fetching a published dataset from remote
  ZIP archives without downloading them whole.

and :mod:`~kalecancer.loaddata.splitting`, whose splitter objects follow
scikit-learn's shape.

The tabular API layers three objects with three lifetimes, which is what keeps
fold-local statistics out of shared state:

* a :class:`~kalecancer.loaddata.tabular_access.Cohort` is an index, built once and read by
  every fold, never mutated;
* a :class:`~kalecancer.loaddata.multimodal_access.Preprocessor` is fitted state belonging
  to exactly one fold;
* a :class:`~kalecancer.loaddata.tabular_access.CohortView` pairs a row subset with one
  preprocessor, and is the ``torch.utils.data.Dataset`` a loader iterates.

**Dataset-specific preparation does not live here.** Which columns define an
endpoint, how a slide filename encodes its patient, which patients a published split
assigns where -- all of that belongs to the dataset that defines it, in
``examples/``. This package supplies the mechanisms those decisions are expressed
with.

``CohortDataModule`` and ``release_workers`` are deliberately *not* imported
eagerly: they pull in Lightning, which costs about two seconds on top of torch, and
loading data should not have to pay for the training stack. Importing either by name
from this package works and pulls it in on demand.
"""

from kalecancer.loaddata.archive_access import ArchiveDataset, DatasetAccessError, RemoteArchive
from kalecancer.loaddata.multimodal_access import (
    ColumnTarget,
    FeatureBagSource,
    ModalitySource,
    MultimodalDataset,
    PatientBatch,
    PatientSample,
    SurvivalTarget,
    Target,
    VectorSource,
    collate_ragged,
    collate_samples,
    release_workers,
)
from kalecancer.loaddata.splitting import (
    SPLIT_NAMES,
    CohortSplitter,
    CrossValidation,
    HoldOut,
    Predefined,
    Split,
    SplitError,
    composite_labels,
    train_test_split,
)
from kalecancer.loaddata.tabular_access import Cohort, CohortView, LeakageError, NotFittedError, TabularCohort
from kalecancer.loaddata.wsi_access import (
    InvalidFeatureFileError,
    SlideIdentifierError,
    inspect_feature_bag,
    parse_patient_id,
    read_feature_bag,
    slide_table,
)

__all__ = [
    "ArchiveDataset",
    "Cohort",
    "CohortSplitter",
    "CohortView",
    "ColumnTarget",
    "CrossValidation",
    "DatasetAccessError",
    "FeatureBagSource",
    "HoldOut",
    "InvalidFeatureFileError",
    "LeakageError",
    "ModalitySource",
    "MultimodalDataset",
    "NotFittedError",
    "PatientBatch",
    "PatientSample",
    "Predefined",
    "RemoteArchive",
    "SPLIT_NAMES",
    "SlideIdentifierError",
    "Split",
    "SplitError",
    "SurvivalTarget",
    "TabularCohort",
    "Target",
    "VectorSource",
    "collate_ragged",
    "collate_samples",
    "composite_labels",
    "inspect_feature_bag",
    "parse_patient_id",
    "read_feature_bag",
    "release_workers",
    "slide_table",
    "train_test_split",
]


def __dir__():
    return sorted(__all__)
