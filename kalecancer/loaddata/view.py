"""The fold-local dataset: a row subset of a cohort, under one fitted preprocessor.

The only ``torch.utils.data.Dataset`` in the package. A cohort is an index, not a
dataset; pairing it with rows and a preprocessor is what a ``DataLoader`` can iterate.

Cheap -- an index array and two references -- so one per fold costs nothing, with no
cloning and nothing to keep in sync.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from kalecancer.loaddata.sample import PatientSample

if TYPE_CHECKING:  # import only for type checking -- avoids a cycle with base.py
    from kalecancer.loaddata.base import Cohort
    from kalecancer.loaddata.protocols import Preprocessor


class CohortView(TorchDataset):
    """A torch Dataset over some of a cohort's rows, under one preprocessor.

    Prefer :meth:`Cohort.view` to constructing this directly.

    Args:
        cohort (Cohort): Held by reference and never mutated, so several folds' views
            share one cohort safely.
        indices (Sequence[int] | np.ndarray): Positional indices into
            ``cohort.identifiers``.
        preprocessor (Preprocessor | None): This fold's fitted state.
    """

    def __init__(
        self,
        cohort: Cohort,
        indices: Sequence[int] | np.ndarray,
        preprocessor: Preprocessor | None,
    ):
        self.cohort = cohort
        self.indices = np.asarray(indices, dtype=int)
        self.preprocessor = preprocessor

        # Caching is the cohort's decision, not the view's: one with a stochastic
        # payload never offers a bulk path, so a tile draw can never be frozen here.
        self._cache = cohort.payload_bulk(self.identifiers, preprocessor)
        if self._cache is not None:
            self._check_cache_alignment()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> PatientSample:
        """Return one training item. ``i`` is a position within this view, not the cohort."""
        identifier = self.cohort.identifiers[self.indices[i]]
        if self._cache is None:
            modalities = self.cohort.payload(identifier, self.preprocessor)
        else:
            modalities = {name: values[i] for name, values in self._cache.items()}
        return PatientSample(
            patient_id=identifier,
            modalities=modalities,
            present=self.cohort.present(identifier),
            target={} if self.cohort.target is None else self.cohort.target.for_(identifier),
        )

    @property
    def identifiers(self) -> list[str]:
        """Identifiers of this view's rows, in order."""
        return [self.cohort.identifiers[i] for i in self.indices]

    @property
    def feature_names(self) -> dict[str, list[str]]:
        """Post-encoding feature names per modality, from this fold's preprocessor.

        Passed through unchanged -- the preprocessor already keys them by modality, so
        there is nothing to reshape and nothing to guess.
        """
        if self.preprocessor is None:
            return {}
        return {name: list(values) for name, values in self.preprocessor.feature_names.items()}

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _check_cache_alignment(self) -> None:
        """Verify an eager cache lines up, row for row, with this view's patients.

        The cache is read positionally while identifiers come from ``self.indices``, so
        a cohort returning rows in another order would pair patients with the wrong
        features -- silently. This is the one door in the design that has to be
        positional, so it is the one that gets checked.

        Row count always, plus the first row against ``payload()``. The spot check is
        valid because implementing ``payload_bulk`` asserts ``payload`` is deterministic.
        """
        assert self._cache is not None  # only called when there is one
        identifiers = self.identifiers

        for name, values in self._cache.items():
            if len(values) != len(identifiers):
                raise ValueError(
                    f"{type(self.cohort).__name__}.payload_bulk returned {len(values)} rows "
                    f"for modality '{name}' but was asked for {len(identifiers)}. Rows are "
                    f"read positionally, so a mismatched block pairs patients with the wrong "
                    f"features. Return exactly the identifiers you were given, in order."
                )

        if not identifiers:
            return
        expected = self.cohort.payload(identifiers[0], self.preprocessor)
        for name, values in self._cache.items():
            if name in expected and not _rows_equal(values[0], expected[name]):
                raise ValueError(
                    f"{type(self.cohort).__name__}.payload_bulk disagrees with payload() on "
                    f"the first row of modality '{name}' (patient {identifiers[0]!r}). The "
                    f"bulk block must return the identifiers it was given, in that order, and "
                    f"must agree with the per-sample path."
                )

    def __repr__(self) -> str:
        status = "no transforms" if self.preprocessor is None else "fitted"
        return f"{type(self).__name__}({len(self)} of {len(self.cohort)} samples | {status})"


def _rows_equal(left: Tensor, right: Tensor) -> bool:
    """Exact equality, counting NaN as equal to NaN.

    ``torch.equal`` calls any NaN tensor unequal to itself, and NaN is reachable --
    ``StandardScaler`` with no imputer propagates it -- so a plain comparison would
    raise on perfectly aligned data.
    """
    if left.shape != right.shape:
        return False
    return bool(((left == right) | (left.isnan() & right.isnan())).all())
