"""The fold-local dataset: a row subset of a cohort, under one fitted preprocessor.

This is the only ``torch.utils.data.Dataset`` in the package. A cohort is an index
and is never a dataset; pairing it with a subset of rows and a preprocessor is what
produces something a ``DataLoader`` can iterate.

A view is cheap -- an index array and two references -- so building one per fold
costs nothing, and there is no cloning, no copy-on-fit, and nothing to keep in sync.
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

    Inherits ``torch.utils.data.Dataset`` so ``DataLoader``, ``Subset`` and
    ``ConcatDataset`` accept it without adapters.

    Args:
        cohort (Cohort): The cohort to read from. Held by reference and never
            mutated -- several folds' views share one cohort safely.
        indices (Sequence[int] | np.ndarray): Positional indices into
            ``cohort.identifiers``.
        preprocessor (Preprocessor | None): This fold's fitted state. ``None`` when
            the cohort declared nothing to fit.

    Note:
        Prefer :meth:`Cohort.view` to constructing this directly.
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

        # Caches only when the cohort volunteers an eager path. A cohort whose
        # payload is stochastic -- a slide cohort resampling tiles each epoch --
        # never does, so a view over one can never freeze a single draw for the
        # whole run. Opting in is the cohort's decision, not the view's.
        self._cache = cohort.payload_bulk(self.identifiers, preprocessor)
        if self._cache is not None:
            self._check_cache_alignment()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> PatientSample:
        """Return one training item.

        Args:
            i (int): Position within this view, not within the cohort.

        Returns:
            PatientSample: Features, availability flags, and the target's values.
        """
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

        Passed through unchanged: the preprocessor already keys them by modality
        (see :class:`~kalecancer.loaddata.protocols.Preprocessor`), so there is
        nothing here to reshape and nothing to guess. A view over a cohort with no
        fitted state has no feature names to report.
        """
        if self.preprocessor is None:
            return {}
        return {name: list(values) for name, values in self.preprocessor.feature_names.items()}

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _check_cache_alignment(self) -> None:
        """Verify an eager cache lines up, row for row, with this view's patients.

        :meth:`~kalecancer.loaddata.base.Cohort.payload_bulk` returns a block that
        is read positionally, while each patient's identifier comes from
        ``self.indices``. A cohort returning rows in any other order -- or returning
        the whole cohort when asked for a subset -- would therefore pair every
        patient with someone else's features, train perfectly, and mean nothing.
        It is the same failure that keying by identifier exists to prevent, arriving
        through the one door that has to be positional.

        Checked once per view, which is once per fold: the row count always, and the
        first row against :meth:`~kalecancer.loaddata.base.Cohort.payload`. The spot
        check is valid precisely because implementing ``payload_bulk`` asserts that
        ``payload`` is deterministic, so the two paths must agree.
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

    ``torch.equal`` reports any tensor containing NaN as unequal to itself, and NaN
    is reachable in a feature matrix -- ``continuous_transform=StandardScaler()``
    with no imputer propagates it -- so a plain comparison would raise on data that
    is perfectly aligned.
    """
    if left.shape != right.shape:
        return False
    return bool(((left == right) | (left.isnan() & right.isnan())).all())
