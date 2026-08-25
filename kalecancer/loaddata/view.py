"""The fold-local dataset: a named subset of a cohort, under one fitted preprocessor.

The only ``torch.utils.data.Dataset`` in the package. A cohort is an index, not a
dataset; pairing it with rows and a preprocessor is what a ``DataLoader`` can iterate.

Cheap -- a list of identifiers and two references -- so one per fold costs nothing,
with no cloning and nothing to keep in sync.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from kalecancer.loaddata.sample import PatientBatch, PatientSample, collate_samples

if TYPE_CHECKING:  # import only for type checking -- avoids a cycle with base.py
    from kalecancer.loaddata.base import Cohort, Identifiers
    from kalecancer.loaddata.protocols import Preprocessor


class CohortView(TorchDataset):
    """A torch Dataset over some of a cohort's rows, under one preprocessor.

    Prefer :meth:`Cohort.view` to constructing this directly.

    Args:
        cohort (Cohort): Held by reference and never mutated, so several folds' views
            share one cohort safely.
        identifiers (Identifiers): Which samples, named. Validated against the cohort.
        preprocessor (Preprocessor | None): This fold's fitted state.
    """

    def __init__(
        self,
        cohort: Cohort,
        identifiers: Identifiers,
        preprocessor: Preprocessor | None,
    ):
        self.cohort = cohort
        self.identifiers = cohort.check_identifiers(identifiers)
        self.preprocessor = preprocessor

        # Caching is the cohort's decision, not the view's: one with a stochastic
        # payload never offers a bulk path, so a tile draw can never be frozen here.
        self._cache = cohort.payload_bulk(self.identifiers, preprocessor)
        if self._cache is not None:
            self._check_cache_alignment()

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, i: int) -> PatientSample:
        """Return one training item.

        ``i`` is a position within this view -- torch's ``Dataset`` contract, and the
        only place in the API where a sample is reached for by number.
        """
        identifier = self.identifiers[i]
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

    def batch(self) -> PatientBatch:
        """Every sample in this view, collated into one :class:`PatientBatch`.

        The bulk counterpart to iterating: features per modality, the target, the
        identifiers and the padding masks, all in one object -- the same one a
        ``DataLoader`` produces, so code written against a batch works either way.

        For fitting an embedder's context, for a full-batch loss, or for anything
        scikit-learn shaped. **Materialises the whole view**, so it suits a clinical
        table and not a cohort of slides.

        A cohort offering a bulk path already holds the block this would rebuild, so
        that case is served from it rather than sliced apart and stacked back together.
        The two routes must agree; a test pins them field for field.

        Raises:
            ValueError: If the view is empty.
        """
        if self._cache is None or not self.identifiers:
            return collate_samples([self[i] for i in range(len(self))])

        target = self.cohort.target
        return PatientBatch(
            patient_id=list(self.identifiers),
            # Cloned, not handed out: the block is read again by every __getitem__,
            # so an in-place op on the batch would rewrite this view's own features.
            modalities={name: block.clone() for name, block in self._cache.items()},
            present=self._present_bulk(),
            # No pad_mask: a bulk block arrives as one stacked tensor, so its rows are
            # fixed-width by construction and there is nothing ragged to mask.
            target={} if target is None else target.values_for(self.identifiers),
        )

    def _present_bulk(self) -> dict[str, Tensor]:
        """Availability per modality, stacked over this view's samples."""
        flags = [self.cohort.present(identifier) for identifier in self.identifiers]
        return {name: torch.stack([flag[name] for flag in flags]) for name in flags[0]}

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

        The cache is read positionally while the rows were asked for by name, so a
        cohort returning them in another order would pair patients with the wrong
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
