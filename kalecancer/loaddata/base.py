"""Abstract base for datasets keyed by a sample identifier.

Loading happens in two distinct phases, and keeping them apart is what lets the
same base class serve a 400-row clinical table and a cohort of gigapixel slides:

1. **Index** -- read once at construction by :meth:`_load_index`. It populates
   ``self.identifiers`` and nothing else. This is cheap for every modality:
   parsing a CSV for tabular data, listing a feature cache or reading a manifest
   for WSI. It must never touch payload.
2. **Payload** -- read per sample by :meth:`get_by_id`. This is where a slide's
   feature bag is actually pulled off disk.

Stateful preprocessing sits between the two. :meth:`fit_on` fits transforms on a
set of indices and returns a *new* instance; the original is never mutated, so
cross-validation folds cannot share fitted state or contaminate one another.

Subclasses implement :meth:`_load_index` and :meth:`get_by_id`. Those with
fitted state (scalers, imputers, in-context example sets) also override
:meth:`fit_on`, :meth:`transform` and :attr:`is_fitted`.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

if TYPE_CHECKING:  # import only for type checking
    import pandas as pd


class NotFittedError(RuntimeError):
    """Raised when transformed values are requested before transforms are fitted."""


@runtime_checkable
class Target(Protocol):
    """What a dataset requires of a supervision target.

    Declared here, in the data layer, so that concrete targets can live wherever
    they belong -- ``SurvivalTarget`` alongside the survival heads, losses and
    metrics that consume it -- without the data layer having to import from the
    modelling layer. The dependency arrow points one way: implementations import
    this protocol, never the reverse.

    A target is bound **once**, at load time, and is keyed by identifier rather
    than row position. That is what lets a dataset be split, subset and
    recombined without the target ever needing realignment.

    The corollary is that anything derived from a *training fold* is not a target
    and must not be attached to one. Discrete-time survival bin edges are the case
    to watch: they are quantiles of the uncensored event times of one fold, so they
    are fitted state belonging in ``prepdata`` under the same clone-per-fold
    discipline as a scaler. Putting them here leaks the held-out fold's time
    distribution, silently, because it never looks like preprocessing.

    Note:
        These two methods are what :class:`BaseDataset` itself calls. Concrete
        datasets may require more of their targets -- ``TabularDataset.split`` uses
        ``events_for`` to stratify and ``__repr__`` uses ``summarise`` -- but those
        are not universal to every kind of target, so they are deliberately not
        part of this contract.
    """

    def bind(self, frame: pd.DataFrame, identifier: str) -> None:
        """Extract and validate supervision values from ``frame``, keyed by identifier.

        Called once during index loading. Implementations should validate here and
        raise rather than defer a mis-specified target to training time.

        Args:
            frame (pd.DataFrame): Table containing the target columns.
            identifier (str): Column holding sample identifiers.
        """
        ...

    def for_(self, identifier) -> dict:
        """Return this sample's supervision values, to be merged into the item dict.

        Args:
            identifier: Sample identifier.

        Returns:
            dict: Named tensors, e.g. ``{"time": Tensor, "event": Tensor}``. Keys must
            not collide with a modality name or with ``"patient_id"`` / ``"mask"``.
        """
        ...


class BaseDataset(TorchDataset, ABC):
    """Base class for identifier-keyed datasets.

    Inherits :class:`torch.utils.data.Dataset`, so ``DataLoader``, ``Subset``,
    ``random_split`` and ``ConcatDataset`` work without adapters.

    ``self.identifiers`` is the single ordering authority. Composite datasets
    must reach into their components via :meth:`get_by_id` and never by
    positional index -- otherwise two components can silently disagree about
    which sample row ``5`` refers to.

    Args:
        path (str | Path | None, optional): Source to read the index from. ``None``
            for composite datasets, which take their index from their components.
        name (str, optional): Key under which this dataset's features appear in the
            item dict returned by :meth:`__getitem__`. Defaults to ``"features"``.
        target (Target | None, optional): Supervision target, e.g. a ``SurvivalTarget``.
            See :class:`Target` for the contract. ``None`` for pure feature providers
            such as a slide dataset, which carry features but no supervision.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        name: str = "features",
        target: Target | None = None,
    ):
        self.path: Path | None = Path(path) if path is not None else None
        self.name = name
        self.target = target
        self.identifiers: list = []
        self._row_of: dict = {}

        if self._has_index_source():
            self._load_index()
            self._reindex()

    # ------------------------------------------------------------------ #
    # subclass contract
    # ------------------------------------------------------------------ #

    def _has_index_source(self) -> bool:
        """Whether there is anything for :meth:`_load_index` to read at construction.

        A path is the usual source but not the only one: a subclass may accept an
        already-loaded table instead of a file, and a composite takes its index from
        its components and has no source of its own. Override wherever ``self.path``
        is not the whole story.
        """
        return self.path is not None

    @abstractmethod
    def _load_index(self) -> None:
        """Read the index from ``self.path`` and populate ``self.identifiers``.

        Called once, at construction. Must not read payload data: for a slide
        dataset this reads a manifest or lists a feature cache, it does not open
        any images.
        """

    @abstractmethod
    def get_by_id(self, identifier) -> Tensor:
        """Return the feature tensor for one sample.

        Shapes are modality-specific: ``(d,)`` for a tabular row, ``(n_tiles, d)``
        for a slide's feature bag.

        Args:
            identifier: Sample identifier, as found in ``self.identifiers``.

        Returns:
            Tensor: This sample's features.
        """

    # ------------------------------------------------------------------ #
    # fold discipline -- fit returns a new object, never mutates in place
    # ------------------------------------------------------------------ #

    @property
    def is_fitted(self) -> bool:
        """Whether this dataset is ready to serve transformed values.

        ``True`` on the base class, which holds no fitted state. Subclasses with
        stateful transforms override this.
        """
        return True

    def subset(self, indices: Sequence[int]) -> BaseDataset:
        """Return a new dataset restricted to ``indices``, carrying any fitted state.

        Args:
            indices (Sequence[int]): Positional indices into ``self.identifiers``.

        Returns:
            BaseDataset: A new instance. ``self`` is unchanged.
        """
        clone = copy.copy(self)
        clone.identifiers = [self.identifiers[i] for i in indices]
        clone._reindex()
        return clone

    def fit_on(self, indices: Sequence[int]) -> BaseDataset:
        """Fit stateful transforms on ``indices`` and return a new dataset over them.

        The base implementation has nothing to fit and just subsets. Override in
        subclasses that hold scalers, imputers, or in-context example sets.

        Args:
            indices (Sequence[int]): Positional indices of the training rows.

        Returns:
            BaseDataset: A new, fitted instance restricted to ``indices``.
        """
        return self.subset(indices)

    def transform(self, other: BaseDataset) -> BaseDataset:
        """Apply this dataset's fitted transforms to another dataset's samples.

        This is how held-out data is prepared: fit on the training fold, then
        transform the validation fold or the test split with those statistics.

        Args:
            other (BaseDataset): Dataset whose samples should be transformed.

        Returns:
            BaseDataset: A new instance over ``other``'s samples, using ``self``'s
            fitted state.

        Raises:
            NotFittedError: If ``self`` has unfitted transforms.
        """
        if not self.is_fitted:
            raise NotFittedError(f"{type(self).__name__} has unfitted transforms; call fit_on() first.")
        return other

    @property
    def raw(self) -> BaseDataset:
        """A view of this dataset with all transforms disabled.

        For exploration and sanity checks. The base class has no transforms, so
        this returns ``self``.
        """
        return self

    # ------------------------------------------------------------------ #
    # torch.utils.data.Dataset interface
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.identifiers)

    def __getitem__(self, idx: int) -> dict:
        """Return one training item.

        Args:
            idx (int): Positional index into ``self.identifiers``.

        Returns:
            dict: ``{self.name: Tensor, "patient_id": str}``, plus whatever the
            target contributes (e.g. ``"time"`` and ``"event"``). Dicts of tensors
            are handled by PyTorch's ``default_collate`` without a custom
            ``collate_fn``.
        """
        identifier = self.identifiers[idx]
        item = {self.name: self.get_by_id(identifier), "patient_id": identifier}
        if self.target is not None:
            item.update(self.target.for_(identifier))
        return item

    def __contains__(self, identifier) -> bool:
        return identifier in self._row_of

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _reindex(self) -> None:
        """Rebuild the identifier to local-row lookup. Call after changing identifiers."""
        self._row_of = {identifier: i for i, identifier in enumerate(self.identifiers)}
