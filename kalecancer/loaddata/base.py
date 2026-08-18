"""The cohort: an index over samples, built once and read by every fold.

Loading happens in two distinct phases, and keeping them apart is what lets one
base class serve a 400-row clinical table and a cohort of gigapixel slides:

1. **Index** -- read once at construction by :meth:`Cohort._load_index`. It
   populates ``self.identifiers`` and nothing else. This is cheap for every
   modality: parsing a CSV for tabular data, reading a tile manifest for WSI. It
   must never touch payload.
2. **Payload** -- read per sample by :meth:`Cohort.payload`. This is where a
   slide is opened and a tile bag is actually pulled off disk.

Fitted state sits between the two, and belongs to neither. A cohort is never
fitted: :meth:`Cohort.fit_preprocessor` returns a *separate*
:class:`~kalecancer.loaddata.protocols.Preprocessor` artifact scoped to the rows
it was given, and :meth:`Cohort.view` pairs a row subset with one of those
artifacts to produce a :class:`~kalecancer.loaddata.view.CohortView` -- the only
``torch.utils.data.Dataset`` in this package.

That three-way split is the design. The cohort is shared and immutable; the
preprocessor and the view live for exactly one fold and may be mutated freely,
because nothing outside that fold can see them. You cannot build a view without
naming a preprocessor, so "which statistics is this fold using" is never implicit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor

from kalecancer.loaddata.protocols import Preprocessor, Target, check_target
from kalecancer.loaddata.view import CohortView

#: Positional indices into ``Cohort.identifiers``. A numpy array is as natural a
#: caller here as a list -- ``split()`` returns arrays and sklearn's splitters
#: yield them -- so the alias admits both rather than forcing conversions at every
#: call site.
Indices = Sequence[int] | np.ndarray


class NotFittedError(RuntimeError):
    """Raised when transformed values are requested before transforms are fitted."""


class LeakageError(RuntimeError):
    """Raised when a preprocessor's fitted rows overlap rows it is applied to as held-out data.

    This is not a corner case caught defensively. It is the single most likely way
    to produce a result that looks good, runs clean, and is wrong.
    """


class Cohort(ABC):
    """An identifier-keyed index over samples. Built once, read by every fold.

    ``self.identifiers`` is the single ordering authority. A composite cohort must
    reach into its components by identifier and never by positional index --
    otherwise two components can silently disagree about which sample row ``5``
    refers to, which trains perfectly and means nothing.

    Args:
        path (str | Path | None, optional): Source to read the index from. ``None``
            for composite cohorts, which take their index from their components,
            and for subclasses given an already-loaded object instead of a file.
        name (str, optional): Key under which this cohort's features appear in
            ``PatientSample.modalities``. Defaults to ``"features"``.
        target (Target | None, optional): Supervision target. ``None`` for a pure
            feature provider, such as a slide cohort that carries no labels of its
            own. See :class:`~kalecancer.loaddata.protocols.Target`.

    Raises:
        TypeError: If ``target`` does not satisfy the ``Target`` contract.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        name: str = "features",
        target: Target | None = None,
    ):
        if target is not None:
            check_target(target)
        self.path: Path | None = Path(path) if path is not None else None
        self.name = name
        self.target = target
        self.identifiers: list[str] = []
        self._row_of: dict[str, int] = {}

        if self._has_index_source():
            self._load_index()
            self._reindex()

    # ------------------------------------------------------------------ #
    # subclass contract -- three methods
    # ------------------------------------------------------------------ #

    def _has_index_source(self) -> bool:
        """Whether there is anything for :meth:`_load_index` to read at construction.

        A path is the usual source but not the only one: a subclass may accept an
        already-loaded table, and a composite takes its index from its components.
        Override wherever ``self.path`` is not the whole story.
        """
        return self.path is not None

    @abstractmethod
    def _load_index(self) -> None:
        """Read the index and populate ``self.identifiers``.

        Called once, at construction. Must not read payload: for a slide cohort
        this reads a tile manifest, it does not open any images.
        """

    @abstractmethod
    def fit_preprocessor(self, indices: Indices) -> Preprocessor | None:
        """Fit transforms on the given rows only, and return them as a new artifact.

        The cohort is not modified. Fitting is scoped to ``indices``, so a fold is
        fitted by passing that fold's training indices and can never see statistics
        computed outside it.

        Args:
            indices (Indices): Positional indices into ``self.identifiers``.

        Returns:
            Preprocessor | None: The fitted artifact, or ``None`` when this cohort
            declares nothing to fit. ``None`` is a legitimate answer, not a failure
            -- a view built with it serves untransformed values.
        """

    @abstractmethod
    def payload(self, identifier: str, prep: Preprocessor | None) -> dict[str, Tensor]:
        """Return this cohort's contribution to ``PatientSample.modalities``.

        Lazy by contract. This is where a slide is opened, a memmap is read, or a
        tile bag is sampled, and it runs inside a DataLoader worker. It **may be
        stochastic** -- a slide cohort resamples tiles every epoch -- so nothing
        may assume two calls with the same arguments return the same tensor.

        Args:
            identifier (str): Sample identifier, as found in ``self.identifiers``.
            prep (Preprocessor | None): This fold's fitted state.

        Returns:
            dict[str, Tensor]: Usually ``{self.name: tensor}``. Shapes are
            modality-specific: ``(d,)`` for a tabular row, ``(n_tiles, d)`` for a
            slide's feature bag.
        """

    # ------------------------------------------------------------------ #
    # optional hooks
    # ------------------------------------------------------------------ #

    def payload_bulk(self, identifiers: Sequence[str], prep: Preprocessor | None) -> dict[str, Tensor] | None:
        """Transform every sample in one call, for modalities cheap enough to allow it.

        Returns ``None`` by default, and :class:`~kalecancer.loaddata.view.CohortView`
        falls back to per-sample :meth:`payload`.

        ``TabularCohort`` overrides this because calling scikit-learn's ``transform``
        once per row would dominate the training loop for a table small enough to
        transform whole. **A slide cohort must not override it**: its payload is
        stochastic, so caching would silently freeze one tile draw for an entire
        run, and its bags do not fit in memory anyway.

        Lazy is the contract; this is the exception, and opting in is how a view
        learns it is allowed to cache.

        **Rows must come back in the order of ``identifiers``.** A view reads the
        block positionally while taking each patient's identifier from its own
        index, so a cohort returning rows in any other order pairs every patient
        with someone else's features -- which trains perfectly and means nothing.
        :class:`~kalecancer.loaddata.view.CohortView` checks the row count and spot
        checks the first row against :meth:`payload`, but the contract is yours to
        keep.

        Implementing this also asserts that :meth:`payload` is **deterministic** for
        this cohort: the two paths must agree, and a view will hold the cached block
        for its whole lifetime. An empty ``identifiers`` must return empty tensors
        rather than raising -- an empty fold is a legitimate thing to build.
        """
        return None

    def present(self, identifier: str) -> dict[str, Tensor]:
        """Whether each of this cohort's modalities is available for one sample.

        A single-modality cohort holds data for every identifier it indexed, so the
        default is unconditionally ``True``. Composite cohorts override this: they
        may carry a patient who has a clinical record but no usable slide.
        """
        return {self.name: torch.tensor(True)}

    # ------------------------------------------------------------------ #
    # shared
    # ------------------------------------------------------------------ #

    def view(self, indices: Indices, preprocessor: Preprocessor | None) -> CohortView:
        """Pair a row subset with a fitted preprocessor to make a torch Dataset.

        Args:
            indices (Indices): Positional indices into ``self.identifiers``.
            preprocessor (Preprocessor | None): This fold's fitted state, from
                :meth:`fit_preprocessor`. Required, even when it is ``None``:
                passing it explicitly is what keeps a fold's provenance visible at
                the call site rather than buried in object state.

        Returns:
            CohortView: A dataset over those rows under that preprocessor.
        """
        return CohortView(self, indices, preprocessor)

    def split(
        self,
        test_size: float = 0.2,
        random_state: int | None = None,
        stratify: bool | np.ndarray = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split into two sets of **indices**, stratified where possible.

        Returns indices rather than cohorts so that it composes with scikit-learn's
        splitters and so that a cohort is never something you have to re-split.

        Stratifying matters at clinical cohort sizes: an unstratified 20% split of a
        few hundred patients can easily land with a badly skewed event rate.

        Args:
            test_size (float, optional): Proportion held out. Defaults to 0.2.
            random_state (int | None, optional): Seed.
            stratify (bool | np.ndarray, optional): ``True`` asks the target for
                labels to stratify on, ``False`` disables it, or pass an array to
                stratify on something of your own choosing. Defaults to ``True``.

        Returns:
            tuple[np.ndarray, np.ndarray]: Train and test indices, each sorted.

        Raises:
            TypeError: If ``stratify=True`` but the target cannot supply labels.
        """
        labels = self._stratify_labels(stratify)
        train_idx, test_idx = train_test_split(
            np.arange(len(self)), test_size=test_size, random_state=random_state, stratify=labels
        )
        return np.sort(train_idx), np.sort(test_idx)

    def _stratify_labels(self, stratify: bool | np.ndarray) -> np.ndarray | None:
        """Resolve the ``stratify`` argument of :meth:`split` to labels or ``None``."""
        if stratify is False:
            return None
        if not isinstance(stratify, bool):
            return np.asarray(stratify)
        if self.target is None:
            return None

        # Asked for by capability rather than required of every Target. What is worth
        # stratifying on is task-specific -- the event indicator for survival, the
        # class for classification -- and there is no sensible universal answer, so
        # this stays an optional extension rather than bloating the contract.
        labels_for = getattr(self.target, "stratify_labels", None)
        if labels_for is None:
            raise TypeError(
                f"stratify=True needs a target providing stratify_labels(identifiers), "
                f"which {type(self.target).__name__} does not. Pass stratify=False, or "
                f"pass an array to stratify on directly."
            )
        return np.asarray(labels_for(self.identifiers))

    def index_of(self, identifiers: Sequence[str]) -> np.ndarray:
        """Positional indices for ``identifiers``. The identifier-to-position bridge."""
        return np.array([self._row_of[i] for i in identifiers], dtype=int)

    def __len__(self) -> int:
        return len(self.identifiers)

    def __contains__(self, identifier: str) -> bool:
        return identifier in self._row_of

    def _reindex(self) -> None:
        """Rebuild the identifier-to-row lookup. Call after changing identifiers."""
        self._row_of = {identifier: i for i, identifier in enumerate(self.identifiers)}

    def __repr__(self) -> str:
        parts = [f"{len(self)} samples", f"name={self.name!r}"]
        if self.target is not None:
            summarise = getattr(self.target, "summarise", None)
            parts.append(summarise(self.identifiers) if summarise else type(self.target).__name__)
        return f"{type(self).__name__}({' | '.join(parts)})"
