"""The cohort: an index over samples, built once and read by every fold.

Loading is two phases, which is what lets one base class serve a 400-row clinical
table and a cohort of gigapixel slides: :meth:`Cohort._load_index` runs once and
populates ``identifiers`` only; :meth:`Cohort.payload` reads one sample on demand.

Fitted state belongs to neither. A cohort is never fitted --
:meth:`Cohort.fit_preprocessor` returns a separate artifact scoped to the rows it
was given, and :meth:`Cohort.view` pairs a row subset with one to make a
:class:`~kalecancer.loaddata.view.CohortView`, the only ``Dataset`` in the package.

The cohort is shared and never mutated; preprocessors and views live for one fold.
You cannot build a view without naming a preprocessor, so which statistics a fold
uses is never implicit.
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
    """Raised when a preprocessor's fitted rows overlap rows it is applied to as held out."""


class Cohort(ABC):
    """An identifier-keyed index over samples. Built once, read by every fold.

    ``self.identifiers`` is the single ordering authority. A composite must reach into
    its components by identifier, never by position, or two components can disagree
    about which sample row ``5`` is -- which trains perfectly and means nothing.

    Args:
        path (str | Path | None, optional): Source for the index. ``None`` for
            composites, and for subclasses given an already-loaded object.
        name (str, optional): Key for this cohort's features in
            ``PatientSample.modalities``. Defaults to ``"features"``.
        target (Target | None, optional): Supervision. ``None`` for a pure feature
            provider, such as a slide cohort carrying no labels of its own.

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

        Override wherever ``self.path`` is not the whole story -- a subclass given a
        loaded table, or a composite taking its index from its components.
        """
        return self.path is not None

    @abstractmethod
    def _load_index(self) -> None:
        """Read the index and populate ``self.identifiers``.

        Called once, at construction. Must not read payload -- for a slide cohort this
        reads a tile manifest and opens no images.
        """

    @abstractmethod
    def fit_preprocessor(self, indices: Indices) -> Preprocessor | None:
        """Fit transforms on the given rows only, and return them as a new artifact.

        The cohort is not modified, so a fold can never see statistics from outside it.

        Args:
            indices (Indices): Positional indices, normally one fold's training rows.

        Returns:
            Preprocessor | None: The fitted artifact, or ``None`` when this cohort has
            nothing to fit. ``None`` is a legitimate answer, not a failure.
        """

    @abstractmethod
    def payload(self, identifier: str, prep: Preprocessor | None) -> dict[str, Tensor]:
        """Return this cohort's contribution to ``PatientSample.modalities``.

        Lazy by contract, and runs inside a DataLoader worker. **May be stochastic** --
        a slide cohort resamples tiles every epoch -- so nothing may assume two calls
        return the same tensor.

        Returns:
            dict[str, Tensor]: Usually ``{self.name: tensor}``. Shapes are
            modality-specific: ``(d,)`` for a tabular row, ``(n_tiles, d)`` for a bag.
        """

    # ------------------------------------------------------------------ #
    # optional hooks
    # ------------------------------------------------------------------ #

    def payload_bulk(self, identifiers: Sequence[str], prep: Preprocessor | None) -> dict[str, Tensor] | None:
        """Transform every sample in one call, for modalities cheap enough to allow it.

        ``None`` by default, and a view falls back to per-sample :meth:`payload`.
        Overriding it is how a cohort opts into caching. ``TabularCohort`` does, because
        per-row scikit-learn calls would dominate the loop. **A slide cohort must not**:
        its payload is stochastic, so a cached block would freeze one tile draw for a
        whole run.

        Implementing this asserts three things. Rows come back in ``identifiers`` order
        -- a view reads the block positionally, so any other order pairs patients with
        the wrong features. :meth:`payload` is deterministic, since the two paths must
        agree. An empty ``identifiers`` returns empty tensors rather than raising.
        """
        return None

    def present(self, identifier: str) -> dict[str, Tensor]:
        """Whether each of this cohort's modalities is available for one sample.

        Unconditionally ``True`` here, since a single-modality cohort holds data for
        every identifier it indexed. Composites override it: a patient may have a
        clinical record but no usable slide.
        """
        return {self.name: torch.tensor(True)}

    # ------------------------------------------------------------------ #
    # shared
    # ------------------------------------------------------------------ #

    def view(self, indices: Indices, preprocessor: Preprocessor | None) -> CohortView:
        """Pair a row subset with a fitted preprocessor to make a torch Dataset.

        ``preprocessor`` is required even when ``None``: passing it explicitly keeps a
        fold's provenance at the call site rather than buried in object state.
        """
        return CohortView(self, indices, preprocessor)

    def split(
        self,
        test_size: float = 0.2,
        random_state: int | None = None,
        *,
        stratify: bool | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Split into two sets of **indices**, balanced on what you name.

        Indices rather than cohorts, so this composes with scikit-learn's splitters.

        Args:
            test_size (float, optional): Proportion held out. Defaults to 0.2.
            random_state (int | None, optional): Seed.
            stratify (bool | np.ndarray): **Required, no default.** ``True`` asks the
                target for labels, ``False`` disables it, or pass an array. Required
                because it changes the numbers you report and leaves no trace when
                wrong -- an unstratified 20% split of a few hundred patients can land
                several points off on the event rate.

        Returns:
            tuple[np.ndarray, np.ndarray]: Train and test indices, each sorted.

        Raises:
            TypeError: If ``stratify=True`` but no target can supply labels.
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
            raise TypeError(
                "stratify=True needs a target to take labels from, and this cohort has "
                "none. Pass stratify=False to split at random, or pass an array to "
                "balance on something of your own choosing."
            )

        # Optional extension, not part of the Target contract: what is worth
        # stratifying on is task-specific and has no universal answer.
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
