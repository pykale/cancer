"""Lightning wiring: one fold's views, wrapped in DataLoaders.

Fold-local: it fits this fold's preprocessor, builds its views, and hands Lightning
the loaders. Nothing it touches outlives the fold, and the cohort it shares it only
reads -- which is what makes ``setup()`` assigning to ``self`` safe. Mutation is not
the hazard; mutation of shared state is.

Cross-validation is one of these per fold, alongside one ``Trainer`` and one
``LightningModule``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import lightning as L
import numpy as np
from torch.utils.data import DataLoader

from kalecancer.loaddata.base import Cohort, Indices, LeakageError
from kalecancer.loaddata.protocols import Preprocessor
from kalecancer.loaddata.sample import collate_samples
from kalecancer.loaddata.view import CohortView

#: Accepted values for ``batch_size``. ``"full"`` means the whole training split.
BatchSize = int | Literal["full"]


class CohortDataModule(L.LightningDataModule):
    """Wraps one fold's views in DataLoaders. Fits nothing that outlives the fold.

    Args:
        cohort (Cohort): Read, never mutated, so every fold may hold the same instance.
        train_idx (Indices): Training rows. The only rows fitted on.
        val_idx (Indices | None, optional): Validation rows.
        test_idx (Indices | None, optional): Test rows.
        batch_size (int | "full"): **Required, no default.** With a Cox head the partial
            likelihood is averaged within a batch, so ``64`` and ``"full"`` optimise
            measurably different objectives -- a modelling decision wearing a loader
            parameter's clothes. ``"full"`` means ``len(train_idx)``.
        num_workers (int, optional): ``0`` suits a clinical table, where worker startup
            costs more than it saves. Slide payloads want several. Defaults to 0.
        pin_memory (bool, optional): Defaults to ``False``.
        shuffle (bool, optional): Shuffles the *training* loader only. Defaults to ``True``.
        drop_last (bool, optional): Drop a short final training batch. Defaults to ``False``.
        preprocessor (Preprocessor | None, optional): Reuse an already-fitted one instead
            of fitting from ``train_idx`` -- for refitting a final model, or scoring a
            checkpoint against the transforms it was trained with. This is the path
            :meth:`check_no_leak` exists for.
        collate_fn (Callable, optional): Defaults to :func:`collate_samples`; torch's
            default collate cannot handle a ``PatientSample``.

    Raises:
        ValueError: If the splits overlap, or ``batch_size`` is invalid.

    Example:
        >>> train_idx, test_idx = cohort.split(test_size=0.2, random_state=0, stratify=True)
        >>> dm = CohortDataModule(cohort, train_idx, test_idx=test_idx, batch_size="full")
        >>> L.Trainer(max_epochs=50).fit(model, datamodule=dm)
    """

    def __init__(
        self,
        cohort: Cohort,
        train_idx: Indices,
        val_idx: Indices | None = None,
        test_idx: Indices | None = None,
        *,
        batch_size: BatchSize,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        drop_last: bool = False,
        preprocessor: Preprocessor | None = None,
        collate_fn: Callable = collate_samples,
    ):
        super().__init__()
        self.cohort = cohort
        self.train_idx = np.asarray(train_idx, dtype=int)
        self.val_idx = None if val_idx is None else np.asarray(val_idx, dtype=int)
        self.test_idx = None if test_idx is None else np.asarray(test_idx, dtype=int)
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last

        _check_batch_size(batch_size)
        self._check_splits_disjoint()

        # For checkpoints only; the attributes above are what the code reads. batch_size
        # matters here: a Cox metric is not reproducible without the batch it was measured at.
        self.save_hyperparameters("batch_size", "num_workers", "pin_memory", "shuffle", "drop_last")

        self.preprocessor = preprocessor
        self._supplied_preprocessor = preprocessor is not None
        self.train_ds: CohortView | None = None
        self.val_ds: CohortView | None = None
        self.test_ds: CohortView | None = None
        self._prepared = False

    # ------------------------------------------------------------------ #
    # Lightning lifecycle
    # ------------------------------------------------------------------ #

    def setup(self, stage: str | None = None) -> None:
        """Fit this fold's preprocessor and build its views.

        Idempotent, since Lightning may call it more than once per ``Trainer``.

        Fitting belongs here and **not** in ``prepare_data()``, which runs on rank 0
        only: under DDP the other ranks would train with no preprocessor at all.
        """
        if self._prepared:
            return
        if not self._supplied_preprocessor:
            self.preprocessor = self.cohort.fit_preprocessor(self.train_idx)
        self.check_no_leak()
        self.train_ds = self.cohort.view(self.train_idx, self.preprocessor)
        self.val_ds = None if self.val_idx is None else self.cohort.view(self.val_idx, self.preprocessor)
        self.test_ds = None if self.test_idx is None else self.cohort.view(self.test_idx, self.preprocessor)
        self._prepared = True

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=self.shuffle, drop_last=self.drop_last)

    def val_dataloader(self) -> DataLoader | list:
        """The validation loader, or ``[]`` when no validation split was given.

        ``[]`` rather than ``None``, which Lightning 2.x rejects with a ``TypeError``
        during the sanity check. Having no validation split should not need a flag.
        """
        return [] if self.val_ds is None else self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """The test loader.

        Raises when there is no test split, the opposite of :meth:`val_dataloader` and
        deliberately so: ``Trainer.test()`` is an explicit request for a result, and
        returning nothing would let it succeed having evaluated no rows.

        Raises:
            ValueError: If the data module was built without ``test_idx``.
        """
        if self.test_ds is None:
            raise ValueError(
                "Trainer.test() was called but this CohortDataModule has no test split. "
                "Pass test_idx=... when constructing it."
            )
        return self._loader(self.test_ds, shuffle=False)

    # ------------------------------------------------------------------ #
    # guards
    # ------------------------------------------------------------------ #

    def check_no_leak(self) -> None:
        """Assert no held-out row's statistics are baked into the preprocessor.

        Holds by construction in the default flow; it earns its keep when a
        ``preprocessor`` is *supplied*, the only way held-out statistics can reach a
        fold. Compares identifiers, not positions, so it survives composition.

        Raises:
            LeakageError: If a validation or test identifier appears in ``fitted_on``.
        """
        fitted_on = getattr(self.preprocessor, "fitted_on", None)
        if not fitted_on:
            return  # nothing fitted, or a passthrough, which carries no row's data
        for label, indices in (("validation", self.val_idx), ("test", self.test_idx)):
            if indices is None:
                continue
            held_out = {self.cohort.identifiers[i] for i in indices}
            overlap = held_out & fitted_on
            if overlap:
                raise LeakageError(
                    f"{len(overlap)} {label} patient(s) are among the rows this fold's "
                    f"preprocessor was fitted on, e.g. {sorted(overlap)[:5]}. Held-out "
                    f"data must be transformed with the training fold's statistics, never "
                    f"fitted on them. Fit with cohort.fit_preprocessor(train_idx) and pass "
                    f"that same preprocessor to every view."
                )

    def _check_splits_disjoint(self) -> None:
        """Refuse overlapping splits, whether or not anything is fitted.

        Independent of :meth:`check_no_leak`, which cannot see this for a passthrough
        cohort -- the rows would still be evaluated as held out having been trained on.
        """
        named = [("train", self.train_idx), ("validation", self.val_idx), ("test", self.test_idx)]
        present = [(label, idx) for label, idx in named if idx is not None]
        for i, (left_label, left) in enumerate(present):
            for right_label, right in present[i + 1 :]:
                shared = np.intersect1d(left, right)
                if shared.size:
                    raise ValueError(
                        f"{left_label} and {right_label} splits share {shared.size} row(s), "
                        f"e.g. positions {shared[:5].tolist()}. Evaluating on rows that were "
                        f"trained on reports a training score as a held-out one."
                    )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _loader(self, dataset: CohortView | None, *, shuffle: bool, drop_last: bool = False) -> DataLoader:
        if dataset is None:  # pragma: no cover - guarded by the callers above
            raise RuntimeError("setup() has not run; Lightning calls it, or call it yourself.")
        return DataLoader(
            dataset,
            batch_size=self._resolved_batch_size(),
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def _resolved_batch_size(self) -> int:
        """``"full"`` resolved against the training split size."""
        size = self.batch_size
        return len(self.train_idx) if size == "full" else int(size)

    def __repr__(self) -> str:
        sizes = [f"train={len(self.train_idx)}"]
        if self.val_idx is not None:
            sizes.append(f"val={len(self.val_idx)}")
        if self.test_idx is not None:
            sizes.append(f"test={len(self.test_idx)}")
        sizes.append(f"batch_size={self.batch_size!r}")
        return f"{type(self).__name__}({' | '.join(sizes)})"


def _check_batch_size(batch_size: BatchSize) -> None:
    if batch_size == "full":
        return
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(
            f"batch_size must be a positive integer or 'full', not {batch_size!r}. There is "
            f"no default: with a Cox head the partial likelihood is averaged within a batch, "
            f"so batch size selects the risk-set approximation and belongs in your script."
        )
