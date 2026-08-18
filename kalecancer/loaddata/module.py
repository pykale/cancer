"""Lightning wiring: one fold's views, wrapped in DataLoaders.

A :class:`CohortDataModule` is a *fold-local* object. It fits this fold's
preprocessor, builds this fold's views, and hands Lightning its loaders. Nothing it
touches outlives the fold, and the one thing it shares -- the cohort -- it only
reads. That is what makes ``setup()`` assigning to ``self`` safe here: mutation is
not the hazard, mutation of shared state is.

Cross-validation therefore means one of these per fold, alongside one ``Trainer``
and one ``LightningModule``. The sequencing that a user would otherwise have to get
right by hand -- fit on train, apply to val, never the reverse -- happens in one
place, once, and is checked by :meth:`CohortDataModule.check_no_leak`.
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
        cohort (Cohort): The shared cohort. Read, never mutated, so every fold's
            data module may hold the same instance.
        train_idx (Indices): Positional indices of the training rows. The only
            rows the preprocessor is fitted on.
        val_idx (Indices | None, optional): Validation rows.
        test_idx (Indices | None, optional): Test rows.
        batch_size (int | "full"): **Required, no default.** With a Cox head this
            selects the risk-set approximation -- the partial likelihood is averaged
            within a batch, so ``64`` and ``"full"`` optimise measurably different
            objectives. That makes it a modelling decision wearing a loader
            parameter's clothes, and a hidden default would put it beyond the reach
            of anyone reading the script. ``"full"`` means ``len(train_idx)``.
        num_workers (int, optional): DataLoader worker processes. ``0`` is right for
            a clinical table, where worker startup costs more than it saves. Slide
            payloads want several. Defaults to 0.
        pin_memory (bool, optional): Defaults to ``False``.
        shuffle (bool, optional): Shuffle the *training* loader. Validation and test
            loaders are never shuffled. Defaults to ``True``.
        drop_last (bool, optional): Drop a short final training batch. Defaults to
            ``False``.
        preprocessor (Preprocessor | None, optional): Reuse an already-fitted
            preprocessor instead of fitting one from ``train_idx``. ``None`` (the
            default) fits a fresh one, which is what a cross-validation fold wants.
            Supply one when a later stage must apply an earlier stage's statistics
            -- refitting a final model, or scoring a saved checkpoint against the
            transforms it was trained with. This is the path
            :meth:`check_no_leak` exists for: a supplied preprocessor is the only
            way held-out rows can end up inside one.
        collate_fn (Callable, optional): Defaults to
            :func:`~kalecancer.loaddata.sample.collate_samples`, which pads ragged
            modalities. ``torch``'s default collate cannot handle a ``PatientSample``.

    Raises:
        ValueError: If the splits overlap, or if ``batch_size`` is not a positive
            integer or ``"full"``.

    Example:
        >>> train_idx, test_idx = cohort.split(test_size=0.2, random_state=0)
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

        # Recorded into checkpoints for provenance, separately from the attributes
        # above which the code actually reads. batch_size especially: with per-batch
        # Cox averaging a reported metric is not reproducible from the reported
        # numbers unless the batch size travels with them.
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

        Idempotent: Lightning may call this more than once per ``Trainer``, and
        refitting would be both wasteful and, with a stochastic transform, wrong.

        Fitting belongs here and **not** in ``prepare_data()``. That hook runs on
        rank 0 only and its assignments are invisible to the other processes, so
        under DDP the remaining ranks would train with no preprocessor at all -- a
        silent wrong answer rather than a crash.
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

        ``[]`` rather than ``None``: Lightning 2.x rejects ``None`` from this hook
        with a ``TypeError`` during the sanity check, but reads an empty list as
        "there is no validation here" and skips it. Not having a validation split is
        an ordinary thing to want, so it should not need a Trainer flag to express.
        """
        return [] if self.val_ds is None else self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """The test loader.

        Raises rather than returning ``[]`` when no test split was given -- the
        opposite of :meth:`val_dataloader`, deliberately. Lightning only calls this
        from ``Trainer.test()``, which is an explicit request for a test result, so
        returning nothing would let that call succeed while evaluating no rows and
        reporting no metrics. A skipped validation is a choice; a silent no-op test
        is a bug you find days later.

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

        In the default flow this holds by construction -- :meth:`setup` fits on
        ``train_idx``, and :meth:`_check_splits_disjoint` has already established
        that the splits do not overlap. It earns its keep when a ``preprocessor`` is
        *supplied*, which is the only way statistics from held-out rows can reach a
        fold. It compares identifiers rather than positions, so it stays correct
        across subsetting, recombination and multimodal composition.

        Raises:
            LeakageError: If any validation or test identifier appears in the
                preprocessor's ``fitted_on``.
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

        Independent of :meth:`check_no_leak`, which cannot see this when a cohort
        declares no stateful transforms: the rows would still be evaluated as
        held-out having been trained on.
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
