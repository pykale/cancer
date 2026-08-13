from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import Dataset as TorchDataset


class BaseDataset(TorchDataset, ABC):
    """Abstract base for datasets keyed by a sample identifier.

    Subclasses implement ``_loader`` and ``get_by_id``.

    Loading is lazy and happens at most once per instance. Call``load_data`` to load data
    DataLoader with workers, since an unloaded dataset is otherwise parsed once per worker process.

    Inherits ``torch.utils.data.Dataset``, so DataLoader, Subset, random_split
    and ConcatDataset work without adapters.

    Args:
        path (str | Path | None, optional): Path to the data file. None for composite datasets that
            have no source of their own.
    """

    def __init__(
        self,
        path: str | Path | None = None,
    ):
        self.path: Path | None = Path(path) if path is not None else None
        self.identifiers: list = []
        self._loaded = False

    def load_data(self) -> None:
        """Read the data once. Later calls are ignored."""
        if self._loaded:
            return
        self._loader()
        self._loaded = True

    @abstractmethod
    def _loader(self) -> None:
        """Read from ``self.path`` and populate ``self.identifiers``."""

    @abstractmethod
    def get_by_id(self, identifier):
        """Return single using the identifier."""

    def __len__(self) -> int:
        self.load_data()
        return len(self.identifiers)

    def __getitem__(self, idx):
        self.load_data()
        return self.get_by_id(self.identifiers[idx])
