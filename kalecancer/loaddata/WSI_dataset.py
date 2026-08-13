from pathlib import Path

from kalecancer.loaddata.dataset import BaseDataset


class WSIDataset(BaseDataset):
    """Whole-slide images: one file per sample, identifier taken from the filename."""

    path: Path

    def __init__(
        self,
        path: str | Path,
    ):
        super().__init__(path)
        """need to figure out how we will find identifiers and target from files"""

    def _loader(self) -> None:
        """probably some sort of generatorm as the data is too big to load in RAM"""
        raise NotImplementedError("WSIDataset.load_data is not yet implemented.")

    def get_by_id(self, identifier):
        """probably finds the file and returns a tensore of the image and the target"""
        raise NotImplementedError("WSIDataset.get_by_id is not yet implemented.")
