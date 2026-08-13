from kalecancer.loaddata.dataset import BaseDataset


class MultiModalDataset(BaseDataset):
    """Still in development: Several modalities aligned on their shared identifier.

    Args:
        components (dict): Maps a modality name to a :class:`BaseDataset`.
        target_source (str, optional): Name of the component whose target is used for the whole
            sample. Keeps the target defined in one place rather than restated per modality, where
            the copies can drift apart.
        require_all (bool): Keep only identifiers present in every component when True; otherwise
            keep every identifier present in at least one and return None for absent modalities.
    """

    def __init__(self, components: dict[str, BaseDataset], target_source: str | None = None, require_all: bool = True):
        super().__init__(path=None)

    def _loader(self) -> None:
        """Should load all components and align their identifiers."""
        raise NotImplementedError("MultiModalDataset.load_data is not yet implemented.")

    def get_by_id(self, identifier):
        raise NotImplementedError("MultiModalDataset.get_by_id is not yet implemented.")
