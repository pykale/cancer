"""Access to the public HANCOCK head and neck cancer dataset.

Shared by every HANCOCK example, so the archive layout and the published train/test
assignment are described once rather than copied into each. Fetching, caching and
member selection come from
:class:`~kalecancer.loaddata.archive_access.ArchiveDataset`.
"""

from examples.hancock.dataset import HancockDataset, fetch_for, official_split, split_for

__all__ = ["HancockDataset", "fetch_for", "official_split", "split_for"]
