"""Fitted transforms.

Everything here is *fold-local state*. A preprocessor is fitted on one fold's rows
and belongs to that fold; it is never attached to a cohort, which is shared. See
:class:`kalecancer.loaddata.multimodal_access.Preprocessor` for the contract.
"""

from kalecancer.prepdata.tabular import TabularPreprocessor

__all__ = ["TabularPreprocessor"]
