"""The item type and its collation.

``collate_samples`` is not optional plumbing: ``default_collate`` refuses a
dataclass outright, so this function is the only thing between a
:class:`PatientSample` and a ``DataLoader``. It is also where variable-length
slide bags will be padded, so its ragged handling is tested now, before the
modality that needs it exists.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import default_collate

from kalecancer.loaddata.sample import PatientBatch, PatientSample, collate_samples


def sample(pid: str, *, tiles: int | None = None, target: bool = True) -> PatientSample:
    modalities = {"clinical": torch.arange(4, dtype=torch.float32)}
    present = {"clinical": torch.tensor(True)}
    if tiles is not None:
        modalities["wsi"] = torch.ones(tiles, 3)
        present["wsi"] = torch.tensor(tiles > 0)
    return PatientSample(
        patient_id=pid,
        modalities=modalities,
        present=present,
        target={"time": torch.tensor(1.0), "event": torch.tensor(0.0)} if target else {},
    )


# =========================================================================== #
# why this function has to exist
# =========================================================================== #


def test_default_collate_cannot_handle_a_patient_sample():
    """The measured fact the module rests on.

    If a future torch learns to collate dataclasses this fails; reconsider whether
    ``collate_samples`` is still needed, rather than deleting the test.
    """
    with pytest.raises(TypeError, match="batch must contain"):
        default_collate([sample("a"), sample("b")])


def test_lightning_can_move_a_batch_to_a_device():
    """``transfer_batch_to_device`` recurses into dataclasses, so we need no hook."""
    from lightning.fabric.utilities.apply_func import move_data_to_device

    moved = move_data_to_device(collate_samples([sample("a"), sample("b")]), "cpu")
    assert isinstance(moved, PatientBatch)
    assert moved.modalities["clinical"].shape == (2, 4)


# =========================================================================== #
# collation
# =========================================================================== #


def test_fixed_width_modalities_stack_with_no_padding():
    batch = collate_samples([sample("a"), sample("b"), sample("c")])

    assert batch.patient_id == ["a", "b", "c"]
    assert batch.modalities["clinical"].shape == (3, 4)
    assert batch.present["clinical"].dtype is torch.bool
    assert batch.present["clinical"].shape == (3,)
    assert batch.pad_mask == {}, "nothing ragged means no padding bookkeeping"
    assert len(batch) == 3


def test_targets_are_batched_by_key():
    batch = collate_samples([sample("a"), sample("b")])
    assert set(batch.target) == {"time", "event"}
    assert batch.target["time"].shape == (2,)


def test_a_cohort_without_a_target_collates_to_an_empty_target_dict():
    batch = collate_samples([sample("a", target=False), sample("b", target=False)])
    assert batch.target == {}


def test_ragged_modalities_are_padded_and_masked():
    """Variable-length bags: the case an attention aggregator has to be told about."""
    batch = collate_samples([sample("a", tiles=2), sample("b", tiles=5), sample("c", tiles=3)])

    assert batch.modalities["wsi"].shape == (3, 5, 3)
    assert batch.pad_mask["wsi"].shape == (3, 5)
    assert batch.pad_mask["wsi"].sum(dim=1).tolist() == [2, 5, 3]
    assert "clinical" not in batch.pad_mask, "only ragged modalities get a mask"


def test_padding_is_zero_and_confined_to_the_tail():
    batch = collate_samples([sample("a", tiles=2), sample("b", tiles=4)])
    padded = batch.modalities["wsi"][0]

    assert torch.equal(padded[:2], torch.ones(2, 3))
    assert torch.equal(padded[2:], torch.zeros(2, 3))


def test_present_and_pad_mask_are_different_axes():
    """Why they are not both called "mask": one flag per patient per modality, versus
    one per tile within a bag. Conflating them gives a fusion layer a plausible but
    wrong shape."""
    batch = collate_samples([sample("a", tiles=1), sample("b", tiles=3)])

    assert batch.present["wsi"].shape == (2,)
    assert batch.pad_mask["wsi"].shape == (2, 3)


# =========================================================================== #
# refusals
# =========================================================================== #


def test_empty_batch_raises():
    with pytest.raises(ValueError, match="empty list"):
        collate_samples([])


def test_samples_disagreeing_about_modalities_raise():
    """A cohort that emits a modality for some patients and not others is broken.

    Silently collating the intersection would drop a modality for the whole batch
    and train a model that never sees it.
    """
    with pytest.raises(ValueError, match="disagree about modalities"):
        collate_samples([sample("a"), sample("b", tiles=2)])


def test_samples_disagreeing_about_target_keys_raise():
    with pytest.raises(ValueError, match="disagree about target"):
        collate_samples([sample("a"), sample("b", target=False)])


def test_a_trailing_shape_mismatch_is_not_padded_over():
    """Ragged means ragged in the *bag* axis only.

    A differing feature width is a one-hot encoder fitted on the wrong rows, or a
    modality wired up wrongly. Padding it would turn a loud shape error into a
    quietly wrong tensor.
    """
    a = PatientSample("a", {"x": torch.ones(2, 3)}, {"x": torch.tensor(True)})
    b = PatientSample("b", {"x": torch.ones(2, 4)}, {"x": torch.tensor(True)})

    with pytest.raises(ValueError, match="more than the first axis"):
        collate_samples([a, b])
