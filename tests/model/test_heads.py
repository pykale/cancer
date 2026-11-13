import pytest
import torch

from kalecancer.model import LinearClassifier


def test_linear_classifier_shape():
    model = LinearClassifier(in_dim=10, out_dim=5)
    scores = model(torch.randn(16, 10))
    assert model.fc.weight.shape == (5, 10)
    assert scores.shape == (16, 5)


def test_linear_classifier_no_bias():
    assert LinearClassifier(in_dim=10, out_dim=5, bias=False).fc.bias is None


def test_linear_classifier_initialization():
    torch.manual_seed(2021)
    model = LinearClassifier(in_dim=10, out_dim=5)
    assert torch.equal(model.fc.bias, torch.zeros(5))
    # Xavier normal over (out_dim, in_dim): std = sqrt(2 / (in_dim + out_dim))
    assert model.fc.weight.std().item() == pytest.approx((2 / 15) ** 0.5, rel=0.25)
