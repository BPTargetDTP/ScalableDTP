# TODO: Add tests for the angle and distance computation.
import torch
from torch import Tensor
from .metrics import compute_dist_angle


def test_zero_when_identical():
    a = torch.ones([3, 123, 456])
    b = a.clone()
    distance, angle = compute_dist_angle(a, b)
    assert distance == 0
    assert angle == 0
