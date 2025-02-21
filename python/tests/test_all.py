import pytest
import grid_weights


def test_sum_as_string():
    assert grid_weights.sum_as_string(1, 1) == "2"
