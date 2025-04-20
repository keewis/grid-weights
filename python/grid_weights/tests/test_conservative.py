import numpy as np
import pytest

from grid_weights import conservative

da = pytest.importorskip("dask.array")


@pytest.mark.parametrize(
    ["args", "expected"],
    (
        pytest.param((np.ones((2,))), "none"),
        pytest.param((da.ones((2,), chunks=(1,))), "all"),
        pytest.param((np.ones((2,)), np.zeros((2,))), "none"),
        pytest.param((da.ones((2,), chunks=(1,)), da.zeros((2,), chunks=(1,))), "all"),
        pytest.param((np.ones((2,)), da.zeros((2,), chunks=(1,))), "inconsistent"),
    ),
)
def test_evaluate_chunked(args, expected):
    actual = conservative.evaluate_chunked(*args)

    assert actual == expected
