from importlib.metadata import PackageNotFoundError, version

from grid_weights.api import Algorithms, Index, create_index, weights
from grid_weights.conservative import conservative_weights
from grid_weights.grid_weights import conservative_regridding

try:
    __version__ = version("grid-weights")
except PackageNotFoundError:  # noqa # pragma: no cover
    # package is not installed
    __version__ = "9999"

__all__ = [
    "conservative_regridding",
    "conservative_weights",
    "Algorithms",
    "create_index",
    "weights",
    "Index",
]
