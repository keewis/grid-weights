"""xarray api draft

# TODO: figure out how to construct geometries
source_geoms = ...
target_geoms = ...

algorithms = Algorithms.by_variable({
    "air_temperature": "conservative", "mask": "nearest"
})
indexed_cells = (
    grid_indexing.create_index(source_geoms, kind="rtree")
    .query(target_geoms, modes=algorithms.indexing_modes())
)
weights = grid_indexing.weights(source_geoms, target_geoms, indexed_cells)
regridded = algorithms.regrid(ds, weights)
"""

import copy
from collections import Counter
from collections.abc import Hashable
from dataclasses import dataclass

import xarray as xr
from tlz.itertoolz import concat
from xarray.namedarray.utils import either_dict_or_kwargs

from grid_weights.conservative import conservative_weights

implemented_algorithms = {
    "conservative": conservative_weights,
}


@dataclass
class Algorithms:
    variables: dict[Hashable, str]

    def __post_init__(self):
        # check that all algorithms are known
        unknown_algorithms = [
            name
            for name in self.variables.values()
            if name not in implemented_algorithms
        ]
        if unknown_algorithms:
            raise ValueError(f"unknown algorithms: {', '.join(unknown_algorithms)}")

    @classmethod
    def by_variable(
        cls,
        ds: xr.Dataset,
        variables: dict[Hashable, str] = None,
        default: str = None,
        **variable_kwargs,
    ):
        passed_variables = either_dict_or_kwargs(
            variables, variable_kwargs, func_name="Algorithms.by_variable"
        )

        missing_variables = [
            name for name in ds.data_vars if name not in passed_variables
        ]
        if missing_variables and default is None:
            raise ValueError(
                f"no configuration for {missing_variables} and no default set"
            )

        variables = passed_variables | {k: default for k in missing_variables}

        return cls(variables)

    @classmethod
    def by_algorithm(
        cls,
        ds: xr.Dataset,
        algorithms: dict[str, Hashable | list[Hashable]] = None,
        default: str = None,
        **algorithm_kwargs,
    ):
        passed_algorithms = copy.deepcopy(
            either_dict_or_kwargs(
                algorithms, algorithm_kwargs, func_name="Algorithms.by_algorithm"
            )
        )

        # check for duplicates
        counter = Counter(concat(passed_algorithms.values()))
        duplicates = [name for name, count in counter.items() if count > 1]
        if duplicates:
            raise ValueError(f"variables {', '.join(duplicates)} appear more than once")

        variables = {var: name for name, vars in algorithms.items() for var in vars}

        missing_variables = [name for name in ds.data_vars if name not in variables]
        if missing_variables and default is None:
            raise ValueError(
                f"no configuration for {missing_variables} and no default set"
            )
        elif missing_variables:
            variables.update({var: default for var in missing_variables})

        return cls(variables)

    def unique(self):
        return list(self.algorithms.values())

    def regrid(self, ds, weights):
        pass


def create_index(source_geoms, *, kind="rtree"):
    pass


def weights(source_geoms, target_geoms, indexed_cells):
    pass
