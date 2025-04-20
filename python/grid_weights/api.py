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

import geoarrow.rust.core as geoarrow
import numpy as np
import xarray as xr
from grid_indexing import RTree
from grid_indexing.distributed import DistributedRTree
from tlz.itertoolz import concat
from xarray.namedarray.utils import either_dict_or_kwargs

from grid_weights.conservative import conservative_weights

try:
    import dask.array as da

    dask_array_type = (da.Array,)
except ImportError:
    dask_array_type = ()

_implemented_algorithms = {
    "conservative": conservative_weights,
}
_indexing_methods = {
    "conservative": "overlaps",
}


@dataclass
class Algorithms:
    variables: dict[Hashable, str]

    def __post_init__(self):
        # check that all algorithms are known
        unknown_algorithms = [
            name
            for name in self.variables.values()
            if name not in _implemented_algorithms
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
        return list(dict.fromkeys(self.variables.values()))

    def regrid(self, ds, weights):
        def _regrid(arr):
            algorithm = self.variables[arr.name]

            return xr.dot(
                arr.variable, weights[algorithm], dims=weights.attrs["source_dims"]
            )

        to_regrid = ds.rename_dims(
            {dim: f"source_{dim}" for dim in ds.dims if f"source_{dim}" in weights.dims}
        )
        regridded = to_regrid.map(_regrid)

        return (
            regridded.assign_coords(
                weights.coords.to_dataset()
                .drop_dims(weights.attrs["source_dims"])
                .coords
            )
            .assign_coords(to_regrid.drop_dims(weights.attrs["source_dims"]).coords)
            .pipe(removeprefix_coords_and_dims, "target")
        )


def prefix_coords_and_dims(coords, prefix):
    return (
        coords.to_dataset()
        .rename_dims({dim: f"{prefix}_{dim}" for dim in coords.dims})
        .rename_vars({var: f"{prefix}_{var}" for var in coords.variables})
        .coords
    )


def removeprefix_coords_and_dims(ds, prefix):
    prefix_ = f"{prefix}_"
    return ds.rename_dims(
        {dim: dim.removeprefix(prefix_) for dim in ds.dims if dim.startswith(prefix_)}
    ).rename_vars(
        {
            var: var.removeprefix(prefix_)
            for var in ds.variables
            if var.startswith(prefix_)
        }
    )


@dataclass
class Index:
    index: RTree | DistributedRTree
    source_geoms: xr.DataArray

    def query(self, target_geoms, *, methods):
        indexing_methods = list(
            dict.fromkeys(_indexing_methods[method] for method in methods)
        )
        if isinstance(self.index, RTree):
            raw_geoms = geoarrow.from_shapely(target_geoms.data.flatten())
            kwargs = {"shape": target_geoms.shape}
        else:
            raw_geoms = target_geoms.data
            kwargs = {}

        results = {
            method: self.index.query(raw_geoms, method=method, **kwargs)
            for method in indexing_methods
        }

        target_coords = prefix_coords_and_dims(target_geoms.coords, "target")
        source_coords = prefix_coords_and_dims(self.source_geoms.coords, "source")

        dims = [f"target_{dim}" for dim in target_geoms.dims] + [
            f"source_{dim}" for dim in self.source_geoms.dims
        ]

        return xr.Dataset(
            {method: (dims, data) for method, data in results.items()},
            coords=source_coords.assign(target_coords),
            attrs={"algorithms": methods},
        )


def create_index(source_geoms):
    raw_geoms = source_geoms.data

    if isinstance(raw_geoms, dask_array_type):
        index = DistributedRTree(raw_geoms)
    elif isinstance(raw_geoms, np.ndarray):
        index = RTree.from_shapely(raw_geoms)

    return Index(index, source_geoms)


def weights(source_geoms, target_geoms, indexed_cells):
    algorithms = indexed_cells.attrs["algorithms"]

    results = {}
    for algorithm in algorithms:
        indexing_method = _indexing_methods[algorithm]

        func = _implemented_algorithms[algorithm]
        indexed_cells_ = indexed_cells[indexing_method]

        raw_weights = func(source_geoms.data, target_geoms.data, indexed_cells_.data)
        results[algorithm] = (indexed_cells_.dims, raw_weights)

    source_dims = [dim for dim in indexed_cells.dims if dim.startswith("source_")]
    return xr.Dataset(
        results, coords=indexed_cells.coords, attrs={"source_dims": source_dims}
    )
