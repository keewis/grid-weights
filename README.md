# grid-weights: fast and scalable generalized regridding

`grid-weights` aims to close a gap in the geospatial python ecosystem: the interpolation of geospatial data to/from arbitrary and potentially larger-than-memory grids. Both the input and output grids are represented by n-dimensional arrays of polygons.

Supported interpolation algorithms:

- first-order conservative (area-weighted sum)

## Typical usage

```python
import grid_weights

source_geoms = ...  # source cell boundary polygons
target_geoms = ...  # target cell boundary polygons

algorithms = grid_weights.Algorithms.by_variable(ds, default="conservative")
indexed_cells = grid_weights.create_index(source_geoms).query(
    target_geoms, methods=algorithms.indexing_methods()
)
weights = grid_weights.weights(source_geoms, target_geoms, indexed_cells)

regridded = algorithms.regrid(ds, weights)
```

## How this works

`grid-weights` allows configuring interpolation algorithms on a per-variable basis. For example, it allows selecting "nearest-neighbour" interpolation for a boolean mask, "majority vote" for categorical data, and "conservative" for a continuous physical quantity (once those algorithms are actually implemented).

The procedure works as follows:

1. use the R\*Tree implementation of the [grid-indexing](https://github.com/keewis/grid-indexing) package to compute a sparse boolean mask representing the source cells interacting with the target cells
2. use that sparse boolean mask and the source and target cell boundary polygons to compute the interpolation weights
3. For each variable, apply the appropriate interpolation weights using a sparse matrix multiplication

`grid-weights` is implemented in rust and makes use of the following crates:

- `geo`
- `geographiclib-rs`
- `geoarrow-rs`
- `numpy`
- `pyo3`
- `pyo3-arrow`
