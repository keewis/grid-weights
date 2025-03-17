import sparse
import geoarrow.rust.core as geoarrow
from grid_indexing.distributed import ChunkGrid
import numpy as np
from grid_weights.grid_weights import conservative_regridding

try:
    import dask.array as da

    dask_array_type = da.Array
except ImportError:
    dask_array_type = ()


def evaluate_chunked(*args):
    chunked = [isinstance(arg, dask_array_type) for arg in args]

    if all(chunked):
        return "all"
    elif not any(chunked):
        return "none"
    else:
        return "inconsistent"


def sparse_as_contiguous(arr):
    return sparse.GCXS.from_coo(arr.tocoo(), compressed_axes=(0,))


def sparse_broadcasted_norm(arr, axis):
    import dask.array as da

    norms = np.sum(arr, axis=axis)

    def sparse_broadcast(chunk, norm_chunk):
        if chunk.shape == ():
            return sparse.GCXS.from_coo(sparse.empty((), dtype=chunk.dtype))

        if chunk.nnz == 0:
            return sparse.GCXS.from_coo(sparse.full_like(chunk.tocoo(), fill_value=1))

        norms_ = np.reshape(norm_chunk, -1).todense()

        coo = np.reshape(chunk.tocoo(), norms_.shape + (-1,))
        chunk_ = sparse.GCXS.from_coo(coo, compressed_axes=(0,))

        repeats = np.diff(chunk_.indptr)
        extended_norms = np.repeat(norms_, repeats)

        broadcasted_ = sparse.GCXS(
            (extended_norms, chunk_.indices, chunk_.indptr),
            shape=chunk_.shape,
            compressed_axes=(0,),
            fill_value=1,
        )
        coo = np.reshape(broadcasted_.tocoo(), chunk.shape)

        return sparse.GCXS.from_coo(coo, compressed_axes=(0,))

    return da.blockwise(
        sparse_broadcast,
        "ijk",
        arr,
        "ijk",
        norms,
        "i",
        meta=sparse.GCXS.from_coo(sparse.empty((), dtype=arr.dtype)),
    )


def sparse_normalize(arr, axis):
    return arr / sparse_broadcasted_norm(arr, axis)


def _compute_chunk(source, target, mask, **kwargs):
    if mask.nnz == 0:
        # no overlap short-circuit
        return sparse.full_like(mask, fill_value=0, dtype="float64")

    output_shape = target.shape + source.shape

    source_geoms = geoarrow.from_shapely(source.flatten())
    target_geoms = geoarrow.from_shapely(target.flatten())

    mask_shape = (len(target_geoms), len(source_geoms))
    mask_reshaped = sparse_as_contiguous(np.reshape(mask, mask_shape))

    weights = conservative_regridding(source_geoms, target_geoms, mask_reshaped)

    return np.reshape(weights, output_shape)


def conservative_weights(source_cells, target_cells, overlapping_cells):
    """Compute conservative regridding weights

    Also called "area-weighted sum", this algorithm assigns weights based on the size of
    overlap between source and target cells (bigger overlaps result in a bigger weight).

    Parameters
    ----------
    source_cells : array-like
        The source cell boundaries as `shapely` polygons.
    target_cells : array-like
        The target cell boundaries as `shapely` polygons.
    overlapping_cells : array-like
        Sparse matrix of booleans. Used to vastly speed up the computation of weights and
        reduce the memory usage.

    Returns
    -------
    weights : array-like
        Sparse matrix of regridding weights.

    Raises
    ------
    ValueError
        In case one array is chunked and the others aren't, or if all but one array are
        chunked.
    ValueError
        If the chunking schemes don't match.
    """
    import dask.array as da
    import dask

    if (
        evaluate_chunked(source_cells, target_cells, overlapping_cells)
        == "inconsistent"
    ):
        raise ValueError("If one argument is chunked, all arguments must be chunked.")

    source_grid = ChunkGrid.from_dask(source_cells)
    target_grid = ChunkGrid.from_dask(target_cells)
    mask_grid = ChunkGrid.from_dask(overlapping_cells)

    output_grid_shape = target_grid.grid_shape + source_grid.grid_shape
    output_grid = np.full(output_grid_shape, dtype=object, fill_value=None)
    meta = sparse.GCXS.from_coo(sparse.empty((), dtype="float64"))

    for (
        target_indices,
        target_chunk_shape,
        target_chunk,
    ) in target_grid.flat_iter_chunks():
        for (
            source_indices,
            source_chunk_shape,
            source_chunk,
        ) in source_grid.flat_iter_chunks():
            indices = target_indices + source_indices

            chunk_shape = target_chunk_shape + source_chunk_shape

            mask_chunk = mask_grid[indices]
            task = dask.delayed(_compute_chunk)(
                source_chunk, target_chunk, mask_chunk, indices=indices
            )

            output_grid[indices] = da.from_delayed(
                task, shape=chunk_shape, dtype="float64", meta=meta
            )

    result = da.block(output_grid.tolist())

    return sparse_normalize(
        result, axis=tuple(range(result.ndim)[-source_cells.ndim :])
    )
