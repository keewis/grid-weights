use geo::{algorithm::BooleanOps, Area};
use geoarrow::{array::PolygonArray, trait_::ArrayAccessor};
use numpy::{PyArray1 as PyNumpyArray1, PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3_arrow::PyArray;

pub trait AsPolygonArray {
    fn into_polygon_array(self) -> PyResult<PolygonArray>;
}

impl AsPolygonArray for PyArray {
    fn into_polygon_array(self) -> PyResult<PolygonArray> {
        let (array, field) = self.into_inner();

        let polygons = PolygonArray::try_from((array.as_ref(), field.as_ref()));

        match polygons {
            Ok(p) => Ok(p),
            Err(error) => Err(PyOSError::new_err(error.to_string())),
        }
    }
}

fn cast(vec: Vec<i64>) -> Vec<usize> {
    vec.into_iter().map(|c| c as usize).collect()
}

fn uncompress_sparse_indices(
    indices: Vec<usize>,
    index_ptr: Vec<usize>,
) -> Vec<(usize, Vec<usize>)> {
    index_ptr
        .iter()
        .scan(0, |prev, cur| {
            let previous = *prev;
            *prev = *cur;

            Some((previous, *cur))
        })
        .skip(1)
        .enumerate()
        .map(|it| {
            let row_index = it.0;
            let col_range = it.1;
            let lower = col_range.0;
            let upper = col_range.1;
            let col_indexes = &indices[lower..upper];
            (row_index, col_indexes.to_vec())
        })
        .collect()
}

fn pyarray_as_vec(data: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let pyarray = data.downcast::<PyArrayDyn<i64>>()?;

    Ok(pyarray.readonly().as_slice()?.to_vec())
}

#[pyfunction]
fn conservative_regridding(
    source_cells: PyArray,
    target_cells: PyArray,
    overlapping_cells: Py<PyAny>,
) -> PyResult<PyObject> {
    let source_polygons: Vec<_> = source_cells
        .into_polygon_array()?
        .iter_geo_values()
        .collect();
    let target_polygons: Vec<_> = target_cells
        .into_polygon_array()?
        .iter_geo_values()
        .collect();

    // algorithm:
    // 1. iterate over the sparse matrix → pair of indices into source / target cells
    // 2. extract the cells
    // 3. compute the overlap between source and target cell
    // 4. normalize
    // 5. construct sparse matrix
    let indices: Vec<(usize, Vec<usize>)> =
        Python::with_gil(|py| -> PyResult<Vec<(usize, Vec<usize>)>> {
            let indices: Vec<usize> = cast(pyarray_as_vec(
                overlapping_cells.getattr(py, "indices")?.bind(py),
            )?);
            let index_ptr: Vec<usize> = cast(pyarray_as_vec(
                overlapping_cells.getattr(py, "indptr")?.bind(py),
            )?);

            Ok(uncompress_sparse_indices(indices, index_ptr))
        })?;

    let weights = indices
        .iter()
        .map(|it| {
            let source_indices = &it.1;
            let target_cell = &target_polygons[it.0];
            let target_area = target_cell.unsigned_area();

            // TODO: normalize the weights
            let weights = source_indices
                .iter()
                .map(|index| {
                    let source_cell = &source_polygons[*index];

                    target_cell.intersection(source_cell).unsigned_area() / target_area
                })
                .collect::<Vec<_>>();

            let sum: f64 = weights.iter().sum();

            weights.into_iter().map(|w| w / sum).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Python::with_gil(|py| {
        let arg = (
            PyNumpyArray1::from_vec(py, weights.into_iter().flatten().collect()),
            overlapping_cells.getattr(py, "indices")?,
            overlapping_cells.getattr(py, "indptr")?,
        );
        let sparse = PyModule::import(py, "sparse")?;
        let args = (arg,);
        let kwargs = [
            ("shape", overlapping_cells.getattr(py, "shape")?),
            (
                "compressed_axes",
                overlapping_cells.getattr(py, "compressed_axes")?,
            ),
        ]
        .into_py_dict(py)?;

        Ok(sparse.getattr("GCXS")?.call(args, Some(&kwargs))?.unbind())
    })
}

/// A Python module implemented in Rust.
#[pymodule]
fn grid_weights(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(conservative_regridding, m)?)?;
    Ok(())
}
