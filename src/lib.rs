use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
// use numpy::PyArray2;
use pyo3::prelude::*;
// use pyo3::types::PyList;
mod kde_funcs;

#[pyfunction]
#[pyo3(name = "epanechnikov_kde_rs")]
fn epanechnikov_kde_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    n_threads: usize,
    n_chunk: usize,
) -> Bound<'py, PyArray1<f64>> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let n_dim: usize = x.shape()[1];
    let res = match n_dim {
        1 => kde_funcs::epanechnikov_kde::<1>(x, points, lamdaopt, n_threads, n_chunk),
        2 => kde_funcs::epanechnikov_kde::<2>(x, points, lamdaopt, n_threads, n_chunk),
        3 => kde_funcs::epanechnikov_kde::<3>(x, points, lamdaopt, n_threads, n_chunk),
        4 => kde_funcs::epanechnikov_kde::<4>(x, points, lamdaopt, n_threads, n_chunk),
        5 => kde_funcs::epanechnikov_kde::<5>(x, points, lamdaopt, n_threads, n_chunk),
        6 => kde_funcs::epanechnikov_kde::<6>(x, points, lamdaopt, n_threads, n_chunk),
        7 => kde_funcs::epanechnikov_kde::<7>(x, points, lamdaopt, n_threads, n_chunk),
        _ => panic!("Unsupported dimension: {}", n_dim),
    };
    res.to_pyarray_bound(py)
}

#[pyfunction]
#[pyo3(name = "epanechnikov_kde_weights_rs")]
fn epanechnikov_kde_weights_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    n_threads: usize,
    n_chunk: usize,
) -> Bound<'py, PyArray1<f64>> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let weights = weights.as_array();
    let n_dim: usize = x.shape()[1];
    let res = match n_dim {
        1 => kde_funcs::epanechnikov_kde_weights::<1>(
            x, points, lamdaopt, weights, n_threads, n_chunk,
        ),
        2 => kde_funcs::epanechnikov_kde_weights::<2>(
            x, points, lamdaopt, weights, n_threads, n_chunk,
        ),
        3 => kde_funcs::epanechnikov_kde_weights::<3>(
            x, points, lamdaopt, weights, n_threads, n_chunk,
        ),
        4 => kde_funcs::epanechnikov_kde_weights::<4>(
            x, points, lamdaopt, weights, n_threads, n_chunk,
        ),
        5 => kde_funcs::epanechnikov_kde_weights::<5>(
            x, points, lamdaopt, weights, n_threads, n_chunk,
        ),
        6 => kde_funcs::epanechnikov_kde_weights::<6>(
            x, points, lamdaopt, weights, n_threads, n_chunk,
        ),
        7 => kde_funcs::epanechnikov_kde_weights::<7>(
            x, points, lamdaopt, weights, n_threads, n_chunk,
        ),
        _ => panic!("Unsupported dimension: {}", n_dim),
    };
    res.to_pyarray_bound(py)
}

/*
#[pyfunction]
#[pyo3(name = "epanechnikov_kde_rev_weights")]
fn epanechnikov_kde_rev_weights_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    n_threads: usize,
) -> &'py PyArray1<f64> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let weights = weights.as_array();
    let res =
        kde_funcs::epanechnikov_kde_rev_weights_3d(x, points, lamdaopt, weights, n_threads);
    res.to_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "epanechnikov_kde_rev_weights_groups")]
fn epanechnikov_kde_rev_weights_groups_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    points: PyReadonlyArray2<f64>,
    lamdaopt: PyReadonlyArray1<f64>,
    weights: PyReadonlyArray1<f64>,
    group_inds: PyReadonlyArray1<usize>,
    n_groups: usize,
    n_threads: usize,
) -> &'py PyArray2<f64> {
    let x = x.as_array();
    let points = points.as_array();
    let lamdaopt = lamdaopt.as_array();
    let weights = weights.as_array();
    let group_inds = group_inds.as_array();

    let res = kde_funcs::epanechnikov_kde_rev_weights_groups(
        x, points, lamdaopt, weights, group_inds, n_groups, n_threads,
    );
    res.to_pyarray(py)
} */

/* #[pyfunction]
#[pyo3(name = "epanechnikov_kde_rev_weights_multi")]
fn epanechnikov_kde_rev_weights_multi_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    // multi_points: Vec<PyReadonlyArray2<'py, f64>>,
    multi_points: Vec<PyReadonlyArray2<f64>>,
    multi_lamdaopt: Vec<PyReadonlyArray1<f64>>,
    multi_weights: Vec<PyReadonlyArray1<f64>>,
    n_threads: usize,
) -> &'py PyArray2<f64> {
    let x = x.as_array();
    println!("In rust trans multi");

    let vec_multi_points = multi_points.iter().map(|item| item.as_array()).collect();
    let vec_multi_weights = multi_weights.iter().map(|item| item.as_array()).collect();
    let vec_multi_lamdaopt = multi_lamdaopt.iter().map(|item| item.as_array()).collect();

    let res = kde_funcs::epanechnikov_kde_rev_weights_multi(
        x,
        vec_multi_points,
        vec_multi_lamdaopt,
        vec_multi_weights,
        n_threads,
    );
    res.to_pyarray(py)
} */

#[pymodule]
fn spyders(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(epanechnikov_kde_py, m)?)?;
    m.add_function(wrap_pyfunction!(epanechnikov_kde_weights_py, m)?)?;
    // m.add_function(wrap_pyfunction!(epanechnikov_kde_rev_weights_groups_py, m)?)?;
    Ok(())
}
