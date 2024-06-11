// use itertools::izip;
use kiddo::float::kdtree::KdTree as float_KdTree;
use kiddo::SquaredEuclidean;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis, Zip};
// use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
// use ndarray_stats::{self, QuantileExt};
use spfunc::gamma;
use std::f64::consts::PI;
fn create_pool(n_threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .unwrap()
}

pub fn epanechnikov_kde<const N_DIM: usize>(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    n_threads: usize,
) -> Array1<f64> {
    let x_shape = x.shape();
    let n_x: usize = x_shape[0];
    let n_dim_x: usize = x_shape[1];
    let n_dim_points: usize = points.shape()[1];
    assert_eq!(n_dim_x, N_DIM);
    assert_eq!(n_dim_points, N_DIM);

    let mut rhos = Array1::<f64>::zeros(n_x);
    let lamdaopt2: Array1<f64> = lamdaopt.map(|&x| x * x);
    let inv_lamdaopt_pow: Array1<f64> = lamdaopt.map(|&x| x.powi(-(N_DIM as i32)));
    let n_chunk: usize = std::cmp::max(std::cmp::min(n_x / n_threads, 50_000), 10_000);

    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_small)| {
                let mut stars_kdtree: float_KdTree<f64, usize, N_DIM, 256, u32> =
                    float_KdTree::with_capacity(n_chunk);
                for (idx, jvec) in x_small.axis_iter(Axis(0)).enumerate() {
                    stars_kdtree.add(unsafe { &*(jvec.as_ptr() as *const [f64; N_DIM]) }, idx)
                }

                Zip::from(points.axis_iter(Axis(0)))
                    .and(&lamdaopt2)
                    .and(&inv_lamdaopt_pow)
                    .for_each(|p_row, lamda2, inv_lamda| {
                        let neighbours = stars_kdtree.within_unsorted::<SquaredEuclidean>(
                            unsafe { &*(p_row.as_ptr() as *const [f64; N_DIM]) },
                            *lamda2,
                        );
                        for neigh in neighbours {
                            let t_2 = neigh.distance / lamda2;
                            rhos_small[neigh.item] += (1. - t_2) * inv_lamda;
                        }
                    });
            });
    });

    //
    let vd = PI.powf(N_DIM as f64 / 2.) / gamma::gamma(N_DIM as f64 / 2. + 1.);
    rhos *= (N_DIM as f64 + 2.) / (2. * vd);
    rhos
}

/* pub fn epanechnikov_kde_3d_weights(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    n_threads: usize,
    weights: ArrayView1<f64>,
) -> Array1<f64> {
    let x_shape = x.shape();
    let n_x: usize = x_shape[0];
    let n_dim: usize = x_shape[1];

    let points_shape = points.shape();
    let n_points: usize = points_shape[0];
    let n_dim_points: usize = points_shape[1];

    assert_eq!(n_dim, 3); //else ndarray to array3 is not allowed!
    assert_eq!(n_dim_points, 3); //else ndarray to array3 is not allowed!
                                 //
    let mut rhos = Array1::<f64>::zeros(n_x);

    let mut kdtree: float_KdTree<f64, usize, 3, 256, u32> = float_KdTree::with_capacity(n_points);
    for (idx, jvec) in points.axis_iter(Axis(0)).enumerate() {
        kdtree.add(ndarray_to_array3(jvec.to_slice().unwrap()), idx)
    }

    // Could chunk to seperate max distts!
    let max_dist2: f64 = (lamdaopt.max().unwrap()).powi(2);

    create_pool(n_threads).install(|| {
        Zip::from(x.axis_iter(Axis(0)))
            .and(&mut rhos)
            .into_par_iter()
            .for_each(|(x_row, rho)| {
                let neighbours = kdtree.within_unsorted::<SquaredEuclidean>(
                    ndarray_to_array3(x_row.to_slice().unwrap()),
                    max_dist2,
                );
                for neigh in neighbours {
                    let lamda = unsafe { *lamdaopt.uget(neigh.item) };
                    let w = unsafe { *weights.uget(neigh.item) };
                    let t_2 = neigh.distance / (lamda).powi(2);
                    if t_2 < 1. {
                        *rho += w * (1. - t_2) / (lamda.powi(n_dim as i32));
                    }
                }
            });
    });

    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    let constant_factor = (n_dim as f64 + 2.) / (2. * vd);
    // (1. / n_points as f64)
    rhos *= constant_factor;
    rhos
}


pub fn epanechnikov_kde_3d_rev_weights(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    n_threads: usize,
) -> Array1<f64> {
    let x_shape = x.shape();
    let n_x: usize = x_shape[0];
    let n_dim: usize = x_shape[1];
    let points_shape = points.shape();
    // let n_points: usize = points_shape[0];
    let n_dim_points: usize = points_shape[1];
    assert_eq!(n_dim, 3); //else ndarray to array3 is not allowed!
    assert_eq!(n_dim_points, 3); //else ndarray to array3 is not allowed!

    let mut rhos = Array1::<f64>::zeros(n_x);
    let lamdaopt_sigma2: Array1<f64> = lamdaopt.map(|&x| x * x);
    let w_inv_lamdaopt_pow: Array1<f64> = lamdaopt.map(|&x| x.powi(-(n_dim as i32))) * weights;

    let n_chunk: usize = std::cmp::max(std::cmp::min(n_x / n_threads, 50_000), 10_000);

    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_small)| {
                // let mut stars_kdtree: KdTree<f64, 3> = KdTree::with_capacity(n_chunk);
                let mut stars_kdtree: float_KdTree<f64, usize, 3, 256, u32> =
                    float_KdTree::with_capacity(n_chunk);
                for (idx, jvec) in x_small.axis_iter(Axis(0)).enumerate() {
                    stars_kdtree.add(ndarray_to_array3(jvec.to_slice().unwrap()), idx)
                }

                Zip::from(points.axis_iter(Axis(0)))
                    .and(&lamdaopt_sigma2)
                    .and(&w_inv_lamdaopt_pow)
                    .for_each(|p_row, lamda_s2, w_inv_lamda| {
                        let neighbours = stars_kdtree.within_unsorted::<SquaredEuclidean>(
                            ndarray_to_array3(p_row.to_slice().unwrap()),
                            *lamda_s2,
                        );
                        for neigh in neighbours {
                            let t_2 = neigh.distance / lamda_s2;
                            rhos_small[neigh.item] += (1. - t_2) * w_inv_lamda;
                        }
                    });
            });
    });

    //
    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    rhos *= (n_dim as f64 + 2.) / (2. * vd); //can
                                             // *(1. / n_points as f64)
    rhos
}

pub fn epanechnikov_kde_3d_rev_weights_groups(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    group_inds: ArrayView1<usize>,
    n_groups: usize,
    n_threads: usize,
) -> Array2<f64> {
    let x_shape = x.shape();
    let n_x: usize = x_shape[0];
    let n_dim: usize = x_shape[1];
    let points_shape = points.shape();
    // let n_points: usize = points_shape[0];
    let n_dim_points: usize = points_shape[1];
    assert_eq!(n_dim, 3); //else ndarray to array3 is not allowed!
    assert_eq!(n_dim_points, 3); //else ndarray to array3 is not allowed!

    let mut rhos_2d = Array2::<f64>::zeros((n_x, n_groups)); // C vs F array?
                                                                 //
    let lamdaopt_sigma2: Array1<f64> = lamdaopt.map(|&x| x * x);
    let w_inv_lamdaopt_pow: Array1<f64> = lamdaopt.map(|&x| x.powi(-(n_dim as i32))) * weights;

    let n_chunk: usize = std::cmp::max(std::cmp::min(n_x / n_threads, 50_000), 10_000);

    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos_2d.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_2d_small)| {
                let mut stars_kdtree: float_KdTree<f64, usize, 3, 256, u32> =
                    float_KdTree::with_capacity(n_chunk);
                for (idx, jvec) in x_small.axis_iter(Axis(0)).enumerate() {
                    stars_kdtree.add(new_ndarray_to_array3(jvec), idx)
                }

                Zip::from(points.axis_iter(Axis(0)))
                    .and(&lamdaopt_sigma2)
                    .and(&w_inv_lamdaopt_pow)
                    .and(&group_inds)
                    .for_each(|p_row, lamda_s2, w_inv_lamda, g_ind| {
                        let neighbours = stars_kdtree.within_unsorted::<SquaredEuclidean>(
                            new_ndarray_to_array3(p_row),
                            *lamda_s2,
                        );
                        for neigh in neighbours {
                            let t_2 = neigh.distance / lamda_s2;
                            unsafe {
                                *rhos_2d_small.uget_mut((neigh.item, *g_ind)) +=
                                    (1. - t_2) * w_inv_lamda;
                            }
                        }
                    });
            });
    });

    //
    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    rhos_2d *= (n_dim as f64 + 2.) / (2. * vd); //can
                                                // *(1. / n_points as f64)
    rhos_2d
}

pub fn epanechnikov_kde_3d_rev_weights_multi(
    x: ArrayView2<f64>,
    multi_points: Vec<ArrayView2<f64>>,
    multi_lamdaopt: Vec<ArrayView1<f64>>,
    multi_weights: Vec<ArrayView1<f64>>,
    n_threads: usize,
) -> Array2<f64> {
    let x_shape = x.shape();
    let n_x: usize = x_shape[0];
    let n_dim: usize = x_shape[1];
    assert_eq!(n_dim, 3); //else ndarray to array3 is not allowed!

    let n_groups = multi_points.len();

    let mut rhos_2d = Array2::<f64>::zeros((n_x, n_groups)); // C vs F array?

    let mut multi_lamdaopt_sigma2: Vec<Array1<f64>> = Vec::new();
    let mut multi_w_inv_lamdaopt_pow: Vec<Array1<f64>> = Vec::new();

    for (points, lamdaopt, weights) in izip!(
        multi_points.iter(),
        multi_lamdaopt.iter(),
        multi_weights.iter(),
    ) {
        let points_shape = points.shape();
        let n_dim_points: usize = points_shape[1];
        assert_eq!(n_dim_points, 3); //else ndarray to array3 is not allowed!
        multi_lamdaopt_sigma2.push(lamdaopt.map(|&x| x * x));
        multi_w_inv_lamdaopt_pow.push(lamdaopt.map(|&x| x.powi(-(n_dim as i32))) * weights);
    }

    let n_chunk: usize = std::cmp::max(std::cmp::min(n_x / n_threads, 50_000), 10_000);

    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos_2d.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_2d_small)| {
                let mut stars_kdtree: float_KdTree<f64, usize, 3, 256, u32> =
                    float_KdTree::with_capacity(n_chunk);
                for (idx, jvec) in x_small.axis_iter(Axis(0)).enumerate() {
                    stars_kdtree.add(ndarray_to_array3(jvec.to_slice().unwrap()), idx)
                }

                for (mut rhos_small, points, lamdaopt_sigma2, w_inv_lamdaopt_pow) in izip!(
                    rhos_2d_small.axis_iter_mut(Axis(1)),
                    multi_points.iter(),
                    multi_lamdaopt_sigma2.iter(),
                    multi_w_inv_lamdaopt_pow.iter(),
                ) {
                    Zip::from(points.axis_iter(Axis(0)))
                        .and(lamdaopt_sigma2)
                        .and(w_inv_lamdaopt_pow)
                        .for_each(|p_row, lamda_s2, w_inv_lamda| {
                            let neighbours = stars_kdtree.within_unsorted::<SquaredEuclidean>(
                                ndarray_to_array3(p_row.to_slice().unwrap()),
                                *lamda_s2,
                            );
                            for neigh in neighbours {
                                let t_2 = neigh.distance / lamda_s2;
                                // rhos_small[neigh.item] += (1. - t_2) * w_inv_lamda;
                                // Not much difference!
                                unsafe {
                                    *rhos_small.uget_mut(neigh.item) += (1. - t_2) * w_inv_lamda;
                                }
                            }
                        });
                }
            });
    });

    //
    let vd = PI.powf(n_dim as f64 / 2.) / gamma::gamma(n_dim as f64 / 2. + 1.);
    rhos_2d *= (n_dim as f64 + 2.) / (2. * vd); //can
    rhos_2d
} */
