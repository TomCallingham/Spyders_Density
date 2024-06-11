// use itertools::izip;
use kiddo::float::kdtree::KdTree as float_KdTree;
use kiddo::SquaredEuclidean;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
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
    n_chunk: usize,
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
                        let inv_l2 = 1./ lamda2;
                        for neigh in neighbours {
                            let t_2 = neigh.distance *inv_l2;
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

pub fn epanechnikov_kde_weights<const N_DIM: usize>(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    n_threads: usize,
    n_chunk: usize,
) -> Array1<f64> {
    let x_shape = x.shape();
    let n_x: usize = x_shape[0];
    let n_dim_x: usize = x_shape[1];
    let n_dim_points: usize = points.shape()[1];
    assert_eq!(n_dim_x, N_DIM);
    assert_eq!(n_dim_points, N_DIM);

    let mut rhos = Array1::<f64>::zeros(n_x);
    let lamdaopt2: Array1<f64> = lamdaopt.map(|&x| x * x);
    let w_inv_lamdaopt_pow: Array1<f64> = lamdaopt.map(|&x| x.powi(-(N_DIM as i32))) * weights;

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
                    .and(&w_inv_lamdaopt_pow)
                    .for_each(|p_row, lamda2, w_inv_lamda| {
                        let neighbours = stars_kdtree.within_unsorted::<SquaredEuclidean>(
                            unsafe { &*(p_row.as_ptr() as *const [f64; N_DIM]) },
                            *lamda2,
                        );

                        let inv_l2 = 1./ lamda2;
                        for neigh in neighbours {
                            let t_2 = neigh.distance*inv_l2;
                            rhos_small[neigh.item] += (1. - t_2) * w_inv_lamda;
                        }
                    });
            });
    });

    //
    let vd = PI.powf(N_DIM as f64 / 2.) / gamma::gamma(N_DIM as f64 / 2. + 1.);
    rhos *= (N_DIM as f64 + 2.) / (2. * vd);
    rhos
}

pub fn epanechnikov_kde_groups<const N_DIM: usize>(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    group_inds: ArrayView1<usize>,
    n_groups: usize,
    n_threads: usize,
    n_chunk: usize,
) -> Array2<f64> {
    //
    let x_shape = x.shape();
    let n_x: usize = x_shape[0];
    let n_dim_x: usize = x_shape[1];
    let n_dim_points: usize = points.shape()[1];
    assert_eq!(n_dim_x, N_DIM);
    assert_eq!(n_dim_points, N_DIM);

    let mut rhos_2d = Array2::<f64>::zeros((n_x, n_groups)); // C vs F array?
                                                                 //
    let lamdaopt2: Array1<f64> = lamdaopt.map(|&x| x * x);
    let inv_lamdaopt_pow: Array1<f64> = lamdaopt.map(|&x| x.powi(-(N_DIM as i32)));

    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos_2d.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_2d_small)| {
                let mut stars_kdtree: float_KdTree<f64, usize, N_DIM, 256, u32> =
                    float_KdTree::with_capacity(n_chunk);

                for (idx, jvec) in x_small.axis_iter(Axis(0)).enumerate() {
                    stars_kdtree.add(unsafe { &*(jvec.as_ptr() as *const [f64; N_DIM]) }, idx)
                }

                Zip::from(points.axis_iter(Axis(0)))
                    .and(&lamdaopt2)
                    .and(&inv_lamdaopt_pow)
                    .and(&group_inds)
                    .for_each(|p_row, lamda2, inv_lamda, g_ind| {
                        let neighbours = stars_kdtree.within_unsorted::<SquaredEuclidean>(
                            unsafe { &*(p_row.as_ptr() as *const [f64; N_DIM]) },
                            *lamda2,
                        );
                        let inv_l2 = 1./ lamda2;
                        for neigh in neighbours {
                            let t_2 = neigh.distance*inv_l2;
                            unsafe {
                                *rhos_2d_small.uget_mut((neigh.item, *g_ind)) +=
                                    (1. - t_2) * inv_lamda;
                            }
                        }
                    });
            });
    });

    //
    let vd = PI.powf(N_DIM as f64 / 2.) / gamma::gamma(N_DIM as f64 / 2. + 1.);
    rhos_2d *= (N_DIM as f64 + 2.) / (2. * vd); 
    rhos_2d
}

pub fn epanechnikov_kde_weights_groups<const N_DIM: usize>(
    x: ArrayView2<f64>,
    points: ArrayView2<f64>,
    lamdaopt: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    group_inds: ArrayView1<usize>,
    n_groups: usize,
    n_threads: usize,
    n_chunk: usize,
) -> Array2<f64> {
    //
    let x_shape = x.shape();
    let n_x: usize = x_shape[0];
    let n_dim_x: usize = x_shape[1];
    let n_dim_points: usize = points.shape()[1];
    assert_eq!(n_dim_x, N_DIM);
    assert_eq!(n_dim_points, N_DIM);

    let mut rhos_2d = Array2::<f64>::zeros((n_x, n_groups)); // C vs F array?
                                                                 //
    let lamdaopt2: Array1<f64> = lamdaopt.map(|&x| x * x);
    let w_inv_lamdaopt_pow: Array1<f64> = lamdaopt.map(|&x| x.powi(-(N_DIM as i32))) * weights;

    create_pool(n_threads).install(|| {
        x.axis_chunks_iter(Axis(0), n_chunk)
            .into_par_iter()
            .zip(rhos_2d.axis_chunks_iter_mut(Axis(0), n_chunk))
            .for_each(|(x_small, mut rhos_2d_small)| {
                let mut stars_kdtree: float_KdTree<f64, usize, N_DIM, 256, u32> =
                    float_KdTree::with_capacity(n_chunk);

                for (idx, jvec) in x_small.axis_iter(Axis(0)).enumerate() {
                    stars_kdtree.add(unsafe { &*(jvec.as_ptr() as *const [f64; N_DIM]) }, idx)
                }

                Zip::from(points.axis_iter(Axis(0)))
                    .and(&lamdaopt2)
                    .and(&w_inv_lamdaopt_pow)
                    .and(&group_inds)
                    .for_each(|p_row, lamda2, w_inv_lamda, g_ind| {
                        let neighbours = stars_kdtree.within_unsorted::<SquaredEuclidean>(
                            unsafe { &*(p_row.as_ptr() as *const [f64; N_DIM]) },
                            *lamda2,
                        );
                        let inv_l2 = 1./ lamda2;
                        for neigh in neighbours {
                            let t_2 = neigh.distance*inv_l2;
                            unsafe {
                                *rhos_2d_small.uget_mut((neigh.item, *g_ind)) +=
                                    (1. - t_2) * w_inv_lamda;
                            }
                        }
                    });
            });
    });

    //
    let vd = PI.powf(N_DIM as f64 / 2.) / gamma::gamma(N_DIM as f64 / 2. + 1.);
    rhos_2d *= (N_DIM as f64 + 2.) / (2. * vd); 
    rhos_2d
}

