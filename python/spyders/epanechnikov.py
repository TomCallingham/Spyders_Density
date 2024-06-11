from .spyders import (
    epanechnikov_kde_rs,
    epanechnikov_kde_weights_rs,
    epanechnikov_kde_groups_rs,
    epanechnikov_kde_weights_groups_rs,
)
import numpy as np


def epanechnikov_kde(
    x: np.ndarray,
    points: np.ndarray,
    lambdaopt: np.ndarray,
    weights: np.ndarray | None = None,
    n_threads: int = 8,
    n_chunk: int | None = None,
) -> np.ndarray:
    """
    Epanechnikov Kernel Density Estimator
    ----
    Parameters (numpy arrays)
    ----
    x [N,n_dim]: Points to evaluate density at.
    points [N,n_dim]: Background density points.
    lambdaopt [N]: The smoothing window for each background point.
    weights [N] (optional): The weighting of each background point.
    n_threads: Number of threads to use.
    n_chunk: (optional) Number of points used by each thread.
    ----
    Output (numpy arrays)
    ----
    Density evaluated at points

    """

    n_x = x.shape[0]
    n_chunk = n_chunk if n_chunk is not None else estimate_n_chunk(n_x, n_threads)
    if weights is None:
        dens = epanechnikov_kde_rs(x, points, lambdaopt, n_threads, n_chunk)
    else:
        dens = epanechnikov_kde_weights_rs(x, points, lambdaopt, weights, n_threads, n_chunk)
    return dens


def epanechnikov_kde_groups(
    x: np.ndarray,
    points: np.ndarray,
    lambdaopt: np.ndarray,
    weights: np.ndarray | None = None,
    n_threads: int = 8,
    n_chunk: int | None = None,
) -> np.ndarray:
    """
    Epanechnikov Kernel Density Estimator
    ----
    Parameters (numpy arrays)
    ----
    x [N,n_dim]: Points to evaluate density at.
    points [N,n_dim]: Background density points.
    lambdaopt [N]: The smoothing window for each background point.
    weights [N] (optional): The weighting of each background point.
    n_threads: Number of threads to use.
    n_chunk: (optional) Number of points used by each thread.
    ----
    Output (numpy arrays)
    ----
    Density evaluated at points

    """

    n_x = x.shape[0]
    n_chunk = n_chunk if n_chunk is not None else estimate_n_chunk(n_x, n_threads)
    if weights is None:
        dens = epanechnikov_kde_groups_rs(x, points, lambdaopt, n_threads, n_chunk)
    else:
        dens = epanechnikov_kde_weights_groups_rs(x, points, lambdaopt, weights, n_threads, n_chunk)
    return dens


def estimate_n_chunk(n_x: int, n_threads: int) -> int:
    n_chunk = np.max([np.min([int(n_x / n_threads), 50_000]), 10_000])
    return n_chunk
