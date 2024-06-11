from .spyders import epanechnikov_kde_rs
import numpy as np


def epanechnikov_kde(x: np.ndarray, points: np.ndarray, lambdaopt: np.ndarray, n_threads=8, n_chunk=None) -> np.ndarray:
    n_x = x.shape[0]
    n_chunk = n_chunk if n_chunk is not None else np.max(np.min([n_x / n_threads, 50_000]), 10_000)
    dens = epanechnikov_kde_rs(x, points, lambdaopt, n_threads, n_chunk)
    return dens
