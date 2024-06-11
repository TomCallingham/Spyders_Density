import numpy as np
from .epanechnikov import epanechnikov_kde


class MBEdens:
    """
    Creates a density estimator from a set of points X, with optional weights w.
    Using a Modified Breiman Density Estimator with a variable Epanechnikov kernel.
    Based on: https://ui.adsabs.harvard.edu/abs/2011A%26A...531A.114F/abstract
    ----
    Parameters (numpy arrays)
    ----
    X [N,n_dim]: Points to evaluate density at.
    weights [N] (optional): The weighting of each background point.
    n_iter (optional,5): Iterations to find smoothing windows.
    p_window (optional,20): percentage window to find scaling.
    n_threads (optional,20): Number of threads to use.
    verbose (optional): Print iteration loops.
    """

    def __init__(
        self,
        X: np.ndarray,
        weights: np.ndarray | None = None,
        n_iter: int = 5,
        p_window: float = 20,
        sigma_type="min",
        n_threads: int = 20,
        verbose: bool = False,
    ):
        self.n_threads = n_threads
        self.n_iter = n_iter
        self.p_window = p_window
        self.verbose = verbose

        self.weights = None

        self.n_points, self.ndim = X.shape
        self.points = X

        alpha = None
        self.alpha = 1.0 / self.ndim if alpha is None else alpha

        self.sigma_type = sigma_type
        self.sigmaopt = self.calc_sigma(X)

        self.calc_lambdaopt()

        # find smoothing first, set weights after!
        if weights is not None:
            assert len(weights) == X.shape[0]
            self.weights = weights.astype(np.float64)

    def calc_lambdaopt(self) -> None:
        self.lambdaopt = np.ones(self.n_points, dtype=float)
        if self.verbose:
            print(f"Iterating {self.n_iter} to find density params")
        for i in range(self.n_iter):
            if self.verbose:
                print(f"Iterating to find density: {i+1}/{self.n_iter}")
            pilot_rho = self.find_dens(self.points)
            g = np.exp(np.mean(np.log(pilot_rho)))
            new_lambdaopt = (pilot_rho / g) ** -self.alpha
            if self.verbose:
                print("med diff:", np.median(np.abs(new_lambdaopt - self.lambdaopt)))
                print(f" min labdopt: {np.min(new_lambdaopt)}, max {np.max(new_lambdaopt)}")
            self.lambdaopt = new_lambdaopt

    def calc_sigma(self, X) -> float:
        if X.shape[0] == 0:
            return 1

        P = np.percentile(X, np.array([self.p_window, 100 - self.p_window]), axis=0)
        sigmas = (P[1] - P[0]) / np.log(self.n_points)
        if self.sigma_type == "geo":
            sigma = np.exp(np.mean(np.log(sigmas)))
        elif self.sigma_type == "min":
            sigma = np.min(sigmas)
        elif self.sigma_type == "max":
            sigma = np.max(sigmas)
        else:
            raise KeyError(f"sigma type not recognised : {self.sigma_type}")
        if self.verbose:
            print(f"Sigma used {self.sigma_type}: {sigma}")
        return sigma

    def find_dens(self, X) -> np.ndarray:
        """
        Parameters (numpy arrays)
        x [N,n_dim]: Points to evaluate density at.
        ----
        Output (numpy arrays)
        Density evaluated at points
        """
        assert X.shape[1] == self.ndim
        dens = epanechnikov_kde(X, self.points, self.lambdaopt * self.sigmaopt, self.weights, self.n_threads)
        return dens
