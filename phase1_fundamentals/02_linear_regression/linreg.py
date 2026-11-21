"""Linear regression with a polynomial basis: ML closed form, ridge, and LMS.

Notation follows B&B §4.1 (see notes/phase1-linear-regression.md):
    N  data points, M basis functions (polynomial order M - 1), phi_0 = 1.
    Phi (N, M) design matrix, Phi[n, j] = phi_j(x_n).

Implement every function below. Do not use np.polyfit or sklearn.
"""

import numpy as np


def poly_features(x: np.ndarray, M: int) -> np.ndarray:
    """Polynomial design matrix.

    Args:
        x: (N,) inputs.
        M: number of basis functions (>= 1).

    Returns:
        Phi: (N, M), column j = x**j for j = 0..M-1.
    """
    raise NotImplementedError


def fit_ml(Phi: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Maximum-likelihood (least-squares) weights.

    Must not form an explicit matrix inverse.

    Args:
        Phi: (N, M) design matrix.
        t: (N,) targets.

    Returns:
        w: (M,) weights minimising 0.5 * ||t - Phi w||^2.
    """
    raise NotImplementedError


def fit_ridge(Phi: np.ndarray, t: np.ndarray, lam: float) -> np.ndarray:
    """Regularised least-squares (ridge / MAP) weights.

    Args:
        Phi: (N, M) design matrix.
        t: (N,) targets.
        lam: regularisation coefficient lambda > 0 (penalises all weights, incl. w_0).

    Returns:
        w: (M,) weights minimising 0.5 * ||t - Phi w||^2 + 0.5 * lam * ||w||^2.
    """
    raise NotImplementedError


def noise_var(Phi: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    """ML estimate of the noise variance 1 / beta_ML.

    Args:
        Phi: (N, M) design matrix.
        t: (N,) targets.
        w: (M,) fitted weights.

    Returns:
        Mean squared residual (scalar).
    """
    raise NotImplementedError


def rms(Phi: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    """Root-mean-square error E_RMS = sqrt(mean((Phi w - t)^2)).

    Args:
        Phi: (N, M) design matrix.
        t: (N,) targets.
        w: (M,) weights.

    Returns:
        E_RMS (scalar).
    """
    raise NotImplementedError


def fit_lms(
    Phi: np.ndarray,
    t: np.ndarray,
    lr: float,
    n_epochs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Least-mean-squares (sequential SGD) on the sum-of-squares error.

    Start from w = 0. Each epoch visits every data point once, in an order
    shuffled with `rng`, and applies one update per point.

    Args:
        Phi: (N, M) design matrix.
        t: (N,) targets.
        lr: learning rate eta.
        n_epochs: number of passes over the data.
        rng: random generator for the shuffling.

    Returns:
        w: (M,) weights after training. May contain inf/nan if lr is too large.
    """
    raise NotImplementedError
