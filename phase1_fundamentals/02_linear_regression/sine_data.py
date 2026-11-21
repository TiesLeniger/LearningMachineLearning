"""Synthetic data for the B&B curve-fitting example: t = sin(2*pi*x) + Gaussian noise."""

import numpy as np


def true_function(x: np.ndarray) -> np.ndarray:
    """Noise-free target sin(2*pi*x)."""
    return np.sin(2 * np.pi * x)


def make_sine_data(
    N: int, noise_std: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw N points with x ~ U[0, 1] and t = sin(2*pi*x) + N(0, noise_std^2).

    Returns:
        x: (N,) inputs, sorted ascending (convenient for plotting).
        t: (N,) noisy targets.
    """
    x = np.sort(rng.uniform(0.0, 1.0, size=N))
    t = true_function(x) + rng.normal(0.0, noise_std, size=N)
    return x, t
