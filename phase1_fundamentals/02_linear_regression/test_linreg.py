"""Acceptance tests for exercise 1.2 (linear regression). See README.md for the spec."""

import numpy as np
import pytest

from linreg import fit_lms, fit_ml, fit_ridge, noise_var, poly_features, rms
from sine_data import make_sine_data

SEED = 0


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def noiseless(rng):
    """Targets exactly in the column space of Phi: t = Phi w_true."""
    x = np.linspace(-1.0, 1.0, 20)
    M = 4
    w_true = rng.normal(size=M)
    Phi = poly_features(x, M)
    return Phi, Phi @ w_true, w_true


@pytest.fixture
def noisy(rng):
    x, t = make_sine_data(N=20, noise_std=0.3, rng=rng)
    return poly_features(x, 4), t


# --- poly_features ---------------------------------------------------------

def test_poly_features_shape_and_values():
    x = np.array([0.0, 0.5, 2.0])
    Phi = poly_features(x, 3)
    assert Phi.shape == (3, 3)
    expected = np.array([[1.0, 0.0, 0.0], [1.0, 0.5, 0.25], [1.0, 2.0, 4.0]])
    np.testing.assert_allclose(Phi, expected)


def test_poly_features_bias_column():
    Phi = poly_features(np.linspace(0, 1, 7), 1)
    assert Phi.shape == (7, 1)
    np.testing.assert_array_equal(Phi, 1.0)


# --- fit_ml ------------------------------------------------------------------

def test_fit_ml_exact_recovery(noiseless, monkeypatch):
    def no_inverse(*args, **kwargs):
        raise AssertionError("fit_ml must not form an explicit inverse")

    monkeypatch.setattr(np.linalg, "inv", no_inverse)
    Phi, t, w_true = noiseless
    w = fit_ml(Phi, t)
    assert w.shape == w_true.shape
    np.testing.assert_allclose(w, w_true, atol=1e-10)


def test_fit_ml_residual_orthogonal_to_columns(noisy):
    Phi, t = noisy
    r = t - Phi @ fit_ml(Phi, t)
    np.testing.assert_allclose(Phi.T @ r, 0.0, atol=1e-10)


def test_fit_ml_residuals_sum_to_zero(noisy):
    Phi, t = noisy
    r = t - Phi @ fit_ml(Phi, t)
    assert abs(r.sum()) < 1e-10


# --- fit_ridge ---------------------------------------------------------------

def test_ridge_small_lambda_matches_ml(noisy):
    Phi, t = noisy
    np.testing.assert_allclose(fit_ridge(Phi, t, 1e-12), fit_ml(Phi, t), atol=1e-6)


def test_ridge_norm_decreases_with_lambda(noisy):
    Phi, t = noisy
    norms = [np.linalg.norm(fit_ridge(Phi, t, lam)) for lam in np.logspace(-4, 3, 8)]
    assert all(a > b for a, b in zip(norms, norms[1:]))


def test_ridge_stationarity(noisy):
    # Gradient of the regularised error is zero: Phi^T (t - Phi w) = lam * w.
    Phi, t = noisy
    lam = 0.1
    w = fit_ridge(Phi, t, lam)
    np.testing.assert_allclose(Phi.T @ (t - Phi @ w), lam * w, atol=1e-10)


def test_ridge_more_basis_functions_than_points(rng):
    x, t = make_sine_data(N=5, noise_std=0.1, rng=rng)
    Phi = poly_features(x, 10)  # M > N: Phi^T Phi is singular
    lam = 1e-3
    w = fit_ridge(Phi, t, lam)
    assert w.shape == (10,)
    assert np.all(np.isfinite(w))
    np.testing.assert_allclose(Phi.T @ (t - Phi @ w), lam * w, atol=1e-8)


# --- noise_var and rms -------------------------------------------------------

def test_noise_var_recovers_true_variance(rng):
    sigma = 0.3
    x, t = make_sine_data(N=10_000, noise_std=sigma, rng=rng)
    Phi = poly_features(x, 6)  # order 5: captures sin(2 pi x) on [0, 1] well
    v = noise_var(Phi, t, fit_ml(Phi, t))
    assert v == pytest.approx(sigma**2, rel=0.05)


def test_rms_known_value():
    Phi = np.ones((4, 1))
    t = np.array([1.0, 3.0, 1.0, 3.0])
    assert rms(Phi, t, np.array([2.0])) == pytest.approx(1.0)
    assert rms(Phi, t, np.array([1.0])) == pytest.approx(np.sqrt(2.0))


# --- fit_lms -----------------------------------------------------------------

@pytest.fixture
def lms_problem(rng):
    x = np.linspace(-1.0, 1.0, 50)
    Phi = poly_features(x, 3)
    t = Phi @ rng.normal(size=3)  # noiseless: SGD can converge exactly
    norm2 = np.sum(Phi**2, axis=1)  # ||phi_n||^2 per data point
    return Phi, t, norm2.min(), norm2.max()


def test_lms_converges_to_ml(lms_problem, rng):
    Phi, t, min_norm2, max_norm2 = lms_problem
    w = fit_lms(Phi, t, lr=0.1 / max_norm2, n_epochs=500, rng=rng)
    np.testing.assert_allclose(w, fit_ml(Phi, t), atol=1e-3)


def test_lms_is_reproducible(lms_problem):
    Phi, t, min_norm2, max_norm2 = lms_problem
    w_a = fit_lms(Phi, t, lr=0.1 / max_norm2, n_epochs=5, rng=np.random.default_rng(1))
    w_b = fit_lms(Phi, t, lr=0.1 / max_norm2, n_epochs=5, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(w_a, w_b)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_lms_diverges_for_large_learning_rate(lms_problem, rng):
    Phi, t, min_norm2, max_norm2 = lms_problem
    # lr > 2 / ||phi_n||^2 for every n: each update amplifies its own residual.
    w = fit_lms(Phi, t, lr=2.5 / min_norm2, n_epochs=200, rng=rng)
    assert not np.all(np.isfinite(w)) or np.linalg.norm(w) > 1e6
