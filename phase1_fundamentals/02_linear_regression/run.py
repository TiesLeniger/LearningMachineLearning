"""Exercise 1.2 experiment: reproduce B&B's polynomial overfitting example.

Usage:
    uv run python phase1_fundamentals/02_linear_regression/run.py

Writes results/fits.png and results/rms_vs_M.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from linreg import fit_ml, poly_features, rms  # noqa: F401  (used in your loop)
from sine_data import make_sine_data, true_function  # noqa: F401

SEED = 0
N_TRAIN = 10
N_TEST = 100
NOISE_STD = 0.3
M_PLOT = [1, 2, 4, 10]  # number of basis functions (order M - 1)
M_RANGE = list(range(1, 11))

RESULTS = Path(__file__).parent / "results"


# --- plotting (scaffolding) --------------------------------------------------

def plot_fits(x, t, fits, path):
    """Grid of fitted curves.

    Args:
        x, t: (N,) training data.
        fits: dict {M: (x_grid (G,), y_grid (G,))} with the fitted curve per M.
        path: output file.
    """
    fig, axes = plt.subplots(1, len(fits), figsize=(3.2 * len(fits), 3), sharey=True)
    x_true = np.linspace(0, 1, 200)
    for ax, (M, (xg, yg)) in zip(np.atleast_1d(axes), fits.items()):
        ax.plot(x_true, true_function(x_true), "g-", lw=1, label=r"$\sin 2\pi x$")
        ax.plot(xg, yg, "r-", lw=1.5, label="fit")
        ax.plot(x, t, "o", mfc="none", mec="b", label="train")
        ax.set_title(f"M = {M} (order {M - 1})")
        ax.set_xlabel("x")
        ax.set_ylim(-1.5, 1.5)
    np.atleast_1d(axes)[0].set_ylabel("t")
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rms(M_values, rms_train, rms_test, path):
    """Train and test E_RMS against number of basis functions M."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(M_values, rms_train, "o-", mfc="none", label="train")
    ax.plot(M_values, rms_test, "s-", mfc="none", label="test")
    ax.set_xlabel("M (number of basis functions)")
    ax.set_ylabel(r"$E_{\mathrm{RMS}}$")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- experiment (Ties) -------------------------------------------------------

def main():
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    x_train, t_train = make_sine_data(N_TRAIN, NOISE_STD, rng)
    x_test, t_test = make_sine_data(N_TEST, NOISE_STD, rng)

    # TODO(Ties):
    # 1. For each M in M_PLOT: fit with fit_ml, evaluate the fit on a dense grid
    #    x_grid = np.linspace(0, 1, 200), collect into fits = {M: (x_grid, y_grid)}.
    # 2. For each M in M_RANGE: fit on the training set, compute train and test E_RMS.
    # 3. Call plot_fits(...) and plot_rms(...) with RESULTS / "fits.png" and
    #    RESULTS / "rms_vs_M.png"; print a small table of M, train RMS, test RMS.
    raise NotImplementedError


if __name__ == "__main__":
    main()
