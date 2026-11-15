"""Environment smoke check: versions, torch matmul + autograd, plot to file.

Run:  uv run python phase0_setup/01_environment/check_env.py
"""

from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open windows
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

RESULTS = Path(__file__).parent / "results"


def versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
        "torch_threads": str(torch.get_num_threads()),
        "cuda_available": str(torch.cuda.is_available()),
    }


def seeded_sample(seed: int, n: int = 5) -> tuple[np.ndarray, torch.Tensor]:
    """Draw n numbers from NumPy and torch with a fixed seed."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    return rng.standard_normal(n), torch.randn(n)


def torch_smoke(seed: int = 0) -> float:
    """Matmul vs NumPy, and autograd of f(x) = sum(x^3) vs analytic 3x^2.

    Returns the max absolute gradient error.
    """
    torch.manual_seed(seed)
    a = torch.randn(64, 32, dtype=torch.float64)
    b = torch.randn(32, 16, dtype=torch.float64)
    np.testing.assert_allclose((a @ b).numpy(), a.numpy() @ b.numpy(), rtol=1e-12)

    x = torch.randn(10, dtype=torch.float64, requires_grad=True)
    (x**3).sum().backward()
    return (x.grad - 3 * x.detach() ** 2).abs().max().item()


def make_plot(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 2 * np.pi, 200)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(x, np.sin(x), label="sin(x)")
    ax.set_xlabel("x")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for k, v in versions().items():
        print(f"{k:>15}: {v}")
    print(f"{'autograd err':>15}: {torch_smoke():.2e}")
    print(f"{'plot':>15}: {make_plot(RESULTS / 'smoke.png')}")
