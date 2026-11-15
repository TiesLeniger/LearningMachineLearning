import numpy as np
import torch

from check_env import make_plot, seeded_sample, torch_smoke, versions


def test_python_version():
    assert versions()["python"].startswith("3.12")


def test_torch_is_cpu_build():
    # CPU wheels carry a "+cpu" local version tag on Windows/Linux.
    assert "+cpu" in torch.__version__ or not torch.cuda.is_available()


def test_seed_reproducible():
    np_a, t_a = seeded_sample(42)
    np_b, t_b = seeded_sample(42)
    np.testing.assert_array_equal(np_a, np_b)
    assert torch.equal(t_a, t_b)


def test_seed_changes_output():
    np_a, _ = seeded_sample(1)
    np_b, _ = seeded_sample(2)
    assert not np.array_equal(np_a, np_b)


def test_autograd_matches_analytic():
    assert torch_smoke() < 1e-12


def test_plot_written(tmp_path):
    path = make_plot(tmp_path / "smoke.png")
    assert path.exists() and path.stat().st_size > 1000
