# 0.1 Python environment

**What:** Reproducible Python 3.12 environment managed by `uv`: NumPy, SciPy, matplotlib, CPU-only PyTorch, pytest.

**Why:** One lockfile (`uv.lock`) pins every version, so laptop and tower get identical environments. CPU wheels keep the install small; the tower can switch to a CUDA index later.

## Usage
```
uv sync                                                   # create .venv from pyproject.toml + uv.lock
uv run pytest                                             # run all tests
uv run python phase0_setup/01_environment/check_env.py    # print versions, write results/smoke.png
uv add <pkg>                                              # add a dependency (updates lockfile)
```

## Conventions
- Test files need unique names repo-wide (`test_<exercise>.py`): pytest's default import mode clashes on duplicate basenames. Same for helper modules imported by tests.
- Plots use the `Agg` backend and go to `<exercise>/results/`.

## Result
Verified 2026-11-15 on the laptop (i7, CPU only).

| Package    | Version     |
|------------|-------------|
| Python     | 3.12.16     |
| NumPy      | 2.5.4       |
| SciPy      | 1.18.1      |
| matplotlib | 3.11.2      |
| PyTorch    | 2.15.0+cpu  |

- `uv run pytest`: 6 passed
- Autograd check (d/dx Σx³ vs 3x²): max error 0.0
- Smoke plot: `results/smoke.png`
