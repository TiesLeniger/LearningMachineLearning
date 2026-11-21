# 1.2 Linear regression

**What:** Polynomial-basis linear regression in NumPy, three ways: maximum likelihood (closed form), ridge (regularized closed form) and LMS (sequential SGD). Reproduces B&B's overfitting experiment: $t = \sin(2\pi x) + \epsilon$, $N = 10$.

**Why:** The closed-form solution, the projection geometry, and the link between regularization and conditioning come back in every later model (GPs at 1.10, polynomial response surfaces in MDAO). LMS is the simplest case of the SGD used to train neural nets.

Theory: [notes/phase1-linear-regression.md](../../notes/phase1-linear-regression.md) (B&B §4.1).

## Spec

**`linreg.py`** (Ties writes):

| Function | Input → Output |
|---|---|
| `poly_features(x, M)` | `x (N,)` → `Phi (N, M)`, columns $x^0, \dots, x^{M-1}$ |
| `fit_ml(Phi, t)` | → `w (M,)`, least squares without an explicit inverse |
| `fit_ridge(Phi, t, lam)` | → `w (M,)` |
| `noise_var(Phi, t, w)` | → scalar $1/\beta_{\text{ML}}$ |
| `rms(Phi, t, w)` | → scalar $E_{\text{RMS}}$ |
| `fit_lms(Phi, t, lr, n_epochs, rng)` | → `w (M,)`; start at 0, shuffle each epoch with `rng` |

$M$ is the number of basis functions (B&B §4.1 convention); polynomial order is $M - 1$. No `np.polyfit` or scikit-learn.

**Acceptance tests** (`test_linreg.py`):
1. Exact recovery: noiseless data from a known `w` → `fit_ml` returns it to `1e-10`, without `np.linalg.inv`.
2. Orthogonality: $\boldsymbol{\Phi}^T(\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}) = \mathbf{0}$; residuals sum to 0.
3. Ridge: `lam=1e-12` matches ML; $\|\mathbf{w}\|$ decreases with $\lambda$; stationarity $\boldsymbol{\Phi}^T(\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}) = \lambda\mathbf{w}$; finite solution for $M > N$.
4. `noise_var` within 5% of the true $\sigma^2$ for $N = 10^4$.
5. LMS: converges to `fit_ml` (`1e-3`) for `lr = 0.1/max‖φ_n‖²`; reproducible with a fixed seed; diverges for `lr = 2.5/min‖φ_n‖²`.

**Deliverables:** `run.py` (experiment loop by Ties) writes `results/fits.png` (fits for $M \in \{1, 2, 4, 10\}$) and `results/rms_vs_M.png` (train/test $E_{\text{RMS}}$ vs. $M$).

## Usage
```
uv run pytest phase1_fundamentals/02_linear_regression
uv run python phase1_fundamentals/02_linear_regression/run.py
```

## Result
<!-- Fill in after running: figures, RMS table, what you observe (where does test error turn up? what happens to |w| at M = 10?). -->
