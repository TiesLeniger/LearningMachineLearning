# 1.2 Linear regression

**Source:** B&B §4.1, pp. a–b · Date: 2026-11-21

## Core idea
A linear model is linear in the *parameters*, not in the input: $y = \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})$ with fixed nonlinear basis functions $\phi_j$. Under Gaussian noise, maximum likelihood is least squares, which has a closed-form solution: the orthogonal projection of the target vector onto the column space of the design matrix. Regularization adds $\lambda\mathbf{I}$, which both reduces overfitting and makes the problem well conditioned. For large or streaming data, the same objective is minimized by sequential gradient steps (LMS).

## Math

Model, $\phi_0(\mathbf{x}) = 1$ (bias):

$$
y(\mathbf{x}, \mathbf{w}) = \sum_{j=0}^{M-1} w_j\,\phi_j(\mathbf{x}) = \mathbf{w}^T\boldsymbol{\phi}(\mathbf{x})
$$

Design matrix, shape $(N, M)$: row $n$ = data point, column $j$ = basis function.

$$
\Phi_{nj} = \phi_j(\mathbf{x}_n)
$$

Objective and gradient:

$$
E(\mathbf{w}) = \tfrac12\|\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}\|^2, \qquad
\nabla_{\mathbf{w}} E = -\boldsymbol{\Phi}^T(\mathbf{t} - \boldsymbol{\Phi}\mathbf{w})
$$

Normal equations and solution ($\boldsymbol{\Phi}^\dagger$ = Moore–Penrose pseudo-inverse):

$$
\boldsymbol{\Phi}^T\boldsymbol{\Phi}\,\mathbf{w}_{\text{ML}} = \boldsymbol{\Phi}^T\mathbf{t}
\quad\Rightarrow\quad
\mathbf{w}_{\text{ML}} = (\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbf{t} = \boldsymbol{\Phi}^\dagger\mathbf{t}
$$

Noise variance (from 1.1):

$$
\frac{1}{\beta_{\text{ML}}} = \frac1N\sum_n \big(t_n - \mathbf{w}_{\text{ML}}^T\boldsymbol{\phi}(\mathbf{x}_n)\big)^2
$$

Geometry (§4.1.5):
- $\mathbf{y} = \boldsymbol{\Phi}\mathbf{w}_{\text{ML}}$ lies in the column space of $\boldsymbol{\Phi}$ (dimension $M$ if full column rank). $\mathbf{y} = \mathbf{P}\mathbf{t}$, $\mathbf{P} = \boldsymbol{\Phi}(\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T$: orthogonal projection.
- The normal equations *are* the orthogonality condition: $\boldsymbol{\Phi}^T(\mathbf{t} - \boldsymbol{\Phi}\mathbf{w}_{\text{ML}}) = \mathbf{0}$ → residual $\perp$ every column of $\boldsymbol{\Phi}$.
- With $\phi_0 = 1$: residuals sum to zero.

Bias (§4.1.4):

$$
w_0 = \bar{t} - \sum_{j=1}^{M-1} w_j\,\bar{\phi}_j, \qquad \bar{\phi}_j = \tfrac1N\textstyle\sum_n \phi_j(\mathbf{x}_n)
$$

- Equivalent: mean prediction = mean target; the fit passes through the data centroid (same fact as residuals summing to zero).
- Centre the data → $w_0 = \bar{t}$, other weights fit to centred data. Reason to leave $w_0$ unregularized (1.3).

Regularized least squares / ridge (§4.1.6), $\widetilde{E} = E + \frac{\lambda}{2}\mathbf{w}^T\mathbf{w}$:

$$
\mathbf{w} = (\lambda\mathbf{I} + \boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^T\mathbf{t}
$$

- This is the MAP estimate ($\lambda = \alpha/\beta$), no longer $\mathbf{w}_{\text{ML}}$.
- Eigenvalues of $\boldsymbol{\Phi}^T\boldsymbol{\Phi}$ are $\sigma_i^2 \ge 0$ ($\sigma_i$: singular values of $\boldsymbol{\Phi}$). Adding $\lambda\mathbf{I}$ → $\sigma_i^2 + \lambda \ge \lambda > 0$: always invertible (also for $M > N$ or collinear basis functions); $\text{cond} = \frac{\sigma_{\max}^2+\lambda}{\sigma_{\min}^2+\lambda}$ stays bounded.

Sequential learning, LMS / Widrow–Hoff (§4.1.3): SGD on $E_n = \frac12(t_n - \mathbf{w}^T\boldsymbol{\phi}_n)^2$:

$$
\mathbf{w}^{(\tau+1)} = \mathbf{w}^{(\tau)} + \eta\,\big(t_n - \mathbf{w}^{(\tau)T}\boldsymbol{\phi}_n\big)\boldsymbol{\phi}_n
$$

- Sign check: prediction too low ($t_n > y_n$) → step along $+\boldsymbol{\phi}_n$ → $y_n$ increases.
- Residual on point $n$ scales by $(1 - \eta\|\boldsymbol{\phi}_n\|^2)$ per update → that update amplifies it if $\eta > 2/\|\boldsymbol{\phi}_n\|^2$. Safe: $\eta < 2/\max_n\|\boldsymbol{\phi}_n\|^2$. Divergence is guaranteed only when this fails for *all* points ($\eta > 2/\min_n\|\boldsymbol{\phi}_n\|^2$); in between, contracting points can still hold it stable. Too large: overshoot, oscillation, divergence. Too small: slow.

## Assumptions and failure modes
- **Gaussian, homoscedastic noise** (constant $\beta$); otherwise least squares is not ML.
- **Basis is fixed by hand:** model is only as good as $\boldsymbol{\phi}$. Number of basis functions grows exponentially with input dimension (motivation for neural nets, B&B ch. 6).
- **Ill-conditioning:** $\text{cond}(\boldsymbol{\Phi}^T\boldsymbol{\Phi}) = \text{cond}(\boldsymbol{\Phi})^2$. Polynomial basis (Vandermonde) on unscaled inputs is badly conditioned. Never form the inverse explicitly; solve with `np.linalg.lstsq` (SVD) or QR. Scale inputs (e.g. to $[-1, 1]$).
- **$M > N$ or collinear columns:** $\boldsymbol{\Phi}^T\boldsymbol{\Phi}$ singular; ML has no unique solution. Ridge fixes it.

## Link to code
Exercise `phase1_fundamentals/02_linear_regression/`:

| Math | Code | Shape |
|------|------|-------|
| $\mathbf{x}_n$ | `x` | `(N,)` (1-D input) |
| $\mathbf{t}$ | `t` | `(N,)` |
| $\boldsymbol{\Phi}$ | `Phi` | `(N, M)` |
| $\mathbf{w}$ | `w` | `(M,)` |
| $\lambda$ | `lam` | scalar |
| $1/\beta_{\text{ML}}$ | `noise_var` | scalar |
| $\eta$ | `lr` | scalar |

## Engineering link
- Polynomial response surfaces in MDAO are exactly this model with a polynomial basis; ridge = standard fix for ill-conditioned response surfaces.
- Normal equations / projection = the same least squares used to identify aerodynamic derivatives ($C_{L_\alpha}$, $C_{m_q}$) from flight-test data (equation-error method).
- LMS = online parameter identification / adaptive filtering; same stability limit on the gain.

## Open questions
- Decision theory: optimal predictor under squared loss, $y^*(x) = \mathbb{E}[t \mid x]$ (B&B §4.2). Quiz at 1.3.
