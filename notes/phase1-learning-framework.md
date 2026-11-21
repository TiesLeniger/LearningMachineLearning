# 1.1 Learning framework

**Source:** B&B ch. 1 (§1.2 tutorial example), ch. 2 (§2.1–2.6), pp. 1–58 · Date: 2026-11-21

## Core idea
Supervised learning: given a training set of inputs $x_n$ with known targets $t_n$, fit a model $y(x, \mathbf{w})$ by minimizing an error function over the parameters $\mathbf{w}$ (*training*). The goal is *generalization*: accurate predictions on new inputs not seen during training. Regression has continuous $t$, classification discrete $t$. The error function is not arbitrary: sum-of-squares follows from maximum likelihood under Gaussian noise.

## Math

Model (polynomial, order $M$, linear in $\mathbf{w}$):

$$
y(x, \mathbf{w}) = \sum_{j=0}^{M} w_j x^j
$$

Objective (sum-of-squares, optionally regularized):

$$
E(\mathbf{w}) = \frac12 \sum_{n=1}^{N} \big(y(x_n, \mathbf{w}) - t_n\big)^2, \qquad
\widetilde{E}(\mathbf{w}) = E(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|^2
$$

Generalization metric (same units as $t$, comparable across $N$):

$$
E_{\text{RMS}} = \sqrt{\tfrac{1}{N}\textstyle\sum_n \big(y(x_n, \mathbf{w}) - t_n\big)^2}
$$

Probabilistic view: $t = y(x, \mathbf{w}) + \epsilon$, $\epsilon \sim \mathcal{N}(0, \beta^{-1})$, so $\beta^{-1} = \sigma^2$ (precision = 1/variance). Log-likelihood:

$$
\ln p(\mathbf{t} \mid \mathbf{x}, \mathbf{w}, \beta) = -\frac{\beta}{2} \sum_{n=1}^{N} \big(y(x_n, \mathbf{w}) - t_n\big)^2 + \frac{N}{2}\ln\beta - \frac{N}{2}\ln(2\pi)
$$

- Only the first term depends on $\mathbf{w}$; maximizing the log-likelihood = minimizing its *negative* = minimizing $E(\mathbf{w})$. So $\mathbf{w}_{\text{ML}} = \arg\min E(\mathbf{w})$.
- Then maximizing over $\beta$: the noise variance is the mean squared residual

$$
\frac{1}{\beta_{\text{ML}}} = \frac{1}{N} \sum_{n=1}^{N} \big(y(x_n, \mathbf{w}_{\text{ML}}) - t_n\big)^2
$$

Bias of ML (Gaussian, $\mu$ and $\sigma^2$ both estimated from the same data):

$$
\mu_{\text{ML}} = \frac1N\sum_n x_n, \qquad
\sigma^2_{\text{ML}} = \frac1N\sum_n (x_n - \mu_{\text{ML}})^2, \qquad
\mathbb{E}[\sigma^2_{\text{ML}}] = \frac{N-1}{N}\,\sigma^2
$$

- Residuals are measured against $\mu_{\text{ML}}$, which was fitted to the same points, so they are systematically too small: variance is underestimated. Unbiased: divide by $N-1$.
- Negligible for large $N$. But with many fitted parameters the same effect grows: this bias is the root of overfitting under ML (B&B §2.3).

Why products and logs: likelihood factorizes because targets are i.i.d. given the inputs. $\ln$ is strictly monotonic (same argmax), turns the product into a sum (gradient = sum over data, basis of SGD), and avoids underflow ($\approx 10^{-2000}$ for $N = 1000$).

Change of variables, $y = g(x)$ monotonic (B&B §2.4). Mass conservation $p_y|dy| = p_x|dx|$:

$$
p_y(y) = p_x\big(g^{-1}(y)\big)\left|\frac{dx}{dy}\right|
$$

- The mode is *not* invariant under nonlinear $g$: $p_y'(y) = p_x'(x)(dx/dy)^2 + p_x(x)\,d^2x/dy^2$; second term $\neq 0$ unless $g$ linear. Example: $x \sim U[1, 10]$, $y = \ln x$ → $p_y \propto e^y$, peak at $\ln 10$. Reverse: $p_x = 2x$ on $[0,1]$ (mode at 1), $y = x^2$ → $p_y = 2\sqrt{y}\cdot\frac{1}{2\sqrt{y}} = 1$, uniform, mode gone.
- Recipe: invert $x = g^{-1}(y)$ → compute $|dx/dy|$ → substitute → check range and $\int p_y = 1$.
- Invariant under monotonic $g$: probabilities of intervals, quantiles (median). Not invariant: density values, mode.

KL divergence and cross-entropy (B&B §2.5):

$$
\mathrm{KL}(p\|q) = -\int p \ln \frac{q}{p}\,dx = \underbrace{-\int p \ln q\,dx}_{\text{cross-entropy}} - H[p] \;\ge 0
$$

- $= 0$ iff $p = q$; $\ge 0$ by Jensen. *Not symmetric*: not a distance.
- With $q = q(x\mid\boldsymbol\theta)$ and samples from $p$: $-\int p\ln q \approx -\frac1N\sum_n \ln q(x_n\mid\boldsymbol\theta)$. Minimizing KL = minimizing NLL = ML. Returns as cross-entropy loss at 1.5.

Bayes and MAP (B&B §2.6):

$$
\underbrace{p(\mathbf{w}\mid D)}_{\text{posterior}} = \frac{\overbrace{p(D\mid\mathbf{w})}^{\text{likelihood}}\;\overbrace{p(\mathbf{w})}^{\text{prior}}}{\underbrace{p(D)}_{\text{evidence}}}
$$

- ML maximizes the likelihood. MAP maximizes the posterior; $p(D)$ is constant in $\mathbf{w}$ (but returns for hyperparameter selection in GPs, 1.10).
- Gaussian prior $p(\mathbf{w}) \propto \exp(-\frac{\alpha}{2}\mathbf{w}^T\mathbf{w})$:

$$
-\ln p(\mathbf{w}\mid D) = \frac{\beta}{2}\sum_n \big(y(x_n,\mathbf{w}) - t_n\big)^2 + \frac{\alpha}{2}\mathbf{w}^T\mathbf{w} + \text{const}
\;\Rightarrow\; \lambda = \alpha/\beta
$$

  Regularization = prior belief that weights are small; strength = noise variance / prior variance.

Full Bayesian prediction: average over all $\mathbf{w}$, weighted by the posterior:

$$
p(t \mid x, D) = \int p(t \mid x, \mathbf{w})\, p(\mathbf{w} \mid D)\, d\mathbf{w}
$$

- Gives a predictive *distribution*, not a point: its variance = noise ($\beta^{-1}$) + parameter uncertainty, and the latter grows away from the training data.

## Assumptions and failure modes
- **Overfitting:** $M = 9$, $N = 10$ → 10 parameters through 10 points, training error 0. Curve interpolates the noise, weights become huge, wild oscillation between points → high test error.
- **Remedies:** more data ($N \uparrow$), lower model order ($M \downarrow$), regularization ($\lambda > 0$).
- **Regularization limits:** $\lambda \to 0$ → unregularized, overfits. $\lambda \to \infty$ → all weights shrink to 0, $y \approx 0$ (not a line: $w_1$ is penalized too) → underfits.
- **Choosing $\lambda$ (or $M$):** on a separate validation set, never on training error (training error always decreases as $\lambda \to 0$, so it would pick $\lambda = 0$). Data scarce → cross-validation (step 1.4).
- **Gaussian noise assumption:** sum-of-squares is optimal only if the noise is Gaussian with constant variance. Outliers / heavy tails → least squares is pulled off.
- **MAP depends on parametrization** (mode not invariant, §2.4): fitting in Re vs. $\log$ Re can give different MAP estimates. ML does not have this problem.
- **Point estimates (ML, MAP) carry no uncertainty:** a surrogate can be confidently wrong far from its samples.

## Link to code
Step 1.2 (`phase1_<name>/02_linear_regression/`) uses:

| Math | Code | Shape |
|------|------|-------|
| $x_n$ | `x` | `(N,)` |
| $\mathbf{t}$ | `t` | `(N,)` |
| $\boldsymbol{\Phi}$, $\Phi_{nj} = x_n^j$ | `Phi` | `(N, M+1)` |
| $\mathbf{w}$ | `w` | `(M+1,)` |
| $\lambda$ | `lam` | scalar |
| $1/\beta_{\text{ML}}$ | `noise_var` | scalar |

## Engineering link
- Least-squares polynomial fit = fitting a drag polar or lift curve to wind-tunnel / XFOIL data; $1/\beta_{\text{ML}}$ estimates the measurement noise.
- $\frac{\lambda}{2}\|\mathbf{w}\|^2$ = Tikhonov regularization in inverse problems.
- Overfitting = polynomial response-surface surrogates in MDAO that pass through every sample but oscillate between them; validation set = independent check runs.
- Predictive uncertainty = Kriging / GP surrogates in MDAO: sample the next expensive CFD run where the surrogate is uncertain or promising (efficient global optimization, expected improvement), and flag optima the optimizer found by exploiting surrogate error.

## Open questions
- Decision theory: optimal predictor under squared loss, $y^*(x) = \mathbb{E}[t \mid x]$, and the irreducible noise term. B&B §4.2; quiz at 1.2/1.3.
