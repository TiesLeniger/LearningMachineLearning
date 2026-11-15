# Topic map

Each item: theory (book) -> exercise (you code) -> tests. Portfolio projects are marked **[P]**.

## Phase 0: Setup
- 0.1 Python environment (venv or conda), CPU PyTorch, NumPy/SciPy, matplotlib, pytest
- 0.2 Git + GitHub repo, commit conventions
- 0.3 Claude Code workflow: CLAUDE.md, PROGRESS.md, review loop
- 0.4 Tool check: XFOIL, AVL, OpenMDAO, Gymnasium installed and smoke-tested

## Phase 1: ML fundamentals
- 1.0 Choose a backbone book; set up the notes format
- 1.1 Learning framework: supervised learning, loss, risk, generalization
- 1.2 Linear regression: least squares, MLE view, basis functions. *Exercise: NumPy, closed form + gradient descent*
- 1.3 Regularization and the bias-variance trade-off, MAP view (ridge = Gaussian prior). *Exercise: polynomial fit on noisy airfoil polar data*
- 1.4 Model selection: train/val/test, k-fold cross-validation
- 1.5 Classification: logistic regression, cross-entropy. *Exercise: stall / no-stall classifier*
- 1.6 Optimization: SGD, momentum, Adam, learning-rate schedules
- 1.7 Neural networks: MLP, backpropagation by hand. *Exercise: NumPy MLP with own backprop, gradient check*
- 1.8 PyTorch: tensors, autograd, nn.Module, training loop. *Exercise: port the 1.7 MLP*
- 1.9 Practicalities: normalization, initialization, overfitting diagnostics, early stopping
- 1.10 Gaussian processes: kernels, posterior, marginal likelihood, hyperparameters. *Exercise: GP from scratch in NumPy*

## Phase 2: Surrogates & MDAO [P]
- 2.1 Design of experiments: Latin hypercube, Sobol sequences
- 2.2 Data generation: batch XFOIL (and/or AVL) runs over airfoil/wing parameters, handling failed runs
- 2.3 GP surrogate for Cl/Cd/Cm; uncertainty and calibration
- 2.4 NN surrogate; compare against the GP (accuracy, data efficiency, cost)
- 2.5 Bayesian optimization: acquisition functions (EI, UCB), constrained BO
- 2.6 OpenMDAO integration: surrogate as a component in a wing design loop
- 2.7 Write-up: README with results, plots, lessons learned

## Phase 3: C++ basics
- 3.1 Toolchain on Windows (MSVC or MinGW, CMake), hello world, compile/link basics
- 3.2 Core language: types, references/pointers, functions, classes, std::vector, headers
- 3.3 Eigen for linear algebra
- 3.4 RK4 integrator; point-mass then longitudinal 3-DOF aircraft model
- 3.5 Validate against a Python implementation of the same model
- 3.6 pybind11 binding; benchmark against Python/NumPy

## Phase 4: Control & RL [P]
- 4.1 MDPs, returns, Bellman equations, dynamic programming
- 4.2 Monte Carlo and TD learning, Q-learning. *Exercise: tabular, on a discretized toy problem*
- 4.3 Function approximation, DQN
- 4.4 Policy gradients: REINFORCE, actor-critic, PPO; SAC for continuous control
- 4.5 Own Gymnasium env: longitudinal pitch/altitude hold (Python dynamics, later the Phase 3 C++ core)
- 4.6 Baselines: PID and LQR on the same env
- 4.7 Train PPO/SAC (own implementation first, then Stable-Baselines3 for comparison)
- 4.8 Move to JSBSim: 6-DOF aircraft, attitude-hold task
- 4.9 Robustness: noise, disturbances, model mismatch
- 4.10 Write-up: RL vs classical control comparison

## Parked
- Physics-informed ML: PINNs, neural operators, reduced-order models (after Phase 2)
- Health monitoring: dropped for now (candidate data: NASA C-MAPSS)
