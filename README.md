# Machine Learning for Aerospace

[![tests](https://github.com/TiesLeniger/LearningMachineLearning/actions/workflows/tests.yml/badge.svg)](https://github.com/TiesLeniger/LearningMachineLearning/actions/workflows/tests.yml)

Self-study project: ML fundamentals applied to aerospace simulation.

## Tracks
1. **Surrogate modelling & MDAO**: GP and neural-network surrogates trained on XFOIL/VLM data, Bayesian optimization, OpenMDAO integration.
2. **Control & RL**: learned flight controllers in Gymnasium and JSBSim, benchmarked against PID/LQR.
3. **C++ for simulation**: small RK4 flight dynamics model with Python bindings via pybind11.

## Structure
```
notes/                theory notes and topic map
phase0_setup/         environment and tooling
phase1_fundamentals/  ML basics, implemented from scratch
phase2_surrogates/    portfolio project 1
phase3_cpp/           C++ basics and simulation
phase4_rl/            portfolio project 2
```

Curriculum: `notes/topic-map.md`. Status: `PROGRESS.md`.

## Stack
Python, PyTorch, NumPy/SciPy, Gymnasium, JSBSim, OpenMDAO, XFOIL/AVL, C++17, Eigen, pybind11, pytest.
