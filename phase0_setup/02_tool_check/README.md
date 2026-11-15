# 0.4 Tool check

**What:** Smoke tests for OpenMDAO, Gymnasium, XFOIL and AVL, each checked against a known physical or analytic result.

**Why:** Phase 2 (surrogates/MDAO) and Phase 4 (RL) depend on these tools. A wrong install or a silently different binary should fail here, not halfway through a data-generation run.

## Tests
| Test | Case | Check |
|------|------|-------|
| `test_openmdao_smoke.py`  | Paraboloid f = (x−3)² + xy + (y+4)² − 3, SLSQP | Analytic optimum x = 20/3, y = −22/3, f = −82/3 |
| `test_gymnasium_smoke.py` | CartPole-v1 | Space shapes; same seed → identical trajectory |
| `test_xfoil_smoke.py`     | NACA 0012, Re = 1e6, viscous | Cl(0°) = 0 (symmetry); 0.50 < Cl(5°) < 0.62 |
| `test_avl_smoke.py`       | Rectangular flat plate, AR = 8 | CL linear in α; CLα below lifting-line bound; span efficiency 0.9 < e ≤ 1 |

XFOIL/AVL tests are skipped when the executable is not found (e.g. on CI). Lookup: `tool_paths.find_exe`, see `tools/README.md`.

## Result
Verified 2026-11-15 on the laptop. OpenMDAO 3.45.2, Gymnasium 1.4.0, XFOIL 6.99, AVL 3.40. `uv run pytest`: 12 passed.

| Quantity | Value | Reference |
|----------|-------|-----------|
| XFOIL Cl(0°) | 0.0000 | 0 |
| XFOIL Cl(5°) | 0.557 | thin-airfoil 2πα = 0.548 |
| XFOIL Cd(5°) | 0.0085 | |
| AVL CLα | 4.57 /rad | lifting line 5.03, Helmbold 4.91 |
| AVL e | 0.97 | 1 (elliptic) |

**Open point:** AVL's CLα is mesh-converged (Nchord 1–12, Nspan 24–48 all give 4.55–4.57) but 7 % below Helmbold's estimate. The gap shrinks with AR (AR = 100: 6.05 vs 6.16). Cause not verified. Worth understanding before AVL is used to generate surrogate training data.
