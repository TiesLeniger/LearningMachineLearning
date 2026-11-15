"""XFOIL smoke test: NACA 0012, Re = 1e6, viscous.

Expected: Cl(0 deg) = 0 by symmetry; Cl(5 deg) ~ 0.55 (thin-airfoil 2*pi*alpha = 0.548,
viscous decambering vs. thickness roughly cancel at this Re).
"""

import subprocess

import numpy as np
import pytest

from tool_paths import find_exe, requires_xfoil

SCRIPT = """\
PLOP
G

NACA 0012
OPER
VISC 1e6
ITER 200
PACC
polar.dat

ALFA 0
ALFA 5

QUIT
"""


def read_polar(path) -> np.ndarray:
    """Rows of [alpha, CL, CD, CDp, CM, Top_Xtr, Bot_Xtr] below the dashed header line."""
    lines = path.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("---")) + 1
    return np.array([[float(v) for v in ln.split()] for ln in lines[start:] if ln.strip()])


@requires_xfoil
def test_naca0012_lift(tmp_path):
    # cwd = tmp_path keeps the polar filename short (XFOIL truncates long paths).
    subprocess.run(
        [str(find_exe("xfoil"))], input=SCRIPT, text=True, cwd=tmp_path,
        capture_output=True, timeout=60, check=True,
    )
    polar = read_polar(tmp_path / "polar.dat")
    cl = dict(zip(polar[:, 0], polar[:, 1]))

    assert set(cl) == {0.0, 5.0}, "both points must converge"
    assert cl[0.0] == pytest.approx(0.0, abs=1e-3)
    assert 0.50 < cl[5.0] < 0.62
