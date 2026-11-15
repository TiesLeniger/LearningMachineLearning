"""AVL smoke test: rectangular flat-plate wing, AR = 8, incompressible.

Physics checks (robust bounds, not exact values):
- CL linear in alpha (VLM is a linear method).
- CL_alpha below the elliptic lifting-line value 2*pi*AR/(AR+2) = 5.03 /rad,
  and within 15 % of it. AVL gives ~4.57 /rad, mesh-converged (see README).
- Span efficiency e = CL^2 / (pi AR CDi) in (0.9, 1.0]: rectangular is near-elliptic.
"""

import math
import re
import subprocess

import pytest

from tool_paths import find_exe, requires_avl

AR = 8.0

GEOMETRY = f"""\
Rectangular wing AR {AR:.0f}
#Mach
0.0
#IYsym IZsym Zsym
0 0 0.0
#Sref Cref Bref
{AR:.1f} 1.0 {AR:.1f}
#Xref Yref Zref
0.25 0.0 0.0
SURFACE
Wing
#Nchord Cspace Nspan Sspace
12 1.0 24 -2.0
YDUPLICATE
0.0
SECTION
#Xle Yle Zle Chord Ainc
0.0 0.0 0.0 1.0 0.0
SECTION
0.0 {AR / 2:.1f} 0.0 1.0 0.0
"""

SCRIPT = """\
LOAD wing.avl
OPER
A A 2.5
X
A A 5
X

QUIT
"""


@requires_avl
def test_rect_wing_lift_slope(tmp_path):
    (tmp_path / "wing.avl").write_text(GEOMETRY)
    out = subprocess.run(
        [str(find_exe("avl"))], input=SCRIPT, text=True, cwd=tmp_path,
        capture_output=True, timeout=60, check=True,
    ).stdout

    cl_25, cl_5 = (float(v) for v in re.findall(r"CLtot\s*=\s*(-?\d+\.\d+)", out))
    cdi_5 = float(re.findall(r"CDff\s*=\s*(-?\d+\.\d+)", out)[-1])

    cl_alpha = cl_5 / math.radians(5.0)  # flat plate: CL(0) = 0
    cl_alpha_ll = 2 * math.pi * AR / (AR + 2)
    e = cl_5**2 / (math.pi * AR * cdi_5)

    assert cl_5 / cl_25 == pytest.approx(2.0, rel=0.01)
    assert 0.85 * cl_alpha_ll < cl_alpha < cl_alpha_ll
    assert 0.9 < e <= 1.0
