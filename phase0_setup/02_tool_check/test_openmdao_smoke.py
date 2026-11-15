"""OpenMDAO paraboloid tutorial: f = (x-3)^2 + xy + (y+4)^2 - 3.

Analytic optimum: grad f = 0 -> x = 20/3, y = -22/3, f = -82/3.
"""

import openmdao.api as om
import pytest


def test_paraboloid_optimum(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # OpenMDAO writes <name>_out/ to cwd

    prob = om.Problem(reports=None)
    prob.model.add_subsystem(
        "parab", om.ExecComp("f = (x - 3)**2 + x*y + (y + 4)**2 - 3"), promotes=["*"]
    )
    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1e-10, disp=False)
    prob.model.add_design_var("x", lower=-50, upper=50)
    prob.model.add_design_var("y", lower=-50, upper=50)
    prob.model.add_objective("f")
    prob.setup()
    prob.set_val("x", 3.0)
    prob.set_val("y", -4.0)
    prob.run_driver()

    assert prob.get_val("x")[0] == pytest.approx(20 / 3, abs=1e-5)
    assert prob.get_val("y")[0] == pytest.approx(-22 / 3, abs=1e-5)
    assert prob.get_val("f")[0] == pytest.approx(-82 / 3, abs=1e-8)
