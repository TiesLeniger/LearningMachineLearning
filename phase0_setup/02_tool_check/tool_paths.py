"""Locate external executables: env var -> repo tools/ -> PATH."""

from pathlib import Path
import os
import shutil

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_exe(name: str) -> Path | None:
    env = os.environ.get(f"{name.upper()}_EXE")
    if env and Path(env).is_file():
        return Path(env)
    for candidate in (REPO_ROOT / "tools" / name / f"{name}.exe", REPO_ROOT / "tools" / name / name):
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    return Path(found) if found else None


requires_xfoil = pytest.mark.skipif(find_exe("xfoil") is None, reason="XFOIL not found")
requires_avl = pytest.mark.skipif(find_exe("avl") is None, reason="AVL not found")
