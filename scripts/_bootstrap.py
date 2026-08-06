"""Make ``g1sim`` importable when a script in this directory is run directly.

``python isaac_task_planning/scripts/teleop_g1_apartment.py`` puts *this* directory
on ``sys.path``, not the repo root, so ``import g1sim`` would fail. Every script
imports this module first to fix that. (Once the project is pip-installed, this
becomes a no-op and can go.)
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
