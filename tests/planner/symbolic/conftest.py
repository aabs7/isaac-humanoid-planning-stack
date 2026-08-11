"""Fixtures for the symbolic (railroad) planner tests.

``smap`` (the synthetic apartment) and ``env`` (a ``MockSkills`` robot in it) come from
``tests/conftest.py``; these build the railroad layer on top. Everything is sim-free, so
the whole file runs in well under a second -- which is the point of testing the planner
against the mock before the simulator.
"""

from __future__ import annotations

import pytest

from g1sim.task.symbolic import G1Environment, SemanticDomain


@pytest.fixture
def domain(smap):
    """Every pickable object in the synthetic apartment: 3 rooms + 5 surfaces, 4 objects."""
    return SemanticDomain.build(smap)


@pytest.fixture
def sym_env(env, domain):
    """The railroad environment wired to the mock robot. Shares ``smap`` with ``domain``
    and ``env``, so a test can assert against the map after planning."""
    return G1Environment(env, domain, verbose=False)
