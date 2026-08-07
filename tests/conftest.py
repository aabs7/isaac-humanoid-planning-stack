"""Fixtures shared by every test area. Area-specific fixtures live in that area's
own ``conftest.py``; reusable non-fixture helpers live in ``tests/helpers/``."""

from __future__ import annotations

import pytest

from g1sim.skills.mock import MockSkills
from tests.helpers.tiny_map import LIVINGROOM_START_XY, build_tiny_map


@pytest.fixture
def smap():
    """The synthetic apartment (see tests/helpers/tiny_map.py), fresh per test."""
    return build_tiny_map()


@pytest.fixture
def env(smap):
    """A virtual robot in the synthetic apartment -- the same skills object the
    planner drives in the sim, minus Isaac. Shares ``smap``, so a test can assert on
    the map directly after driving the robot."""
    return MockSkills(smap, start_xy=LIVINGROOM_START_XY, verbose=False)
