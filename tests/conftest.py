"""Fixtures shared by every test area. Area-specific fixtures live in that area's
own ``conftest.py``; reusable non-fixture helpers live in ``tests/helpers/``."""

from __future__ import annotations

import pytest

from tests.helpers.tiny_map import build_tiny_map


@pytest.fixture
def smap():
    """The synthetic apartment (see tests/helpers/tiny_map.py), fresh per test."""
    return build_tiny_map()
