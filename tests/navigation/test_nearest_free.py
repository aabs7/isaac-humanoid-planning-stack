"""``nearest_free`` must tolerate a cell outside the grid.

The robot can leave the mapped area -- it did, in the first symbolic sim run, ending up
~13 m south of the map's edge after repeated stuck-recoveries. Passing that cell straight
into a numpy index has two distinct failure modes, and the wrong-answer one is the
dangerous half.
"""

import numpy as np

from g1sim.navigation.path_planning import nearest_free


def _free_grid():
    """A 20x30 grid, free everywhere except a blocked band across the middle rows."""
    free = np.ones((20, 30), dtype=bool)
    free[8:12, :] = False
    return free


def test_in_bounds_free_cell_is_returned_unchanged():
    assert nearest_free(_free_grid(), (3, 4)) == (3, 4)


def test_in_bounds_blocked_cell_snaps_to_free_space():
    i, j = nearest_free(_free_grid(), (10, 15))
    assert _free_grid()[i, j]


def test_a_large_negative_cell_does_not_raise():
    """This was an IndexError that killed a 5-minute sim run outright."""
    i, j = nearest_free(_free_grid(), (-267, 4))
    assert (0 <= i < 20) and (0 <= j < 30)


def test_a_small_negative_cell_does_not_wrap_to_the_far_side():
    """The subtler half: numpy accepts -3 as "row 17" and returns a confident, wrong
    answer on the opposite side of the map. Clamping must send it to the near edge."""
    i, _ = nearest_free(_free_grid(), (-3, 4))
    assert i == 0, f"expected the near edge, got row {i} (wrapped to the far side?)"


def test_a_cell_past_the_far_edge_is_clamped():
    i, j = nearest_free(_free_grid(), (999, 999))
    assert (i, j) == (19, 29)
