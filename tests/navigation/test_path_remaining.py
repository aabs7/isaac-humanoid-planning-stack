"""Progress along the planned path.

``path_remaining`` is what the goto stall-detector judges the robot by. The property
that matters is the one straight-line-to-goal does not have: while the robot follows a
detour *around* an obstacle, remaining path length falls monotonically even though the
distance to the goal is flat or growing.
"""

from __future__ import annotations

import math

import pytest

from g1sim.navigation.path_planning import path_remaining


def test_a_straight_path_measures_the_distance_left_to_walk():
    path = [(0.0, 0.0), (3.0, 0.0), (5.0, 0.0)]
    assert path_remaining(path, 0.0, 0.0, (5.0, 0.0)) == pytest.approx(5.0)
    assert path_remaining(path, 2.0, 0.0, (5.0, 0.0)) == pytest.approx(3.0)


def test_it_counts_the_corners_not_the_crow_flight():
    """An L-shaped route is longer than the straight line, and must be reported so --
    otherwise the robot is judged against a distance it was never going to travel."""
    path = [(0.0, 0.0), (0.0, 4.0), (3.0, 4.0)]
    goal = (3.0, 4.0)
    assert path_remaining(path, 0.0, 0.0, goal) == pytest.approx(7.0)
    assert math.hypot(*goal) == pytest.approx(5.0)          # the crow flight


def test_walking_a_detour_reduces_remaining_path_while_the_straight_line_grows():
    """The exact false-stall the old metric produced: rounding an obstacle means
    heading *away* from the goal for a while."""
    goal = (4.0, 0.0)
    path = [(0.0, 0.0), (0.0, -3.0), (4.0, -3.0), (4.0, 0.0)]   # around a wall
    walk = [(0.0, 0.0), (0.0, -1.0), (0.0, -2.0), (0.0, -3.0)]

    straight = [math.hypot(goal[0] - x, goal[1] - y) for x, y in walk]
    along = [path_remaining(path, x, y, goal) for x, y in walk]

    assert straight == sorted(straight), "straight-line distance grows on the detour"
    assert along == sorted(along, reverse=True), "path progress must still fall"
    assert along[0] - along[-1] == pytest.approx(3.0)           # 3 m actually walked


def test_it_falls_back_to_the_straight_line_without_a_plan():
    assert path_remaining([], 0.0, 0.0, (3.0, 4.0)) == pytest.approx(5.0)
    assert path_remaining([(0.0, 0.0)], 0.0, 0.0, (3.0, 4.0)) == pytest.approx(5.0)


def test_progress_is_measured_from_the_robot_not_the_first_waypoint():
    """waypoints[0] is the robot's own cell and goes stale the moment it moves; the
    live pose is what counts."""
    path = [(0.0, 0.0), (5.0, 0.0)]
    assert path_remaining(path, 4.0, 0.0, (5.0, 0.0)) == pytest.approx(1.0)
