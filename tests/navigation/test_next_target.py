"""Path following must advance past waypoints it has reached.

The bug this guards: the follower always steered at ``waypoints[1]``. Once the robot was
within the unicycle controller's arrival radius of that waypoint, ``command_to`` returned
``arrived`` and its stand-still command, so the robot stopped in the middle of an
unfinished path. Re-planning was supposed to move the target on, and usually did -- but not
in a doorway, where 0.35 m of obstacle inflation leaves a channel a cell or two wide,
string-pulling loses line of sight immediately, and every re-plan produces another waypoint
just as close. The robot then stood until the stall detector reversed it out.
"""

import math

from g1sim.navigation.path_planning import next_target

TOL = 0.3           # WaypointNavigator.goal_tol as RobotSkills configures it
GOAL = (10.0, 0.0)


def test_steers_at_the_next_waypoint_when_it_is_far():
    path = [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0)]
    assert next_target(path, 0.0, 0.0, GOAL, TOL) == (3.0, 0.0)


def test_skips_a_waypoint_the_robot_is_already_sitting_on():
    """The core fix: waypoints[1] is reached, so steer at waypoints[2]."""
    path = [(0.0, 0.0), (0.1, 0.0), (6.0, 0.0)]
    assert next_target(path, 0.0, 0.0, GOAL, TOL) == (6.0, 0.0)


def test_a_doorway_of_densely_packed_waypoints_still_yields_a_reachable_target():
    """The exact failing geometry: string-pulling could only keep near-adjacent cells, so
    every waypoint for the next third of a metre is inside the arrival radius. The chosen
    target must be outside it, or the controller stands still."""
    path = [(0.0, 0.0)] + [(0.05 * i, 0.0) for i in range(1, 8)] + [(4.0, 0.0)]
    target = next_target(path, 0.0, 0.0, GOAL, TOL)
    assert math.hypot(target[0], target[1]) > TOL, (
        "target inside the arrival radius: the controller would command a full stop")


def test_falls_back_to_the_last_waypoint_once_all_are_reached():
    """At the end of the path every waypoint is within tolerance; the caller's own arrival
    check is what ends the run, so returning the final waypoint is correct."""
    path = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)]
    assert next_target(path, 0.0, 0.0, GOAL, TOL) == (0.2, 0.0)


def test_a_single_waypoint_path_returns_that_waypoint():
    assert next_target([(1.0, 2.0)], 0.0, 0.0, GOAL, TOL) == (1.0, 2.0)


def test_an_empty_path_falls_back_to_the_goal():
    """plan_path returns [] when A* finds nothing; heading straight at the goal is the
    optimistic behaviour the rest of the stack relies on."""
    assert next_target([], 0.0, 0.0, GOAL, TOL) == GOAL


def test_the_chosen_target_always_commands_motion():
    """The property that actually matters, over a range of robot positions along a path
    with mixed spacing: whatever is returned is either outside the arrival radius, or the
    whole remaining path is (meaning we have arrived)."""
    path = [(0.0, 0.0), (0.2, 0.0), (0.35, 0.0), (2.0, 0.0), (2.1, 0.0), (5.0, 0.0)]
    for k in range(60):
        px = 0.1 * k
        target = next_target(path, px, 0.0, GOAL, TOL)
        moves = math.hypot(target[0] - px, target[1]) > TOL
        arrived = all(math.hypot(w[0] - px, w[1]) <= TOL for w in path[1:])
        assert moves or arrived, f"stalled at px={px:.1f} with target {target}"
