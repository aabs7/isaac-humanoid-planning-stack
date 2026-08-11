"""Move-duration models -- the numbers railroad plans against.

An operator's duration is a Python callable, so this is the whole of the geometric
knowledge the planner gets: ``move_time(robot, from, to) -> seconds``. Returning
``float("inf")`` is railroad's sanctioned way to say *impossible*; such actions are
dropped during grounding.

Two models, and a real reason for the default:

:func:`euclidean_move_time` is straight-line distance over walking speed. It ignores
walls, which A* does not.

:class:`AStarMoveTime` runs our actual navigation planner (``plan_path`` over the
robot's sensed occupancy grid) so planned cost matches what walking will really cost.
It is **not** the default, because duration callables are evaluated **during
grounding, once per binding**: 45 locations is 1980 ordered pairs, and an A* over a
0.05 m grid of this apartment is not free. Memoization halves it (the metric is
symmetric) but the first ``get_actions()`` would still stall for a long time. Opt in
when the plan quality matters more than startup latency, or pre-warm the cache for the
handful of locations a task actually involves.

The honest limitation of the default: two locations either side of a wall look closer
than they are, so the planner can pick a worse ordering. It cannot make a plan
*invalid* -- ``goto_*`` still routes around the wall, it just takes longer than the
plan predicted.
"""

from __future__ import annotations

from typing import Callable

from g1sim.navigation.path_planning import path_remaining, plan_path

# Sustained walking speed of the locomotion policy (m/s), measured over the goto runs.
# Only ever a scale factor on cost, so precision here buys nothing.
WALK_SPEED = 0.6

# Charged on top of travel for arriving somewhere: the final creep up to an object plus
# turning to face it (`goto_object` does both). Without it, adjacent locations cost ~0
# and the planner treats hopping between them as free.
ARRIVAL_TIME = 3.0


def euclidean_move_time(domain, speed: float = WALK_SPEED,
                        arrival: float = ARRIVAL_TIME) -> Callable[[str, str, str], float]:
    """Straight-line duration model. Fast enough to ground the full domain."""
    def move_time(robot: str, loc_from: str, loc_to: str) -> float:
        if loc_from == loc_to:
            return float("inf")          # a self-move is never useful; drop the action
        return domain.distance(loc_from, loc_to) / speed + arrival
    return move_time


class AStarMoveTime:
    """Duration model backed by the robot's own A* planner and sensed occupancy grid.

    Needs a live ``mapper`` (so: sim only, after some lidar has been integrated).
    Memoized and symmetric. Unreachable pairs cost ``inf``, which drops the action --
    note that early in a run, when little has been sensed, "unreachable" mostly means
    "unobserved", so prefer :func:`euclidean_move_time` until the map has filled in.
    """

    def __init__(self, domain, mapper, *, speed: float = WALK_SPEED,
                 arrival: float = ARRIVAL_TIME, robot_radius_m: float = 0.35):
        self.domain = domain
        self.mapper = mapper
        self.speed = speed
        self.arrival = arrival
        self.robot_radius_m = robot_radius_m
        self._cache: dict = {}
        self.misses = 0

    def __call__(self, robot: str, loc_from: str, loc_to: str) -> float:
        if loc_from == loc_to:
            return float("inf")
        key = (loc_from, loc_to) if loc_from < loc_to else (loc_to, loc_from)
        if key not in self._cache:
            self.misses += 1
            self._cache[key] = self._path_length(*key)
        length = self._cache[key]
        if length == float("inf"):
            return float("inf")
        return length / self.speed + self.arrival

    def _path_length(self, a: str, b: str) -> float:
        start, goal = self.domain.locations[a], self.domain.locations[b]
        waypoints, _, _ = plan_path(self.mapper, start, goal,
                                   robot_radius_m=self.robot_radius_m)
        if not waypoints:
            return float("inf")
        return path_remaining(waypoints, start[0], start[1], goal)

    def prewarm(self, locations) -> None:
        """Compute and cache every pair among ``locations``. Use this to pay the A*
        bill for a task's handful of locations instead of all 1980 pairs."""
        locs = list(locations)
        for i, a in enumerate(locs):
            for b in locs[i + 1:]:
                self(self.domain.robot, a, b)
