"""Can the robot's arm actually get to this object, from somewhere it could stand?

Sim-free. ``PICK_RADIUS`` answers a different and much weaker question -- how close the
*base* parks to an object's footprint -- and it is measured in 2D from the pelvis. An arm
hangs off a shoulder about 0.29 m above the pelvis, so what decides a grasp is the **3D
shoulder-to-object distance from a position the robot can legally occupy**. That distance
had never been measured in this stack until ``scripts/reach_probe_g1_apartment.py``, which
found ``cup_0005`` reachable with 7 mm to spare (0.693 m against a ~0.70 m arm) while
``PICK_RADIUS`` reported a comfortable 0.698 m of a 1.00 m budget.

**Two things the probe taught, both the hard way.**

*Furniture bounding boxes are not obstacles.* The robot was measured standing at
``xy_dist = 0.000`` from ``cabinet_0009`` -- i.e. inside its 5.37 m^2 axis-aligned box. That
box is a 2.4 x 2.2 m overstatement of an L-shaped counter run, and the robot stands in the
open middle of it. So an AABB-based base-exclusion test rejects stances the robot demonstrably
occupies. Only a sensed occupancy grid knows where the solid parts are, and that is not
available sim-free -- hence :func:`best_stance` takes a ``free_xy`` predicate and defaults to
the weakest honest one, "inside some room".

*A ``goto`` result is not a reachability verdict.* ``goto_object`` stops the moment its
``reach`` argument is satisfied, so a robot that halts 0.99 m from a cup asked to stop within
1.00 m has told you nothing about whether it could have got closer. Measuring true reach
needs a stance that presses in until it is blocked.

So: this module computes what reach a *given* stance requires (exact, :func:`required_reach`)
and searches for the best stance under whatever free-space knowledge the caller has
(optimistic by default, :func:`best_stance`). A ``False`` from the default configuration is
strong evidence -- nothing about walls or clutter would make it *better*. A ``True`` means
"worth walking to", not "guaranteed".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

# Shoulder height above the floor while standing. Measured, not guessed: the probe reported
# the shoulder 0.03 m above a 0.99 m object top and 0.11 m above a 0.91 m one -- z = 1.02 m
# at both stances, with the pelvis at 0.72-0.73 m.
SHOULDER_Z = 1.02

# Reach from shoulder to hand. The G1's arm is ~0.60-0.70 m; 0.70 is the optimistic end, kept
# so that a "not reachable" verdict is never an artefact of a pessimistic constant.
ARM_REACH = 0.70


@dataclass(frozen=True)
class Reachability:
    """Whether an object can be grasped, and the best stance found for doing so."""
    object: str
    reachable: bool
    required_reach: float            # shoulder -> object bbox, from the best stance found
    stance: Optional[tuple]          # (x, y) the base would occupy, or None if none is legal
    reason: str

    def __bool__(self) -> bool:
        return self.reachable


def aabb_distance(point, bbox_min, bbox_max) -> float:
    """3D distance from a point to an axis-aligned box (0 inside).

    The 3D sibling of :meth:`SemanticObject.xy_dist`. Reach must be judged in 3D: a cup on a
    0.9 m counter is nearly overhead at close range, and the XY distance flatters it.
    """
    d = [max(bbox_min[i] - point[i], 0.0, point[i] - bbox_max[i]) for i in range(3)]
    return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])


def required_reach(o, stance_xy, *, shoulder_z: float = SHOULDER_Z) -> float:
    """Shoulder-to-object distance if the base stands at ``stance_xy``.

    Exact and assumption-free apart from the shoulder model: the shoulder is taken to be
    directly above the base centre. The real one is ~0.16 m to one side, which for a target
    straight ahead makes the true distance slightly *worse*, so this errs optimistic in the
    same direction as everything else here.
    """
    return aabb_distance((stance_xy[0], stance_xy[1], shoulder_z), o.bbox_min, o.bbox_max)


def in_a_room(smap) -> Callable[[float, float], bool]:
    """The weakest honest free-space test: the base must be inside some room polygon.

    Deliberately does *not* treat furniture bounding boxes as obstacles -- see the module
    docstring; the robot was measured standing inside one.
    """
    def free(x: float, y: float) -> bool:
        return smap.room_at(x, y) is not None
    return free


def occupancy_free(mapper, robot_radius: float = 0.35) -> Callable[[float, float], bool]:
    """Free-space test backed by the robot's sensed occupancy grid -- the accurate one.

    Only available with a live mapper (so: in the simulator, after some lidar). Use it when
    you have it; the difference from :func:`in_a_room` is precisely the clutter and wall
    geometry that bounding boxes cannot express.
    """
    from g1sim.navigation.path_planning import inflate

    free_grid = ~inflate(mapper.occupied(), int(round(robot_radius / mapper.res)))

    def free(x: float, y: float) -> bool:
        i, j = mapper.world_to_cell(x, y)
        if not mapper.in_bounds(i, j):
            return False
        return bool(free_grid[i, j])
    return free


def best_stance(smap, o, *, arm_reach: float = ARM_REACH,
                shoulder_z: float = SHOULDER_Z,
                free_xy: Optional[Callable[[float, float], bool]] = None,
                angles: int = 72, step: float = 0.05,
                min_radius: float = 0.20, max_radius: float = 1.60) -> Reachability:
    """Search around ``o`` for the stance whose shoulder comes closest to it.

    Samples a polar grid centred on the object's footprint, keeps the closest point that
    ``free_xy`` accepts, and reports the reach that stance would require. ``free_xy``
    defaults to :func:`in_a_room`; pass :func:`occupancy_free` when a mapper exists.
    """
    free = free_xy if free_xy is not None else in_a_room(smap)
    cx, cy = o.xy

    best, best_d = None, float("inf")
    radius = min_radius
    while radius <= max_radius:
        for i in range(angles):
            th = 2.0 * math.pi * i / angles
            x, y = cx + radius * math.cos(th), cy + radius * math.sin(th)
            if not free(x, y):
                continue
            d = required_reach(o, (x, y), shoulder_z=shoulder_z)
            if d < best_d:
                best_d, best = d, (x, y)
        radius += step

    if best is None:
        return Reachability(o.name, False, float("inf"), None,
                            "no legal stance anywhere around it")
    if best_d <= arm_reach:
        return Reachability(o.name, True, best_d, best,
                            f"shoulder comes within {best_d:.3f} m from "
                            f"({best[0]:.2f}, {best[1]:.2f})")
    return Reachability(o.name, False, best_d, best,
                        f"{best_d - arm_reach:.3f} m beyond the arm even from the best "
                        f"stance ({best[0]:.2f}, {best[1]:.2f}) -- it sits too far in from "
                        f"the edge of whatever it rests on")


def graspable_objects(smap, names=None, **kw) -> dict:
    """Reachability for every pickable object (or just ``names``), keyed by object name."""
    targets = (list(smap.small_objects()) if names is None
               else [smap.get(n) for n in names])
    return {o.name: best_stance(smap, o, **kw) for o in targets if o is not None}
