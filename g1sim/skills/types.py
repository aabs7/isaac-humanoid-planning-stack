"""Sim-free skill primitives shared by the real skills, the mock skills, and the
planner.

These live *outside* :mod:`g1sim.skills.robot` because that module imports ``isaaclab``
at import time (so it can only be imported after the sim app launches). The planner
and the sim-free ``MockSkills`` need the result type and reach geometry without
standing up Isaac, so the pieces with no sim dependency live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Magic-grasp reach geometry. Reach is measured to an object's *footprint* (its XY
# bounding box), NOT its centre: a robot at a table's edge is ~0 m from the table
# even though its centre is a metre away. 0.5 m is then an achievable, physically
# meaningful "at arm's reach". Shared so the mock enforces the same precondition the
# real sim does (you must be at the object to pick/place).
PICK_RADIUS = 0.50        # robot must be within this of an object's footprint to pick (m)
PLACE_RADIUS = 0.50       # robot must be within this of the place target to place (m)


def dropped_pose(o, target_xy, surface_z: float):
    """Where object ``o`` ends up when ``place`` sets it down at ``target_xy`` on a
    surface whose top is at ``surface_z``. Returns ``(position, bbox_min, bbox_max)``
    ready for :meth:`SemanticMap.relocate`.

    The object's origin-to-base offset is preserved, so its base rests exactly on the
    surface, and the **whole bounding box translates with the origin**. That last part
    is load-bearing: reach is measured to an object's footprint (``xy_dist``), and
    ``goto_object`` derives its approach point from the same box, so a bbox left
    behind at the old location sends the robot to the wrong room to pick the object
    up again.

    Shared by both skill implementations -- this arithmetic used to be copied into
    each, which is how they came to disagree with reality in the same way twice.
    """
    base_offset = o.position[2] - o.bbox_min[2]
    position = (float(target_xy[0]), float(target_xy[1]), surface_z + base_offset)
    delta = tuple(position[i] - o.position[i] for i in range(3))
    return (position,
            tuple(o.bbox_min[i] + delta[i] for i in range(3)),
            tuple(o.bbox_max[i] + delta[i] for i in range(3)))


@dataclass
class SkillResult:
    """Outcome of a skill call. Truthy iff the skill succeeded, so callers can do
    ``if skills.pick("cup"):``. ``data`` carries skill-specific extras (e.g. scan
    hits, the resolved object)."""
    ok: bool
    skill: str
    detail: str = ""
    data: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        tag = "OK " if self.ok else "FAIL"
        return f"[{tag}] {self.skill}: {self.detail}"
