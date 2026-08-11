"""How the robot *holds* an object -- the one part of pick/place that is a physics question.

Split out of :mod:`g1sim.skills.robot` so a real, physical grasp can be added beside the
magic one and selected per run. The seam is deliberately narrow. A strategy owns four
things and nothing else:

    attach(host, o)                    make the object come with the hand
    release(host, o, position)         let go, and report where it actually ended up
    on_control(host)                   before joint targets are flushed to the sim
    on_tick(host)                      after the physics substeps

The reach preconditions, the place-target resolution, the drop arithmetic
(:func:`~g1sim.skills.types.dropped_pose`) and the semantic-map mutations (``set_carried`` /
``relocate``) all stay in ``RobotSkills``, shared by every strategy. That is not tidiness:
``dropped_pose``'s docstring records that this arithmetic was once copied into two skill
implementations and came to disagree with reality in the same way twice. A strategy that
owned the map mutation would let that happen a third time.

Two bits of shape exist only to anticipate the physical strategy, and are cheap now versus
expensive later:

* **Two hooks, not one.** A real arm has to write joint targets *before*
  ``scene.write_data_to_sim()``, while the magic carry teleports a prim *after* the substeps.
  ``MagicGrasp.on_control`` is simply ``pass``.
* **:class:`GraspOutcome` carries a pose.** A magic release ends exactly where it was told;
  a real one ends wherever the object settles. Callers should not have to know which, so
  ``release`` reports the achieved pose either way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from g1sim.sim.scene import APARTMENT_PRIM

# Magic-carry pose: where a held object rides relative to the robot base each tick.
CARRY_FORWARD = 0.35      # metres in front of the base (out past the chest)
CARRY_Z = 0.95            # world height it is carried at (~chest)


@dataclass
class GraspOutcome:
    """Result of an attach or a release."""
    ok: bool
    detail: str = ""
    data: dict = field(default_factory=dict)
    # Where the object actually ended up, when the strategy measured it. ``None`` means
    # "no better information than the caller's prediction".
    position: Optional[tuple] = None

    def __bool__(self) -> bool:
        return self.ok


def sim_prim_path(semantic_prim_path: str, apartment_prim: str = APARTMENT_PRIM) -> str:
    """Semantic-map prim path -> the path on the running stage.

    The map records paths rooted at the apartment USD's default prim (``/Root/...``); on the
    stage that default prim is collapsed onto ``/World/Apartment``.
    """
    suffix = semantic_prim_path
    if suffix.startswith("/Root"):
        suffix = suffix[len("/Root"):]
    return apartment_prim + suffix


class GraspStrategy(Protocol):
    """What ``RobotSkills`` needs from a way of holding things.

    ``host`` is the ``RobotSkills`` instance -- passed per call rather than held, so a
    strategy stays cheap to construct and drivable by a fake in tests.
    """

    name: str

    def attach(self, host, o) -> GraspOutcome:
        """Begin holding ``o``."""

    def release(self, host, o, position) -> GraspOutcome:
        """Let ``o`` go, aiming for world ``position``."""

    def on_control(self, host) -> None:
        """Called once per control tick, before joint targets reach the sim."""

    def on_tick(self, host) -> None:
        """Called once per control tick, after the physics substeps."""


class MagicGrasp:
    """Holding by fiat: the object's prim is driven to the robot's chest every tick.

    Not a grasp at all -- the hand is never involved and the object need not be anywhere
    near it. It cannot fail for a physical reason, which is exactly why the planner's
    failure handling above the seam went so long without being exercised by anything real.
    Kept because it is fast and reliable, so a multi-object fetch task can run end to end
    while a real grasp is still unreliable.

    Collision is disabled while carried, and not merely for appearances: an enabled follower
    collider 0.35 m ahead of the robot becomes an obstacle it can never walk past.
    """

    name = "magic"

    def __init__(self, stage):
        self.stage = stage
        self._translate_op = None

    # -- GraspStrategy ----------------------------------------------------
    def attach(self, host, o) -> GraspOutcome:
        self._translate_op = self._grab_prim(host, o)
        self.on_tick(host)                    # snap it to the chest immediately
        return GraspOutcome(True, f"holding {o.name}")

    def release(self, host, o, position) -> GraspOutcome:
        self._set_translate(self._translate_op, position)
        self._release_prim(o)
        self._translate_op = None
        # A magic release ends exactly where it was told, so there is nothing to measure.
        return GraspOutcome(True, "", position=tuple(position))

    def on_control(self, host) -> None:
        """Nothing to do: the magic carry writes no joint targets."""

    def on_tick(self, host) -> None:
        """Drive the held object to the robot's chest this tick (the render follows)."""
        if host.held is None or self._translate_op is None:
            return
        x, y, h = host.pose()
        self._set_translate(self._translate_op,
                            (x + CARRY_FORWARD * math.cos(h),
                             y + CARRY_FORWARD * math.sin(h), CARRY_Z))

    # -- prim manipulation ------------------------------------------------
    def _grab_prim(self, host, o):
        """Fetch the object's ``xformOp:translate`` op (so carry/release can drive it) and
        disable its collision. Returns the op, or None if the prim/op cannot be found --
        the carry then stays purely logical, and the semantic-map relocation still happens."""
        prim = self.stage.GetPrimAtPath(sim_prim_path(o.prim_path))
        if not prim or not prim.IsValid():
            host._log(f"[skill] warn: carried prim not found on stage for {o.name}")
            return None
        self._set_collision(prim, False)
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                return op
        return xf.AddTranslateOp()   # no translate op authored; add one so we can drive it

    def _release_prim(self, o):
        prim = self.stage.GetPrimAtPath(sim_prim_path(o.prim_path))
        if prim and prim.IsValid():
            self._set_collision(prim, True)

    @staticmethod
    def _set_collision(prim, enabled: bool):
        """Toggle collision on every collider under ``prim`` (both the UsdPhysics and PhysX
        schemas, whichever is authored)."""
        try:
            from pxr import PhysxSchema
        except Exception:
            PhysxSchema = None
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                try:
                    UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(enabled)
                except Exception:
                    pass
            if PhysxSchema is not None and p.HasAPI(PhysxSchema.PhysxCollisionAPI):
                try:
                    p.GetAttribute("physxCollision:collisionEnabled").Set(enabled)
                except Exception:
                    pass

    @staticmethod
    def _set_translate(op, xyz):
        if op is None:
            return
        try:
            op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
        except Exception:
            pass
