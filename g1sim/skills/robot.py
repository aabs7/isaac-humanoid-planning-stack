"""Skill API -- the composable verbs a task planner calls.

This is the seam the whole project is organized around: the planner (LLM/PDDL,
later) speaks only in skills -- ``goto``, ``scan``, ``pick``, ``place`` -- and each
returns a :class:`SkillResult` (success/failure + detail). Everything sim-specific
(the locomotion policy, the occupancy mapper, moving USD prims) lives *below* this
line, so the same planner can eventually drive real hardware by swapping only the
skill *implementations* and the localization source. Keep that boundary clean.

Phase-0 scope (stubs, to be deepened later):
  * ``goto(x, y)`` / ``goto_room`` / ``goto_object`` -- real closed-loop navigation
    with optimistic online mapping + A* obstacle avoidance (reuses the mapping/
    planning stack).
  * ``scan()`` -- stub "perception": reports nearby objects from the ground-truth
    semantic map. Later: rotate in place and run detection on the RGB-D camera.
  * ``pick`` / ``place`` -- *magic* grasp: ``pick`` disables the object's collision
    and carries it at the robot's chest (its prim is driven there each tick), and
    ``place`` re-enables collision and sets it down on the destination surface/floor.
    The authoritative result is the updated semantic-map state (held / relocated).
    Later: real IK grasp from the loco-manipulation env.

Import only after the sim app is launched.
"""

from __future__ import annotations

import math
from typing import Optional

import isaaclab.sim as sim_utils
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from g1sim.sim.locomotion import CONTROL_HZ
from g1sim.navigation.waypoint import WaypointNavigator
from g1sim.perception.mapping import OccupancyGridMapper
from g1sim.navigation.path_planning import plan_path
from g1sim.sim.scene import APARTMENT_PRIM
from g1sim.skills.types import SkillResult, PICK_RADIUS, PLACE_RADIUS

# Online-nav loop cadences (control ticks). Mirror the tuning in the standalone
# map_and_navigate entry point so behavior is identical wherever goto runs.
_INTEGRATE_EVERY = 3     # ticks between lidar fusions
_REPLAN_EVERY = 30       # ticks between A* re-plans (~0.6 s)

# Magic-grasp geometry (PICK_RADIUS / PLACE_RADIUS) now lives in
# ``g1sim.skills.types`` so the sim-free planner + mock can share it; imported above.
MAX_RECOVERIES = 8        # give up a goto after this many failed stuck-recoveries
# Carry pose: where a held object rides relative to the robot base each tick.
CARRY_FORWARD = 0.35      # metres in front of the base (out past the chest)
CARRY_Z = 0.95            # world height it is carried at (~chest)


def _sim_prim_path(semantic_prim_path: str, apartment_prim: str = APARTMENT_PRIM) -> str:
    """Translate a semantic-map prim path (parsed from the raw USD, rooted at
    ``/Root``) to its on-stage path. The apartment USD is referenced under
    ``apartment_prim`` and its default prim ``/Root`` collapses onto that target,
    so ``/Root/Meshes/...`` lives at ``<apartment_prim>/Meshes/...``."""
    suffix = semantic_prim_path
    if suffix.startswith("/Root"):
        suffix = suffix[len("/Root"):]
    return apartment_prim + suffix


class RobotSkills:
    """Bind the low-level sim stack (locomotion, sensors, mapper) + the semantic map
    into the verb set a planner drives. One instance per running world."""

    def __init__(self, sim, scene, controller, semantic_map, *,
                 app=None, use_lidar: bool = True, on_step=None, verbose: bool = True):
        self.sim = sim
        self.app = app                    # simulation_app, for the GUI-close check
        self.scene = scene
        self.robot = scene["robot"]
        self.controller = controller
        self.smap = semantic_map
        self.nav = WaypointNavigator(controller, goal_tol=0.3)
        self.mapper = OccupancyGridMapper() if (use_lidar and "lidar" in scene.sensors) else None
        self.on_step = on_step            # called(self) after each control tick (GUI hooks)
        self.verbose = verbose
        self.stage = sim_utils.get_current_stage()

        # Carry state (magic grasp).
        self.held = None                  # SemanticObject being carried, or None
        self._carry_translate_op = None   # cached xformOp:translate of the carried prim

        # Exposed for a live map window: last plan the online goto produced.
        self.last_free = None
        self.last_waypoints = None
        self.last_goal = None

    # -- logging ----------------------------------------------------------
    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # -- pose -------------------------------------------------------------
    def pose(self):
        """Robot base pose ``(x, y, heading)`` in world."""
        return self.nav.pose2d()

    def xy(self):
        x, y, _ = self.pose()
        return (x, y)

    # -- low-level sim stepping ------------------------------------------
    def step(self, command):
        """Advance one control tick: run the policy for ``command`` and step the
        sim ``decimation`` times, then update the carried object and any GUI hook."""
        self.controller.step(command)
        for _ in range(self.controller.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())
        self._carry_follow()
        if self.on_step is not None:
            self.on_step(self)

    def idle(self, seconds: float):
        """Stand in place (still running the balance policy) for ``seconds``."""
        for _ in range(int(seconds * CONTROL_HZ)):
            if not self._running():
                break
            self.step(self.controller.command())

    def _running(self) -> bool:
        return self.app.is_running() if self.app is not None else True

    # -- sensing ----------------------------------------------------------
    def _integrate_lidar(self):
        if self.mapper is None or "lidar" not in self.scene.sensors:
            return
        lidar = self.scene["lidar"].data
        if lidar.ray_hits_w is not None:
            self.mapper.integrate(
                lidar.ray_hits_w.torch[0].detach().cpu().numpy(),
                sensor_xyz=lidar.pos_w.torch[0].detach().cpu().numpy())

    # ===================================================================
    # SKILLS
    # ===================================================================
    def goto(self, x: float, y: float, *, timeout_s: float = 90.0, goal_tol: float = 0.35,
             reach_xy=None, reach_obj=None, reach_dist: float = 0.0) -> SkillResult:
        """Navigate to world ``(x, y)``. With a lidar, uses optimistic online
        mapping + A* to route around sensed furniture; otherwise walks straight.

        Early-stop options (used by ``goto_object`` to halt at arm's reach without
        burrowing into furniture inflation): ``reach_xy`` succeeds within
        ``reach_dist`` of a point; ``reach_obj`` succeeds within ``reach_dist`` of a
        SemanticObject's *footprint*. A stuck-detector backs the robot out and
        re-plans if it stops making progress (doorways, tight corners)."""
        goal = (float(x), float(y))
        self.last_goal = goal
        if self.mapper is None:
            return self._goto_straight(goal, timeout_s, goal_tol, reach_xy, reach_obj, reach_dist)

        waypoints, free, _ = plan_path(self.mapper, self.xy(), goal)
        self.last_free, self.last_waypoints = free, waypoints
        max_ticks = int(timeout_s * CONTROL_HZ)
        replans = 0
        best_goal_dist = math.hypot(goal[0] - self.xy()[0], goal[1] - self.xy()[1])
        stuck_since = 0
        recoveries = 0
        for tick in range(max_ticks):
            if not self._running():
                return SkillResult(False, "goto", "sim closed")
            if tick % _INTEGRATE_EVERY == 0:
                self._integrate_lidar()
            if tick % _REPLAN_EVERY == 0:
                wps, free, _ = plan_path(self.mapper, self.xy(), goal)
                self.last_free = free
                if wps:
                    waypoints, replans = wps, replans + 1
            self.last_waypoints = waypoints

            px, py = self.xy()
            if self._reached(px, py, reach_xy, reach_obj, reach_dist):
                return SkillResult(True, "goto", f"within {reach_dist:.2f} m of target at ({px:.2f}, {py:.2f})")
            reach = waypoints[-1] if waypoints else goal
            if math.hypot(reach[0] - px, reach[1] - py) < goal_tol:
                resid = math.hypot(goal[0] - px, goal[1] - py)
                if resid < goal_tol:
                    return SkillResult(True, "goto", f"reached ({px:.2f}, {py:.2f}) in {replans} replans")
                return SkillResult(True, "goto",
                                   f"reached closest free point ({px:.2f}, {py:.2f}); "
                                   f"goal {resid:.2f} m inside an obstacle")

            # Stuck detection by *progress toward the goal* over ~1.5 seconds (catches
            # oscillation, where the robot moves but nets no ground toward the goal).
            if tick - stuck_since >= 1.5 * CONTROL_HZ:
                goal_dist = math.hypot(goal[0] - px, goal[1] - py)
                if goal_dist > best_goal_dist - 0.15:
                    recoveries += 1
                    if recoveries > MAX_RECOVERIES:
                        return SkillResult(False, "goto",
                                           f"stuck at ({px:.2f}, {py:.2f}) after {recoveries} recoveries")
                    self._recover(toward=goal)
                    wps, free, _ = plan_path(self.mapper, self.xy(), goal)
                    self.last_free = free
                    if wps:
                        waypoints = wps
                else:
                    recoveries = 0        # made progress; reset the recovery budget
                best_goal_dist = min(best_goal_dist, goal_dist)
                stuck_since = tick

            target = (waypoints[1] if len(waypoints) > 1
                      else (waypoints[0] if waypoints else goal))
            command, _, _ = self.nav.command_to(target)
            self.step(command)
        return SkillResult(False, "goto", f"timeout after {timeout_s:.0f}s at ({px:.2f}, {py:.2f})")

    def _reached(self, px, py, reach_xy, reach_obj, reach_dist) -> bool:
        if reach_obj is not None and reach_obj.xy_dist(px, py) <= reach_dist:
            return True
        if reach_xy is not None and math.hypot(reach_xy[0] - px, reach_xy[1] - py) <= reach_dist:
            return True
        return False

    def _recover(self, toward=None, back_ticks: int = 30, push_ticks: int = 60):
        """Un-stick: back up a little to break contact, rotate to face ``toward``
        (the goal -- i.e. the way out of a dead-end), then push forward through it.
        Cheap and stateless; the caller re-plans afterwards."""
        self._log("[skill] goto: stuck -> recovering (back up, turn to goal, push)")
        for _ in range(back_ticks):
            if not self._running():
                return
            self.step(self.controller.command(vx=-0.25, vy=0.0, wz=0.0))
        if toward is not None:
            self._face(toward[0], toward[1], tol=0.25, max_ticks=60)
        for _ in range(push_ticks):
            if not self._running():
                return
            self.step(self.controller.command(vx=0.4, vy=0.0, wz=0.0))

    def _goto_straight(self, goal, timeout_s, goal_tol, reach_xy=None, reach_obj=None, reach_dist=0.0):
        max_ticks = int(timeout_s * CONTROL_HZ)
        for _ in range(max_ticks):
            if not self._running():
                return SkillResult(False, "goto", "sim closed")
            px, py = self.xy()
            if self._reached(px, py, reach_xy, reach_obj, reach_dist):
                return SkillResult(True, "goto", f"within {reach_dist:.2f} m at ({px:.2f}, {py:.2f})")
            command, dist, arrived = self.nav.command_to(goal)
            if dist < goal_tol:
                return SkillResult(True, "goto", f"reached ({px:.2f}, {py:.2f})")
            self.step(command)
        return SkillResult(False, "goto", f"timeout after {timeout_s:.0f}s")

    def goto_room(self, room: str, **kw) -> SkillResult:
        """Navigate to a representative interior point of ``room``."""
        pt = self.smap.navigable_point(room)
        if pt is None:
            return SkillResult(False, "goto_room", f"unknown room '{room}'")
        self._log(f"[skill] goto_room({room}) -> ({pt[0]:.2f}, {pt[1]:.2f})")
        res = self.goto(pt[0], pt[1], **kw)
        res.skill = "goto_room"
        res.data["room"] = room
        return res

    def _resolve_object(self, obj):
        """Accept an object name, a category (nearest of that category is chosen),
        or a SemanticObject; return the SemanticObject or None."""
        if hasattr(obj, "prim_path"):
            return obj
        found = self.smap.get(obj)
        if found is not None:
            return found
        return self.smap.nearest(obj, self.xy())

    def goto_object(self, obj, *, standoff: float = 0.45, goal_tol: float = 0.3, **kw) -> SkillResult:
        """Navigate up to ``obj`` (by name, category, or SemanticObject) and face it,
        stopping within ``PICK_RADIUS`` of its footprint.

        The approach point is ``standoff`` metres out from the object's *nearest
        footprint edge* toward the robot -- i.e. on the robot's own, already-open
        side. Aiming at the edge (not the object's centre, which sits inside the
        furniture's obstacle inflation and can snap to the wrong side of a wall) lets
        the robot pull right up to a table/cabinet and be within reach of it."""
        o = self._resolve_object(obj)
        if o is None:
            return SkillResult(False, "goto_object", f"no object matching '{obj}'")
        rx, ry = self.xy()
        # Closest point on the object's footprint to the robot, then step `standoff`
        # back toward the robot to get a stand pose just outside the obstacle.
        nx = min(max(rx, o.bbox_min[0]), o.bbox_max[0])
        ny = min(max(ry, o.bbox_min[1]), o.bbox_max[1])
        dx, dy = rx - nx, ry - ny
        d = math.hypot(dx, dy)
        if d < 1e-3:
            dx, dy = rx - o.xy[0], ry - o.xy[1]
            d = math.hypot(dx, dy) or 1.0
        ax, ay = nx + dx / d * standoff, ny + dy / d * standoff
        self._log(f"[skill] goto_object({o.name} in {o.room}) -> approach ({ax:.2f}, {ay:.2f}), "
                  f"stop within {PICK_RADIUS:.2f} m of footprint")
        res = self.goto(ax, ay, goal_tol=goal_tol, reach_obj=o, reach_dist=PICK_RADIUS, **kw)
        # The planner's robot-radius inflation won't let A* route the final ~0.3 m up
        # to the object, so goto can stop just outside reach. Close that last gap with
        # a short open-loop creep (safe: the physical collider stops the robot, and a
        # magic pick lifts the object away immediately after).
        self._creep_to(o, PICK_RADIUS)
        self._face(o.xy[0], o.xy[1])
        px, py = self.xy()
        dd = o.xy_dist(px, py)
        ok = dd <= PICK_RADIUS
        return SkillResult(ok, "goto_object",
                           f"{'within' if ok else 'stopped'} {dd:.2f} m of {o.name} "
                           f"at ({px:.2f}, {py:.2f})", data={"object": o.name})

    def _creep_to(self, o, reach, max_ticks=90):
        """Nose straight toward object ``o`` until within ``reach`` of its footprint
        or blocked. Bypasses the map (used for the final approach where planning
        inflation would otherwise stop the robot short)."""
        for _ in range(max_ticks):
            if not self._running():
                return
            px, py = self.xy()
            if o.xy_dist(px, py) <= reach:
                return
            self._face(o.xy[0], o.xy[1], tol=0.3, max_ticks=20)
            self.step(self.controller.command(vx=0.3, vy=0.0, wz=0.0))

    def _face(self, tx, ty, tol=0.15, max_ticks=200):
        """Turn in place to face world point (tx, ty)."""
        for _ in range(max_ticks):
            if not self._running():
                return
            x, y, h = self.pose()
            yaw_err = math.atan2(ty - y, tx - x) - h
            yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))
            if abs(yaw_err) < tol:
                return
            wz = max(-1.0, min(1.0, 1.5 * yaw_err))
            self.step(self.controller.command(vx=0.0, vy=0.0, wz=wz))

    def scan(self, radius: float = 3.0) -> SkillResult:
        """Stub perception: report objects within ``radius`` of the robot, from the
        (ground-truth) semantic map. Stands in for rotating + running detection on
        the RGB-D camera; the return shape (list of labeled poses) is what a real
        perception ``scan`` will produce too."""
        rx, ry = self.xy()
        hits = []
        for o in self.smap.objects.values():
            d = math.hypot(o.xy[0] - rx, o.xy[1] - ry)
            if d <= radius:
                hits.append((o.name, o.category, o.room, round(d, 2)))
        hits.sort(key=lambda h: h[3])
        self._log(f"[skill] scan: {len(hits)} objects within {radius:.1f} m")
        for name, cat, room, d in hits[:12]:
            self._log(f"          {name:24s} ({cat}) in {room} @ {d} m")
        return SkillResult(True, "scan", f"{len(hits)} objects within {radius:.1f} m",
                           data={"hits": hits})

    def pick(self, obj) -> SkillResult:
        """Magic-grasp ``obj`` (name/category/SemanticObject). Requires the robot to
        already be within ``PICK_RADIUS`` (compose ``goto_object`` first). On success
        the object is marked held and thereafter carried in front of the chest."""
        if self.held is not None:
            return SkillResult(False, "pick", f"already holding {self.held.name}")
        o = self._resolve_object(obj)
        if o is None:
            return SkillResult(False, "pick", f"no object matching '{obj}'")
        rx, ry = self.xy()
        d = o.xy_dist(rx, ry)                      # distance to the object's footprint
        if d > PICK_RADIUS:
            return SkillResult(False, "pick",
                               f"{o.name} is {d:.2f} m from reach (> {PICK_RADIUS} m); goto it first")

        prev_room = o.room
        self.held = o
        self._carry_translate_op = self._grab_prim(o)   # also disables its collision
        self._carry_follow()                            # snap it to the chest immediately
        self.smap.set_carried(o)                         # graph: it left its surface/room
        self._log(f"[skill] pick({o.name}) OK (was in {prev_room})")
        return SkillResult(True, "pick", f"holding {o.name}", data={"object": o.name})

    def place(self, location) -> SkillResult:
        """Set the held object down at ``location``: a room name (dropped on the floor
        at its nav point), a world ``(x, y)`` tuple, or another object's name (placed
        on that object's top surface). Requires being within ``PLACE_RADIUS``."""
        if self.held is None:
            return SkillResult(False, "place", "not holding anything")

        target_xy, surface_z, where, surf = self._resolve_place(location)
        if target_xy is None:
            return SkillResult(False, "place", f"cannot resolve place location '{location}'")
        rx, ry = self.xy()
        # Reach to a surface object is measured to its footprint; a bare point/room
        # floor target is measured to the point.
        d = surf.xy_dist(rx, ry) if surf is not None else math.hypot(target_xy[0] - rx, target_xy[1] - ry)
        if d > PLACE_RADIUS:
            return SkillResult(False, "place",
                               f"place target is {d:.2f} m from reach (> {PLACE_RADIUS} m); goto it first")

        o = self.held
        # Keep the object's origin-to-base offset so its base rests on the surface.
        base_offset = o.position[2] - o.bbox_min[2]
        drop = (target_xy[0], target_xy[1], surface_z + base_offset)
        self._set_prim_translate(self._carry_translate_op, drop)
        self._release_prim(o)

        # Update the authoritative semantic-map state (pose, room, and 'on' edges):
        # if placed on an object, `surf` becomes its new support; else free-standing.
        dz = drop[2] - o.position[2]
        new_min = (o.bbox_min[0], o.bbox_min[1], o.bbox_min[2] + dz)
        new_max = (o.bbox_max[0], o.bbox_max[1], o.bbox_max[2] + dz)
        self.smap.relocate(o, drop, new_min, new_max, on_surface=surf)

        self.held = None
        self._carry_translate_op = None
        self._log(f"[skill] place({where}) OK: {o.name} now at "
                  f"({drop[0]:.2f}, {drop[1]:.2f}, {drop[2]:.2f}) in {o.room}")
        return SkillResult(True, "place", f"{o.name} placed at {where}",
                           data={"object": o.name, "at": drop})

    # -- place-location resolution ---------------------------------------
    def _resolve_place(self, location):
        """Return (target_xy, surface_z, human_label, surface_obj_or_None) for a
        place target. ``surface_obj`` is set when placing ON an object, so ``place``
        can measure reach to its footprint."""
        if isinstance(location, (tuple, list)) and len(location) == 2:
            return (float(location[0]), float(location[1])), 0.0, f"({location[0]:.2f}, {location[1]:.2f})", None
        if isinstance(location, str):
            if self.smap.get(location) is not None:            # place ON an object (surface)
                s = self.smap.get(location)
                return s.xy, s.top_z, f"on {location}", s
            pt = self.smap.navigable_point(location)           # place in a room (floor)
            if pt is not None:
                return pt, 0.0, f"in {location}", None
        if hasattr(location, "xy"):                            # a SemanticObject surface
            return location.xy, location.top_z, f"on {location.name}", location
        return None, None, str(location), None

    # -- carried-prim manipulation ---------------------------------------
    # Visible magic carry: while held, the object's prim is driven to the robot's
    # chest every tick (setting its translate op moves the render). Its collision is
    # disabled on pick so the following collider can't wedge the walking robot (an
    # enabled follower 0.35 m ahead becomes an obstacle it can never pass), and
    # re-enabled on place so the object rests solidly again.
    def _grab_prim(self, o):
        """Fetch the object's ``xformOp:translate`` op (so carry/place can drive it)
        and disable its collision. Returns the op, or None if the prim/op can't be
        found (carry then stays logical -- the semantic-map relocation still happens)."""
        prim = self.stage.GetPrimAtPath(_sim_prim_path(o.prim_path))
        if not prim or not prim.IsValid():
            self._log(f"[skill] warn: carried prim not found on stage for {o.name}")
            return None
        self._set_collision(prim, False)
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                return op
        return xf.AddTranslateOp()  # no translate op authored; add one so we can drive it

    def _release_prim(self, o):
        prim = self.stage.GetPrimAtPath(_sim_prim_path(o.prim_path))
        if prim and prim.IsValid():
            self._set_collision(prim, True)

    def _set_collision(self, prim, enabled: bool):
        """Toggle collision on every collider under ``prim`` (both the UsdPhysics and
        PhysX schemas, whichever is authored)."""
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

    def _carry_follow(self):
        """Drive the held object to the robot's chest this tick (render follows)."""
        if self.held is None or self._carry_translate_op is None:
            return
        x, y, h = self.pose()
        self._set_prim_translate(self._carry_translate_op,
                                 (x + CARRY_FORWARD * math.cos(h),
                                  y + CARRY_FORWARD * math.sin(h), CARRY_Z))

    @staticmethod
    def _set_prim_translate(op, xyz):
        if op is None:
            return
        try:
            op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
        except Exception:
            pass
