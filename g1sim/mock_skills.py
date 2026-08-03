"""Sim-free stand-in for :class:`g1sim.skills.RobotSkills`.

Same interface the planner drives -- ``.smap``, ``.xy()``, ``.held`` and
``scan/goto_room/goto_object/pick/place`` returning :class:`SkillResult` -- but backed
by a virtual robot pose and the semantic map alone (no Isaac, no locomotion). This is
what lets the LLM planner and its prompts be developed and unit-tested in seconds
instead of minutes-per-sim-launch; the *identical* planner then drives the real
skills. It mirrors the real skills' reach preconditions and pick/place map mutations
so the reasoning it exercises is faithful.
"""

from __future__ import annotations

import math

from g1sim.skill_types import SkillResult, PICK_RADIUS, PLACE_RADIUS


class MockSkills:
    def __init__(self, smap, start_xy=(0.0, 0.0), *, verbose: bool = True):
        self.smap = smap
        self._xy = (float(start_xy[0]), float(start_xy[1]))
        self.held = None
        self.verbose = verbose

    # -- interface parity with RobotSkills --------------------------------
    def xy(self):
        return self._xy

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _resolve_object(self, ref):
        if hasattr(ref, "prim_path"):
            return ref
        o = self.smap.get(ref)
        if o is not None:
            return o
        return self.smap.nearest(ref, self._xy)

    # -- skills -----------------------------------------------------------
    def scan(self, radius: float = 3.0) -> SkillResult:
        rx, ry = self._xy
        hits = []
        for o in self.smap.objects.values():
            d = math.hypot(o.xy[0] - rx, o.xy[1] - ry)
            if d <= radius:
                hits.append((o.name, o.category, o.room, round(d, 2)))
        hits.sort(key=lambda h: h[3])
        return SkillResult(True, "scan", f"{len(hits)} objects within {radius:.1f} m",
                           data={"hits": hits})

    def goto_room(self, room: str, **kw) -> SkillResult:
        pt = self.smap.navigable_point(room)
        if pt is None:
            return SkillResult(False, "goto_room", f"unknown room '{room}'")
        self._xy = (pt[0], pt[1])
        return SkillResult(True, "goto_room", f"at {room} ({pt[0]:.2f}, {pt[1]:.2f})",
                           data={"room": room})

    def goto_object(self, obj, **kw) -> SkillResult:
        o = self._resolve_object(obj)
        if o is None:
            return SkillResult(False, "goto_object", f"no object matching '{obj}'")
        # Stand at the object's nearest footprint edge -> within PICK_RADIUS of it,
        # so a following pick's reach precondition is satisfied (as in the real sim).
        rx, ry = self._xy
        nx = min(max(rx, o.bbox_min[0]), o.bbox_max[0])
        ny = min(max(ry, o.bbox_min[1]), o.bbox_max[1])
        self._xy = (nx, ny)
        return SkillResult(True, "goto_object", f"within reach of {o.name}",
                           data={"object": o.name})

    def pick(self, obj) -> SkillResult:
        if self.held is not None:
            return SkillResult(False, "pick", f"already holding {self.held.name}")
        o = self._resolve_object(obj)
        if o is None:
            return SkillResult(False, "pick", f"no object matching '{obj}'")
        d = o.xy_dist(*self._xy)
        if d > PICK_RADIUS:
            return SkillResult(False, "pick",
                               f"{o.name} is {d:.2f} m from reach (> {PICK_RADIUS} m); "
                               f"goto_object it first")
        self.held = o
        self.smap.set_carried(o)      # graph: it left its surface/room
        return SkillResult(True, "pick", f"holding {o.name}", data={"object": o.name})

    def place(self, location) -> SkillResult:
        if self.held is None:
            return SkillResult(False, "place", "not holding anything")
        target_xy, surface_z, where, surf = self._resolve_place(location)
        if target_xy is None:
            return SkillResult(False, "place", f"cannot resolve place location '{location}'")
        rx, ry = self._xy
        d = surf.xy_dist(rx, ry) if surf is not None else math.hypot(
            target_xy[0] - rx, target_xy[1] - ry)
        if d > PLACE_RADIUS:
            return SkillResult(False, "place",
                               f"place target is {d:.2f} m from reach (> {PLACE_RADIUS} m); "
                               f"go to it first")
        o = self.held
        # Same drop math + map mutation as RobotSkills.place: keep the object's
        # origin-to-base offset so its base rests on the surface, shift its bbox, and
        # let the map re-wire room + 'on' edges (surf is the surface when placing ON one).
        base_offset = o.position[2] - o.bbox_min[2]
        drop = (target_xy[0], target_xy[1], surface_z + base_offset)
        dz = drop[2] - o.position[2]
        new_min = (o.bbox_min[0], o.bbox_min[1], o.bbox_min[2] + dz)
        new_max = (o.bbox_max[0], o.bbox_max[1], o.bbox_max[2] + dz)
        self.smap.relocate(o, drop, new_min, new_max, on_surface=surf)
        self.held = None
        return SkillResult(True, "place", f"{o.name} placed {where}",
                           data={"object": o.name, "at": drop})

    # -- helpers (mirror RobotSkills) -------------------------------------
    def _resolve_place(self, location):
        if isinstance(location, (tuple, list)) and len(location) == 2:
            return ((float(location[0]), float(location[1])), 0.0,
                    f"({location[0]:.2f}, {location[1]:.2f})", None)
        if isinstance(location, str):
            s = self.smap.get(location)
            if s is not None:                                   # place ON an object
                return s.xy, s.top_z, f"on {location}", s
            pt = self.smap.navigable_point(location)            # place in a room
            if pt is not None:
                return pt, 0.0, f"in {location}", None
        if hasattr(location, "xy"):
            return location.xy, location.top_z, f"on {location.name}", location
        return None, None, str(location), None
