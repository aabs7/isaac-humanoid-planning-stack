"""Semantic map of the apartment, built straight from the USD.

A *semantic map* is the planner's model of "what is where": a set of rooms, and a
set of objects each tagged with a category, a room, and a world pose. In Phase 0
we get this for free from the ground-truth USD (this module); later phases replace
the source with an on-robot perception module that back-projects detections into
the map. The *interface* the planner sees -- :class:`SemanticMap` and its query
methods -- is meant to stay identical across that swap, so keep this module free of
any perception/USD specifics leaking into the query API.

Deliberately dependency-light: it needs only ``pxr`` (USD) + stdlib, NOT isaaclab
or a running sim. So it can be built and queried offline/on hardware, and is fast
(a fraction of a second). Import it any time::

    from g1sim.perception.semantic_map import SemanticMap
    smap = SemanticMap.build()            # parse the default apartment USD
    smap.save("sensor_output/semantic_map.json")
    cups = smap.find("cup")               # every cup, nearest-first from a point
    table = smap.nearest("dining_table", (7.5, 0.0))

Coordinate frame: the USD's ``/Root`` frame, which is the same world frame the
robot navigates in (the apartment is referenced into the sim without offset), so
object XY positions are directly usable as navigation goals.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

# Defaults describe the one apartment we ship with. They are just defaults --
# every builder argument can be overridden for another house (Phase 2).
DEFAULT_USD = "/home/abhish/isaac/InteriorAgent/kujiale_0021/kujiale_0021.usda"
DEFAULT_ROOMS_JSON = "/home/abhish/isaac/InteriorAgent/kujiale_0021/rooms.json"

# Scopes under Meshes that are building structure / fixtures, not rooms full of
# objects. Everything else under Meshes is treated as a room scope -- derived, not
# hard-coded to this house's room names, so the builder generalizes.
STRUCTURAL_SCOPES = {"ceiling", "wall", "floor", "other"}

# An object counts as a *support surface* (something other things sit ON) if it has
# a big enough top footprint at a plausible surface height. Derived from geometry,
# not a category whitelist, so it generalizes to other houses (Phase 2).
SUPPORT_MIN_AREA = 0.10      # m^2 top footprint
SUPPORT_MIN_TOP_Z = 0.10     # m -- a surface has to be off the floor
SUPPORT_MAX_TOP_Z = 1.60     # m -- above this it's a shelf top/ceiling fixture, skip
ON_Z_TOL = 0.18              # m -- how far an object's base may sit above a top and still be "on" it

_TRAILING_INDEX = re.compile(r"_\d+$")


def _category_of(prim_name: str) -> str:
    """``cup_0003`` -> ``cup``; ``bathroom_product_0001`` -> ``bathroom_product``.
    Strips the trailing per-instance ``_NNNN`` index the USD authors use."""
    return _TRAILING_INDEX.sub("", prim_name)


def _normalize_room(name: str) -> str:
    """Room label from a scope name or a rooms.json ``room_type``:
    ``livingroom_377`` -> ``livingroom``, ``"living room"`` -> ``livingroom``."""
    return _TRAILING_INDEX.sub("", name).replace(" ", "").lower()


def _is_support(o: "SemanticObject") -> bool:
    """Could this object have things sitting on top of it (table/cabinet/bed/sofa)?"""
    return (o.footprint_area >= SUPPORT_MIN_AREA
            and SUPPORT_MIN_TOP_Z <= o.top_z <= SUPPORT_MAX_TOP_Z)


def _xy_inside_bbox(o: "SemanticObject", surf: "SemanticObject", margin: float = 0.05) -> bool:
    return (surf.bbox_min[0] - margin <= o.xy[0] <= surf.bbox_max[0] + margin
            and surf.bbox_min[1] - margin <= o.xy[1] <= surf.bbox_max[1] + margin)


def _point_in_polygon(x: float, y: float, poly: list) -> bool:
    """Ray-casting point-in-polygon test on an (x, y) vertex list."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


@dataclass
class SemanticObject:
    """One placed object: a semantic label, where it is in the world, and its place
    in the scene graph (which room it is in, what surface supports it, what it
    supports)."""
    name: str                        # unique prim name, e.g. "cup_0001"
    category: str                    # "cup" (name minus the trailing index)
    room: str                        # normalized room, e.g. "livingroom"
    position: tuple                  # world (x, y, z) of the object's origin
    bbox_min: tuple                  # world axis-aligned bounding box, min corner
    bbox_max: tuple                  # world axis-aligned bounding box, max corner
    prim_path: Optional[str] = None    # source USD path (provenance/debugging). For real perception, this is None.
    # Scene-graph edges (filled in after all objects are read):
    supported_by: Optional[str] = None   # name of the surface this rests on ("on" edge)
    supports: list = field(default_factory=list)  # names of objects resting on this one
    held: bool = False                   # True while carried by the robot (not in any room)

    @property
    def xy(self) -> tuple:
        return (self.position[0], self.position[1])

    @property
    def size(self) -> tuple:
        return tuple(self.bbox_max[i] - self.bbox_min[i] for i in range(3))

    @property
    def max_dim(self) -> float:
        return max(self.size)

    @property
    def base_z(self) -> float:
        """Height of the object's underside -- ~0 for floor-standing objects."""
        return self.bbox_min[2]

    @property
    def top_z(self) -> float:
        """Height of the object's top -- a support surface's usable height."""
        return self.bbox_max[2]

    @property
    def footprint_area(self) -> float:
        s = self.size
        return s[0] * s[1]

    def xy_dist(self, px: float, py: float) -> float:
        """Distance in the XY plane from point ``(px, py)`` to this object's
        (axis-aligned) footprint -- 0 if the point is over the footprint. This is
        the physically meaningful reach distance: a robot standing at a table's edge
        is ~0 m from the table object even though its *centre* is a metre away."""
        dx = max(self.bbox_min[0] - px, 0.0, px - self.bbox_max[0])
        dy = max(self.bbox_min[1] - py, 0.0, py - self.bbox_max[1])
        return (dx * dx + dy * dy) ** 0.5


@dataclass
class Room:
    """A room: its footprint (if known) and the objects inside it."""
    name: str                        # normalized, e.g. "kitchen"
    scope: str                       # source scope name, e.g. "kitchen_753"
    polygon: list = field(default_factory=list)   # [(x, y), ...] footprint, may be []
    object_names: list = field(default_factory=list)
    _nav_point: Optional[tuple] = None            # cached representative interior point

    def contains(self, x: float, y: float) -> bool:
        return bool(self.polygon) and _point_in_polygon(x, y, self.polygon)


class SemanticMap:
    """A 3D scene graph of the apartment: an *apartment* root containing *rooms*,
    each containing *objects*, with ``on`` edges linking objects to the support
    surfaces they rest on (``supported_by`` / ``supports``). Every node carries a 3D
    pose + axis-aligned bounding box.

    Build with :meth:`build` (from USD) or :meth:`load` (from saved JSON). Query with
    ``find`` / ``nearest`` / ``objects_in_room`` / ``reachable_in_room`` /
    ``objects_on`` / ``support_of`` / ``navigable_point``; visualize with
    ``describe`` (per-room tally) or ``describe_graph`` (the indented tree).

    (Exported as both ``SemanticMap`` and ``SceneGraph``.)"""

    def __init__(self, rooms: dict, objects: dict):
        self.rooms: dict = rooms          # name -> Room
        self.objects: dict = objects      # name -> SemanticObject

    # ---- construction -------------------------------------------------------
    @classmethod
    def build(cls, usd_path: str = DEFAULT_USD, rooms_json: Optional[str] = DEFAULT_ROOMS_JSON,
              meshes_scope: str = "/Root/Meshes") -> "SemanticMap":
        """Parse ``usd_path`` into a semantic map. ``rooms_json`` (optional) supplies
        room footprints; without it, rooms still exist but have no polygon."""
        from pxr import Usd, UsdGeom  # local import: keep module import isaaclab-free

        stage = Usd.Stage.Open(usd_path)
        if stage is None:
            raise FileNotFoundError(f"could not open USD stage: {usd_path}")
        meshes = stage.GetPrimAtPath(meshes_scope)
        if not meshes:
            raise ValueError(f"no Meshes scope at {meshes_scope} in {usd_path}")

        xform_cache = UsdGeom.XformCache()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        rooms: dict = {}
        objects: dict = {}
        for scope in meshes.GetChildren():
            scope_name = scope.GetName()
            if scope_name in STRUCTURAL_SCOPES:
                continue
            room_name = _normalize_room(scope_name)
            room = rooms.setdefault(room_name, Room(name=room_name, scope=scope_name))
            for obj in scope.GetChildren():
                world = xform_cache.GetLocalToWorldTransform(obj)
                t = world.ExtractTranslation()
                aligned = bbox_cache.ComputeWorldBound(obj).ComputeAlignedRange()
                bmin, bmax = aligned.GetMin(), aligned.GetMax()
                name = obj.GetName()
                objects[name] = SemanticObject(
                    name=name,
                    category=_category_of(name),
                    room=room_name,
                    position=(t[0], t[1], t[2]),
                    bbox_min=(bmin[0], bmin[1], bmin[2]),
                    bbox_max=(bmax[0], bmax[1], bmax[2]),
                    prim_path=str(obj.GetPath()),
                )
                room.object_names.append(name)

        if rooms_json:
            cls._attach_polygons(rooms, rooms_json)
        cls._compute_supports(objects)
        return cls(rooms, objects)

    @staticmethod
    def _compute_supports(objects: dict) -> None:
        """Fill in the ``on`` scene-graph edges: for each object, find the support
        surface whose top it rests on (topmost surface directly below its base, with
        its footprint overlapping that surface). Purely geometric."""
        supports = [o for o in objects.values() if _is_support(o)]
        for o in objects.values():
            best, best_top = None, -1e9
            for surf in supports:
                if surf is o:
                    continue
                # o rests on surf if its base is at surf's top (within tolerance) and
                # its footprint sits over surf. Prefer the highest such surface.
                if abs(o.base_z - surf.top_z) > ON_Z_TOL:
                    continue
                if surf.footprint_area <= o.footprint_area:
                    continue                       # a support must be bigger than what it holds
                if not _xy_inside_bbox(o, surf):
                    continue
                if surf.top_z > best_top:
                    best, best_top = surf, surf.top_z
            if best is not None:
                o.supported_by = best.name
                best.supports.append(o.name)

    @staticmethod
    def _attach_polygons(rooms: dict, rooms_json: str) -> None:
        try:
            with open(rooms_json) as f:
                entries = json.load(f)
        except FileNotFoundError:
            return
        for e in entries:
            rn = _normalize_room(e.get("room_type", ""))
            poly = [(p[0], p[1]) for p in e.get("polygon", [])]
            if rn in rooms:
                rooms[rn].polygon = poly
            else:
                rooms[rn] = Room(name=rn, scope="", polygon=poly)

    # ---- queries ------------------------------------------------------------
    def find(self, category: str, room: Optional[str] = None,
             near: Optional[tuple] = None) -> list:
        """All objects whose category == ``category`` (optionally within ``room``).
        If ``near`` (x, y) is given, results are sorted nearest-first."""
        cat = category.lower()
        hits = [o for o in self.objects.values()
                if o.category == cat and (room is None or o.room == _normalize_room(room))]
        if near is not None:
            hits.sort(key=lambda o: math.hypot(o.xy[0] - near[0], o.xy[1] - near[1]))
        return hits

    def nearest(self, category: str, from_xy: tuple, room: Optional[str] = None) -> Optional[SemanticObject]:
        """The single closest object of ``category`` to ``from_xy`` (or ``None``)."""
        hits = self.find(category, room=room, near=from_xy)
        return hits[0] if hits else None

    def get(self, name: str) -> Optional[SemanticObject]:
        return self.objects.get(name)

    def objects_in_room(self, room: str) -> list:
        rn = _normalize_room(room)
        return [o for o in self.objects.values() if o.room == rn]

    # ---- scene-graph edges --------------------------------------------------
    def objects_on(self, name: str) -> list:
        """Objects resting on the surface ``name`` (its ``supports`` children)."""
        o = self.objects.get(name)
        return [self.objects[n] for n in o.supports] if o else []

    def support_of(self, name: str) -> Optional[SemanticObject]:
        """The surface object ``name`` rests on, or ``None`` if free-standing."""
        o = self.objects.get(name)
        return self.objects.get(o.supported_by) if (o and o.supported_by) else None

    # ---- scene-graph mutation (pick/place keep the graph consistent) --------
    # When the robot moves an object, the semantic map must reflect it: the object
    # leaves its old surface and room and appears at its new one. These methods own
    # that bookkeeping so both the real and mock skills stay consistent by calling
    # them (rather than each hand-editing edges and room lists).
    def reassign_room(self, o: SemanticObject) -> None:
        """Recompute which room ``o`` is in after it moved (polygon test) and fix up
        the room-membership lists. No-op if no room polygon contains it."""
        for name, room in self.rooms.items():
            if room.contains(o.xy[0], o.xy[1]):
                old = self.rooms.get(o.room)
                if old is not None and o.name in old.object_names:
                    old.object_names.remove(o.name)
                o.room = name
                if o.name not in room.object_names:
                    room.object_names.append(o.name)
                return

    def detach(self, o: SemanticObject) -> None:
        """Break the ``on`` edge to whatever currently supports ``o``."""
        if o.supported_by:
            s = self.objects.get(o.supported_by)
            if s is not None and o.name in s.supports:
                s.supports.remove(o.name)
            o.supported_by = None

    def set_carried(self, o: SemanticObject) -> None:
        """Mark ``o`` as picked up: detach it from its surface and remove it from its
        room, so the graph no longer shows it where it was. While carried it belongs
        to no room (``held=True``, ``room=None``)."""
        self.detach(o)
        r = self.rooms.get(o.room)
        if r is not None and o.name in r.object_names:
            r.object_names.remove(o.name)
        o.room = None
        o.held = True

    def relocate(self, o: SemanticObject, position, bbox_min, bbox_max,
                 on_surface: Optional[SemanticObject] = None) -> None:
        """Set ``o`` down at a new pose and re-wire the graph: update its pose, set or
        clear its ``on`` edge (``on_surface`` when placed on a surface, else free-
        standing), reassign its room, and clear the carried flag."""
        o.position = tuple(position)
        o.bbox_min = tuple(bbox_min)
        o.bbox_max = tuple(bbox_max)
        o.held = False
        self.detach(o)                              # clear any stale edge first
        if on_surface is not None:
            o.supported_by = on_surface.name
            if o.name not in on_surface.supports:
                on_surface.supports.append(o.name)
        self.reassign_room(o)

    def room_at(self, x: float, y: float) -> Optional[str]:
        """Name of the room whose footprint contains ``(x, y)``, or ``None`` if the
        point falls outside every room polygon (e.g. in a doorway/between rooms)."""
        for name, room in self.rooms.items():
            if room.contains(x, y):
                return name
        return None

    def nearest_object(self, x: float, y: float, max_dist: float = 2.0):
        """The object whose footprint is closest to ``(x, y)`` within ``max_dist`` m
        (skipping anything currently held), or ``None``. Used to describe where the
        robot is standing ("near dining_table")."""
        best, nearest = max_dist, None
        for o in self.objects.values():
            if o.held:
                continue
            d = o.xy_dist(x, y)
            if d <= best:
                best, nearest = d, o
        return nearest

    def reachable_in_room(self, room: str, from_xy: tuple) -> list:
        """Objects in ``room`` sorted by reach distance (to their footprint, not
        centre) from ``from_xy`` -- i.e. easiest-to-approach first."""
        objs = self.objects_in_room(room)
        objs.sort(key=lambda o: o.xy_dist(from_xy[0], from_xy[1]))
        return objs

    def categories(self) -> dict:
        """Map category -> count, handy for seeing what the environment offers."""
        out: dict = {}
        for o in self.objects.values():
            out[o.category] = out.get(o.category, 0) + 1
        return out

    def small_objects(self, max_dim: float = 0.4) -> list:
        """Objects small enough to plausibly be pick targets (bbox max-dim <= threshold)."""
        return [o for o in self.objects.values() if o.max_dim <= max_dim]

    def navigable_point(self, room: str) -> Optional[tuple]:
        """A representative (x, y) *inside* ``room`` to use as a nav goal. Prefers the
        polygon centroid, but rooms here are L-shaped so the centroid can fall
        outside; then falls back to the mean of the room's object positions (which
        are, by construction, inside the room)."""
        rn = _normalize_room(room)
        r = self.rooms.get(rn)
        if r is None:
            return None
        if r._nav_point is not None:
            return r._nav_point
        pt = None
        if r.polygon:
            cx = sum(p[0] for p in r.polygon) / len(r.polygon)
            cy = sum(p[1] for p in r.polygon) / len(r.polygon)
            if r.contains(cx, cy):
                pt = (cx, cy)
        if pt is None:
            objs = self.objects_in_room(rn)
            if objs:
                pt = (sum(o.xy[0] for o in objs) / len(objs),
                      sum(o.xy[1] for o in objs) / len(objs))
        r._nav_point = pt
        return pt

    def room_names(self) -> list:
        return sorted(self.rooms.keys())

    # ---- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "rooms": {
                n: {"name": r.name, "scope": r.scope, "polygon": r.polygon,
                    "object_names": r.object_names, "nav_point": self.navigable_point(n)}
                for n, r in self.rooms.items()
            },
            "objects": {n: asdict(o) for n, o in self.objects.items()},
        }

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SemanticMap":
        with open(path) as f:
            d = json.load(f)
        rooms = {}
        for n, r in d["rooms"].items():
            room = Room(name=r["name"], scope=r.get("scope", ""),
                        polygon=[tuple(p) for p in r.get("polygon", [])],
                        object_names=r.get("object_names", []))
            np_ = r.get("nav_point")
            room._nav_point = tuple(np_) if np_ else None
            rooms[n] = room
        objects = {}
        for n, o in d["objects"].items():
            objects[n] = SemanticObject(
                name=o["name"], category=o["category"], room=o["room"],
                position=tuple(o["position"]), bbox_min=tuple(o["bbox_min"]),
                bbox_max=tuple(o["bbox_max"]), prim_path=o["prim_path"],
                supported_by=o.get("supported_by"), supports=list(o.get("supports", [])))
        return cls(rooms, objects)

    # ---- human-readable summary ---------------------------------------------
    def describe(self) -> str:
        """One-line-per-room category tally (compact overview)."""
        lines = [f"SceneGraph: {len(self.rooms)} rooms, {len(self.objects)} objects"]
        for name in self.room_names():
            r = self.rooms[name]
            nav = self.navigable_point(name)
            nav_s = f"({nav[0]:.2f}, {nav[1]:.2f})" if nav else "n/a"
            lines.append(f"  {name:12s} nav={nav_s:18s} {len(r.object_names)} objects")
            tally: dict = {}
            for on in r.object_names:
                c = self.objects[on].category
                tally[c] = tally.get(c, 0) + 1
            summary = ", ".join(f"{c}x{n}" if n > 1 else c
                                for c, n in sorted(tally.items()))
            lines.append(f"               {summary}")
        return "\n".join(lines)

    def describe_graph(self, room: Optional[str] = None, without_nav: bool = False) -> str:
        lines = ["apartment"]
        carried = [o for o in self.objects.values() if o.held]
        if carried:
            lines.append("  (carried by robot: "
                         + ", ".join(f"{o.category} [{o.name}]" for o in carried) + ")")
        rooms = [room] if room else self.room_names()
        for rn in rooms:
            rn = _normalize_room(rn)
            r = self.rooms.get(rn)
            if r is None:
                continue
            nav = self.navigable_point(rn)
            nav_s = f"({nav[0]:.2f}, {nav[1]:.2f})" if nav else "n/a"
            line = f"  room:{rn}  nav={nav_s}" if not without_nav else \
                   f"  room:{rn}"
            lines.append(line)

            room_objs = self.objects_in_room(rn)
            supports = [o for o in room_objs if o.supports]
            on_something = {n for o in supports for n in o.supports}
            for surf in sorted(supports, key=lambda o: -o.footprint_area):
                line = f"    {surf.category} [{surf.name}] @({surf.xy[0]:.2f},{surf.xy[1]:.2f}) top_z={surf.top_z:.2f}" if not without_nav else \
                       f"    {surf.category} [{surf.name}]"
                lines.append(line)
                for on in sorted(surf.supports):
                    o = self.objects[on]
                    line = f"      on: {o.category} [{o.name}] @({o.xy[0]:.2f},{o.xy[1]:.2f})" if not without_nav else \
                           f"      on: {o.category} [{o.name}]"
                    lines.append(line)
            free = [o for o in room_objs if o.name not in on_something and not o.supports]
            if free:
                lines.append("    (free-standing / floor: "
                             + ", ".join(sorted(o.category for o in free)) + ")")
        return "\n".join(lines)


# The semantic map *is* a 3D scene graph; expose both names.
SceneGraph = SemanticMap
