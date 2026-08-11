"""Projection of the semantic map onto railroad's symbol vocabulary.

railroad plans over a flat, typed world of *strings*: ``robot``, ``location``,
``object``, related by fluents like ``at cup_0003 dining_table_0001``. Our
:class:`~g1sim.perception.semantic_map.SemanticMap` is continuous -- 3D poses, bounding
boxes and ``on`` edges. This module is the (sim-free, pure) bridge between the two, and
it is the only place that knows how one maps onto the other.

The choices it makes, and why:

**A ``location`` is a support surface or a room.** Not a room alone: "which table?" is
exactly the question we want the planner to answer, and rooms-only pushes it down into
the skills where nothing reasons. Not every object either: a symbol is only useful as a
location if the robot can *stand at it*, which is what ``goto_object`` provides for
furniture-sized things.

**A location's coordinates are the object's centre**, in metres, in our world frame --
*not* railroad's grid cells (``railroad.environment.types.Pose`` is ``(row, col)`` in
cells; we never hand railroad geometry, only scalar costs, precisely to avoid that
conversion living in two places).

**Surfaces and pickable objects are disjoint.** Four objects in this apartment qualify as
both (two books, two trays: small enough to carry, flat enough to stack on). Pickable
wins -- a symbol in both type sets would ground nonsense actions like
``pick g1 tray_0001 tray_0001``.

**Symbols must not contain whitespace.** railroad's grounded action name is
``"<op> <robot> <args...>"`` and *that string is the dispatch key* -- the planner, the
skill lookup and every hygiene filter split it on spaces. A symbol with a space in it
would corrupt the ABI silently, so :meth:`SemanticDomain.build` asserts against it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

from railroad.core import Fluent as F

from g1sim.perception.semantic_map import _is_support

# Objects at most this big (bbox max dimension, metres) are treated as pickable. Matches
# the threshold SemanticMap.small_objects defaults to.
MAX_PICKABLE_DIM = 0.4


@dataclass(frozen=True)
class SemanticDomain:
    """The symbol universe for one apartment: what exists and where it is.

    Immutable. Built once from a semantic map; the *live* world state is read back out
    of the map on demand by :meth:`world_fluents`, which is what lets the environment
    run a closed loop (see :mod:`g1sim.task.symbolic.environment`).
    """

    robot: str
    locations: dict           # location symbol -> (x, y) in metres
    rooms: frozenset          # subset of `locations` that are rooms
    surfaces: frozenset       # subset of `locations` that are support surfaces
    objects: frozenset        # pickable object symbols
    # The map this was projected from, so a domain can answer questions about the objects
    # it names (see relevant_locations). Not part of the domain's identity.
    _smap: object = field(default=None, repr=False, compare=False)

    # ---- construction -------------------------------------------------------
    @classmethod
    def for_task(cls, smap, objects: Iterable[str], destinations: Iterable[str] = (),
                 *, robot: str = "g1", **kw) -> "SemanticDomain":
        """The domain for one task: only its objects, and only the locations it involves.

        This is the constructor to use for planning. Restricting *locations* is what makes
        search reliable, not merely fast, and the reason is worth stating because it is not
        obvious:

        ``move`` is unconstrained -- any location to any other -- so with all 49 locations
        the robot has 48 first moves to choose from, and the relaxed-plan cost of the goal
        is nearly identical from all of them (from a chair the cup is one move away; from
        the living room it is also one move away). The heuristic therefore barely
        discriminates, the cost term dominates, and MCTS commits to whichever neighbour is
        cheapest -- then dithers between two adjacent chairs while the cup sits in the
        kitchen. Observed exactly that: ``cup_0005 -> dining_table_0000`` oscillated between
        ``chair_0004`` and ``chair_0001`` until the stall detector stopped it, and no search
        budget fixed it (120k iterations still chose the chair). Dropping the 40 locations
        the task has no use for removes the decoys instead of trying to out-search them.

        Rooms are always kept, so the robot can still traverse the apartment and place
        things on any floor.

        Note the two-pass build, which is load-bearing: which surfaces exist depends on
        which objects are pickable (a small object that is not pickable becomes eligible as
        a surface), so the object restriction has to be applied *before* asking where the
        task's objects are. ``cup_0005`` rests on ``tray_0002``, which is a surface only
        once ``tray_0002`` itself is not in the pickable set.
        """
        objects = list(objects)
        with_objects = cls.build(smap, robot=robot, objects=objects, **kw)
        keep = with_objects.relevant_locations(objects, destinations)
        return cls.build(smap, robot=robot, objects=objects, locations=keep, **kw)

    def relevant_locations(self, objects: Iterable[str],
                           destinations: Iterable[str] = ()) -> set:
        """The surfaces a task involves: where each of its objects rests now, plus every
        destination it names. Rooms are excluded because :meth:`build` always keeps them."""
        keep = {d for d in destinations if d in self.surfaces}
        for name in objects:
            o = self.smap_object(name)
            if o is not None:
                loc = self.location_of(o)
                if loc in self.surfaces:
                    keep.add(loc)
        return keep

    def smap_object(self, name: str):
        """Look up one of this domain's objects in the map it was projected from."""
        return self._smap.get(name) if self._smap is not None else None

    @classmethod
    def build(cls, smap, *, robot: str = "g1",
              objects: Optional[Iterable[str]] = None,
              locations: Optional[Iterable[str]] = None,
              max_pickable_dim: float = MAX_PICKABLE_DIM) -> "SemanticDomain":
        """Project ``smap`` onto symbols.

        ``objects`` restricts the pickable universe to the named objects. Leaving it
        ``None`` takes every small object: for this apartment, 97 of them and 10710
        grounded actions -- correct, but MCTS pays for the branching factor. Naming just
        the goal's object brings that to 2450.

        ``locations`` restricts the *surfaces* to the named ones; rooms are always kept, so
        the robot can always traverse and can always use a floor. Prefer
        :meth:`for_task`, which derives this from the task -- and read its docstring for why
        the restriction matters for correctness, not just speed.

        Note the object restriction is not purely a subtraction: a small object that is
        *not* pickable becomes eligible to be a *surface* if its geometry qualifies, so
        restricting objects grows the location set slightly (45 -> 49 here) unless
        ``locations`` is given too.
        """
        pickable = {o.name for o in smap.small_objects(max_pickable_dim)}
        # pickable = {o.name for o in smap.objects if o not in smap.rooms and smap._is_}
        if objects is not None:
            requested = set(objects)
            unknown = requested - set(smap.objects)
            if unknown:
                raise ValueError(f"objects not in the semantic map: {sorted(unknown)}")
            pickable = requested

        wanted = None if locations is None else set(locations)

        # Rooms first, so a room name wins any collision with an object name (it cannot
        # happen with USD-derived names, which always carry a trailing _NNNN, but this
        # module must not depend on that). Rooms are never filtered: without them the robot
        # has nowhere to stand and no floor to put anything on.
        coords: dict = {}
        rooms = []
        for name in smap.room_names():
            pt = smap.navigable_point(name)
            if pt is not None:
                coords[name] = (float(pt[0]), float(pt[1]))
                rooms.append(name)

        surfaces = []
        for o in smap.objects.values():
            if o.name in pickable or o.name in coords:
                continue
            if wanted is not None and o.name not in wanted:
                continue
            # A surface needs a room: one outside every room polygon has no approach
            # the navigation stack can reason about.
            if _is_support(o) and o.room:
                coords[o.name] = (float(o.xy[0]), float(o.xy[1]))
                surfaces.append(o.name)

        for sym in (*coords, *pickable, robot):
            if not sym or any(c.isspace() for c in sym):
                raise ValueError(
                    f"symbol {sym!r} contains whitespace; railroad's action-name ABI "
                    f"is space-delimited, so this would corrupt action dispatch")

        return cls(robot=robot, locations=coords, rooms=frozenset(rooms),
                   surfaces=frozenset(surfaces), objects=frozenset(pickable), _smap=smap)

    # ---- queries ------------------------------------------------------------
    def location_of(self, o) -> Optional[str]:
        """The location symbol a semantic object is currently at, or ``None`` if it is
        held (or somewhere with no symbol, e.g. a room we found no nav point for).

        Prefers the ``on`` edge over the room: a cup on the dining table is ``at
        dining_table_0001``, which is the symbol the robot can actually stand at.
        """
        if o.held:
            return None
        if o.supported_by in self.surfaces:
            return o.supported_by
        if o.room in self.rooms:
            return o.room
        return None

    def nearest_location(self, xy) -> str:
        """The location symbol closest to a point -- used to place the robot in the
        symbolic world at startup."""
        if not self.locations:
            raise ValueError("domain has no locations")
        return min(self.locations,
                   key=lambda s: math.hypot(self.locations[s][0] - xy[0],
                                            self.locations[s][1] - xy[1]))

    def distance(self, a: str, b: str) -> float:
        """Straight-line metres between two location symbols."""
        (ax, ay), (bx, by) = self.locations[a], self.locations[b]
        return math.hypot(bx - ax, by - ay)

    def objects_by_type(self) -> dict:
        return {"robot": {self.robot},
                "location": set(self.locations),
                "object": set(self.objects)}

    # ---- state --------------------------------------------------------------
    def world_fluents(self, smap, held=None) -> set:
        """The fluents describing *the world* (as opposed to the robot's status), read
        fresh from the semantic map: where each object is, and what is in the hand.

        This is the observation function. The environment calls it after every skill so
        that what the planner believes tracks what the map -- which both skill
        implementations mutate on pick/place -- actually says. See
        :meth:`G1Environment.observe`.
        """
        fluents = set()
        for name in self.objects:
            o = smap.get(name)
            if o is None:
                continue
            loc = self.location_of(o)
            if loc is not None:
                fluents.add(F(f"at {name} {loc}"))
        if held is not None and held.name in self.objects:
            fluents.add(F(f"holding {self.robot} {held.name}"))
            fluents.add(F(f"hand-full {self.robot}"))
        return fluents

    def initial_fluents(self, smap, robot_xy, held=None) -> set:
        """Full initial state: the robot free at its nearest location, plus the world."""
        return {F(f"at {self.robot} {self.nearest_location(robot_xy)}"),
                F(f"free {self.robot}")} | self.world_fluents(smap, held=held)

    def describe(self) -> str:
        return (f"SemanticDomain: robot={self.robot}, "
                f"{len(self.locations)} locations ({len(self.rooms)} rooms + "
                f"{len(self.surfaces)} surfaces), {len(self.objects)} pickable objects")
