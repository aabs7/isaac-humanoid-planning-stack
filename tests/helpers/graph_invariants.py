"""Structural invariants of a :class:`g1sim.perception.semantic_map.SemanticMap` (a.k.a. the
scene graph), plus helpers for comparing whole graphs.

:func:`assert_graph_consistent` is the workhorse: it encodes what must be true of
*any* scene graph, at rest or mid-task. The scene-graph mutators
(``set_carried`` / ``detach`` / ``relocate`` / ``reassign_room``) hand-maintain four
coupled pieces of state -- ``supported_by``, ``supports``, ``room``, and each
``Room.object_names`` list -- so the rule in the tests is: call this after *every*
mutation. A regression in that bookkeeping is otherwise invisible until the planner
inexplicably can't find an object it just put down.

Kept deliberately free of any fixture specifics so Tier-2/3 tests (mock skills,
real-USD builder) can reuse it unchanged.
"""

from __future__ import annotations


def assert_graph_consistent(smap, ctx: str = "") -> None:
    """Assert every structural invariant of ``smap``. ``ctx`` labels the call site
    (e.g. ``"after place"``) so a failure says *when* the graph broke."""
    where = f" [{ctx}]" if ctx else ""
    objects, rooms = smap.objects, smap.rooms

    for name, o in objects.items():
        assert o.name == name, f"objects[{name!r}].name == {o.name!r}{where}"

        # -- bounding box is not inverted -----------------------------------
        for axis in range(3):
            assert o.bbox_min[axis] <= o.bbox_max[axis], (
                f"{name}: bbox inverted on axis {axis}: "
                f"{o.bbox_min[axis]} > {o.bbox_max[axis]}{where}")

        # -- 'on' edges are symmetric and point at real objects -------------
        assert o.name not in o.supports, f"{name} supports itself{where}"
        assert o.supported_by != o.name, f"{name} is supported by itself{where}"

        if o.supported_by is not None:
            assert o.supported_by in objects, (
                f"{name}.supported_by -> {o.supported_by!r} which is not in the "
                f"map{where}")
            parent = objects[o.supported_by]
            assert o.name in parent.supports, (
                f"one-way 'on' edge: {name}.supported_by == {parent.name} but "
                f"{parent.name}.supports == {parent.supports}{where}")

        for child_name in o.supports:
            assert child_name in objects, (
                f"{name}.supports lists {child_name!r} which is not in the map{where}")
            child = objects[child_name]
            assert child.supported_by == o.name, (
                f"one-way 'on' edge: {name}.supports includes {child_name} but "
                f"{child_name}.supported_by == {child.supported_by!r}{where}")

        assert len(set(o.supports)) == len(o.supports), (
            f"{name}.supports has duplicates: {o.supports}{where}")

        # -- room membership: exactly one room iff not carried --------------
        holders = sorted(r.name for r in rooms.values() if o.name in r.object_names)
        if o.held:
            assert o.room is None, (
                f"{name} is held but still claims room {o.room!r}{where}")
            assert holders == [], (
                f"{name} is held but still listed in rooms {holders}{where}")
        else:
            assert o.room is not None, (
                f"{name} is not held but has no room -- it is orphaned and "
                f"invisible to the planner{where}")
            assert o.room in rooms, (
                f"{name}.room == {o.room!r} which is not a known room "
                f"({sorted(rooms)}){where}")
            assert holders == [o.room], (
                f"{name}.room == {o.room!r} but it is listed in rooms {holders}"
                f"{where}")

    for name, room in rooms.items():
        assert room.name == name, f"rooms[{name!r}].name == {room.name!r}{where}"
        assert len(set(room.object_names)) == len(room.object_names), (
            f"room {name}.object_names has duplicates: {room.object_names}{where}")
        for on in room.object_names:
            assert on in objects, (
                f"room {name} lists {on!r} which is not in the map{where}")
        # objects_in_room() (what describe_graph reads) must agree with the room's
        # own list (what save() writes) -- two code paths, one truth.
        assert {o.name for o in smap.objects_in_room(name)} == set(room.object_names), (
            f"room {name}: objects_in_room() disagrees with object_names{where}")

    _assert_no_support_cycles(smap, where)


def _assert_no_support_cycles(smap, where: str = "") -> None:
    """No object may (transitively) rest on itself."""
    for start in smap.objects.values():
        seen = {start.name}
        cur = start
        while cur.supported_by is not None:
            nxt = smap.objects[cur.supported_by]
            assert nxt.name not in seen, (
                f"support cycle: {start.name} -> ... -> {nxt.name}{where}")
            seen.add(nxt.name)
            cur = nxt


def assert_pose_bbox_agree(smap, tol: float = 1e-6) -> None:
    """Every object's origin XY lies inside its own footprint.

    Separate from :func:`assert_graph_consistent` because it is a statement about
    *geometry*, not graph structure, and real USD pivots can sit outside their
    geometry. It holds by construction for the synthetic fixture, where it catches
    a pose update that moves ``position`` without moving ``bbox``.
    """
    for o in smap.objects.values():
        x, y = o.xy
        assert o.bbox_min[0] - tol <= x <= o.bbox_max[0] + tol, (
            f"{o.name}: origin x={x} outside footprint "
            f"[{o.bbox_min[0]}, {o.bbox_max[0]}] -- stale bbox?")
        assert o.bbox_min[1] - tol <= y <= o.bbox_max[1] + tol, (
            f"{o.name}: origin y={y} outside footprint "
            f"[{o.bbox_min[1]}, {o.bbox_max[1]}] -- stale bbox?")


def graph_signature(smap, *, ndigits: int = 9) -> dict:
    """A hashable-ish, order-insensitive snapshot of the whole graph, for
    ``before == after`` comparisons (round trips, no-op mutations). Uses sets for
    membership lists because their *order* carries no meaning."""
    def rnd(t):
        return tuple(round(float(v), ndigits) for v in t)

    return {
        "rooms": {
            n: (r.scope,
                tuple(rnd(p) for p in r.polygon),
                frozenset(r.object_names))
            for n, r in smap.rooms.items()
        },
        "objects": {
            n: (o.category, o.room, rnd(o.position), rnd(o.bbox_min), rnd(o.bbox_max),
                o.prim_path, o.supported_by, frozenset(o.supports), o.held)
            for n, o in smap.objects.items()
        },
    }
