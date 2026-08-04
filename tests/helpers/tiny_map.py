"""A synthetic apartment used by every sim-free test.

Built by hand rather than parsed from the real ``.usda``: three rectangular rooms
with known polygons, a couple of support surfaces, two same-category objects in
different rooms (the case that trips instance disambiguation), one free-standing
floor object, and -- deliberately -- a gap between two room polygons so "placed in
a doorway" is expressible. Fully controlled geometry means a failure points at the
code under test rather than at whatever the house happens to contain, and the suite
needs no USD, no Isaac, and no I/O.

Layout (x right, y up)::

    y=9  +-----------+
         | bedroom   |          bed_0000, nightstand_0000 <- lamp_0000
    y=5  +-----------+
    y=4  +-----------+  +-----------+
         | livingroom|  | kitchen   |   counter_0000 <- cup_0001
         |  table <- cup_0000, book_0000
         |  chair (floor)          |
    y=0  +-----------+  +-----------+
        x=0         x=4 x=5        x=9
                     ^^^ no room here: DOORWAY_XY
"""

from __future__ import annotations

from g1sim.semantic_map import Room, SemanticMap, SemanticObject

LIVINGROOM_POLY = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
KITCHEN_POLY = [(5.0, 0.0), (9.0, 0.0), (9.0, 4.0), (5.0, 4.0)]
BEDROOM_POLY = [(0.0, 5.0), (4.0, 5.0), (4.0, 9.0), (0.0, 9.0)]

DOORWAY_XY = (4.5, 2.0)             # inside no room polygon
LIVINGROOM_FLOOR_XY = (0.5, 3.5)    # inside livingroom, clear of every surface
LIVINGROOM_START_XY = (0.5, 0.5)    # a robot start pose: in the livingroom, near nothing


def make_object(name, room, xy, *, base_z=0.0, size=(0.4, 0.4, 0.3)) -> SemanticObject:
    """Build a SemanticObject the way the USD builder would: an axis-aligned bbox
    resting with its base at ``base_z``, and an origin at the footprint centre,
    half-way up -- so ``position[2] - bbox_min[2]`` is a realistic non-zero base
    offset, which is what the place() drop math relies on."""
    x, y = xy
    sx, sy, sz = size
    return SemanticObject(
        name=name,
        category=name.rsplit("_", 1)[0],
        room=room,
        position=(x, y, base_z + sz / 2.0),
        bbox_min=(x - sx / 2.0, y - sy / 2.0, base_z),
        bbox_max=(x + sx / 2.0, y + sy / 2.0, base_z + sz),
        prim_path=f"/Root/Meshes/{room}/{name}",
    )


def link_on(child: SemanticObject, surface: SemanticObject) -> None:
    """Wire a consistent ``on`` edge (both directions), as the builder would."""
    child.supported_by = surface.name
    surface.supports.append(child.name)


def build_tiny_map() -> SemanticMap:
    """A fresh 3-room synthetic apartment. Build per-test -- tests mutate it."""
    objects: dict = {}

    def add(name, room, xy, **kw):
        o = make_object(name, room, xy, **kw)
        objects[name] = o
        return o

    # livingroom: a table with two things on it, plus a chair on the floor.
    table = add("dining_table_0000", "livingroom", (2.0, 2.0), size=(2.0, 2.0, 0.75))
    cup_lr = add("cup_0000", "livingroom", (1.6, 2.0), base_z=table.top_z,
                 size=(0.10, 0.10, 0.12))
    book = add("book_0000", "livingroom", (2.4, 2.4), base_z=table.top_z,
               size=(0.20, 0.30, 0.05))
    add("chair_0000", "livingroom", (3.5, 0.6), size=(0.5, 0.5, 0.9))

    # kitchen: a counter with the *second* cup on it.
    counter = add("counter_0000", "kitchen", (7.0, 2.0), size=(2.0, 1.5, 0.9))
    cup_k = add("cup_0001", "kitchen", (6.6, 2.0), base_z=counter.top_z,
                size=(0.10, 0.10, 0.12))

    # bedroom: a bed (low surface) and a nightstand with a lamp on it.
    add("bed_0000", "bedroom", (2.0, 7.0), size=(1.6, 2.0, 0.5))
    stand = add("nightstand_0000", "bedroom", (3.4, 6.0), size=(0.4, 0.4, 0.55))
    lamp = add("lamp_0000", "bedroom", (3.4, 6.0), base_z=stand.top_z,
               size=(0.15, 0.15, 0.30))

    link_on(cup_lr, table)
    link_on(book, table)
    link_on(cup_k, counter)
    link_on(lamp, stand)

    rooms = {
        "livingroom": Room(name="livingroom", scope="livingroom_377",
                           polygon=list(LIVINGROOM_POLY)),
        "kitchen": Room(name="kitchen", scope="kitchen_753",
                        polygon=list(KITCHEN_POLY)),
        "bedroom": Room(name="bedroom", scope="bedroom_120",
                        polygon=list(BEDROOM_POLY)),
    }
    for o in objects.values():
        rooms[o.room].object_names.append(o.name)

    return SemanticMap(rooms, objects)
