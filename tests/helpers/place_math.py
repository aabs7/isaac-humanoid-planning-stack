"""The pose arithmetic ``place()`` applies to a dropped object.

Kept here, mirroring ``RobotSkills.place`` (g1sim/skills/robot.py) and ``MockSkills.place``
(g1sim/skills/mock.py), so scene-graph tests exercise the *real* caller math rather
than an idealized version -- including the quirk that those implementations shift
only the bbox's z.
"""

from __future__ import annotations


def drop_on(o, surface):
    """The pose a ``place(<surface>)`` gives ``o``, exactly as the skills compute it
    (bbox shifted in z only). Returns ``(position, bbox_min, bbox_max)`` ready for
    ``SemanticMap.relocate``."""
    base_offset = o.position[2] - o.bbox_min[2]
    drop = (surface.xy[0], surface.xy[1], surface.top_z + base_offset)
    dz = drop[2] - o.position[2]
    return (drop,
            (o.bbox_min[0], o.bbox_min[1], o.bbox_min[2] + dz),
            (o.bbox_max[0], o.bbox_max[1], o.bbox_max[2] + dz))


def drop_at(o, xy, surface_z: float = 0.0):
    """The pose a ``place(<room or point>)`` gives ``o``: dropped at ``xy`` with its
    base on ``surface_z``, translating the whole bbox -- the behaviour a correct
    place() should have. Returns ``(position, bbox_min, bbox_max)``."""
    base_offset = o.position[2] - o.bbox_min[2]
    drop = (float(xy[0]), float(xy[1]), surface_z + base_offset)
    dx, dy = drop[0] - o.position[0], drop[1] - o.position[1]
    dz = drop[2] - o.position[2]
    return (drop,
            (o.bbox_min[0] + dx, o.bbox_min[1] + dy, o.bbox_min[2] + dz),
            (o.bbox_max[0] + dx, o.bbox_max[1] + dy, o.bbox_max[2] + dz))
