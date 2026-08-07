"""The pose arithmetic ``place()`` applies to a dropped object.

Thin wrappers over :func:`g1sim.skills.types.dropped_pose` -- the function both skill
implementations call -- so scene-graph tests drive ``relocate`` with exactly the
input the real caller produces. Deliberately not a reimplementation: this file used
to mirror the math by hand, which meant it faithfully reproduced a bug (the bbox
shifting only in z) instead of catching it.
"""

from __future__ import annotations

from g1sim.skills.types import dropped_pose


def drop_on(o, surface):
    """The pose a ``place(<surface>)`` gives ``o``: resting on the surface's top, at
    the surface's XY. Returns ``(position, bbox_min, bbox_max)``."""
    return dropped_pose(o, surface.xy, surface.top_z)


def drop_at(o, xy, surface_z: float = 0.0):
    """The pose a ``place(<room or point>)`` gives ``o``: dropped at ``xy`` with its
    base on ``surface_z``. Returns ``(position, bbox_min, bbox_max)``."""
    return dropped_pose(o, xy, surface_z)
