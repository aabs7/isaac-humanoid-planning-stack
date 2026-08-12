"""Testing of whether the arm can get to an object, and it's stance position.
"""

import math

import pytest

from g1sim.skills.reach import (ARM_REACH, SHOULDER_Z, aabb_distance, best_stance,
                                in_a_room, required_reach)
from tests.helpers.tiny_map import make_object


# Real bounding boxes from the kujiale_0021 semantic map, and stances the probe actually
# drove the robot to, with the shoulder-to-object distance Isaac reported there. Inlined
# rather than loaded, so the test needs no build artefact.
CUP_0005 = ((13.5611, 0.4823, 0.9218), (13.6305, 0.5745, 0.9931))
CUP_0000 = ((12.0628, 0.7565, 0.8508), (12.1562, 0.8402, 0.9074))

MEASURED = [
    # (name, bbox, stance the robot stood at, distance Isaac measured there)
    ("cup_0005", CUP_0005, (13.04, 0.02), 0.693),
    ("cup_0000", CUP_0000, (13.01, 0.04), 1.068),
    ("cup_0000", CUP_0000, (12.62, -0.12), 0.971),
]


class Box:
    """Just the bbox fields :func:`required_reach` reads."""

    def __init__(self, bbox):
        self.bbox_min, self.bbox_max = bbox


def test_reach_is_measured_in_3d_not_on_the_floor():
    """Test that 2D distance to an object from the stance position is not the same as the distance from the shoulder of the robot."""
    cup = make_object("cup_0001", "kitchen", (1.0, 0.0), base_z=0.9, size=(0.06, 0.06, 0.06))
    stance = (0.7, 0.0)

    flat = cup.xy_dist(*stance)                       # what PICK_RADIUS would see
    real = required_reach(cup, stance)                # what the shoulder has to span

    assert flat == pytest.approx(0.27, abs=0.01)
    assert real > flat, "the vertical gap to the counter cannot be free"


def test_an_object_at_shoulder_height_needs_only_the_horizontal_distance():
    cup = make_object("cup_0001", "kitchen", (1.0, 0.0),
                      base_z=SHOULDER_Z - 0.03, size=(0.06, 0.06, 0.06))
    assert required_reach(cup, (0.5, 0.0)) == pytest.approx(cup.xy_dist(0.5, 0.0), abs=0.03)


@pytest.mark.parametrize("name,bbox,stance,measured", MEASURED)
def test_the_model_reproduces_what_isaac_measured(name, bbox, stance, measured):
    got = required_reach(Box(bbox), stance)
    assert got == pytest.approx(measured, abs=0.06)
    assert got >= measured - 0.01, "the model should not read shorter than the real arm"


def test_pick_radius_admits_things_the_arm_cannot_reach():
    """Why reach needs its own check. cup_0000 sits 0.994 m from the base footprint --
    comfortably inside PICK_RADIUS -- while the shoulder is a full 0.27 m beyond the arm.
    A planner using PICK_RADIUS as its only precondition will keep choosing it."""
    from g1sim.skills.types import PICK_RADIUS

    stance = (12.62, -0.12)
    assert required_reach(Box(CUP_0000), stance) > ARM_REACH
    assert 0.994 < PICK_RADIUS, "…and the base-level precondition is satisfied there"


def test_a_stance_outside_every_room_is_not_a_stance(smap):
    from tests.helpers.tiny_map import DOORWAY_XY
    free = in_a_room(smap)
    assert not free(*DOORWAY_XY), "the gap between room polygons is not standable"
    assert free(2.0, 2.0), "the middle of the livingroom is"


def test_best_stance_finds_a_close_one_for_a_reachable_object(smap):
    cup = smap.get("cup_0001")                       # on the kitchen counter
    r = best_stance(smap, cup)

    assert r.reachable and r.stance is not None
    assert r.required_reach <= ARM_REACH
    # And the stance it picked really does require that reach -- the search and the
    # primitive must not disagree.
    assert required_reach(cup, r.stance) == pytest.approx(r.required_reach, abs=1e-9)


def test_aabb_distance_is_zero_inside_and_symmetric_outside():
    lo, hi = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    assert aabb_distance((0.5, 0.5, 0.5), lo, hi) == 0.0
    assert aabb_distance((2.0, 0.5, 0.5), lo, hi) == pytest.approx(1.0)
    assert aabb_distance((-1.0, 0.5, 0.5), lo, hi) == pytest.approx(1.0)
    # Corner: the diagonal, not the axis distance.
    assert aabb_distance((2.0, 2.0, 0.5), lo, hi) == pytest.approx(math.hypot(1.0, 1.0))
