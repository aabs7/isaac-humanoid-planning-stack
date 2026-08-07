"""What ``place`` leaves behind in the map.

``MockSkills`` and ``RobotSkills`` share their drop arithmetic
(:func:`g1sim.skills.types.dropped_pose`), so exercising the mock pins the pose the
real robot writes too. The scene-graph suite checks ``relocate`` given correct input;
this checks that ``place`` *produces* correct input -- the seam where a cross-room
move used to leave an object's footprint behind in the room it came from.
"""

from __future__ import annotations

import pytest

from tests.helpers.graph_invariants import assert_graph_consistent, assert_pose_bbox_agree


def test_an_object_can_be_fetched_again_after_a_cross_room_place(env, smap):
    """The consequence that matters: move a cup to another room, then go back for it.
    Reach is measured to the footprint, so if the bbox stayed behind the robot walks
    to the *old* room, believes it has arrived, and magic-grasps the object across the
    apartment."""
    env.goto_object("cup_0000")
    assert env.pick("cup_0000").ok
    env.goto_object("counter_0000")
    assert env.place("counter_0000").ok

    assert env.goto_object("cup_0000").ok
    assert smap.room_at(*env.xy()) == "kitchen", "walked to the wrong room to fetch it"
    assert env.pick("cup_0000").ok


def test_place_leaves_the_object_where_it_says_it_did(env, smap):
    counter = smap.get("counter_0000")
    env.goto_object("cup_0000")
    env.pick("cup_0000")
    env.goto_object("counter_0000")

    result = env.place("counter_0000")

    cup = smap.get("cup_0000")
    assert result.data["at"] == cup.position                 # the reported pose is real
    assert cup.xy_dist(*counter.xy) == pytest.approx(0.0)    # footprint is on the counter
    assert cup.base_z == pytest.approx(counter.top_z)        # resting on its top
    assert cup.size == pytest.approx((0.10, 0.10, 0.12))     # not resized in transit
    assert_pose_bbox_agree(smap)
    assert_graph_consistent(smap, "after place")


def test_a_floor_place_moves_the_footprint_to_that_room(env, smap):
    env.goto_object("cup_0000")
    env.pick("cup_0000")
    env.goto_room("bedroom")

    assert env.place("bedroom").ok

    cup = smap.get("cup_0000")
    assert cup.room == "bedroom"
    assert smap.room_at(*cup.xy) == "bedroom", "footprint left outside the bedroom"
    assert cup.base_z == pytest.approx(0.0)
    assert_pose_bbox_agree(smap)


def test_arriving_at_an_object_is_always_close_enough_to_place_on_it():
    """``goto_object`` stops as soon as it is within PICK_RADIUS, so any place
    threshold tighter than that opens a dead band: goto_object reports success, place
    refuses with "goto it first", and the planner loops on advice it has already
    taken. The two constants are coupled even though they read as independent."""
    from g1sim.skills.types import PICK_RADIUS, PLACE_RADIUS
    assert PLACE_RADIUS >= PICK_RADIUS, (
        f"place needs {PLACE_RADIUS} m but goto_object only guarantees {PICK_RADIUS} m")


def test_the_goto_object_then_place_sequence_closes(env, smap):
    """End to end over the seam that deadlocked: approach a surface, then place on it
    from wherever the approach happened to stop."""
    env.goto_object("cup_0000")
    assert env.pick("cup_0000").ok
    env.goto_object("counter_0000")

    result = env.place("counter_0000")

    assert result.ok, result.detail
    assert smap.get("cup_0000").supported_by == "counter_0000"
