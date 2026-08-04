"""Scene-graph invariants: the mutation API that pick/place drive.

``SemanticMap`` is the authoritative state of the whole system -- the planner's
observation (``describe_graph``), its completion check (``_targets_satisfied``), and
the skills' preconditions all read it. Its mutators (``set_carried`` / ``detach`` /
``relocate`` / ``reassign_room``) hand-maintain four coupled fields, so every test
here ends by asserting the graph is still structurally sound
(:func:`assert_graph_consistent`). Break the bookkeeping and the failure shows up
here, not three phases later as a planner that can't find what it just put down.

Tests marked ``xfail(strict=True)`` document invariants that do **not** hold today;
fixing the underlying bug turns them into unexpected passes, which fails the suite
and tells you to flip the marker.
"""

from __future__ import annotations

import pytest

from g1sim.semantic_map import Room, SemanticMap
from tests.helpers.graph_invariants import (assert_graph_consistent,
                                            assert_pose_bbox_agree, graph_signature)
from tests.helpers.place_math import drop_at, drop_on
from tests.helpers.tiny_map import DOORWAY_XY, LIVINGROOM_FLOOR_XY


# ---------------------------------------------------------------------------
# Fixture sanity -- if these fail, every test below is measuring the wrong thing.
# ---------------------------------------------------------------------------
def test_fixture_is_a_consistent_graph(smap):
    assert_graph_consistent(smap, "fresh fixture")
    assert_pose_bbox_agree(smap)
    assert smap.room_names() == ["bedroom", "kitchen", "livingroom"]
    assert smap.categories()["cup"] == 2          # same category, two rooms


def test_room_at_locates_points_and_reports_none_between_rooms(smap):
    assert smap.room_at(2.0, 2.0) == "livingroom"
    assert smap.room_at(7.0, 2.0) == "kitchen"
    assert smap.room_at(2.0, 7.0) == "bedroom"
    assert smap.room_at(*DOORWAY_XY) is None      # the gap between the polygons


def test_room_without_a_polygon_contains_nothing():
    assert Room(name="balcony", scope="balcony_1").contains(0.0, 0.0) is False


# ---------------------------------------------------------------------------
# detach -- breaking one 'on' edge
# ---------------------------------------------------------------------------
def test_detach_breaks_the_edge_in_both_directions(smap):
    cup, table = smap.get("cup_0000"), smap.get("dining_table_0000")
    assert cup.supported_by == table.name

    smap.detach(cup)

    assert cup.supported_by is None
    assert cup.name not in table.supports
    assert "book_0000" in table.supports          # siblings untouched
    assert cup.room == "livingroom"               # detach does NOT move it
    assert_graph_consistent(smap, "after detach")


def test_detach_of_a_free_standing_object_is_a_noop(smap):
    before = graph_signature(smap)
    smap.detach(smap.get("chair_0000"))
    assert graph_signature(smap) == before


def test_detach_is_idempotent(smap):
    cup = smap.get("cup_0000")
    smap.detach(cup)
    after_first = graph_signature(smap)
    smap.detach(cup)
    assert graph_signature(smap) == after_first
    assert_graph_consistent(smap, "after double detach")


def test_detach_tolerates_a_dangling_edge(smap):
    """A ``supported_by`` pointing at a vanished object must not crash detach -- objects can be removed from the map (scene-graph updates) while an edge lingers."""
    cup = smap.get("cup_0000")
    smap.get("dining_table_0000").supports.remove(cup.name)
    cup.supported_by = "ghost_table_9999"

    smap.detach(cup)

    assert cup.supported_by is None
    assert_graph_consistent(smap, "after detaching a dangling edge")


# ---------------------------------------------------------------------------
# set_carried -- what pick() does to the graph
# ---------------------------------------------------------------------------
def test_set_carried_removes_object_from_its_surface_and_room(smap):
    cup, table = smap.get("cup_0000"), smap.get("dining_table_0000")

    smap.set_carried(cup)

    assert cup.held is True
    assert cup.room is None
    assert cup.supported_by is None
    assert cup.name not in table.supports
    assert cup.name not in smap.rooms["livingroom"].object_names
    assert_graph_consistent(smap, "after set_carried")


def test_set_carried_works_for_a_free_standing_object(smap):
    chair = smap.get("chair_0000")
    assert chair.supported_by is None

    smap.set_carried(chair)

    assert chair.held is True and chair.room is None
    assert chair.name not in smap.rooms["livingroom"].object_names
    assert_graph_consistent(smap, "after carrying a floor object")


def test_set_carried_is_idempotent(smap):
    cup = smap.get("cup_0000")
    smap.set_carried(cup)
    after_first = graph_signature(smap)

    smap.set_carried(cup)

    assert graph_signature(smap) == after_first
    assert_graph_consistent(smap, "after double set_carried")


def test_a_carried_object_disappears_from_every_room_query(smap):
    """The planner only ever sees the graph through these queries, so a held object
    leaking into them is what makes an LLM try to walk to something in its hands."""
    cup, table = smap.get("cup_0000"), smap.get("dining_table_0000")

    smap.set_carried(cup)

    assert cup.name not in {o.name for o in smap.objects_in_room("livingroom")}
    assert cup.name not in {o.name for o in smap.reachable_in_room("livingroom", (0.0, 0.0))}
    assert cup.name not in {o.name for o in smap.objects_on(table.name)}
    assert smap.nearest_object(cup.xy[0], cup.xy[1], max_dist=2.0) is not cup

    graph = smap.describe_graph()
    assert "carried by robot" in graph
    assert cup.name in graph.split("\n")[1]        # ...on the carried line, and
    # nowhere in the room listing itself (everything after the "room:" header).
    livingroom = smap.describe_graph(room="livingroom")
    assert cup.name not in livingroom.split("room:livingroom", 1)[1]


# ---------------------------------------------------------------------------
# relocate -- what place() does to the graph
# ---------------------------------------------------------------------------
def test_relocate_onto_another_surface_rewires_both_edges(smap):
    cup = smap.get("cup_0000")
    table, counter = smap.get("dining_table_0000"), smap.get("counter_0000")
    smap.set_carried(cup)

    smap.relocate(cup, *drop_on(cup, counter), on_surface=counter)

    assert cup.held is False
    assert cup.supported_by == counter.name
    assert cup.name in counter.supports
    assert cup.name not in table.supports
    assert_graph_consistent(smap, "after place-on-surface")


def test_relocate_onto_a_surface_puts_the_object_in_that_surfaces_room(smap):
    """Cross-room place: the object must end up in the destination room's list, or
    the planner's completion check ('is it in the kitchen?') can never fire."""
    cup, counter = smap.get("cup_0000"), smap.get("counter_0000")
    smap.set_carried(cup)

    smap.relocate(cup, *drop_on(cup, counter), on_surface=counter)

    assert cup.room == counter.room == "kitchen"
    assert cup.name in smap.rooms["kitchen"].object_names
    assert cup.name not in smap.rooms["livingroom"].object_names
    assert_graph_consistent(smap, "after cross-room place")


def test_relocate_rests_the_object_on_the_surface_top(smap):
    cup, counter = smap.get("cup_0000"), smap.get("counter_0000")
    smap.set_carried(cup)

    smap.relocate(cup, *drop_on(cup, counter), on_surface=counter)

    assert cup.base_z == pytest.approx(counter.top_z)
    assert cup.size == pytest.approx((0.10, 0.10, 0.12))   # not squashed or grown


def test_relocate_to_a_room_floor_clears_the_support_edge(smap):
    cup, table = smap.get("cup_0000"), smap.get("dining_table_0000")
    smap.set_carried(cup)

    smap.relocate(cup, *drop_at(cup, LIVINGROOM_FLOOR_XY, surface_z=0.0))

    assert cup.supported_by is None
    assert cup.name not in table.supports
    assert cup.room == "livingroom"
    assert cup.base_z == pytest.approx(0.0)
    assert_graph_consistent(smap, "after floor place")
    assert_pose_bbox_agree(smap)


def test_relocate_clears_a_stale_edge_even_without_set_carried(smap):
    """``relocate`` must be safe on its own -- it detaches first, so a caller that
    skips ``set_carried`` can't leave the old surface still claiming the object."""
    cup, table = smap.get("cup_0000"), smap.get("dining_table_0000")

    smap.relocate(cup, *drop_at(cup, LIVINGROOM_FLOOR_XY, surface_z=0.0))

    assert cup.supported_by is None
    assert cup.name not in table.supports
    assert_graph_consistent(smap, "after bare relocate")


def test_relocate_onto_the_same_surface_does_not_duplicate_the_edge(smap):
    cup, table = smap.get("cup_0000"), smap.get("dining_table_0000")

    smap.relocate(cup, *drop_on(cup, table), on_surface=table)

    assert table.supports.count(cup.name) == 1
    assert_graph_consistent(smap, "after re-place on the same surface")


def test_relocate_stores_the_given_pose_verbatim_and_clears_held(smap):
    cup = smap.get("cup_0000")
    smap.set_carried(cup)
    pos, bmin, bmax = (1.0, 2.0, 3.0), (0.9, 1.9, 2.9), (1.1, 2.1, 3.1)

    smap.relocate(cup, pos, bmin, bmax)

    assert cup.position == pos and cup.bbox_min == bmin and cup.bbox_max == bmax
    assert cup.held is False


def test_two_objects_can_rest_on_the_same_surface(smap):
    table, counter = smap.get("dining_table_0000"), smap.get("counter_0000")
    other_cup = smap.get("cup_0001")
    smap.set_carried(other_cup)

    smap.relocate(other_cup, *drop_on(other_cup, table), on_surface=table)

    assert set(table.supports) == {"cup_0000", "book_0000", "cup_0001"}
    assert other_cup.name not in counter.supports
    assert_graph_consistent(smap, "after stacking a second object")


# ---------------------------------------------------------------------------
# Round trips -- the strongest statement about the mutation API
# ---------------------------------------------------------------------------
def test_pick_then_place_back_restores_the_graph_exactly(smap):
    cup, table = smap.get("cup_0000"), smap.get("dining_table_0000")
    before = graph_signature(smap)
    pose = (cup.position, cup.bbox_min, cup.bbox_max)

    smap.set_carried(cup)
    smap.relocate(cup, *pose, on_surface=table)

    assert graph_signature(smap) == before
    assert_graph_consistent(smap, "after pick/place round trip")


def test_a_chain_of_moves_keeps_the_graph_consistent_and_is_reversible(smap):
    cup, table = smap.get("cup_0000"), smap.get("dining_table_0000")
    before = graph_signature(smap)
    origin = (cup.position, cup.bbox_min, cup.bbox_max)

    for dest in ("counter_0000", "nightstand_0000", "bed_0000"):
        surface = smap.get(dest)
        smap.set_carried(cup)
        assert_graph_consistent(smap, f"carrying toward {dest}")
        smap.relocate(cup, *drop_on(cup, surface), on_surface=surface)
        assert_graph_consistent(smap, f"placed on {dest}")
        assert cup.supported_by == dest and cup.room == surface.room

    smap.set_carried(cup)
    smap.relocate(cup, *origin, on_surface=table)

    assert graph_signature(smap) == before


def test_json_round_trip_preserves_the_whole_graph(tmp_path, smap):
    path = tmp_path / "map.json"
    smap.save(str(path))

    loaded = SemanticMap.load(str(path))

    assert loaded.to_dict() == smap.to_dict()
    assert_graph_consistent(loaded, "after save/load")


def test_json_round_trip_after_a_move_preserves_the_new_state(tmp_path, smap):
    cup, counter = smap.get("cup_0000"), smap.get("counter_0000")
    smap.set_carried(cup)
    smap.relocate(cup, *drop_on(cup, counter), on_surface=counter)
    path = tmp_path / "moved.json"
    smap.save(str(path))

    loaded = SemanticMap.load(str(path))

    assert loaded.get("cup_0000").supported_by == "counter_0000"
    assert loaded.get("cup_0000").room == "kitchen"
    assert loaded.to_dict() == smap.to_dict()
    assert_graph_consistent(loaded, "after save/load of a moved object")


# ---------------------------------------------------------------------------
# reassign_room
# ---------------------------------------------------------------------------
def test_reassign_room_moves_membership_when_the_position_changed(smap):
    chair = smap.get("chair_0000")
    chair.position = (7.0, 2.0, chair.position[2])     # teleport into the kitchen

    smap.reassign_room(chair)

    assert chair.room == "kitchen"
    assert chair.name in smap.rooms["kitchen"].object_names
    assert chair.name not in smap.rooms["livingroom"].object_names
    assert_graph_consistent(smap, "after reassign_room")


def test_reassign_room_is_a_noop_when_the_point_is_in_no_room(smap):
    """Documents current behaviour: outside every polygon the object keeps its old
    room. Benign here, but see the orphan test below for where it bites."""
    chair = smap.get("chair_0000")
    chair.position = (DOORWAY_XY[0], DOORWAY_XY[1], chair.position[2])

    smap.reassign_room(chair)

    assert chair.room == "livingroom"
    assert chair.name in smap.rooms["livingroom"].object_names


# ---------------------------------------------------------------------------
# Known gaps -- current behaviour pinned, intended invariant xfailed.
# ---------------------------------------------------------------------------
def test_placing_between_rooms_orphans_the_object_today(smap):
    """A place in a doorway leaves the object held=False with room=None and in no
    room's list, so ``describe_graph`` (the planner's only observation) stops
    mentioning it: the goal can never be judged satisfied and the run burns its
    step budget. Pinning the behaviour so a fix is a deliberate, visible change."""
    cup = smap.get("cup_0000")
    smap.set_carried(cup)

    smap.relocate(cup, *drop_at(cup, DOORWAY_XY, surface_z=0.0))

    assert smap.room_at(*DOORWAY_XY) is None
    assert cup.held is False
    assert cup.room is None
    assert all(cup.name not in r.object_names for r in smap.rooms.values())
    assert cup.name not in smap.describe_graph()


@pytest.mark.xfail(strict=True, reason="relocate outside every room polygon leaves "
                                       "the object orphaned (room=None, in no room "
                                       "list) -- see reassign_room()")
def test_placing_between_rooms_should_keep_the_graph_consistent(smap):
    cup = smap.get("cup_0000")
    smap.set_carried(cup)
    smap.relocate(cup, *drop_at(cup, DOORWAY_XY, surface_z=0.0))
    assert_graph_consistent(smap, "after placing between rooms")


@pytest.mark.xfail(strict=True, reason="place() shifts only the bbox's z, so after a "
                                       "cross-room place the footprint is left at the "
                                       "old location (skills.py / mock_skills.py)")
def test_place_on_a_distant_surface_should_move_the_footprint_too(smap):
    """``place`` translates ``position`` fully but the bbox only in z, so a cup moved
    from the livingroom table to the kitchen counter keeps a livingroom footprint.
    Everything reach-related reads that footprint -- ``xy_dist``, ``nearest_object``,
    ``goto_object``'s approach point -- so the robot would walk to the wrong room to
    pick the cup back up."""
    cup, counter = smap.get("cup_0000"), smap.get("counter_0000")
    smap.set_carried(cup)

    smap.relocate(cup, *drop_on(cup, counter), on_surface=counter)

    assert_pose_bbox_agree(smap)
    assert cup.xy_dist(*counter.xy) == pytest.approx(0.0)
    assert smap.nearest_object(counter.xy[0], counter.xy[1], max_dist=0.5) is not None


@pytest.mark.xfail(strict=True, reason="SemanticMap.load() never restores the 'held' "
                                       "flag, so a map saved mid-carry comes back "
                                       "claiming empty hands")
def test_json_round_trip_preserves_the_held_flag(tmp_path, smap):
    smap.set_carried(smap.get("cup_0000"))
    path = tmp_path / "carrying.json"
    smap.save(str(path))

    loaded = SemanticMap.load(str(path))

    assert loaded.get("cup_0000").held is True
    assert loaded.to_dict() == smap.to_dict()


# ---------------------------------------------------------------------------
# Query inverses -- the graph read paths the planner and skills rely on
# ---------------------------------------------------------------------------
def test_objects_on_and_support_of_are_inverses(smap):
    for o in smap.objects.values():
        for child in smap.objects_on(o.name):
            assert smap.support_of(child.name) is o
        parent = smap.support_of(o.name)
        if parent is not None:
            assert o.name in {c.name for c in smap.objects_on(parent.name)}


def test_objects_on_an_unknown_name_is_empty_not_an_error(smap):
    assert smap.objects_on("no_such_object_0000") == []
    assert smap.support_of("no_such_object_0000") is None


def test_describe_graph_shows_a_moved_object_under_its_new_surface(smap):
    cup, counter = smap.get("cup_0000"), smap.get("counter_0000")
    smap.set_carried(cup)
    smap.relocate(cup, *drop_on(cup, counter), on_surface=counter)

    kitchen = smap.describe_graph(room="kitchen")
    assert "on: cup [cup_0000]" in kitchen
    assert "cup_0000" not in smap.describe_graph(room="livingroom")


def test_describe_graph_lists_a_floor_object_as_free_standing(smap):
    livingroom = smap.describe_graph(room="livingroom")
    assert "free-standing / floor" in livingroom
    assert "chair" in livingroom.split("free-standing / floor")[1]
