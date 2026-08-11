"""The map -> symbols projection: what becomes a location, an object, a fluent."""

import pytest
from railroad.core import Fluent as F

from g1sim.task.symbolic import SemanticDomain


def test_rooms_and_surfaces_partition_the_locations(domain):
    assert set(domain.rooms) == {"livingroom", "kitchen", "bedroom"}
    assert set(domain.rooms) | set(domain.surfaces) == set(domain.locations)
    assert "dining_table_0000" in domain.surfaces
    assert "counter_0000" in domain.surfaces


def test_pickable_objects_are_the_small_ones(domain):
    assert set(domain.objects) == {"cup_0000", "cup_0001", "book_0000", "lamp_0000"}


def test_pickable_and_location_symbols_are_disjoint(domain):
    """A symbol in both type sets would ground nonsense like `pick g1 book book`."""
    assert not (set(domain.objects) & set(domain.locations))


def test_a_pickable_object_never_becomes_a_surface(smap):
    """book_0000 is flat and stackable but small enough to carry; pickable must win."""
    domain = SemanticDomain.build(smap)
    assert "book_0000" in domain.objects
    assert "book_0000" not in domain.locations


def test_location_of_prefers_the_support_edge_over_the_room(smap, domain):
    cup = smap.get("cup_0000")
    assert cup.supported_by == "dining_table_0000"
    assert domain.location_of(cup) == "dining_table_0000"


def test_location_of_falls_back_to_the_room(smap, domain):
    """An object standing on the floor is located by its room."""
    from tests.helpers.tiny_map import LIVINGROOM_FLOOR_XY
    cup = smap.get("cup_0000")
    smap.relocate(cup, (*LIVINGROOM_FLOOR_XY, 0.06),
                  (LIVINGROOM_FLOOR_XY[0] - 0.05, LIVINGROOM_FLOOR_XY[1] - 0.05, 0.0),
                  (LIVINGROOM_FLOOR_XY[0] + 0.05, LIVINGROOM_FLOOR_XY[1] + 0.05, 0.12))
    assert cup.supported_by is None
    assert domain.location_of(cup) == "livingroom"


def test_held_object_has_no_location(smap, domain):
    cup = smap.get("cup_0000")
    smap.set_carried(cup)
    assert domain.location_of(cup) is None


def test_world_fluents_track_the_map(smap, domain):
    assert F("at cup_0000 dining_table_0000") in domain.world_fluents(smap)
    cup = smap.get("cup_0000")
    smap.set_carried(cup)
    after = domain.world_fluents(smap, held=cup)
    assert F("at cup_0000 dining_table_0000") not in after
    assert F("holding g1 cup_0000") in after
    assert F("hand-full g1") in after


def test_initial_fluents_place_the_robot_at_its_nearest_location(smap, domain):
    # The nightstand, unlike the counter, is not sitting on its room's nav point, so
    # "nearest" is unambiguous here.
    fluents = domain.initial_fluents(smap, smap.get("nightstand_0000").xy)
    assert F("at g1 nightstand_0000") in fluents
    assert F("free g1") in fluents


def test_restricting_the_object_universe(smap):
    domain = SemanticDomain.build(smap, objects=["cup_0001"])
    assert set(domain.objects) == {"cup_0001"}
    # The others are no longer pickable, so those that qualify geometrically become
    # surfaces instead -- which is the whole reason restriction shrinks the problem.
    assert "cup_0000" not in domain.objects


def test_unknown_object_is_rejected(smap):
    with pytest.raises(ValueError, match="not in the semantic map"):
        SemanticDomain.build(smap, objects=["nonexistent_0001"])


def test_whitespace_in_a_symbol_is_rejected(smap):
    """railroad's grounded action name is space-delimited and *is* the dispatch key, so a
    symbol containing a space would silently corrupt every lookup."""
    o = smap.get("cup_0000")
    o.name = "bad name"
    smap.objects["bad name"] = o
    with pytest.raises(ValueError, match="whitespace"):
        SemanticDomain.build(smap, objects=["bad name"])


def test_distance_is_metres_between_locations(domain):
    d = domain.distance("livingroom", "kitchen")
    assert 2.0 < d < 8.0            # the two room polygons are ~5 m apart


# -- task scoping ---------------------------------------------------------
def test_for_task_keeps_every_room_but_only_relevant_surfaces(smap):
    """Rooms always survive -- the robot needs somewhere to stand and a floor to use."""
    domain = SemanticDomain.for_task(smap, ["cup_0001"], ["dining_table_0000"])

    assert set(domain.rooms) == {"livingroom", "kitchen", "bedroom"}
    # Where the cup is now, and where it is going. Nothing else.
    assert set(domain.surfaces) == {"counter_0000", "dining_table_0000"}
    assert "bed_0000" not in domain.locations
    assert "chair_0000" not in domain.locations


def test_for_task_finds_a_surface_that_only_exists_once_objects_are_restricted(smap):
    """The two-pass build, which is load-bearing.

    Some objects are *both* small enough to carry and flat enough to stack on -- the real
    apartment has four (two books, two trays), and ``cup_0005`` rests on one of them. Such an
    object is pickable in the full domain, so it is not a surface there and a cup on top of
    it has no surface symbol; restricting the object set makes it a surface. Scoping
    therefore has to apply the object restriction *before* asking where the task's objects
    are, or it drops the very location it needs. Here the book is resized to qualify, since
    the synthetic apartment has no such object by default."""
    book = smap.get("book_0000")
    book.bbox_min = (book.xy[0] - 0.175, book.xy[1] - 0.175, book.bbox_min[2])
    book.bbox_max = (book.xy[0] + 0.175, book.xy[1] + 0.175, book.bbox_max[2])
    cup = smap.get("cup_0001")
    smap.detach(cup)
    cup.supported_by, book.supports = "book_0000", ["cup_0001"]

    full = SemanticDomain.build(smap)
    assert "book_0000" in full.objects, "the book is carryable, so not a surface"
    assert full.location_of(cup) == "kitchen", "so the cup falls back to its room"

    domain = SemanticDomain.for_task(smap, ["cup_0001"], ["bed_0000"])
    assert "book_0000" in domain.surfaces
    assert domain.location_of(cup) == "book_0000"


def test_for_task_shrinks_the_problem(smap):
    full = SemanticDomain.build(smap)
    scoped = SemanticDomain.for_task(smap, ["cup_0001"], ["dining_table_0000"])
    assert len(scoped.locations) < len(full.locations)
    assert set(scoped.objects) == {"cup_0001"}


def test_relevant_locations_ignores_rooms_and_unknown_names(smap):
    domain = SemanticDomain.build(smap, objects=["cup_0001"])
    keep = domain.relevant_locations(["cup_0001"], ["livingroom", "no_such_thing"])
    assert "livingroom" not in keep          # rooms are kept unconditionally by build()
    assert "no_such_thing" not in keep
    assert "counter_0000" in keep            # where the cup actually is


def test_locations_can_be_restricted_directly(smap):
    domain = SemanticDomain.build(smap, locations=["bed_0000"])
    assert set(domain.surfaces) == {"bed_0000"}
    assert set(domain.rooms) == {"livingroom", "kitchen", "bedroom"}
