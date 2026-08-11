"""The operator set: ``move``, ``pick``, ``place``.

Deliberately written out rather than assembled from ``railroad.operators.core``'s
constructors. Those are close to what we want, but we need two things they don't offer
-- the ``failed-*`` guards (see below) and a place operator that accepts *rooms* as
targets -- and an operator set is the specification of the whole planning problem, so
it earns being readable in one place.

Naming is not free. railroad's C++ core attaches meaning to ``free``, ``waiting``,
``at`` and ``found``, and ``ObjectSearchEnvironment`` adds ``searched``, ``revealed``
and ``holding`` plus the action prefixes ``move`` / ``place`` / ``search``. Everything
here either uses those with their intended meaning or picks a fresh name.

Two families of guard, for two different failure modes:

``just-picked`` / ``just-placed`` (lifted from railroad's own ``_blocking`` operator
variants) stop the planner discovering that pick-then-place-in-the-same-spot is a cheap
way to make the state look busy. They expire 0.1 s after the action, so they constrain
only the immediately following dispatch.

``failed-move`` / ``failed-pick`` / ``failed-place`` are ours, and they exist because
railroad has no notion of an action that *tried and did not work*. Effects are
declarative: once dispatched, they fire. Our skills fail routinely and for real reasons
(nav timeout, an object 0.85 m away when reach is 0.80, a grasp that slips). The
environment asserts one of these fluents when a skill reports failure, and every
operator refuses an action already marked, so the planner is forced to try something
else instead of re-dispatching the identical failing action forever. It is a blunt
instrument -- one failure bans that target permanently -- but a blunt instrument that
terminates beats a subtle one that loops. See the module docstring of
:mod:`g1sim.task.symbolic.environment` for the intended refinement.
"""

from __future__ import annotations

from typing import Callable, List

from railroad.core import Effect, Fluent as F, Operator

# Magic grasp is instantaneous in sim; these are what a real arm would plausibly take,
# and they matter only in that they make pick/place non-free relative to walking.
PICK_TIME = 4.0
PLACE_TIME = 4.0


def build_operators(move_time: Callable[[str, str, str], float], *,
                    pick_time: float = PICK_TIME,
                    place_time: float = PLACE_TIME) -> List[Operator]:
    """The three operators, bound to a duration model from
    :mod:`g1sim.task.symbolic.costs`."""
    return [_move(move_time), _pick(pick_time), _place(place_time)]


def _move(move_time) -> Operator:
    """``move ?r ?from ?to`` -- walk to a location.

    Executed by ``goto_room`` for a room symbol and ``goto_object`` for a surface, both
    of which route with A* across the whole apartment; there is no adjacency
    precondition because there is no adjacency constraint. Unreachability is expressed
    in the *cost* instead: a duration model returning ``inf`` drops the action.
    """
    return Operator(
        name="move",
        parameters=[("?r", "robot"), ("?from", "location"), ("?to", "location")],
        preconditions=[F("at ?r ?from"), F("free ?r"), ~F("failed-move ?r ?to")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?r ?from")}),
            Effect(time=(move_time, ["?r", "?from", "?to"]),
                   resulting_fluents={F("free ?r"), F("at ?r ?to")}),
        ],
    )


def _pick(pick_time: float) -> Operator:
    """``pick ?r ?loc ?obj`` -- lift an object from the location the robot is at.

    ``at ?r ?loc`` plus ``at ?obj ?loc`` is how "in reach" is expressed symbolically:
    standing at a surface puts the robot within ``PICK_RADIUS`` of that surface's
    footprint, and an object resting on it is inside that footprint. That inference is
    sound for 52 of the 55 objects-on-surfaces in this apartment and the remainder come
    back as honest skill failures (see ``g1sim/skills/types.py`` for the geometry).
    """
    return Operator(
        name="pick",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("at ?obj ?loc"),
                       ~F("hand-full ?r"), ~F("just-placed ?r ?obj"),
                       ~F("failed-pick ?r ?obj")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"), F("not at ?obj ?loc")}),
            Effect(time=pick_time,
                   resulting_fluents={F("free ?r"), F("holding ?r ?obj"),
                                      F("hand-full ?r"), F("just-picked ?r ?obj")}),
            Effect(time=pick_time + 0.1,
                   resulting_fluents={~F("just-picked ?r ?obj")}),
        ],
    )


def _place(place_time: float) -> Operator:
    """``place ?r ?loc ?obj`` -- set the carried object down at the current location.

    ``?loc`` may be a surface (the object ends up on top of it) or a room (dropped on
    the floor at the room's nav point); ``RobotSkills.place`` resolves either, so the
    operator does not need to distinguish them.
    """
    return Operator(
        name="place",
        parameters=[("?r", "robot"), ("?loc", "location"), ("?obj", "object")],
        preconditions=[F("at ?r ?loc"), F("free ?r"), F("holding ?r ?obj"),
                       F("hand-full ?r"), ~F("just-picked ?r ?obj"),
                       ~F("failed-place ?r ?loc")],
        effects=[
            Effect(time=0, resulting_fluents={F("not free ?r"),
                                              F("not holding ?r ?obj")}),
            Effect(time=place_time,
                   resulting_fluents={F("free ?r"), F("at ?obj ?loc"),
                                      ~F("hand-full ?r"), F("just-placed ?r ?obj")}),
            Effect(time=place_time + 0.1,
                   resulting_fluents={~F("just-placed ?r ?obj")}),
        ],
    )
