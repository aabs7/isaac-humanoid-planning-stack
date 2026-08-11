"""The seam: grounding, execution through the skills, observation, and failure repair."""

import pytest
from railroad.core import Fluent as F, get_action_by_name

from g1sim.skills.types import SkillResult
from g1sim.task.symbolic import G1Environment, object_at


def _act(sym_env, name):
    """Dispatch one grounded action by name."""
    actions = sym_env.get_actions()
    action = get_action_by_name(actions, name)
    assert action is not None, f"{name!r} was not grounded"
    sym_env.act(action)
    return action


# -- grounding ------------------------------------------------------------
def test_the_expected_actions_are_grounded(sym_env):
    names = {a.name for a in sym_env.get_actions()}
    assert "move g1 livingroom counter_0000" in names
    assert "pick g1 counter_0000 cup_0001" in names
    assert "place g1 dining_table_0000 cup_0001" in names


def test_self_moves_are_dropped(sym_env):
    """A zero-distance move is never useful; the cost model returns inf and grounding
    drops the action."""
    assert not [a for a in sym_env.get_actions()
                if a.name.split()[2] == a.name.split()[3]]


def test_failed_guards_survive_grounding(sym_env):
    """`failed-*` is produced by no operator effect, so grounding's static inference would
    judge it immutable and *compile the guards away* -- after which a failing action would
    be re-dispatched forever. Declaring the predicates runtime-mutated prevents that."""
    assert {"failed-move", "failed-pick", "failed-place"} <= \
        sym_env.runtime_mutated_predicates()
    sym_env.get_actions()       # populates the eliminated set
    assert not ({"failed-move", "failed-pick", "failed-place"}
                & set(sym_env._eliminated_predicates))


# -- execution ------------------------------------------------------------
def test_move_drives_the_robot(sym_env, env):
    _act(sym_env, "move g1 livingroom counter_0000")
    assert F("at g1 counter_0000") in sym_env.state.fluents
    assert F("free g1") in sym_env.state.fluents
    # The mock robot really moved: it is within reach of the counter's footprint.
    assert smap_dist(env, "counter_0000") < 0.8


def test_pick_then_place_moves_the_object_in_the_map(sym_env, env, smap):
    _act(sym_env, "move g1 livingroom counter_0000")
    _act(sym_env, "pick g1 counter_0000 cup_0001")
    assert env.held is not None and env.held.name == "cup_0001"
    assert F("holding g1 cup_0001") in sym_env.state.fluents

    _act(sym_env, "move g1 counter_0000 dining_table_0000")
    _act(sym_env, "place g1 dining_table_0000 cup_0001")
    assert env.held is None
    assert smap.get("cup_0001").supported_by == "dining_table_0000"
    assert F("at cup_0001 dining_table_0000") in sym_env.state.fluents


def test_observation_reads_the_map_not_the_effects(sym_env, smap):
    """The environment's world beliefs come from the semantic map, so a change made
    behind the planner's back is picked up rather than contradicted."""
    assert F("at cup_0000 dining_table_0000") in sym_env.state.fluents
    smap.set_carried(smap.get("cup_0000"))       # something moved the cup
    sym_env.observe()
    assert F("at cup_0000 dining_table_0000") not in sym_env.state.fluents


def smap_dist(env, name):
    return env.smap.get(name).xy_dist(*env.xy())


# -- failure --------------------------------------------------------------
class FailingPick:
    """Wraps a skills object so `pick` always fails, as a real marginal grasp would."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def pick(self, obj):
        return SkillResult(False, "pick", f"{obj} slipped out of the gripper")


def test_a_failed_pick_frees_the_robot_and_bans_the_object(env, domain):
    sym_env = G1Environment(FailingPick(env), domain, verbose=False)
    _act(sym_env, "move g1 livingroom counter_0000")
    _act(sym_env, "pick g1 counter_0000 cup_0001")

    fluents = sym_env.state.fluents
    assert F("free g1") in fluents, "a stuck-busy robot deadlocks the act() loop"
    assert F("failed-pick g1 cup_0001") in fluents
    assert F("hand-full g1") not in fluents
    # The optimistic time-0 effect took the cup off the counter; observation put it back.
    assert F("at cup_0001 counter_0000") in fluents


def test_a_banned_object_is_not_applicable_again(env, domain):
    """The ban makes every pick of that object *inapplicable*, not ungrounded: grounding
    depends only on the object universe and the static facts, so the action still exists
    and it is the precondition check -- the planner's and ours -- that rejects it."""
    sym_env = G1Environment(FailingPick(env), domain, verbose=False)
    _act(sym_env, "move g1 livingroom counter_0000")
    _act(sym_env, "pick g1 counter_0000 cup_0001")

    doomed = [a for a in sym_env.get_actions()
              if a.name.startswith("pick g1") and a.name.endswith("cup_0001")]
    assert doomed, "the actions are still grounded"
    assert not any(sym_env.state.satisfies_precondition(a) for a in doomed)

    # The ban is per (robot, object): walk to the other cup and it is pickable, which
    # would not be true if the ban had leaked onto the pick operator as a whole.
    _act(sym_env, "move g1 counter_0000 dining_table_0000")
    other = get_action_by_name(sym_env.get_actions(), "pick g1 dining_table_0000 cup_0000")
    assert sym_env.state.satisfies_precondition(other)


class FailingMove:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def goto_object(self, obj, **kw):
        return SkillResult(False, "goto_object", "timeout after 90s")

    def goto_room(self, room, **kw):
        return SkillResult(False, "goto_room", "timeout after 90s")


def test_a_failed_move_leaves_the_robot_at_a_real_location(env, domain):
    """The time-0 effect deleted `at g1 livingroom` and the completion effect never
    fired, so without repair the robot is at *no* location and every operator becomes
    inapplicable."""
    sym_env = G1Environment(FailingMove(env), domain, verbose=False)
    _act(sym_env, "move g1 livingroom counter_0000")

    at_robot = [f for f in sym_env.state.fluents if f.name == "at" and f.args[0] == "g1"]
    assert len(at_robot) == 1, f"robot must be at exactly one location, got {at_robot}"
    assert F("free g1") in sym_env.state.fluents
    assert F("failed-move g1 counter_0000") in sym_env.state.fluents
    # Still able to act: something is applicable.
    assert sym_env.get_actions()


# -- goal validation ------------------------------------------------------
def test_goal_naming_an_unknown_symbol_is_rejected(sym_env):
    with pytest.raises(ValueError, match="not in the domain"):
        sym_env.validate_goal(object_at("cup_0001", "no_such_table_0000"))


def test_valid_goal_passes(sym_env):
    sym_env.validate_goal(object_at("cup_0001", "dining_table_0000"))
