"""The parts of the planner that never talk to the model.

Argument grounding, the completion check, the skill catalog and the state text are
plain functions of the semantic map, so they are worth pinning exactly -- unlike the
model's behaviour, which is checked end-to-end in ``test_end_to_end_llm.py``.
"""

from __future__ import annotations

import pytest

from g1sim.task.planner import (RESPONSE_SCHEMA, SKILL_DOCS, SKILLS, _execute,
                           build_state_text, ground)
from g1sim.skills.types import SkillResult

VALID_ARGS = {
    "scan": {},
    "goto_room": {"room": "kitchen"},
    "goto_object": {"object": "cup_0000"},
    "pick": {"object": "cup_0000"},
    "place": {"location": "kitchen"},
}


# ---------------------------------------------------------------------------
# The catalog the model is shown must match what actually runs.
# ---------------------------------------------------------------------------
def test_catalog_schema_and_dispatcher_all_agree(env):
    assert [name for name, _, _ in SKILL_DOCS] == SKILLS
    assert RESPONSE_SCHEMA["properties"]["skill"]["enum"] == SKILLS

    for skill in SKILLS:
        if skill == "finish":
            continue                      # handled by the loop, never dispatched
        assert isinstance(_execute(env, skill, VALID_ARGS[skill]), SkillResult), \
            f"{skill} is offered to the model but _execute cannot run it"


# ---------------------------------------------------------------------------
# ground(): the firewall between a hallucinating model and the robot.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skill, args", [
    ("goto_room", {"room": "kitchen"}),
    ("goto_object", {"object": "cup_0000"}),      # exact name
    ("goto_object", {"object": "cup"}),           # category, two instances
    ("pick", {"object": "cup_0000"}),
    ("place", {"location": "kitchen"}),           # a room floor
    ("place", {"location": "counter_0000"}),      # on top of an object
])
def test_legitimate_arguments_ground(env, skill, args):
    assert ground(env, skill, args) == (True, None)


@pytest.mark.parametrize("skill, args", [
    ("goto_object", {"object": "banana_9999"}),   # invented object
    ("goto_room", {"room": "garage"}),            # invented room
    ("goto_object", {}),                          # no argument at all
    ("pick", {}),
    ("place", {}),
    ("place", {"location": "atlantis"}),
    ("teleport", {}),                             # invented skill
])
def test_bad_arguments_are_rejected_with_a_usable_hint(env, skill, args):
    """The hint is the only thing the model gets back to correct itself from, so an
    empty or silent rejection is as bad as a crash."""
    ok, err = ground(env, skill, args)
    assert ok is False
    assert err and len(err) > 15


def test_the_room_object_mixup_is_named_in_both_directions(env):
    """By far the most common model mistake; the fix has to be in the message."""
    _, to_room = ground(env, "goto_room", {"room": "cup_0000"})
    assert "OBJECT" in to_room and "goto_object" in to_room

    _, to_object = ground(env, "goto_object", {"object": "kitchen"})
    assert "ROOM" in to_object and "goto_room" in to_object


@pytest.mark.parametrize("spelling", ["livingroom", "living room", "LivingRoom"])
def test_ground_accepts_every_room_spelling_the_skill_can_execute(env, spelling):
    """Grounding must accept exactly what the skill accepts. Stricter than the skill
    (as it was, on spelling alone) fails an action that would have worked."""
    assert ground(env, "goto_room", {"room": spelling}) == (True, None)
    assert ground(env, "place", {"location": spelling}) == (True, None)
    assert env.goto_room(spelling).ok, "the skill itself accepts this spelling"


def test_ground_still_rejects_a_spelling_the_skill_cannot_execute(env):
    """The other half of the invariant. ``_normalize_room`` strips case and spaces but
    not underscores, so the skill fails on 'living_room' -- and grounding must fail
    with it rather than paper over a difference the robot cannot honour."""
    assert env.goto_room("living_room").ok is False
    assert ground(env, "goto_room", {"room": "living_room"})[0] is False


# ---------------------------------------------------------------------------
# _targets_satisfied(): the deterministic completion check. The tri-state matters:
# None means "can't tell" and hands completion to the LLM judge instead.
# ---------------------------------------------------------------------------
def test_completion_is_false_until_the_target_arrives(planner, env):
    assert planner._targets_satisfied(env, ["cup_0000"], "kitchen") is False

    env.goto_object("cup_0000")
    env.pick("cup_0000")
    assert planner._targets_satisfied(env, ["cup_0000"], "kitchen") is False  # carried

    env.goto_room("kitchen")
    env.place("kitchen")
    assert planner._targets_satisfied(env, ["cup_0000"], "kitchen") is True


def test_a_surface_destination_needs_the_object_to_rest_on_it(planner, env):
    env.goto_object("cup_0000")
    env.pick("cup_0000")
    env.goto_object("counter_0000")
    env.place("counter_0000")

    assert planner._targets_satisfied(env, ["cup_0000"], "counter_0000") is True
    assert planner._targets_satisfied(env, ["cup_0000"], "nightstand_0000") is False


@pytest.mark.parametrize("targets, destination", [
    ([], "kitchen"),                  # nothing was grounded
    (["cup_0000"], None),             # no destination was grounded
    (["banana_9999"], "kitchen"),     # target is not in the map
    (["cup_0000"], "atlantis"),       # destination is neither room nor object
])
def test_unevaluable_completion_reports_none_so_the_judge_takes_over(
        planner, env, targets, destination):
    assert planner._targets_satisfied(env, targets, destination) is None


# ---------------------------------------------------------------------------
# build_state_text(): the observation the model sees every turn.
# ---------------------------------------------------------------------------
def test_state_text_reports_where_the_robot_is_and_what_it_holds(env):
    text = build_state_text(env)
    assert "Robot position: (0.50, 0.50)" in text
    assert "in the livingroom" in text
    assert "hands are empty" in text
    assert "room:kitchen" in text                 # the whole scene graph is included


def test_a_carried_object_is_announced_and_gone_from_its_old_room(env):
    """Without this the model walks back to fetch what it is already holding."""
    env.goto_object("cup_0000")
    env.pick("cup_0000")

    text = build_state_text(env)

    assert "carrying 'cup_0000'" in text
    assert "cup_0000" not in text.split("room:livingroom", 1)[1]
