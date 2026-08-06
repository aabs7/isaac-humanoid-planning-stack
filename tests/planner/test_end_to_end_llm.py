"""Whole tasks, driven by the real local model.

The planner's job is to turn a sentence into a sequence of skill calls that leaves
the world in the right state, so that is what these assert: the **final scene**, not
which actions were chosen. A different-but-valid route must pass.

They run against ``MockSkills`` -- the real planner, the real prompts, the real
Ollama client, no Isaac -- so a whole task takes seconds instead of minutes. Marked
``llm``: they skip when Ollama is not running.

    pytest -m llm -v          # just these
    pytest -m "not llm"       # the fast deterministic suite
"""

from __future__ import annotations

import pytest

from g1sim.planner import Planner
from tests.helpers.graph_invariants import assert_graph_consistent

pytestmark = pytest.mark.llm


def run(llm, env, goal, max_steps=10):
    return Planner(llm, max_steps=max_steps, verbose=False).run(env, goal)


@pytest.mark.xfail(reason="run() honours finish(success=True) unconditionally, even "
                          "when its own _targets_satisfied() says the target (cup_0000) never "
                          "moved. This is because the model might ask to move cup_0001 instead, and the model claims that cup is moved. "
                          "Not strict: the model sometimes gets the task right and "
                          "the test passes.")
def test_it_carries_an_object_to_another_room(llm, env, smap):
    outcome = run(llm, env, "bring the cup from the livingroom to the kitchen")

    assert env.held is None, "the robot finished still holding the cup"
    # A reported success must not be contradicted by the scene.
    assert smap.get("cup_0000").room == "kitchen", (
        f"planner reported success={outcome['success']} with reason "
        f"{outcome['reason']!r}, but cup_0000 is still in the livingroom")
    assert_graph_consistent(smap, "after an LLM-planned delivery")


def test_it_places_an_object_on_a_named_surface(llm, env, smap):
    outcome = run(llm, env, "put the book on the kitchen counter")

    assert smap.get("book_0000").supported_by == "counter_0000", outcome["reason"]
    assert_graph_consistent(smap, "after an LLM-planned stack")


def test_it_moves_the_instance_the_goal_names_and_leaves_the_other(llm, env, smap):
    """There is a cup in the livingroom and another in the kitchen. Picking the wrong
    instance is the failure mode the up-front goal grounding exists to prevent."""
    run(llm, env, "take the cup that is in the livingroom to the bedroom", max_steps=12)

    assert smap.get("cup_0000").room == "bedroom"
    assert smap.get("cup_0001").room == "kitchen", "the kitchen cup should not move"


def test_an_impossible_goal_fails_without_wrecking_the_scene(llm, env, smap):
    """No banana exists. The planner must give up rather than crash, loop forever, or
    move something else instead."""
    before = {name: o.room for name, o in smap.objects.items()}

    outcome = run(llm, env, "bring me the banana from the garage", max_steps=6)

    assert outcome["success"] is False
    assert {name: o.room for name, o in smap.objects.items()} == before
    assert_graph_consistent(smap, "after an impossible goal")
