"""English -> goal fluents.

The LLM call itself is faked: a canned reply is exactly what the schema guarantees Ollama
returns, so everything worth testing -- symbol resolution, the retry, the impossible-goal
check, the prompt's symbol menu -- is deterministic and runs without a model. There is one
marked test at the bottom that drives the real local model.
"""

import json

import pytest
from railroad.core import Fluent as F

from g1sim.task.symbolic import SemanticDomain, describe_symbols, translate
from g1sim.task.symbolic.goals import (GOAL_SCHEMA, GoalTranslationError,
                                        GoalUnsatisfiable)


class FakeLLM:
    """Returns canned replies in order, recording the messages it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat_json(self, messages, schema):
        assert schema is GOAL_SCHEMA, "the reply must be schema-constrained"
        self.calls.append(messages)
        return self.replies.pop(0)


def reply(*conditions, thought="because"):
    return {"thought": thought, "goal": list(conditions)}


def at(obj, loc, negated=False):
    c = {"predicate": "at", "object": obj, "location": loc}
    if negated:
        c["negated"] = True
    return c


def holding_(obj):
    return {"predicate": "holding", "object": obj}


# -- resolution -----------------------------------------------------------
def test_exact_symbols_translate_to_an_at_fluent(smap, domain):
    llm = FakeLLM(reply(at("cup_0001", "dining_table_0000")))
    task = translate(llm, smap, "put the kitchen cup on the dining table",
                     domain=domain, verbose=False)

    assert task.goal == F("at cup_0001 dining_table_0000")
    assert task.objects == {"cup_0001"}
    assert task.locations == {"dining_table_0000"}
    assert task.attempts == 1


def test_a_holding_goal_needs_no_location(smap, domain):
    llm = FakeLLM(reply(holding_("cup_0001")))
    task = translate(llm, smap, "bring me a cup", domain=domain, verbose=False)

    assert task.goal == F("holding g1 cup_0001")
    assert task.objects == {"cup_0001"}


def test_a_category_resolves_to_a_pickable_instance(smap, domain):
    """The model may answer "cup"; the map has cup_0000 and cup_0001."""
    llm = FakeLLM(reply(holding_("cup")))
    task = translate(llm, smap, "bring me a cup", domain=domain, verbose=False)

    assert task.objects <= {"cup_0000", "cup_0001"}
    assert len(task.objects) == 1


def test_a_room_name_is_normalized(smap, domain):
    """"living room" is how people write it; the map's symbol is "livingroom"."""
    llm = FakeLLM(reply(at("cup_0001", "living room")))
    task = translate(llm, smap, "put the cup in the living room",
                     domain=domain, verbose=False)

    assert task.goal == F("at cup_0001 livingroom")


def test_a_negated_condition_becomes_a_negated_fluent(smap, domain):
    llm = FakeLLM(reply(at("cup_0000", "dining_table_0000", negated=True)))
    task = translate(llm, smap, "get the cup off the dining table",
                     domain=domain, verbose=False)

    assert task.goal == ~F("at cup_0000 dining_table_0000")


def test_several_conditions_are_anded(smap, domain):
    llm = FakeLLM(reply(at("cup_0000", "counter_0000"), at("cup_0001", "bed_0000")))
    task = translate(llm, smap, "put both cups away", domain=domain, verbose=False)

    assert len(task.literals) == 2
    assert task.objects == {"cup_0000", "cup_0001"}
    assert task.goal.evaluate({F("at cup_0000 counter_0000"), F("at cup_0001 bed_0000")})
    assert not task.goal.evaluate({F("at cup_0000 counter_0000")})


# -- rejection and retry --------------------------------------------------
def test_a_hallucinated_object_is_retried_with_the_reason(smap, domain):
    llm = FakeLLM(reply(holding_("unicorn_0001")), reply(holding_("cup_0001")))
    task = translate(llm, smap, "fetch the unicorn", domain=domain, verbose=False)

    assert task.attempts == 2
    assert task.goal == F("holding g1 cup_0001")
    # The retry has to *say* what was wrong, or it is just a re-roll.
    followup = llm.calls[-1][-1]["content"]
    assert "unicorn_0001" in followup and "no pickable object" in followup


def test_a_pickable_object_is_not_accepted_as_a_location(smap, domain):
    """You cannot put the cup on the book: a thing the robot can carry is not a place."""
    llm = FakeLLM(reply(at("cup_0001", "book_0000")), reply(at("cup_0001", "bed_0000")))
    task = translate(llm, smap, "put the cup on the book", domain=domain, verbose=False)

    assert task.goal == F("at cup_0001 bed_0000")
    assert "not a place to put things" in llm.calls[-1][-1]["content"]


def test_a_location_is_not_accepted_as_an_object(smap, domain):
    llm = FakeLLM(reply(holding_("dining_table_0000")), reply(holding_("cup_0001")))
    task = translate(llm, smap, "pick up the dining table", domain=domain, verbose=False)

    assert task.attempts == 2
    assert "not something the robot can pick up" in llm.calls[-1][-1]["content"]


def test_giving_up_raises_with_the_last_reason(smap, domain):
    llm = FakeLLM(*[reply(holding_("unicorn_0001"))] * 3)
    with pytest.raises(GoalTranslationError, match="unicorn_0001"):
        translate(llm, smap, "fetch the unicorn", domain=domain, max_attempts=3,
                  verbose=False)


def test_an_empty_goal_is_rejected(smap, domain):
    llm = FakeLLM({"thought": "nothing to do", "goal": []},
                  reply(holding_("cup_0001")))
    task = translate(llm, smap, "do something", domain=domain, verbose=False)
    assert task.attempts == 2


def test_holding_and_placing_the_same_object_is_rejected(smap, domain):
    """A common LLM habit: stating the intermediate step as well as the end state. `place`
    is what makes `at` true and it empties the hand, so both can never hold at once --
    catching it here saves minutes of the planner searching for the impossible."""
    llm = FakeLLM(reply(holding_("cup_0001"), at("cup_0001", "bed_0000")),
                  reply(at("cup_0001", "bed_0000")))
    task = translate(llm, smap, "pick up the cup and put it on the bed",
                     domain=domain, verbose=False)

    assert task.goal == F("at cup_0001 bed_0000")
    assert "cannot both" in llm.calls[-1][-1]["content"]


# -- the prompt -----------------------------------------------------------
def test_the_symbol_menu_lists_rooms_surfaces_and_objects_by_room(smap, domain):
    menu = describe_symbols(domain, smap)

    assert "livingroom" in menu and "kitchen" in menu
    assert "dining_table_0000" in menu and "counter_0000" in menu
    # A pickable object appears with where it currently sits, which is how people refer to
    # things ("the cup on the counter").
    assert "cup_0001 (on counter_0000)" in menu
    # Surfaces are grouped under their room so "the kitchen cup" is answerable.
    kitchen_line = next(l for l in menu.splitlines() if l.strip().startswith("kitchen:"))
    assert "counter_0000" in kitchen_line


def test_the_menu_reaches_the_model(smap, domain):
    llm = FakeLLM(reply(holding_("cup_0001")))
    translate(llm, smap, "bring me a cup", domain=domain, verbose=False)

    user_msg = llm.calls[0][1]["content"]
    assert "PICKABLE OBJECTS" in user_msg and "cup_0001" in user_msg
    assert "bring me a cup" in user_msg


def test_a_held_object_is_flagged_in_the_menu(smap, domain):
    smap.set_carried(smap.get("cup_0001"))
    assert "CURRENTLY IN THE ROBOT'S HANDS: cup_0001" in describe_symbols(domain, smap)


# -- end to end, through the planner --------------------------------------
@pytest.mark.symbolic
def test_a_translated_goal_plans_and_executes(smap, env):
    """The point of the whole module: English in, robot action out, with no hand-written
    fluents anywhere."""
    from g1sim.task.symbolic import G1Environment
    from g1sim.task.symbolic.planner import solve as run

    full = SemanticDomain.build(smap)
    llm = FakeLLM(reply(at("cup_0001", "dining_table_0000")))
    task = translate(llm, smap, "put the kitchen cup on the dining table",
                     domain=full, verbose=False)

    domain = SemanticDomain.build(smap, objects=sorted(task.objects))
    outcome = run(G1Environment(env, domain, verbose=False), task.goal, verbose=False)

    assert outcome.ok, outcome.reason
    assert smap.get("cup_0001").supported_by == "dining_table_0000"


@pytest.mark.llm
def test_the_real_model_translates_a_fetch_task(llm, smap, domain):
    """Drives the actual local model. Skips cleanly when Ollama is not running."""
    task = translate(llm, smap, "bring me the cup from the kitchen", domain=domain,
                     verbose=False)

    assert task.objects, task
    assert all(o in domain.objects for o in task.objects)
    assert all(l in domain.locations for l in task.locations)
    # The kitchen cup is cup_0001; a model that picks the livingroom one has ignored the
    # room grouping in the menu, which is the thing most worth knowing about.
    assert task.objects == {"cup_0001"}, f"chose {task.objects}: {json.dumps(task.literals)}"


# -- refusal --------------------------------------------------------------
def test_a_refusal_is_reported_not_retried(smap, domain):
    """Without an escape hatch the model substitutes. Observed on the real apartment:
    "fetch the unicorn" came back as `at book_0000 table_0001` -- every name resolved, so
    nothing downstream could catch it, and the robot would have done the wrong task
    confidently. A refusal must also NOT be retried: retrying is pressure to invent."""
    llm = FakeLLM({"thought": "no unicorn here", "impossible": "this home has no unicorn",
                   "goal": []},
                  reply(holding_("cup_0001")))          # would be used if we retried

    with pytest.raises(GoalUnsatisfiable, match="no unicorn"):
        translate(llm, smap, "fetch the unicorn", domain=domain, verbose=False)
    assert len(llm.calls) == 1, "a refusal must be final, not retried"


def test_a_refusal_is_a_goal_translation_error_too(smap, domain):
    """So callers that only care "no goal came out" need no change."""
    llm = FakeLLM({"thought": "n/a", "impossible": "cannot express that", "goal": []})
    with pytest.raises(GoalTranslationError):
        translate(llm, smap, "make me a sandwich", domain=domain, verbose=False)


def test_a_blank_impossible_field_is_not_a_refusal(smap, domain):
    """Models pad optional string fields with "". That must not read as a refusal."""
    llm = FakeLLM({"thought": "fine", "impossible": "  ",
                   "goal": [holding_("cup_0001")]})
    task = translate(llm, smap, "bring me a cup", domain=domain, verbose=False)
    assert task.goal == F("holding g1 cup_0001")


def test_the_prompt_tells_the_model_how_to_refuse(smap, domain):
    llm = FakeLLM(reply(holding_("cup_0001")))
    translate(llm, smap, "bring me a cup", domain=domain, verbose=False)
    system = llm.calls[0][0]["content"]
    assert "impossible" in system and "NEVER substitute" in system
