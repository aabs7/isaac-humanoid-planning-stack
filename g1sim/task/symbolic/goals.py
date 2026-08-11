"""Natural language -> railroad goal fluents.

The symbolic planner searches; it does not interpret. Its goal is a fluent expression like
``F("at cup_0000 table_0002")``, which is precise and unusable as a human interface -- you
have to know that the mug you meant is ``cup_0000`` and the table is ``table_0002``. This
module puts the LLM back in the one place it is genuinely better than search: turning
"bring the mug from the kitchen to the living room table" into that expression, once, before
planning starts.

So the two planners are not rivals here. :mod:`g1sim.task.llm_based` asks a model to choose
*every action*, and lives with a model that can hallucinate its way through a plan.
This asks a model for *one translation* and then hands over to a planner that cannot
hallucinate at all. The LLM's output is also the easiest kind to check: a goal either names
symbols that exist or it does not.

Three guardrails, in the order they matter:

* **The symbol menu.** The prompt lists the apartment's exact rooms, surfaces and pickable
  objects, grouped by room, so the model is choosing from a set rather than inventing names.
* **Schema-constrained output.** The reply is forced into :data:`GOAL_SCHEMA`, so parsing
  cannot fail and the predicate can only be one the operators actually produce.
* **Resolution with an informed retry.** Every name is resolved against the domain -- exact
  symbol first, then category ("mug" -> the nearest ``cup_*``). A name that resolves to
  nothing is not a silent wrong answer: the failure, with the valid options, goes back to
  the model and it tries again. This is the same guard the ReAct planner gets from
  ``ground()``, moved to translation time.

Conjunction and negation only (``at``/``holding``, optionally negated, ANDed). Disjunctive
goals are expressible in railroad but nobody asks for them in English, and a tighter schema
is a model that fails less.
"""

from __future__ import annotations

import functools
import json
import operator
from dataclasses import dataclass, field
from typing import List, Optional

from railroad.core import Fluent as F

from g1sim.perception.semantic_map import _normalize_room
from g1sim.task.symbolic.domain import SemanticDomain

# The two predicates a goal can be built from -- the ones our operators establish. `at`
# takes an object and a location; `holding` takes an object (the robot is implicit, since
# there is one).
PREDICATES = ("at", "holding")

GOAL_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        # The escape hatch, and it is load-bearing. Without a way to say "no", a model asked
        # to fetch something the apartment does not contain will emit a *valid* goal for a
        # different object -- observed: "fetch the unicorn" came back as
        # `at book_0000 table_0001`, with a thought that correctly noted no unicorn exists.
        # Every name resolved, so nothing downstream could catch it, and the robot would
        # have confidently done the wrong task.
        "impossible": {"type": "string"},
        "goal": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "predicate": {"type": "string", "enum": list(PREDICATES)},
                    "object": {"type": "string"},
                    "location": {"type": "string"},
                    "negated": {"type": "boolean"},
                },
                "required": ["predicate", "object"],
            },
        },
    },
    "required": ["thought"],
}

SYSTEM_PROMPT = (
    "You translate a household task, given in plain English, into a formal goal for a "
    "robot task planner. You do NOT plan: you never say how to do the task, only what "
    "must be TRUE when it is done. The planner works out the actions.\n\n"
    "A goal is a list of conditions, all of which must hold (they are ANDed):\n"
    "  {\"predicate\": \"at\", \"object\": <object>, \"location\": <surface or room>}\n"
    "      -- that object is resting there when the task is done.\n"
    "  {\"predicate\": \"holding\", \"object\": <object>}\n"
    "      -- the robot is carrying that object when the task is done.\n"
    "Add \"negated\": true to require the opposite (e.g. an object NOT on a table).\n\n"
    "Rules:\n"
    "- Use ONLY the exact names listed under SYMBOLS below. Never invent a name.\n"
    "- 'object' must come from PICKABLE OBJECTS. Only those can be moved.\n"
    "- 'location' must be a SURFACE or a ROOM. A pickable object is not a location, so "
    "you cannot put something on top of something the robot could carry.\n"
    "- 'take/fetch/bring me X' with no destination is holding(X). 'Put X on Y' or "
    "'bring X to Y' is at(X, Y).\n"
    "- 'clear Y' / 'take everything off Y' is one negated at(...) per object currently "
    "on Y -- read them from the listing.\n"
    "- Choose the instance the request implies: if the task says the kitchen cup, pick a "
    "cup listed under kitchen.\n"
    "- Keep the goal MINIMAL. State the end condition only, never intermediate steps: no "
    "holding(X) alongside at(X, Y), because the robot must put X down to satisfy at.\n"
    "- If the task names something this home does not contain, or asks for something these "
    "two predicates cannot express, set \"impossible\" to a one-sentence explanation and "
    "leave \"goal\" empty. NEVER substitute a different object or place: a wrong goal makes "
    "the robot confidently do the wrong task, which is far worse than admitting it cannot "
    "be done.\n"
    "- 'thought' is one short sentence on which symbols you chose and why."
)


class GoalTranslationError(RuntimeError):
    """The model's goal could not be turned into fluents. The message names what failed
    and lists valid options, so it is also what gets fed back for a retry."""


class GoalUnsatisfiable(GoalTranslationError):
    """The model declined the task: this home cannot satisfy it, or the goal language
    cannot express it.

    Deliberately **not** retried. A retry after a refusal is pressure to invent something,
    and an invented goal is exactly the failure the refusal avoided. Subclasses
    GoalTranslationError so callers that only care "no goal came out of this" need no
    change.
    """


@dataclass
class TranslatedGoal:
    """The result of translating one task."""
    goal: object                       # a railroad Goal (or a single Fluent, which is one)
    literals: List[str] = field(default_factory=list)   # readable, for logging
    objects: set = field(default_factory=set)           # object symbols the goal names
    locations: set = field(default_factory=set)         # location symbols the goal names
    thought: str = ""
    task: str = ""
    attempts: int = 1

    def __str__(self) -> str:
        return " AND ".join(self.literals)


# ---------------------------------------------------------------------------
# The symbol menu
# ---------------------------------------------------------------------------
def describe_symbols(domain: SemanticDomain, smap) -> str:
    """The apartment's symbols, grouped by room, as the prompt sees them.

    Grouping by room is what makes "the kitchen cup" answerable: the model needs to see
    which ``cup_*`` is in the kitchen. Each pickable object also shows what it is currently
    on, because "the mug on the dining table" is how people refer to things.
    """
    lines = ["SYMBOLS -- use these names exactly.", "", "ROOMS (a room is a valid location; "
             "placing there puts the object on its floor):",
             "  " + ", ".join(sorted(domain.rooms))]

    lines.append("\nSURFACES (valid locations; things can be put on top of them):")
    for room in sorted(domain.rooms):
        here = sorted(s for s in domain.surfaces
                      if (smap.get(s) is not None and smap.get(s).room == room))
        if here:
            lines.append(f"  {room}: " + ", ".join(here))

    lines.append("\nPICKABLE OBJECTS (only these can be moved; 'on X' is where it sits now):")
    for room in sorted(domain.rooms):
        here = []
        for name in sorted(domain.objects):
            o = smap.get(name)
            if o is None or o.room != room:
                continue
            here.append(f"{name} (on {o.supported_by})" if o.supported_by else name)
        if here:
            lines.append(f"  {room}: " + ", ".join(here))

    held = [o.name for o in smap.objects.values() if o.held]
    if held:
        lines.append("\nCURRENTLY IN THE ROBOT'S HANDS: " + ", ".join(held))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def translate(llm, smap, task: str, *, domain: Optional[SemanticDomain] = None,
              max_attempts: int = 3, verbose: bool = True) -> TranslatedGoal:
    """Turn ``task`` (plain English) into a railroad goal.

    ``domain`` should be the *unrestricted* domain -- the model has to be able to name any
    pickable object. Restrict the domain afterwards, using
    :attr:`TranslatedGoal.objects`, if you want the smaller search problem.

    Raises:
        GoalTranslationError: if the model cannot produce a resolvable goal in
            ``max_attempts`` tries. The message says which name failed and what was valid.
    """
    domain = domain if domain is not None else SemanticDomain.build(smap)
    menu = describe_symbols(domain, smap)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{menu}\n\nTASK: {task}\n\nGive the goal."},
    ]

    last_error = None
    for attempt in range(1, max_attempts + 1):
        reply = llm.chat_json(messages, GOAL_SCHEMA)
        try:
            result = _build_goal(domain, smap, reply, task)
        except GoalUnsatisfiable:
            raise                       # a refusal is final; see GoalUnsatisfiable
        except GoalTranslationError as e:
            last_error = e
            if verbose:
                print(f"[goal] attempt {attempt} rejected: {e}")
            if attempt == max_attempts:
                break
            # Hand the model its own answer plus the specific reason, so the retry is
            # informed rather than a re-roll of the same dice.
            messages += [
                {"role": "assistant", "content": json.dumps(reply)},
                {"role": "user", "content": f"That goal is invalid: {e}\n"
                                            f"Correct it, using only the listed names. If "
                                            f"the task cannot be done in this home, set "
                                            f"\"impossible\" instead of guessing."},
            ]
            continue

        result.attempts = attempt
        if verbose:
            print(f"[goal] \"{task}\"\n[goal] -> {result}\n[goal]    ({result.thought})")
        return result

    raise GoalTranslationError(
        f"could not translate {task!r} into a goal after {max_attempts} attempts. "
        f"Last problem: {last_error}")


def _build_goal(domain: SemanticDomain, smap, reply: dict, task: str) -> TranslatedGoal:
    """Resolve one schema-valid reply into fluents, or raise with a usable reason."""
    refusal = (reply.get("impossible") or "").strip()
    if refusal:
        raise GoalUnsatisfiable(f"the task cannot be expressed as a goal here: {refusal}")

    specs = reply.get("goal") or []
    if not specs:
        raise GoalTranslationError("the goal list is empty; give at least one condition.")

    literals, fluents, objects, locations = [], [], set(), set()
    for spec in specs:
        predicate = spec.get("predicate")
        obj = _resolve_object(domain, smap, spec.get("object"))
        objects.add(obj)

        if predicate == "holding":
            fluent = F(f"holding {domain.robot} {obj}")
        elif predicate == "at":
            loc = _resolve_location(domain, smap, spec.get("location"))
            locations.add(loc)
            fluent = F(f"at {obj} {loc}")
        else:
            raise GoalTranslationError(
                f"unknown predicate {predicate!r}; use one of {', '.join(PREDICATES)}.")

        if spec.get("negated"):
            fluent = ~fluent
        fluents.append(fluent)
        literals.append(str(fluent))

    # A goal that both holds an object and puts it somewhere can never be satisfied: place
    # is what establishes `at`, and it empties the hand. Catch it here rather than let the
    # planner search for minutes and report an unreachable goal.
    for obj in objects:
        if (F(f"holding {domain.robot} {obj}") in fluents
                and any(f.name == "at" and f.args[0] == obj and not f.negated
                        for f in fluents)):
            raise GoalTranslationError(
                f"the goal both holds {obj} and puts it down somewhere, which cannot both "
                f"be true at the end. Keep only the destination.")

    goal = functools.reduce(operator.and_, fluents) if len(fluents) > 1 else fluents[0]
    return TranslatedGoal(goal=goal, literals=literals, objects=objects,
                          locations=locations, thought=reply.get("thought", ""), task=task)


def _resolve_object(domain: SemanticDomain, smap, ref) -> str:
    """A goal's object name -> an object symbol.

    Accepts an exact symbol, or a category ("cup", "mug" only insofar as the map calls it
    that) resolved to some instance of it that is actually pickable.
    """
    if not isinstance(ref, str) or not ref:
        raise GoalTranslationError("a condition is missing its 'object'.")
    if ref in domain.objects:
        return ref
    # A category: take an instance that is in the pickable universe.
    hits = [o.name for o in smap.find(ref) if o.name in domain.objects]
    if hits:
        return sorted(hits)[0]
    if ref in domain.locations:
        raise GoalTranslationError(
            f"'{ref}' is a location, not something the robot can pick up. Objects are the "
            f"names under PICKABLE OBJECTS.")
    raise GoalTranslationError(
        f"there is no pickable object called '{ref}'. "
        f"Valid categories: {_categories(domain, smap)}.")


def _resolve_location(domain: SemanticDomain, smap, ref) -> str:
    """A goal's location name -> a location symbol (a surface or a room)."""
    if not isinstance(ref, str) or not ref:
        raise GoalTranslationError(
            "an 'at' condition is missing its 'location' (a surface or a room).")
    if ref in domain.locations:
        return ref
    room = _normalize_room(ref)                     # "living room" -> "livingroom"
    if room in domain.rooms:
        return room
    hits = [o.name for o in smap.find(ref) if o.name in domain.surfaces]
    if hits:
        return sorted(hits)[0]
    if ref in domain.objects:
        raise GoalTranslationError(
            f"'{ref}' is a pickable object, so it is not a place to put things. Use a "
            f"surface or a room.")
    raise GoalTranslationError(
        f"there is no location called '{ref}'. Rooms are: "
        f"{', '.join(sorted(domain.rooms))}. Surfaces are named in the listing.")


def _categories(domain: SemanticDomain, smap) -> str:
    cats = sorted({smap.get(n).category for n in domain.objects if smap.get(n)})
    return ", ".join(cats) or "none"
