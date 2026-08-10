"""LLM task planner -- turns a natural-language goal into skill calls (Phase 1.1).

The planner is a thin, *sim-free* translation layer that sits above the skill seam:
given a goal, the current semantic-map state, and the skill catalog, it asks a local
LLM (via :mod:`g1sim.task.llm_based.llm`) for one skill call at a time, executes it
through whatever "skills" object it is handed, reads the :class:`SkillResult`, and
loops -- a ReAct
loop. Because it drives an *interface* (``.smap``, ``.xy()``, ``.held`` + the verb
methods), the identical planner runs against the sim-free ``MockSkills`` and the real
``RobotSkills``. It never imports Isaac.

Two guardrails against the usual LLM-planner failure modes:
  * **Structured output** -- the reply is constrained to :data:`RESPONSE_SCHEMA`, so
    parses are reliable.
  * **Argument grounding** -- every object/room/location argument is checked against
    the semantic map *before* execution (:func:`ground`); a hallucinated target never
    reaches the robot, it comes back as a synthetic failure the model corrects from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from g1sim.task.llm_based.llm import OllamaChat
from g1sim.skills.types import SkillResult

# ---------------------------------------------------------------------------
# Skill catalog -- the LLM-facing contract. One entry per verb the planner may
# emit; descriptions spell out preconditions so the model sequences correctly.
# ---------------------------------------------------------------------------
SKILLS = ["scan", "goto_room", "goto_object", "pick", "place", "finish"]

# Backstop for goal grounding only (the skills do not enforce it): anything bigger
# than this is furniture, not cargo. Deliberately generous -- a chair is ~1.15 m and
# must stay a legal target -- so it catches beds/sofas/curtains/fridges and leaves the
# real work to the "is it holding other things" test in _carryable_targets.
MAX_CARRY_DIM = 1.2       # metres, longest bounding-box edge

SKILL_DOCS = [
    ("scan", "{}",
     "Look around and report objects near the robot. Free; use it to confirm what is here."),
    ("goto_room", '{"room": "<room name>"}',
     "Walk to the middle of a room. Use ONLY as the destination for placing on a "
     "room's FLOOR. Do NOT use it to reach an object -- goto_object already travels to "
     "the object's room by itself, and heading to a small room's centre first can "
     "strand the robot in a dead-end."),
    ("goto_object", '{"object": "<object name or category>"}',
     "Walk ALL THE WAY to an object (from anywhere, across rooms) and stop within arm's "
     "reach of it. This is the ONLY step needed to reach an object -- no goto_room "
     "first. REQUIRED before pick, and before place-on-an-object."),
    ("pick", '{"object": "<object name or category>"}',
     "Grasp an object. Precondition: you must already be at it (goto_object first) and "
     "not already holding something."),
    ("place", '{"location": "<room name, OR an object name to place on top of>"}',
     "Put the held object down -- on the floor of a room, or on top of another object. "
     "Precondition: you are holding something and are within reach of the location "
     "(goto_room / goto_object first)."),
    ("finish", '{"success": true, "reason": "<why>"}',
     "End the task. Use when the goal is achieved, or set success=false when it is "
     "impossible (e.g. the requested object does not exist)."),
]

# JSON schema handed to the LLM as Ollama's `format` -- constrains every reply to one
# well-formed action. `args` is left free-form and validated per-skill by ground().
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "skill": {"type": "string", "enum": SKILLS},
        "args": {"type": "object"},
    },
    "required": ["thought", "skill", "args"],
}


def _catalog_text() -> str:
    return "\n".join(f"- {name}(args={args})  {desc}" for name, args, desc in SKILL_DOCS)


SYSTEM_PROMPT = (
    "You are the task planner for a household humanoid robot. You are given a goal in "
    "plain English, a live description of the environment (a scene graph built from the "
    "robot's map), and a set of skills. Choose ONE skill to call next, each turn, until "
    "the goal is done.\n\n"
    "Skills:\n" + _catalog_text() + "\n\n"
    "Rules:\n"
    "- Only ever reference rooms and objects that appear in the CURRENT STATE. Never "
    "invent an object.\n"
    "- To reach ANY object, call goto_object directly -- it walks the whole way there "
    "itself, across rooms. Never goto_room first just to then goto_object; that wastes "
    "time and can strand the robot in a small room.\n"
    "- To pick an object you must goto_object it first. To place: on a room's floor, "
    "goto_room that room first; on top of an object, goto_object that object first.\n"
    "- You carry only one object at a time.\n"
    "- Read each result. If a skill fails, fix the cause (usually: go to the target "
    "first) and retry, or finish(success=false) if it is truly impossible.\n"
    "- goto_room takes a ROOM name; goto_object takes an OBJECT name/category. Do not "
    "pass a room to goto_object or an object to goto_room.\n"
    "- If an action fails, READ the failure and do something DIFFERENT next -- never "
    "repeat the identical failing action.\n"
    "- Call finish as soon as the goal is achieved. Do not keep acting after that.\n"
    "- Respond with exactly one JSON action: a short thought, the skill, and its args.\n\n"
    "Canonical example -- 'bring the mug from the kitchen to the bedroom':\n"
    '  goto_object {"object": "mug"}      (walks to the kitchen mug)\n'
    '  pick        {"object": "mug"}\n'
    '  goto_room   {"room": "bedroom"}    (destination room, for a floor drop)\n'
    '  place       {"location": "bedroom"}\n'
    '  finish      {"success": true, "reason": "mug delivered"}'
)


# ---------------------------------------------------------------------------
# State serialization + argument grounding (both read only the semantic map).
# ---------------------------------------------------------------------------
def build_state_text(env) -> str:
    """The observation shown to the LLM each turn: the scene graph, plus the robot's
    position and what it is holding. Rebuilt every turn so it reflects moved objects."""
    smap = env.smap
    x, y = env.xy()
    if env.held is not None:
        hands = (f"IN YOUR HANDS: you are carrying '{env.held.name}'. It is NO LONGER on "
                 f"any table or in any room -- it is in your hands. Do NOT pick it or "
                 f"goto it again; walk to the DESTINATION and place() it there.")
    else:
        hands = "IN YOUR HANDS: nothing (hands are empty)."
    return (f"Robot position: ({x:.2f}, {y:.2f}) -- {_robot_location_phrase(smap, x, y)}.\n"
            f"{hands}\n"
            f"Rooms: {', '.join(smap.room_names())}.\n\n"
            f"Scene graph (where each thing IS right now):\n{smap.describe_graph()}")


def _robot_location_phrase(smap, x: float, y: float) -> str:
    """A human/LLM-friendly description of where the robot is standing: the room it is
    in and the nearest furniture, e.g. "in the livingroom, near dining_table
    [dining_table_0000]"."""
    room = smap.room_at(x, y)
    parts = [f"in the {room}" if room else "between rooms (in a doorway/open space)"]
    o = smap.nearest_object(x, y, max_dist=2.0)
    if o is not None:
        rel = "right at" if o.xy_dist(x, y) < 0.6 else "near"
        parts.append(f"{rel} {o.category} [{o.name}]")
    return ", ".join(parts)


def _resolve_object_name(smap, ref: str):
    """An object arg may be an exact name or a category. Return a concrete
    SemanticObject (nearest of a category) or None."""
    o = smap.get(ref)
    if o is not None:
        return o
    hits = smap.find(ref)
    return hits[0] if hits else None


def ground(env, skill: str, args: dict):
    """Validate a proposed action against the map. Return ``(ok, error_msg)``:
    ``(True, None)`` if executable, else ``(False, <hint listing valid options>)``.
    Catches hallucinated objects/rooms before they reach the robot."""
    smap = env.smap
    if skill in ("scan", "finish"):
        return True, None

    rooms_list = ", ".join(smap.room_names())

    if skill == "goto_room":
        room = args.get("room")
        # Ask the resolver the *skill* uses, rather than re-deriving the test here, so
        # grounding accepts exactly what goto_room can execute -- including spellings
        # navigable_point() normalizes ("living room" -> livingroom). Grounding looser
        # than the skill lets a hallucination through; stricter (as this was) fails
        # actions that would have worked, costing the model a turn.
        if isinstance(room, str) and smap.navigable_point(room) is not None:
            return True, None
        if room and _resolve_object_name(smap, room) is not None:  # gave an object
            return False, (f"'{room}' is an OBJECT, not a room. Use goto_object for it. "
                           f"Rooms are: {rooms_list}.")
        return False, f"no room '{room}'. Rooms are: {rooms_list}."

    if skill in ("goto_object", "pick"):
        ref = args.get("object")
        if ref and _resolve_object_name(smap, ref) is not None:
            return True, None
        if ref in smap.room_names():                               # gave a room
            return False, (f"'{ref}' is a ROOM, not an object. To go there use "
                           f"goto_room(room='{ref}'). To place on its floor: "
                           f"goto_room('{ref}') then place('{ref}').")
        cats = ", ".join(sorted(smap.categories())) or "none"
        return False, f"no object '{ref}' in the map. Known object categories: {cats}."

    if skill == "place":
        loc = args.get("location")
        if not loc:
            return False, "place needs a 'location' (a room name or an object name)."
        # Mirror _resolve_place: an object name to stack on, else a room to drop in.
        if isinstance(loc, str) and (smap.get(loc) is not None
                                     or smap.navigable_point(loc) is not None):
            return True, None
        return (False, f"cannot place at '{loc}'. Use a room "
                       f"({', '.join(smap.room_names())}) or an existing object name.")

    return False, f"unknown skill '{skill}'."


def _execute(env, skill: str, args: dict) -> SkillResult:
    """Dispatch a grounded action onto the skills object."""
    if skill == "scan":
        return env.scan()
    if skill == "goto_room":
        return env.goto_room(args["room"])
    if skill == "goto_object":
        return env.goto_object(args["object"])
    if skill == "pick":
        return env.pick(args["object"])
    if skill == "place":
        return env.place(args["location"])
    raise ValueError(f"not executable here: {skill}")


@dataclass
class Step:
    """One planner iteration, for the returned trace / logging."""
    skill: str
    args: dict
    thought: str
    result: SkillResult


class Planner:
    """Drives a goal to completion through a skills object via a ReAct loop."""

    def __init__(self, llm: Optional[OllamaChat] = None, *, max_steps: int = 15,
                 verbose: bool = True, on_action=None):
        self.llm = llm or OllamaChat()
        self.max_steps = max_steps
        self.verbose = verbose
        # Optional observer, called as on_action(step, skill, args, thought, result)
        # once when an action starts (result=None) and again when it finishes. Lets a
        # caller mirror the plan somewhere -- the video overlay uses it to print what
        # action the robot is executing.
        self.on_action = on_action

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _notify(self, step, skill, args, thought, result=None):
        """Fire the observer, never letting it break the run."""
        if self.on_action is None:
            return
        try:
            self.on_action(step, skill, args, thought, result)
        except Exception as e:      # pragma: no cover - a display must not kill a task
            self._log(f"    (on_action observer raised: {e})")

    # A crisp yes/no completion judge, asked in ISOLATION (not inside the noisy action
    # loop). A small model reliably answers "is this goal satisfied?" from the current
    # scene state even when it struggles to spontaneously emit `finish` mid-loop --
    # especially with multiple same-category objects, where it loses track of which
    # instance it just moved.
    _DONE_SCHEMA = {
        "type": "object",
        "properties": {"done": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["done", "reason"],
    }

    def _check_done(self, env, goal: str):
        q = (f"GOAL: {goal}\n\nCURRENT STATE:\n{build_state_text(env)}\n\n"
             "Considering ONLY the current state above, is the GOAL now fully "
             "satisfied? An object that started elsewhere and is now in the requested "
             "place counts as done, even if other similar objects exist elsewhere.")
        msgs = [{"role": "system",
                 "content": "You judge whether a household-robot task goal is already "
                            "satisfied by the current scene. Answer strictly from the state."},
                {"role": "user", "content": q}]
        try:
            ans = self.llm.chat_json(msgs, self._DONE_SCHEMA)
        except Exception:
            return False, ""
        return bool(ans.get("done")), ans.get("reason", "")

    # Up-front goal grounding: resolve the NL goal to CONCRETE object name(s) + one
    # destination, done ONCE at the start while the scene is still in its initial
    # configuration (so "the lamp on the balcony table" maps unambiguously to a single
    # id, before any move erases that provenance). This makes the actor target a
    # specific instance and turns completion into a deterministic check -- the key to
    # handling multiple same-category objects on a small model.
    _GROUND_SCHEMA = {
        "type": "object",
        "properties": {
            "targets": {"type": "array", "items": {"type": "string"}},
            "destination": {"type": "string"},
        },
        "required": ["targets", "destination"],
    }

    def _ground_goal(self, env, goal: str):
        q = (f"GOAL: {goal}\n\nSCENE (objects shown as category [exact_name]):\n"
             f"{env.smap.describe_graph()}\n\n"
             "Which EXACT object name(s) must be PICKED UP AND MOVED to satisfy this "
             "goal? List only the items to carry -- NOT the furniture/surface they "
             "currently sit on, and NOT the source location. If the goal names a SOURCE "
             "(e.g. 'from the balcony', 'on the kitchen table'), the target MUST be an "
             "object CURRENTLY located there in the scene above -- pick the instance in "
             "that room/on that surface, not a same-category object elsewhere. Then give "
             "the SINGLE destination: a room name (to set on its floor), or an object "
             "name (to place them ON TOP OF). Use the exact names in [brackets]. If it is "
             "not a move/place task, return empty targets.")
        msgs = [{"role": "system",
                 "content": "You map a natural-language household task to the concrete "
                            "objects and destination present in the scene. Use exact names."},
                {"role": "user", "content": q}]
        try:
            ans = self.llm.chat_json(msgs, self._GROUND_SCHEMA)
        except Exception:
            return [], None
        destination = self._resolve_destination(env, ans.get("destination", ""))
        return self._carryable_targets(env, ans.get("targets", []), destination), destination

    def _carryable_targets(self, env, names, destination):
        """Keep only the proposed targets the robot could actually pick up and carry.

        The model reliably names the right object but often *also* names the furniture
        it is sitting on -- "the lamp from the balcony table" comes back as
        ``[table_lamp_0002, table_0001]`` despite the prompt saying not to. One bogus
        target poisons the whole run: :meth:`_targets_satisfied` requires *every*
        target to reach the destination, so an un-carryable one makes the goal
        permanently unsatisfiable, the deterministic completion check can never fire,
        and the task can only end by exhausting the step budget.

        The load-bearing rule is "is this thing currently holding other things" -- that
        is what identifies the surface the real target rests on. Size is only a
        backstop for bare furniture (beds, sofas, curtains); it cannot do the main job,
        because chairs (up to 1.15 m here) and side tables (up to 1.17 m) overlap.
        """
        smap = env.smap
        kept, seen = [], set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            o = smap.get(name)
            if o is None:
                why = "not in the map"
            elif name == destination:
                why = "it is the destination, not cargo"
            elif o.supports:
                why = f"it is a surface holding {len(o.supports)} other object(s)"
            elif o.max_dim > MAX_CARRY_DIM:
                why = f"too big to carry ({o.max_dim:.2f} m > {MAX_CARRY_DIM} m)"
            else:
                kept.append(name)
                continue
            self._log(f"[grounding] dropped target {name!r}: {why}")
        return kept

    def _resolve_destination(self, env, dest: str):
        """Normalize a destination string to a known room name or object name."""
        if not dest:
            return None
        if dest in env.smap.room_names() or env.smap.get(dest) is not None:
            return dest
        cand = dest.lower().replace(" ", "").replace("_", "")
        for r in env.smap.room_names():
            if r.replace("_", "") == cand:
                return r
        return dest

    def _targets_satisfied(self, env, targets, destination):
        """Deterministic completion: are all target objects at the destination (in the
        room, or resting ON the destination object)? Returns True/False, or None when
        it can't be evaluated (unknown targets/destination) so the caller falls back to
        the LLM judge."""
        if not targets or not destination:
            return None
        dest_is_room = destination in env.smap.room_names()
        dest_is_obj = env.smap.get(destination) is not None
        if not (dest_is_room or dest_is_obj):
            return None
        for t in targets:
            o = env.smap.get(t)
            if o is None:
                return None
            if dest_is_room and o.room != destination:
                return False
            if dest_is_obj and o.supported_by != destination:
                return False
        return True

    def run(self, env, goal: str) -> dict:
        """Execute ``goal`` against ``env`` (a RobotSkills/MockSkills). Returns
        ``{"success": bool, "reason": str, "steps": [Step, ...]}``."""
        transcript: list[str] = []
        steps: list[Step] = []
        last_sig = None           # signature of the previous action
        repeats = 0               # identical actions emitted in a row (ok OR fail)
        self._log(f"\n=== PLAN GOAL: {goal} ===")

        # Ground the goal to concrete target(s) + destination up front (see
        # _ground_goal). Gives the actor a specific instance to move and enables a
        # deterministic completion check.
        targets, destination = self._ground_goal(env, goal)
        # Guard against degenerate grounding: if the "targets" are ALREADY at the
        # destination before we do anything, grounding mis-resolved (e.g. picked a
        # same-category object already in the destination room). Discard it and fall
        # back to LLM-judged completion so we don't declare a no-op success.
        if self._targets_satisfied(env, targets, destination) is True:
            self._log(f"[grounding] discarded (targets already at {destination!r} "
                      f"before acting): {targets}")
            targets = []
        self._log(f"[grounding] targets={targets} destination={destination!r}")
        plan_hint = ""
        if targets and destination:
            plan_hint = (f"RESOLVED TASK: move exactly [{', '.join(targets)}] to "
                         f"'{destination}'. Refer to the target by that exact name; do "
                         f"not move any other object.\n\n")

        for i in range(self.max_steps):
            user = (f"GOAL: {goal}\n\n" + plan_hint +
                    f"CURRENT STATE:\n{build_state_text(env)}\n\n"
                    f"HISTORY (skill -> result), oldest first:\n"
                    + ("\n".join(transcript) if transcript else "(nothing yet)")
                    + "\n\nChoose the next single action.")
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user}]
            action = self.llm.chat_json(messages, RESPONSE_SCHEMA)
            skill = action.get("skill", "")
            args = action.get("args") or {}
            thought = action.get("thought", "")
            self._log(f"\n[{i+1}] think: {thought}\n    -> {skill}({args})")
            self._notify(i + 1, skill, args, thought)      # starting

            if skill == "finish":
                success = bool(args.get("success", True))
                reason = args.get("reason", "")
                self._log(f"    finish(success={success}): {reason}")
                return {"success": success, "reason": reason, "steps": steps}

            ok, err = ground(env, skill, args)
            if not ok:
                res = SkillResult(False, skill, err)  # synthetic failure, fed back
                self._log(f"    grounding: {err}")
            else:
                res = _execute(env, skill, args)
            self._log(f"    {res}")
            self._notify(i + 1, skill, args, thought, res)   # finished
            steps.append(Step(skill, args, thought, res))

            # Loop-breaker: if the model keeps emitting the SAME action -- even a
            # *successful* one that doesn't advance the goal (e.g. goto_object to a
            # thing it's already at) -- nudge it, then bail rather than burn the budget.
            sig = f"{skill}:{sorted(args.items())}"
            repeats = repeats + 1 if sig == last_sig else 0
            last_sig = sig

            # Completion: prefer the deterministic check against the grounded targets;
            # fall back to the LLM judge (after a place) only when the goal couldn't be
            # grounded to concrete targets.
            sat = self._targets_satisfied(env, targets, destination)
            if sat is True:
                self._log(f"    done: all targets at {destination}")
                return {"success": True,
                        "reason": f"all targets {targets} are at '{destination}'",
                        "steps": steps}
            if sat is None and skill == "place" and res.ok:
                done, why = self._check_done(env, goal)
                self._log(f"    done-check: {done} ({why})")
                if done:
                    return {"success": True, "reason": why or "goal satisfied",
                            "steps": steps}

            note = ""
            if skill == "place" and res.ok:
                note = ("  <-- Item is now placed and your hands are empty. If the GOAL is "
                        "satisfied, call finish(success=true) NOW; do not move other items.")
            elif repeats >= 1:
                note = ("  <-- YOU ALREADY DID THIS EXACT ACTION and it did not advance "
                        "the goal. Do something DIFFERENT: if you are holding the item, "
                        "goto the destination room and place() it; else finish(success=false).")
            transcript.append(
                f"{skill}({args}) -> {'OK' if res.ok else 'FAIL'}: {res.detail}{note}")

            if repeats >= 2:   # 3 identical actions in a row -> stuck, give up
                self._log("    (aborting: same action repeated 3x with no progress)")
                return {"success": False,
                        "reason": f"stuck repeating an action with no progress: {skill}({args})",
                        "steps": steps}

        self._log("    (step budget exhausted)")
        return {"success": False, "reason": f"did not finish within {self.max_steps} steps",
                "steps": steps}
