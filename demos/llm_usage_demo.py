import argparse
import sys
from pathlib import Path

# This script lives in a subdirectory, so the repo root (which holds g1sim/) is not
# on sys.path when it is run directly. Put it there.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from g1sim.task.llm_based.llm import OllamaChat
from g1sim.task.llm_based.planner import Planner, SKILLS, build_state_text, SYSTEM_PROMPT
from g1sim.perception.semantic_map import SemanticMap
from g1sim.skills.mock import MockSkills


def build_env(args):
    """Build the environment for the demo."""
    smap = SemanticMap.load(args.map_json) if args.map_json else SemanticMap.build()
    start = (0.0, 0.0)
    return MockSkills(smap, start_xy=start, verbose=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-json", default=None, help="Load a prebuilt scene graph JSON.")
    parser.add_argument("--goal", default="Bring cup from the kitchen to the living room.", help="The goal of the task.")
    args = parser.parse_args()

    env = build_env(args)
    llm = OllamaChat()

    # 1. Action call: ask the LLM for the next action based on the current state and goal.
    _RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "skill": {"type": "string", "enum": SKILLS},
            "args": {"type": "object"},
        },
        "required": ["thought", "skill", "args"],
    }
    user = (f"GOAL: {args.goal}\n\n"
            f"CURRENT STATE:\n{build_state_text(env)}\n\n"
            f"HISTORY (skill -> result), oldest first:\n"
            + ("\n""(nothing yet)")
            + "\n\nChoose the next single action.")

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    action = llm.chat_json(messages, _RESPONSE_SCHEMA)
    print(f'Action call: {action}')

    # 2. Grounding call: ask the LLM to ground the action's targets in the environment.
    _GROUND_SCHEMA = {
        "type": "object",
        "properties": {
            "targets": {"type": "array", "items": {"type": "string"}},
            "destination": {"type": "string"},
        },
        "required": ["targets", "destination"],
    }
    user = (f"GOAL: {args.goal}\n\nSCENE (objects shown as category [exact_name]):\n"
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
    messages = [{"role": "system",
                "content": "You map a natural-language household task to the concrete "
                        "objects and destination present in the scene. Use exact names."},
            {"role": "user", "content": user}]
    ans = llm.chat_json(messages, _GROUND_SCHEMA)
    print(f'Grounding call: {ans}')

    # 3. Done call: ask the LLM if the goal is satisfied in the current state.
    _DONE_SCHEMA = {
        "type": "object",
        "properties": {"done": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["done", "reason"],
    }
    user = (f"GOAL: {args.goal}\n\nCURRENT STATE:\n{build_state_text(env)}\n\n"
            "Considering ONLY the current state above, is the GOAL now fully "
            "satisfied? An object that started elsewhere and is now in the requested "
            "place counts as done, even if other similar objects exist elsewhere.")
    messages = [{"role": "system",
                "content": "You judge whether a household-robot task goal is already "
                        "satisfied by the current scene. Answer strictly from the state."},
            {"role": "user", "content": user}]
    ans = llm.chat_json(messages, _DONE_SCHEMA)
    print(f'Done call: {ans}')
