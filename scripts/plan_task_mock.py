"""Run the LLM planner against the sim-free MockSkills -- fast prompt/iteration loop.

No Isaac, no GPU sim: builds the semantic map from the USD, stands up a virtual robot
(MockSkills), and drives a natural-language goal through the planner. Use this to
develop the planner and its prompts in seconds, then run the identical planner in the
real sim with scripts/plan_task_g1_apartment.py.

    cd ~/isaac && source .venv/bin/activate
    ollama pull qwen2.5:7b-instruct          # one-time
    python isaac_task_planning/scripts/plan_task_mock.py \
        --goal "bring the chair from the balcony to the living room"

Handy checks:
    --goal "put the table lamp on the floor of the kitchen"
    --goal "pick up the banana"              # impossible -> graceful finish(false)
"""

import argparse

# Make g1sim importable when this file is run directly (see scripts/_bootstrap.py).
import _bootstrap  # noqa: F401

from g1sim.perception.semantic_map import SemanticMap
from g1sim.skills.mock import MockSkills
from g1sim.task.llm import OllamaChat, DEFAULT_MODEL
from g1sim.task.planner import Planner


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal", required=True, help="Natural-language task goal.")
    ap.add_argument("--map-json", default=None,
                    help="Load a prebuilt scene graph JSON instead of parsing the USD.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag.")
    ap.add_argument("--start", default="0 0", help="Robot start 'X Y'. Default: 0 0.")
    ap.add_argument("--max-steps", type=int, default=15)
    args = ap.parse_args()

    smap = SemanticMap.load(args.map_json) if args.map_json else SemanticMap.build()
    print(smap.describe())

    llm = OllamaChat(model=args.model)
    if not llm.available():
        print(f"\n[warn] Ollama model '{args.model}' not reachable/pulled at "
              f"{llm.base_url}. Run `ollama serve` and `ollama pull {args.model}`.")

    sx, sy = (float(v) for v in args.start.split())
    env = MockSkills(smap, start_xy=(sx, sy))

    planner = Planner(llm, max_steps=args.max_steps)
    outcome = planner.run(env, args.goal)

    print("\n=== RESULT ===")
    print(f"success={outcome['success']}  reason={outcome['reason']}")
    print(f"steps={len(outcome['steps'])}")
    if env.held is not None:
        print(f"still holding: {env.held.name}")
    print("\nFinal object rooms (movable items):")
    for s in outcome["steps"]:
        if s.skill == "place" and s.result.ok:
            o = smap.get(s.result.data["object"])
            print(f"  {o.name}: now in {o.room} @ ({o.xy[0]:.2f}, {o.xy[1]:.2f})")


if __name__ == "__main__":
    main()
