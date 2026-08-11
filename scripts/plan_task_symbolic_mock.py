"""Run the symbolic planner against the sim-free MockSkills -- fast iteration loop.

No Isaac, no GPU: builds the semantic map, stands up a virtual robot (MockSkills), asks a
local LLM to turn the English goal into fluents, and drives it through railroad's MCTS
planner. Use this to develop the domain, operators, cost model and goal prompt in seconds,
then run the identical planner in the real sim with
``scripts/plan_task_symbolic_g1_apartment.py``.

    cd ~/isaac && source .venv/bin/activate
    ollama pull qwen2.5:7b-instruct          # one-time
    python isaac_task_planning/scripts/plan_task_symbolic_mock.py --list
    python isaac_task_planning/scripts/plan_task_symbolic_mock.py \
        --goal "put the kitchen cup on the living room table"

Handy checks:
    --goal "bring me a cup"                  # no destination -> a 'holding' goal
    --goal "put the cup on the bedroom floor" # a room is a valid destination
    --goal "fetch the unicorn"               # rejected, with the valid categories
    --dry-run                                # translate and stop: the fastest prompt loop
    --all-objects                            # the full ~10700-action domain
"""

import argparse

# Make g1sim importable when this file is run directly (see scripts/_bootstrap.py).
import _bootstrap  # noqa: F401

from g1sim.perception.semantic_map import SemanticMap
from g1sim.skills.mock import MockSkills
from g1sim.task.llm_based.llm import DEFAULT_MODEL, OllamaChat
from g1sim.task.symbolic import (G1Environment, GoalTranslationError, SemanticDomain,
                                 solve, translate)
from g1sim.task.symbolic.planner import MCTS_ITERATIONS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal", help="The task in plain English.")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Translate the goal, print it, and exit without planning.")
    ap.add_argument("--list", action="store_true",
                    help="Print the domain's symbols and exit.")
    ap.add_argument("--map-json", default=None,
                    help="Load a prebuilt scene graph JSON instead of parsing the USD.")
    ap.add_argument("--all-objects", action="store_true",
                    help="Plan over the whole apartment instead of just the task's objects "
                         "and locations. Slower and less reliable; diagnostic only.")
    ap.add_argument("--start", default="7.93 -0.39",
                    help="Robot start 'X Y'. Default: the livingroom nav point.")
    ap.add_argument("--max-dispatches", type=int, default=15)
    ap.add_argument("--mcts-iterations", type=int, default=MCTS_ITERATIONS)
    args = ap.parse_args()

    smap = SemanticMap.load(args.map_json) if args.map_json else SemanticMap.build()
    print(smap.describe())

    # The LLM chooses from the unrestricted domain; the search domain is narrowed after.
    full_domain = SemanticDomain.build(smap)

    if args.list:
        print(full_domain.describe())
        print("\nrooms:     " + ", ".join(sorted(full_domain.rooms)))
        print("\nsurfaces:  " + ", ".join(sorted(full_domain.surfaces)))
        print("\nobjects:   " + ", ".join(sorted(full_domain.objects)))
        return

    if not args.goal:
        ap.error("--goal is required (or use --list to see what is available)")

    llm = OllamaChat(model=args.model)
    if not llm.available():
        raise SystemExit(f"\n[error] Ollama model '{args.model}' not reachable at "
                         f"{llm.base_url}. Run `ollama serve` and `ollama pull {args.model}`.")
    try:
        task = translate(llm, smap, args.goal, domain=full_domain)
    except GoalTranslationError as e:
        raise SystemExit(f"\n[error] {e}")

    if args.dry_run:
        print(f"\n=== GOAL (dry run) ===\n{task}")
        return

    # Only the task's objects and locations -- see SemanticDomain.for_task for why the
    # location restriction is what makes search reliable, not merely fast.
    domain = (SemanticDomain.build(smap) if args.all_objects else
              SemanticDomain.for_task(smap, sorted(task.objects), sorted(task.locations)))
    print(domain.describe())

    sx, sy = (float(v) for v in args.start.split())
    skills = MockSkills(smap, start_xy=(sx, sy), verbose=False)
    env = G1Environment(skills, domain)

    print(f"\n=== TASK: {args.goal}\n=== GOAL: {task}\n")
    print(f"grounded {len(env.get_actions())} actions")

    outcome = solve(env, task.goal, max_dispatches=args.max_dispatches,
                    mcts_iterations=args.mcts_iterations)

    print("\n=== RESULT ===")
    if skills.held is not None:
        print(f"still holding: {skills.held.name}")
    for name in sorted(task.objects):
        o = smap.get(name)
        if o is not None and not o.held:
            print(f"{o.name}: on {o.supported_by or '(floor)'} in {o.room} "
                  f"@ ({o.xy[0]:.2f}, {o.xy[1]:.2f})")
    raise SystemExit(0 if outcome.ok else 1)


if __name__ == "__main__":
    main()
