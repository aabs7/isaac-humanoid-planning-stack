import _bootstrap

'''Railroad planner uses fluents as goal. This demo shows how to translate a plain-English goal into fluents, then plan and act on it.'''

import argparse
from g1sim.task.llm_based.llm import DEFAULT_MODEL, OllamaChat
from g1sim.perception.semantic_map import SemanticMap

from g1sim.task.symbolic.goals import translate, SemanticDomain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, default='Bring cup from the kitchen counter to the dining table.', help="Plain-English goal to translate and plan for.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model to use for translation. Default is qwen2.5:7b-instruct.")
    parser.add_argument("--map-json", default=None, help="Load a prebuilt scene graph JSON.")
    args = parser.parse_args()

    # LLM for translating plain-English goals into fluents and the map
    llm = OllamaChat(model=args.model)
    smap = SemanticMap.load(args.map_json) if args.map_json else SemanticMap.build()


    # You can compute goal fluents using the translate() function, which uses the LLM to translate a plain-English goal into fluents.
    task = translate(llm, smap, args.goal, verbose=False)
    print(f"Goal fluents for task: {task.task}")
    print(task)
    print("------------------")

    ## You can use SemanticDomain.build(smap) to build the domain for the scene represented in semantic map. Domain can be used to compute initial fluents, and get observations too.
    domain = SemanticDomain.build(smap)

    # All the task-relevant objects that you can pick in the domain
    print(f'{domain.objects=}')
    print("-------------------")

    # Define the initial position of the robot
    robot_xy = (1, 1)
    initial_fluents = domain.initial_fluents(smap, robot_xy)
    print(f'{initial_fluents=}')
    print("------------------")


    # Since the complete domain might be too large for the planner, we can build a partial domain that only includes the objects and locations relevant to the task.
    partial_domain_for_planning = SemanticDomain.for_task(smap, sorted(task.objects), sorted(task.locations))
    print(f'{partial_domain_for_planning.objects=}')
    initial_fluents = partial_domain_for_planning.initial_fluents(smap, robot_xy)
    print(f'{initial_fluents=}')
    print("------------------")
