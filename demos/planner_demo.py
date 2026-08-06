import argparse
import sys
from pathlib import Path

# This script lives in a subdirectory, so the repo root (which holds g1sim/) is not
# on sys.path when it is run directly. Put it there.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from g1sim.planner import ground, Planner
from llm_usage_demo import build_env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-json", default=None, help="Load a prebuilt scene graph JSON.")
    args = parser.parse_args()

    env = build_env(args)

    # Grounding examples
    ## There are 5 cups in the apartment (cup_0000, cup_0001, cup_0002, cup_0003, cup_0004) so grounding says (True, None)
    grounding_result = ground(env, "goto_object", {"object": "cup"})
    print(f'Grounding call for cup: {grounding_result}')

    ## There is no towel in the apartment, so grounding says (False, "No object named 'towel' found. .... ")
    grounding_result = ground(env, "goto_object", {"object": "towel"})
    print(f'Grounding call for towel: {grounding_result}')

    ## There is no garage in the apartment, so grounding says (False, "No room named 'garage' found. .... ")
    grounding_result = ground(env, "goto_room", {"room": "garage"})
    print(f'Grounding call for garage: {grounding_result}')

    ## When goto_object is called for kitchen (which is a room), the grounding says (False, "'kitchen' is a room not an object.  instead to go to kitchen ...... ")
    grounding_result = ground(env, "goto_object", {"object": "kitchen"})
    print(f'Grounding call for goto_object kitchen: {grounding_result}')

    grounded_goal = Planner()._ground_goal(env=env, goal="bring the cup from the livingroom to the kitchen")
    print(f'\nGrounded goal: {grounded_goal}')

    # For other usages, refer to tests in tests/planner/
