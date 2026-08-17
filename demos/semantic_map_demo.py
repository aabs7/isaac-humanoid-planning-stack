import _bootstrap

import argparse
from g1sim.task.llm_based.llm import DEFAULT_MODEL, OllamaChat
from g1sim.perception.semantic_map import SemanticMap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, default='Bring cup from the kitchen counter to the dining table.', help="Plain-English goal to translate and plan for.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model to use for translation. Default is qwen2.5:7b-instruct.")
    parser.add_argument("--map-json", default=None, help="Load a prebuilt scene graph JSON.")
    args = parser.parse_args()

    smap = SemanticMap.load(args.map_json) if args.map_json else SemanticMap.build()


    # Describe the semantic map
    print(smap.describe(), "\n\r")

    # Describe graph
    print(smap.describe_graph(without_nav=True))
