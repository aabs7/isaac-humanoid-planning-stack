"""Build the apartment's semantic map from its USD and save it to JSON.

Standalone: parses the USD directly (pxr only), so it needs NO running sim and
finishes in well under a second. This is the Phase-0 stand-in for on-robot
perception -- it turns ground-truth USD geometry into the "what is where" the
skill layer and planner query.

    cd ~/isaac && source .venv/bin/activate
    python isaac_task_planning/build_semantic_map.py
    python isaac_task_planning/build_semantic_map.py --room kitchen   # list one room
    python isaac_task_planning/build_semantic_map.py --find cup       # locate a category
"""

import argparse

from g1sim.semantic_map import SemanticMap, DEFAULT_USD, DEFAULT_ROOMS_JSON

DEFAULT_OUT = "/home/abhish/isaac/isaac_task_planning/sensor_output/semantic_map.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usd", default=DEFAULT_USD)
    ap.add_argument("--rooms-json", default=DEFAULT_ROOMS_JSON)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--room", help="Print a detailed object listing for one room.")
    ap.add_argument("--find", metavar="CATEGORY",
                    help="Print every object of this category with its world position.")
    ap.add_argument("--graph", nargs="?", const="__all__", metavar="ROOM",
                    help="Print the scene-graph tree (optionally for one ROOM).")
    args = ap.parse_args()

    smap = SemanticMap.build(args.usd, args.rooms_json)
    smap.save(args.out)
    print(smap.describe())
    print(f"\n[semantic_map] saved -> {args.out}")

    if args.graph:
        print()
        print(smap.describe_graph(None if args.graph == "__all__" else args.graph))

    if args.room:
        print(f"\n--- {args.room} ---")
        for o in sorted(smap.objects_in_room(args.room), key=lambda o: o.category):
            sz = o.size
            print(f"  {o.name:26s} @ ({o.position[0]:6.2f}, {o.position[1]:6.2f}, "
                  f"{o.position[2]:5.2f})  size=({sz[0]:.2f},{sz[1]:.2f},{sz[2]:.2f})")

    if args.find:
        print(f"\n--- '{args.find}' ---")
        hits = smap.find(args.find)
        if not hits:
            print("  (none found)")
        for o in hits:
            print(f"  {o.name:26s} in {o.room:12s} @ ({o.position[0]:6.2f}, "
                  f"{o.position[1]:6.2f}, {o.position[2]:5.2f})")


if __name__ == "__main__":
    main()
