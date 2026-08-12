"""Walk the apartment once, build an occupancy map from the robot's own lidar, and save it.

The small, honest version of "explore a new environment first". It does not do frontier
exploration -- it visits each room's navigable point in turn, mapping the whole way, because
``goto`` already fuses lidar every third control tick. What comes out is a map built entirely
from the robot's sensors, with no USD geometry read anywhere, which is the property that
makes it worth having: the same artefact a real robot would produce, produced by the code
that would produce it.

Why bother saving one. Several questions need free space and do not need a simulator --
"which objects could this robot actually stand close enough to grasp?" being the immediate
one. Answering those against ground-truth USD geometry is a shortcut that will not exist on
hardware; answering them against a persisted sensor map is the same computation the real
robot would do. One lap, then analyse offline as often as you like.

    cd ~/isaac && source .venv/bin/activate
    python isaac_task_planning/scripts/build_occupancy_map.py --headless
    # then, with no simulator at all:
    python isaac_task_planning/scripts/reachability_report.py

What it is not: complete. The lidar sees what it sees from the floor path the robot walks,
so cabinet interiors, surfaces above the sensor band and anything behind a closed door are
simply absent. Treat the result as a prior, not as truth.
"""

import os

# Make g1sim importable when this file is run directly (see scripts/_bootstrap.py).
import _bootstrap  # noqa: F401

from g1sim.sim.launch import make_parser, launch

parser = make_parser("Map the apartment with the robot's lidar and save the grid.")
parser.add_argument("--map-json", default=None,
                    help="Load a prebuilt scene graph JSON instead of parsing the USD. Used "
                         "only to know which rooms exist and where their nav points are.")
parser.add_argument("--rooms", default=None,
                    help="Comma-separated room order. Default: every room in the map.")
parser.add_argument("--out", default="sensor_output/occupancy_map.npz",
                    help="Where to save the grid.")
parser.add_argument("--settle", type=float, default=1.0,
                    help="Seconds to stand and map on arrival in each room.")
args = parser.parse_args()

simulation_app = launch(args)

# ---- imports that need the running sim app ----
from g1sim.sim.scene import build_world, NAV_LIDAR_TARGETS
from g1sim.sim.locomotion import G1LocomotionPolicy
from g1sim.perception.semantic_map import SemanticMap
from g1sim.skills.robot import RobotSkills


def main():
    smap = SemanticMap.load(args.map_json) if args.map_json else SemanticMap.build()
    print(smap.describe())

    rooms = ([r.strip() for r in args.rooms.split(",") if r.strip()]
             if args.rooms else smap.room_names())

    sim, scene = build_world(args.spawn, device=args.device, sensors="lidar",
                             lidar_targets=NAV_LIDAR_TARGETS)
    controller = G1LocomotionPolicy(scene["robot"], sim.device)
    skills = RobotSkills(sim, scene, controller, smap, app=simulation_app)
    if skills.mapper is None:
        raise SystemExit("[map] no lidar mapper -- this scene has no lidar sensor")

    skills.idle(0.5)
    print(f"\n=== MAPPING LAP over {len(rooms)} rooms: {', '.join(rooms)} ===")

    for room in rooms:
        if not skills._running():
            break
        res = skills.goto_room(room)
        # A failed goto still mapped everything it walked past, so this is worth noting
        # and not worth aborting for.
        print(f"  [{room}] {res}")
        skills.idle(args.settle)
        print(f"      {skills.mapper.describe()}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    skills.mapper.save(args.out)
    print(f"\n[map] {skills.mapper.describe()}")
    print(f"[map] saved -> {os.path.abspath(args.out)}")

    # The map, not the planner's view of it: no inflation, no path. Inflation belongs to
    # whoever is planning, at the moment they plan, for the body they have.
    skills.mapper.save_png(os.path.splitext(args.out)[0] + ".png")


if __name__ == "__main__":
    main()
    simulation_app.close()
