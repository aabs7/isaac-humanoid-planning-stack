"""Measure whether the G1 can actually reach the things it is asked to pick up.

Read-only. This script commands no joints and moves no objects; it walks the robot to a
target exactly as the planner would and then *measures* the geometry nobody in this stack
has ever looked at.

The question it answers. ``PICK_RADIUS`` (1.0 m) governs how close the base parks to an
object's **footprint**, and the comment above it admits the number was forced by A*'s
robot-radius inflation rather than by anything about an arm. But an arm hangs off a
*shoulder*, which sits above and behind the pelvis, so what decides "can I grasp this" is
the **shoulder-to-object** distance in 3D. Nothing in the repo has ever read a link pose --
no ``body_link_pos_w``, no jacobian -- so that distance has been asserted by a constant and
never measured. If it comes back far outside a ~0.7 m arm, the fix is where the robot
*stands*, not which IK solver we pick, and building the arm controller first would produce
something that works perfectly and cannot reach anything.

It also prints the full body list, which is how we learn whether the fingertips are separate
rigid bodies -- contact sensing needs one sensor per finger link and silently reads zeros if
the link names are wrong.

    cd ~/isaac && source .venv/bin/activate
    python isaac_task_planning/scripts/reach_probe_g1_apartment.py \
        --map-json isaac_task_planning/sensor_output/semantic_map.json --headless
    # a different set of targets, and what a 0.35 m standoff would buy:
    python isaac_task_planning/scripts/reach_probe_g1_apartment.py \
        --targets cup_0001,cup_0002 --headless
"""

import math
import statistics

# Make g1sim importable when this file is run directly (see scripts/_bootstrap.py).
import _bootstrap  # noqa: F401

from g1sim.sim.launch import make_parser, launch

parser = make_parser("Measure the G1's real reach to candidate grasp targets.")
parser.add_argument("--targets", default="cup_0005,cup_0000",
                    help="Comma-separated semantic-map object names to probe. The default "
                         "pair are both in the kitchen, so the walk between them is short.")
parser.add_argument("--map-json", default=None,
                    help="Load a prebuilt scene graph JSON instead of parsing the USD.")
parser.add_argument("--arm-reach", type=float, default=0.70,
                    help="Nominal shoulder-to-hand reach (m) to judge targets against. The "
                         "G1's arm is roughly 0.60-0.70 m.")
parser.add_argument("--sway-ticks", type=int, default=100,
                    help="Control ticks to stand still while measuring base sway.")
args = parser.parse_args()

simulation_app = launch(args)

# ---- imports that need the running sim app ----
from g1sim.sim.scene import build_world, NAV_LIDAR_TARGETS
from g1sim.sim.locomotion import G1LocomotionPolicy
from g1sim.perception.semantic_map import SemanticMap
from g1sim.skills.robot import RobotSkills
from g1sim.skills.types import PICK_RADIUS
from g1sim.task.symbolic.skills import SURFACE_REACH

# Body-name patterns worth resolving. Printed with their indices so later stages (IK end
# effector, contact sensors) can use names that are known to exist rather than guessed.
PROBE_BODIES = [".*shoulder.*", ".*wrist_yaw_link", ".*hand_palm_link",
                ".*hand_(index|middle)_1_link", ".*hand_thumb_2_link", "torso_link", "pelvis"]


def aabb_distance(point, bbox_min, bbox_max) -> float:
    """3D distance from a point to an axis-aligned box (0 if inside).

    The 3D sibling of ``SemanticObject.xy_dist``. Reach has to be judged in 3D: a cup on a
    0.9 m counter is nearly overhead at close range, and the XY distance flatters it.
    """
    d = [max(bbox_min[i] - point[i], 0.0, point[i] - bbox_max[i]) for i in range(3)]
    return math.sqrt(sum(v * v for v in d))


def link_pos(robot, index):
    return [float(v) for v in robot.data.body_link_pos_w.torch[0, index, :3]]


def measure(skills, robot, bodies, o, arm_reach):
    """Everything worth knowing about reaching ``o`` from where the robot is standing."""
    px, py, heading = skills.pose()
    pelvis = link_pos(robot, bodies["pelvis"])
    rows = []
    for side in ("left", "right"):
        shoulder = link_pos(robot, bodies[f"{side}_shoulder"])
        wrist = link_pos(robot, bodies[f"{side}_wrist"])
        rows.append({
            "side": side,
            "shoulder": shoulder,
            "wrist": wrist,
            # To the object's *surface*, not its centre -- a hand grasps the near face.
            "to_bbox": aabb_distance(shoulder, o.bbox_min, o.bbox_max),
            "to_centre": math.dist(shoulder, list(o.position)),
            "wrist_to_bbox": aabb_distance(wrist, o.bbox_min, o.bbox_max),
        })
    best = min(rows, key=lambda r: r["to_bbox"])
    return {
        "object": o.name,
        "base_xy": (px, py),
        "heading": heading,
        "pelvis_z": pelvis[2],
        "footprint_dist": o.xy_dist(px, py),      # what PICK_RADIUS actually governs
        "object_top_z": o.top_z,
        "height_above_shoulder": o.top_z - best["shoulder"][2],
        "best_side": best["side"],
        "shoulder_to_bbox": best["to_bbox"],
        "shoulder_to_centre": best["to_centre"],
        "wrist_to_bbox": best["wrist_to_bbox"],
        "in_reach": best["to_bbox"] <= arm_reach,
        "sides": rows,
    }


def report(m, arm_reach):
    verdict = "REACHABLE" if m["in_reach"] else "OUT OF REACH"
    print(f"\n  {m['object']}  [{verdict}]")
    print(f"    base at ({m['base_xy'][0]:.2f}, {m['base_xy'][1]:.2f}), "
          f"pelvis z={m['pelvis_z']:.2f} m, heading {math.degrees(m['heading']):+.0f} deg")
    print(f"    footprint distance (what PICK_RADIUS governs) : {m['footprint_dist']:.3f} m "
          f"(limit {PICK_RADIUS:.2f})")
    print(f"    shoulder -> object bbox  ({m['best_side']} arm)  : {m['shoulder_to_bbox']:.3f} m "
          f"(arm {arm_reach:.2f})")
    print(f"    shoulder -> object centre                     : {m['shoulder_to_centre']:.3f} m")
    print(f"    wrist    -> object bbox (arm at rest)         : {m['wrist_to_bbox']:.3f} m")
    print(f"    object top {m['object_top_z']:.2f} m, "
          f"{m['height_above_shoulder']:+.2f} m relative to the shoulder")
    if not m["in_reach"]:
        print(f"    -> {m['shoulder_to_bbox'] - arm_reach:.3f} m too far. No IK solver fixes "
              f"this; the base has to stand closer.")


def main():
    smap = SemanticMap.load(args.map_json) if args.map_json else SemanticMap.build()
    print(smap.describe())

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    objects = []
    for name in targets:
        o = smap.get(name)
        if o is None:
            print(f"[probe] no object '{name}' in the map -- skipping")
            continue
        objects.append(o)
    if not objects:
        raise SystemExit("[probe] no valid targets")

    sim, scene = build_world(args.spawn, device=args.device, sensors="lidar",
                             lidar_targets=NAV_LIDAR_TARGETS)
    controller = G1LocomotionPolicy(scene["robot"], sim.device)
    robot = scene["robot"]
    skills = RobotSkills(sim, scene, controller, smap, app=simulation_app)

    # ---- what links exist -------------------------------------------------
    print(f"\n=== BODIES ({robot.num_bodies}) ===")
    print("  " + ", ".join(robot.body_names))
    bodies = {}
    for pattern in PROBE_BODIES:
        ids, names = robot.find_bodies(pattern, preserve_order=False)
        print(f"\n  {pattern}")
        for i, n in zip(ids, names):
            print(f"    [{i:2d}] {n}")
    for side in ("left", "right"):
        bodies[f"{side}_shoulder"] = robot.find_bodies(f"{side}_shoulder_pitch_link")[0][0]
        bodies[f"{side}_wrist"] = robot.find_bodies(f"{side}_wrist_yaw_link")[0][0]
        bodies[f"{side}_palm"] = robot.find_bodies(f"{side}_hand_palm_link")[0][0]
    bodies["pelvis"] = robot.find_bodies("pelvis")[0][0]

    skills.idle(0.5)

    # ---- base sway, standing still ---------------------------------------
    xs, ys = [], []
    for _ in range(args.sway_ticks):
        if not skills._running():
            break
        skills.step(controller.command())
        x, y, _ = skills.pose()
        xs.append(x)
        ys.append(y)
    if len(xs) > 2:
        print(f"\n=== BASE SWAY over {len(xs)} ticks standing still ===")
        print(f"  stdev  x {statistics.stdev(xs) * 1000:.1f} mm   "
              f"y {statistics.stdev(ys) * 1000:.1f} mm")
        print(f"  range  x {(max(xs) - min(xs)) * 1000:.1f} mm   "
              f"y {(max(ys) - min(ys)) * 1000:.1f} mm")
        print("  (the hand inherits this; it is added to any IK tracking error)")

    # ---- per target -------------------------------------------------------
    results = []
    for o in objects:
        support = smap.support_of(o.name)
        print(f"\n=== TARGET {o.name} in {o.room}"
              f"{f' (on {support.name})' if support else ''} ===")
        print(f"  size {o.size[0]:.3f} x {o.size[1]:.3f} x {o.size[2]:.3f} m")

        # 1. Stand where the planner would: at the supporting surface.
        if support is not None:
            res = skills.goto_object(support.name, reach=SURFACE_REACH)
            print(f"  [walk to {support.name}] {res}")
            if skills._running():
                m = measure(skills, robot, bodies, o, args.arm_reach)
                m["stance"] = f"at {support.name}"
                report(m, args.arm_reach)
                results.append(m)

        # 2. Then close in on the object itself, as SkillBridge._pick does.
        if not skills._running():
            break
        res = skills.goto_object(o.name, reach=PICK_RADIUS)
        print(f"  [close in on {o.name}] {res}")
        m = measure(skills, robot, bodies, o, args.arm_reach)
        m["stance"] = "closed in on the object"
        report(m, args.arm_reach)
        results.append(m)

    # ---- verdict ----------------------------------------------------------
    print(f"\n=== SUMMARY (arm reach {args.arm_reach:.2f} m) ===")
    print(f"  {'object':<14} {'stance':<28} {'footprint':>10} {'shoulder':>10}  verdict")
    for m in results:
        print(f"  {m['object']:<14} {m['stance']:<28} {m['footprint_dist']:>9.3f}m "
              f"{m['shoulder_to_bbox']:>9.3f}m  "
              f"{'reachable' if m['in_reach'] else 'OUT OF REACH'}")
    reachable = sum(1 for m in results if m["in_reach"])
    print(f"\n  {reachable}/{len(results)} stances put the target inside the arm.")
    if reachable == 0:
        print("  Every stance is out of reach: the standoff, not the solver, is the problem.")


if __name__ == "__main__":
    main()
    simulation_app.close()
