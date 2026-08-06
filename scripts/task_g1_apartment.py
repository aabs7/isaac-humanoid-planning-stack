"""One end-to-end round-trip task for the G1, driven entirely through the skill API.

This is the Phase-0 integration test: prove the full loop
    scan -> goto ROOM -> goto object -> (magic) pick -> goto BACK -> (magic) place
runs on the real sim stack, with the 3D scene graph (built from the USD) telling the
planner what exists and where. The "planner" here is just this scripted sequence;
Phase 1 swaps it for an LLM/PDDL planner emitting the same skill calls.

    cd ~/isaac && source .venv/bin/activate
    # default: go to the balcony, pick a chair, carry it to the living room, headless
    python isaac_task_planning/scripts/task_g1_apartment.py --headless
    # watch it in the GUI, custom task:
    python isaac_task_planning/scripts/task_g1_apartment.py --pick-room kitchen --object fridge --return-to livingroom

The pick target is chosen as the most *reachable* object of ``--object`` in
``--pick-room`` (reach measured to its footprint). Cross-room nav uses optimistic
online mapping + A* with stuck-recovery; narrow doorways (living room<->bedroom) are
still the flaky case, so the reliable default goes via the balcony's wide opening.

The occupancy/plan window (GUI runs only) shows the map the robot builds as it goes.
"""

import os

# Make g1sim importable when this file is run directly (see scripts/_bootstrap.py).
import _bootstrap  # noqa: F401

from g1sim.sim.launch import make_parser, launch

parser = make_parser("End-to-end round-trip pick-and-place task for the G1.")
parser.add_argument("--pick-room", dest="pick_room", default="balcony",
                    help="Room to travel to and pick from. Default: balcony.")
parser.add_argument("--object", default="table_lamp",
                    help="Object category (or exact name) to pick in --pick-room. Default: table_lamp.")
parser.add_argument("--return-to", dest="dest", default="livingroom",
                    help="Where to carry it back and place: a room name (floor), an "
                         "object name (place on it), or 'X Y'. Default: livingroom.")
parser.add_argument("--map-json", default=None,
                    help="Load a prebuilt scene graph JSON instead of parsing the USD.")
parser.add_argument("--goto-timeout", type=float, default=120.0)
parser.add_argument("--outdir", default="/home/abhish/isaac/isaac_task_planning/sensor_output")
args = parser.parse_args()

simulation_app = launch(args)

# ---- imports that need the running sim app ----
from g1sim.sim.scene import build_world, NAV_LIDAR_TARGETS
from g1sim.sim.locomotion import G1LocomotionPolicy
from g1sim.perception.semantic_map import SemanticMap
from g1sim.skills.robot import RobotSkills


def choose_target(smap, room, obj):
    """Pick the concrete object to grab: the most reachable object of category
    ``obj`` in ``room`` (reach measured to its footprint from the room nav point),
    or ``obj`` itself if it is already an exact object name."""
    if smap.get(obj) is not None:
        return obj
    nav = smap.navigable_point(room)
    cands = [o for o in smap.reachable_in_room(room, nav) if o.category == obj.lower()]
    return cands[0].name if cands else None


def parse_dest(dest: str):
    """A destination is 'X Y' -> (x, y) tuple, else a room/object name string."""
    parts = dest.split()
    if len(parts) == 2:
        try:
            return (float(parts[0]), float(parts[1]))
        except ValueError:
            pass
    return dest


class MapWindow:
    """Live occupancy-map + A* plan window (GUI runs only), fed from skill state."""
    def __init__(self, title="Task: Occupancy Map + A* plan", width=740, height=560):
        import omni.ui as ui
        self.prov = ui.ByteImageProvider()
        self.win = ui.Window(title, width=width, height=height)
        with self.win.frame:
            ui.ImageWithProvider(self.prov)

    def update(self, skills):
        import cv2
        import numpy as np
        mapper = skills.mapper
        if mapper is None or skills.last_free is None:
            return
        free = skills.last_free
        occ = mapper.occupied()
        img = np.full((mapper.H, mapper.W, 3), 255, np.uint8)
        img[~free] = (180, 180, 180)
        img[occ] = (30, 30, 30)
        scale = 2
        img = cv2.resize(img, (mapper.W * scale, mapper.H * scale), interpolation=cv2.INTER_NEAREST)

        def px(x, y):
            i, j = mapper.world_to_cell(x, y)
            return int(j * scale), int(i * scale)

        wps = skills.last_waypoints or []
        if len(wps) >= 2:
            for a, b in zip(wps[:-1], wps[1:]):
                cv2.line(img, px(*a), px(*b), (0, 160, 0), 2)
        rx, ry = skills.xy()
        cv2.circle(img, px(rx, ry), 5, (0, 90, 255), -1)
        if skills.last_goal:
            cv2.drawMarker(img, px(*skills.last_goal), (220, 0, 0), cv2.MARKER_STAR, 16, 2)
        if skills.held is not None:
            cv2.circle(img, px(skills.held.xy[0], skills.held.xy[1]), 4, (200, 0, 200), -1)
        img = np.flipud(np.ascontiguousarray(img))
        rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        h, w = rgba.shape[:2]
        self.prov.set_bytes_data(list(rgba.tobytes()), [w, h])


def main():
    os.makedirs(args.outdir, exist_ok=True)

    # Semantic map: the planner's model of what exists and where.
    smap = SemanticMap.load(args.map_json) if args.map_json else SemanticMap.build()
    print(smap.describe())

    # Stand up the world with the lidar (the obstacle sensor for goto's mapping).
    sim, scene = build_world(args.spawn, device=args.device, sensors="lidar",
                             lidar_targets=NAV_LIDAR_TARGETS)
    controller = G1LocomotionPolicy(scene["robot"], sim.device)

    map_window = None if args.headless else MapWindow()
    on_step = (lambda s: None) if map_window is None else _throttled(map_window.update, every=8)

    skills = RobotSkills(sim, scene, controller, smap, app=simulation_app, on_step=on_step)

    dest = parse_dest(args.dest)
    pick_room = args.pick_room

    # Choose the concrete pick target: the most reachable object of --object in the
    # pick room (reach to its footprint from the room's nav point). Falls back to
    # treating --object as an exact object name.
    target = choose_target(smap, pick_room, args.object)
    if target is None:
        print(f"[task] no '{args.object}' found in {pick_room}; aborting.")
        print("\n=== TASK FAILED ===")
        return
    print(f"\n=== TASK: go to {pick_room}, pick {target} ({args.object}), "
          f"carry back to '{args.dest}' ===\n")

    # Let the balance policy settle before walking.
    skills.idle(0.5)

    def goto_back():
        if isinstance(dest, tuple):
            return skills.goto(*dest, timeout_s=args.goto_timeout)
        if smap.get(dest) is not None:                     # place ON a named object
            return skills.goto_object(dest, timeout_s=args.goto_timeout)
        return skills.goto_room(dest, timeout_s=args.goto_timeout)   # place in a room

    # Note: we do NOT goto_room(pick_room) first -- that would send the robot to the
    # room's deep centroid nav point (a dead-end trap in small rooms like the
    # balcony). goto_object travels there and approaches the target from the robot's
    # own side (shallow, near the room entrance), so the exit stays easy.
    steps = [
        ("scan", lambda: skills.scan()),
        ("goto object", lambda: skills.goto_object(target, timeout_s=args.goto_timeout)),
        ("pick", lambda: skills.pick(target)),
        ("goto back", goto_back),
        ("place", lambda: skills.place(dest)),
        ("scan", lambda: skills.scan()),
    ]

    ok = True
    for label, fn in steps:
        print(f"\n--- {label} ---")
        res = fn()
        print(res)
        if not res:
            ok = False
            print(f"[task] ABORT: '{label}' failed.")
            break

    print(f"\n=== TASK {'SUCCEEDED' if ok else 'FAILED'} ===")

    # Save the final map the robot built.
    if skills.mapper is not None and skills.last_free is not None and args.headless:
        _save_map(skills, os.path.join(args.outdir, "task_map_result.png"))

    if not args.headless:
        while simulation_app.is_running():
            if map_window is not None:
                map_window.update(skills)
            skills.step(controller.command())


def _throttled(fn, every=8):
    state = {"i": 0}

    def wrapped(s):
        state["i"] += 1
        if state["i"] % every == 0:
            fn(s)
    return wrapped


def _save_map(skills, path):
    import cv2
    import numpy as np
    mapper = skills.mapper
    occ = mapper.occupied()
    img = np.full((mapper.H, mapper.W, 3), 255, np.uint8)
    img[~skills.last_free] = (180, 180, 180)
    img[occ] = (30, 30, 30)
    img = np.flipud(np.ascontiguousarray(img))
    cv2.imwrite(path, img)
    print(f"[task] saved map -> {path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
