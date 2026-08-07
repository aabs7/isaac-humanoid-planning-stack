"""End-to-end LLM-planned task on the real G1 sim (Phase 1.1).

Same stack as scripts/task_g1_apartment.py (USD scene graph, agile locomotion, optimistic
online mapping + A* nav, magic grasp), but the skill sequence is produced by the LLM
planner from a natural-language --goal instead of being hardcoded. The planner drives
the real RobotSkills through the exact interface it drives MockSkills, so a goal proven
in scripts/plan_task_mock.py runs here unchanged.

    cd ~/isaac && source .venv/bin/activate
    ollama pull qwen2.5:7b-instruct          # one-time, if not already
    # headless, the proven scenario:
    python isaac_task_planning/scripts/plan_task_g1_apartment.py \
        --goal "bring the chair from the balcony to the living room" --headless
    # watch it in the GUI (live occupancy/plan window):
    python isaac_task_planning/scripts/plan_task_g1_apartment.py --goal "..."
    # record a follow-the-robot video (chase cam + sensor panels + occupancy map):
    python isaac_task_planning/scripts/plan_task_g1_apartment.py --goal "..." --headless \
        --video isaac_task_planning/sensor_output/run.mp4
"""

import os

# Make g1sim importable when this file is run directly (see scripts/_bootstrap.py).
import _bootstrap  # noqa: F401

from g1sim.sim.launch import make_parser, launch

parser = make_parser("LLM-planned pick-and-place task for the G1.")
parser.add_argument("--goal", required=True, help="Natural-language task goal.")
parser.add_argument("--map-json", default=None,
                    help="Load a prebuilt scene graph JSON instead of parsing the USD.")
parser.add_argument("--model", default=None, help="Ollama model tag (default: planner's).")
parser.add_argument("--max-steps", type=int, default=15)
parser.add_argument("--outdir", default="/home/abhish/isaac/isaac_task_planning/sensor_output")
parser.add_argument("--video", default=None, metavar="PATH",
                    help="Record a follow-the-robot mp4 (chase cam + sensor panels + "
                         "occupancy map) to PATH. Forces the camera pipeline on.")
parser.add_argument("--video-fps", type=int, default=15, help="Recorded video frame rate.")
parser.add_argument("--video-every", type=int, default=3,
                    help="Capture a frame every N control ticks (default 3 ~ 15 Hz).")
args = parser.parse_args()

# Recording needs the render/camera pipeline; turn it on before the app launches.
if args.video:
    args.enable_cameras = True

simulation_app = launch(args)

# ---- imports that need the running sim app ----
from g1sim.sim.scene import build_world, NAV_LIDAR_TARGETS
from g1sim.sim.locomotion import G1LocomotionPolicy
from g1sim.perception.semantic_map import SemanticMap
from g1sim.skills.robot import RobotSkills
from g1sim.task.llm import OllamaChat, DEFAULT_MODEL
from g1sim.task.planner import Planner

# Live map window + helpers (own module -- importing task_g1_apartment would re-run
# its top-level arg parsing/launch against our argv).
from g1sim.viz.task_map import MapWindow, save_map as _save_map


def main():
    os.makedirs(args.outdir, exist_ok=True)

    smap = SemanticMap.load(args.map_json) if args.map_json else SemanticMap.build()
    print(smap.describe())

    llm = OllamaChat(model=args.model or DEFAULT_MODEL)
    if not llm.available():
        print(f"\n[warn] Ollama model not reachable/pulled at {llm.base_url}. "
              f"Run `ollama serve` and `ollama pull {llm.model}`. Aborting before sim.")
        return

    # Recording needs the RGB-D camera + a chase camera (the "record" variant); a plain
    # run only needs the lidar.
    sensors = "record" if args.video else "lidar"
    sim, scene = build_world(args.spawn, device=args.device, sensors=sensors,
                             lidar_targets=NAV_LIDAR_TARGETS)
    controller = G1LocomotionPolicy(scene["robot"], sim.device)

    map_window = None if args.headless else MapWindow(title="LLM Plan: Occupancy + A*")

    skills = RobotSkills(sim, scene, controller, smap, app=simulation_app)

    # Optional video recorder (needs skills for pose/occupancy).
    recorder = None
    if args.video:
        from g1sim.viz.recorder import ChaseRecorder
        recorder = ChaseRecorder(sim, scene, skills, args.video, goal=args.goal,
                                 fps=args.video_fps)

    # One per-tick hook drives both the live GUI map and the video capture.
    tick = {"i": 0}

    def on_step(s):
        tick["i"] += 1
        if map_window is not None and tick["i"] % 8 == 0:
            map_window.update(s)
        if recorder is not None and tick["i"] % args.video_every == 0:
            recorder.capture()
    skills.on_step = on_step

    skills.idle(0.5)   # let the balance policy settle before walking

    print(f"\n=== LLM TASK: {args.goal} ===\n")
    planner = Planner(llm, max_steps=args.max_steps)
    outcome = {"success": False, "reason": "run did not complete", "steps": []}
    try:
        outcome = planner.run(skills, args.goal)
    finally:
        if recorder is not None:
            skills.idle(0.4)      # a beat on the final state, then flush the video
            recorder.close()
            # Unhook it: the GUI loop below keeps stepping the sim (and firing
            # on_step) for as long as the window is open, and the video is done.
            recorder = None

    print(f"\n=== TASK {'SUCCEEDED' if outcome['success'] else 'FAILED'} ===")
    print(f"reason: {outcome['reason']}  ({len(outcome['steps'])} steps)")

    if skills.mapper is not None and skills.last_free is not None and args.headless:
        _save_map(skills, os.path.join(args.outdir, "plan_task_map_result.png"))

    if not args.headless:
        while simulation_app.is_running():
            if map_window is not None:
                map_window.update(skills)
            skills.step(controller.command())


if __name__ == "__main__":
    main()
    simulation_app.close()
