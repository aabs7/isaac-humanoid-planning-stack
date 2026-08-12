"""End-to-end symbolically-planned task on the real G1 sim.

The counterpart to ``plan_task_g1_apartment.py``: same stack below the skill seam (USD
scene graph, agile locomotion, optimistic online mapping + A* nav, magic grasp), but the
action sequence comes from railroad's MCTS planner searching a symbolic state rather than
from an LLM reading a prompt. The planner drives the real ``RobotSkills`` through the exact
interface it drives ``MockSkills`` with, so a task proven in
``scripts/plan_task_symbolic_mock.py`` runs here unchanged.

The goal is given in plain English and translated once, before planning, by a local LLM
(:mod:`g1sim.task.symbolic.goals`) into the fluent expression railroad searches against.
The LLM chooses *what must be true*; the planner works out every action.

Reaching Ollama is checked before the simulator launches, since that is the most common way
this fails. The goal itself is translated after launch, because it needs the semantic map
and building that from USD must not happen first -- see the comment above ``read_map``. Use
``--dry-run`` to translate without a simulator at all, which is the fast way to check a
phrasing; ``--list`` prints the symbols this apartment offers.

    cd ~/isaac && source .venv/bin/activate
    ollama pull qwen2.5:7b-instruct          # one-time, if not already
    python isaac_task_planning/scripts/plan_task_symbolic_g1_apartment.py \
        --goal "put the kitchen cup on the living room table" --headless
    # just fetch it and hold on to it
    python isaac_task_planning/scripts/plan_task_symbolic_g1_apartment.py \
        --goal "bring me a cup from the kitchen"
    # watch it, or record it
    python isaac_task_planning/scripts/plan_task_symbolic_g1_apartment.py \
        --goal "move the cup to the coffee table" \
        --video isaac_task_planning/sensor_output/symbolic_run.mp4 --headless
"""

import os

# Make g1sim importable when this file is run directly (see scripts/_bootstrap.py).
import _bootstrap  # noqa: F401

from g1sim.sim.launch import make_parser, launch

parser = make_parser("Symbolically-planned (railroad MCTS) pick-and-place for the G1.")
parser.add_argument("--goal", required=True,
                    help="The task in plain English, e.g. \"put the kitchen cup on the "
                         "living room table\". An LLM translates it into goal fluents.")
parser.add_argument("--model", default=None,
                    help="Ollama model tag for the goal translation (default: the "
                         "client's).")
parser.add_argument("--dry-run", action="store_true",
                    help="Translate the goal, print it, and exit without launching the sim.")
parser.add_argument("--list", action="store_true",
                    help="Print the domain's locations and pickable objects, then exit "
                         "(before launching the sim).")
parser.add_argument("--map-json", default=None,
                    help="Load a prebuilt scene graph JSON instead of parsing the USD.")
parser.add_argument("--astar-costs", action="store_true",
                    help="Cost moves with the real A* planner over the sensed occupancy "
                         "grid instead of straight-line distance. Wall-aware, but slow to "
                         "ground and meaningless until the map has been filled in.")
parser.add_argument("--all-objects", action="store_true",
                    help="Plan over the whole apartment (10710 grounded actions vs ~56) "
                         "instead of just the task's objects and locations. Much slower and "
                         "markedly less reliable -- see SemanticDomain.for_task. Diagnostic.")
parser.add_argument("--max-dispatches", type=int, default=15)
parser.add_argument("--mcts-iterations", type=int, default=None,
                    help="Search budget per dispatch (default: the module's tuned value).")
parser.add_argument("--outdir", default="/home/abhish/isaac/isaac_task_planning/sensor_output")
parser.add_argument("--video", default=None, metavar="PATH",
                    help="Record a follow-the-robot mp4. Forces the camera pipeline on.")
parser.add_argument("--video-fps", type=int, default=15)
parser.add_argument("--video-every", type=int, default=3)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# NOTHING here may touch USD before launch(). `SemanticMap.build()` parses the stage with
# standalone ``pxr``, and pxr loaded before SimulationApp starts leaves two USD runtimes in
# one process: Kit then dies during startup with `free(): invalid pointer` -- an abort, not
# an exception you can catch or a message that points at the cause. (SimulationApp does warn,
# in a wall of startup logs, that modules were "loaded before SimulationApp was started".)
#
# So the map, the domain and the goal are all prepared *inside* main(), after the app exists.
# The only things done up front are the ones that cannot import pxr: parsing arguments and
# pinging Ollama over stdlib urllib. That preflight is worth keeping, because a stopped
# Ollama is the most common way this script fails and catching it costs a second.
#
# --list and --dry-run are the exception, and are safe only because they never launch: they
# parse USD and then exit. Do not add work after them that reaches launch().
# ---------------------------------------------------------------------------
from g1sim.task.llm_based.llm import DEFAULT_MODEL, OllamaChat

llm = OllamaChat(model=args.model or DEFAULT_MODEL)
if not llm.available():
    raise SystemExit(f"\n[error] Ollama model '{llm.model}' not reachable at {llm.base_url}. "
                     f"Run `ollama serve` and `ollama pull {llm.model}`.")


def read_map():
    """The semantic map. Parses USD unless --map-json makes it a pure JSON read, so this
    must not be called before launch() except on a path that exits without launching."""
    from g1sim.perception.semantic_map import SemanticMap

    smap = SemanticMap.load(args.map_json) if args.map_json else SemanticMap.build()
    print(smap.describe())
    return smap


def prepare(smap):
    """map -> symbols -> translated goal -> the task-scoped planning domain.

    The menu the LLM chooses from must be the *unrestricted* domain: it has to be able to
    name any pickable object. Only the search domain gets narrowed, to what the goal
    mentions -- which takes grounding from 10710 actions to ~56 and planning from ~2 s to
    under 10 ms. That narrowing matters for *reliability*, not just speed: with all 49
    locations in play the search dithers among cheap irrelevant neighbours and never sets
    out for the object. See SemanticDomain.for_task.
    """
    from g1sim.task.symbolic import GoalTranslationError, SemanticDomain, translate

    full_domain = SemanticDomain.build(smap)
    try:
        task = translate(llm, smap, args.goal, domain=full_domain)
    except GoalTranslationError as e:
        raise SystemExit(f"\n[error] {e}")

    domain = (full_domain if args.all_objects else
              SemanticDomain.for_task(smap, sorted(task.objects), sorted(task.locations)))
    print(domain.describe())
    return task, domain


# ---- the two paths that answer without a simulator ----
if args.list:
    from g1sim.task.symbolic import SemanticDomain

    full_domain = SemanticDomain.build(read_map())
    print(full_domain.describe())
    print("\nrooms:     " + ", ".join(sorted(full_domain.rooms)))
    print("\nsurfaces:  " + ", ".join(sorted(full_domain.surfaces)))
    print("\nobjects:   " + ", ".join(sorted(full_domain.objects)))
    raise SystemExit(0)

if args.dry_run:
    task, _ = prepare(read_map())
    print(f"\n=== GOAL (dry run, sim not launched) ===\n{task}")
    raise SystemExit(0)

if args.video:
    args.enable_cameras = True

simulation_app = launch(args)

# ---- imports that need the running sim app ----
from g1sim.sim.scene import build_world, NAV_LIDAR_TARGETS
from g1sim.sim.locomotion import G1LocomotionPolicy
from g1sim.skills.robot import RobotSkills
from g1sim.task.symbolic import AStarMoveTime, G1Environment, solve
from g1sim.task.symbolic.planner import MCTS_ITERATIONS
from g1sim.viz.task_map import MapWindow


def main():
    os.makedirs(args.outdir, exist_ok=True)
    smap = read_map()
    task, domain = prepare(smap)
    goal = task.goal
    print(f"\n=== TASK: {args.goal}\n=== GOAL: {task}\n")

    sensors = "record" if args.video else "lidar"
    sim, scene = build_world(args.spawn, device=args.device, sensors=sensors,
                             lidar_targets=NAV_LIDAR_TARGETS)
    controller = G1LocomotionPolicy(scene["robot"], sim.device)

    map_window = None if args.headless else MapWindow(title="Symbolic Plan: Occupancy + A*")
    skills = RobotSkills(sim, scene, controller, smap, app=simulation_app)

    recorder = None
    if args.video:
        from g1sim.viz.recorder import ChaseRecorder
        # The overlay shows the English task -- that is what a viewer wants to read; the
        # fluent form is in the log.
        recorder = ChaseRecorder(sim, scene, skills, args.video, goal=args.goal,
                                 fps=args.video_fps)

    tick = {"i": 0}

    def on_step(s):
        tick["i"] += 1
        if map_window is not None and tick["i"] % 8 == 0:
            map_window.update(s)
        if recorder is not None and tick["i"] % args.video_every == 0:
            recorder.capture()
    skills.on_step = on_step

    skills.idle(0.5)      # let the balance policy settle before walking

    # The environment must be built *after* the robot is standing: its initial state reads
    # the robot's pose to decide which location symbol it starts at.
    move_time = AStarMoveTime(domain, skills.mapper) if args.astar_costs else None
    if move_time is not None and skills.mapper is None:
        print("[warn] --astar-costs needs the lidar mapper; falling back to straight-line.")
        move_time = None
    env = G1Environment(skills, domain, move_time=move_time)

    def on_dispatch(step, name, result):
        """Mirror the planner's current action onto the video overlay. Called before the
        action runs (result=None -> "running...") and again after it (-> OK / FAILED)."""
        if recorder is not None:
            verb, _robot, *operands = name.split()
            recorder.set_action(step, verb, operands, "", result)

    outcome = None
    try:
        outcome = solve(env, goal, max_dispatches=args.max_dispatches,
                        mcts_iterations=args.mcts_iterations or MCTS_ITERATIONS,
                        on_dispatch=on_dispatch)
    finally:
        if recorder is not None:
            skills.idle(0.4)
            recorder.close()
            recorder = None

    print(f"\n=== TASK {'SUCCEEDED' if (outcome and outcome.ok) else 'FAILED'} ===")
    if outcome is not None:
        print(outcome)

    if skills.mapper is not None and skills.last_free is not None and args.headless:
        skills.mapper.save_png(os.path.join(args.outdir, "symbolic_task_map_result.png"),
                               free=skills.last_free)

    if not args.headless:
        while simulation_app.is_running():
            if map_window is not None:
                map_window.update(skills)
            skills.step(controller.command())


if __name__ == "__main__":
    main()
    simulation_app.close()
