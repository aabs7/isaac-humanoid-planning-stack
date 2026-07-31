# Navigate the G1 to given (x, y) coordinates in the apartment using the
# pretrained locomotion policy + a closed-loop waypoint controller.
# Thin entry point -- navigation logic lives in g1sim/navigation.py.
#
# Go to one point:
#   cd ~/isaac && source .venv/bin/activate
#   python isaac_task_planning/navigate_g1_apartment.py --goal 3.5 -3.0
#
# Chain several waypoints (repeat --goal):
#   python isaac_task_planning/navigate_g1_apartment.py --goal 3.5 -3.0 --goal 10.0 -3.0
#
# Headless (validate it arrives, then exit):
#   python isaac_task_planning/navigate_g1_apartment.py --headless --goal 3.5 -3.0

from g1sim.launch import make_parser, launch

parser = make_parser("Navigate the G1 to (x, y) goal(s) in the apartment.")
parser.add_argument("--goal", type=float, nargs=2, action="append", metavar=("X", "Y"),
                    help="A goal (x, y) in meters. Repeat for multiple waypoints. "
                         "Default: a point across the living room.")
parser.add_argument("--goal-timeout", type=float, default=60.0, metavar="SEC",
                    help="Give up on a waypoint after this many seconds (default 60).")
args = parser.parse_args()

simulation_app = launch(args)

# ---- Imports that require the running sim app ----
from g1sim.scene import build_world
from g1sim.locomotion import G1LocomotionPolicy
from g1sim.navigation import WaypointNavigator

CONTROL_HZ = 50  # policy control rate (1 tick = decimation sim steps)


def main():
    goals = args.goal if args.goal else [[11.0, 0.0]]

    sim, scene = build_world(args.spawn, device=args.device)
    robot = scene["robot"]
    controller = G1LocomotionPolicy(robot, sim.device)
    nav = WaypointNavigator(controller)
    print(f"[nav] start {tuple(args.spawn)} -> waypoints {[tuple(g) for g in goals]}")

    sim_dt = sim.get_physics_dt()
    max_ticks = int(args.goal_timeout * CONTROL_HZ)
    gi = 0
    ticks_on_goal = 0
    done = False

    while simulation_app.is_running():
        if not done:
            command, dist, arrived = nav.command_to(goals[gi])
            ticks_on_goal += 1

            if ticks_on_goal % CONTROL_HZ == 0:  # ~1 Hz progress print
                x, y, h = nav.pose2d()
                print(f"[nav] -> goal {gi + 1}/{len(goals)} {tuple(goals[gi])}: "
                      f"at ({x:.2f}, {y:.2f}) dist={dist:.2f} m")

            if arrived:
                x, y, _ = nav.pose2d()
                print(f"[nav] reached goal {gi + 1} at ({x:.2f}, {y:.2f})")
                gi += 1
                ticks_on_goal = 0
            elif ticks_on_goal >= max_ticks:
                print(f"[nav] TIMEOUT on goal {gi + 1} (stuck?); moving on")
                gi += 1
                ticks_on_goal = 0

            if gi >= len(goals):
                print("[nav] all waypoints reached; holding position")
                done = True
        else:
            command = controller.command()  # stand in place at the final goal

        controller.step(command)
        for _ in range(controller.decimation):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

        if done and args.headless:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
