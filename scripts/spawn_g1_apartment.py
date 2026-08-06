# Load the apartment world with a Unitree G1 and hold it standing in place.
# Thin entry point -- all reusable logic lives in g1sim/. Standing uses the same
# pretrained locomotion policy with a zero-velocity command, so the humanoid
# actively balances rather than collapsing.
#
# Run:
#   cd ~/isaac && source .venv/bin/activate
#   python isaac_task_planning/scripts/spawn_g1_apartment.py
#
# Headless self-test:
#   python isaac_task_planning/scripts/spawn_g1_apartment.py --headless --smoke 200

# Make g1sim importable when this file is run directly (see scripts/_bootstrap.py).
import _bootstrap  # noqa: F401

from g1sim.sim.launch import make_parser, launch

parser = make_parser("Load the apartment + G1 and stand in place.")
parser.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="Headless self-test: stand for N control steps, report "
                         "base pose, then exit.")
args = parser.parse_args()
simulation_app = launch(args)

# ---- Imports that require the running sim app ----
from g1sim.sim.scene import build_world
from g1sim.sim.locomotion import G1LocomotionPolicy


def main():
    sim, scene = build_world(args.spawn, device=args.device)
    robot = scene["robot"]
    controller = G1LocomotionPolicy(robot, sim.device)
    print(f"[g1] spawned at {tuple(args.spawn)}; standing via locomotion policy")

    sim_dt = sim.get_physics_dt()
    stand = controller.command()  # zero velocity -> balance in place
    start = robot.data.root_pos_w.torch[0, :3].tolist()
    ctrl_step = 0

    while simulation_app.is_running():
        controller.step(stand)
        for _ in range(controller.decimation):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        ctrl_step += 1

        if args.smoke and ctrl_step >= args.smoke:
            p = robot.data.root_pos_w.torch[0, :3].tolist()
            drift = ((p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2) ** 0.5
            print(f"[smoke] {ctrl_step} control steps ({ctrl_step / 50:.1f}s) | "
                  f"base z={p[2]:.3f} m | drift {drift:.2f} m")
            print(f"[smoke] {'STANDING' if p[2] > 0.55 else 'fell'}")
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
