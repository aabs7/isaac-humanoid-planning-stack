# Keyboard-teleop a Unitree G1 around the apartment with the pretrained
# locomotion policy. Thin entry point -- all reusable logic lives in g1sim/.
#
# Run (GUI, keyboard):
#   cd ~/isaac && source .venv/bin/activate
#   python isaac_task_planning/teleop_g1_apartment.py
#
# Keyboard (focus the Isaac Sim window):
#   Up / Down      : walk forward / backward    (also Numpad 8 / 2)
#   Left / Right   : strafe left / right         (also Numpad 4 / 6)
#   Z / X          : turn left / right (yaw)      (also Numpad 7 / 9)
#   L              : stop (zero the command)
#
# Headless self-test (no keyboard, walks forward, reports distance):
#   python isaac_task_planning/teleop_g1_apartment.py --headless --smoke 250

from g1sim.launch import make_parser, launch

parser = make_parser("Keyboard-teleop a G1 in the apartment.")
parser.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="Headless self-test: drive a fixed forward command for N "
                         "control steps, report distance walked, then exit.")
args = parser.parse_args()
simulation_app = launch(args)

# ---- Imports that require the running sim app ----
from g1sim.scene import build_world
from g1sim.locomotion import G1LocomotionPolicy


def main():
    sim, scene = build_world(args.spawn, device=args.device)
    robot = scene["robot"]
    controller = G1LocomotionPolicy(robot, sim.device)
    print(f"[g1] spawned at {tuple(args.spawn)}; loaded pretrained agile locomotion policy")

    # Keyboard device (needs the GUI window). Skipped in headless/smoke mode.
    keyboard = None
    if not args.smoke:
        from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg
        keyboard = Se2Keyboard(Se2KeyboardCfg(
            v_x_sensitivity=0.8, v_y_sensitivity=0.5, omega_z_sensitivity=1.0,
            sim_device=str(sim.device),
        ))
        print(keyboard)  # prints the full key-binding help
        print("[teleop] focus the Isaac Sim window. Arrows=move, Z/X=turn, L=stop.")

    sim_dt = sim.get_physics_dt()
    start_xy = robot.data.root_pos_w.torch[0, :2].tolist()
    ctrl_step = 0

    while simulation_app.is_running():
        # Velocity command: from the keyboard, or a fixed forward walk in smoke mode.
        if keyboard is not None:
            command = controller.command_from_se2(keyboard.advance())
        else:
            command = controller.command(vx=0.5)

        # One 50 Hz control tick, then decimated 200 Hz sim steps.
        controller.step(command)
        for _ in range(controller.decimation):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        ctrl_step += 1

        if args.smoke and ctrl_step >= args.smoke:
            p = robot.data.root_pos_w.torch[0, :3].tolist()
            dist = ((p[0] - start_xy[0]) ** 2 + (p[1] - start_xy[1]) ** 2) ** 0.5
            print(f"[smoke] {ctrl_step} control steps ({ctrl_step / 50:.1f}s) | "
                  f"base z={p[2]:.3f} m | walked {dist:.2f} m from start")
            print(f"[smoke] {'WALKING' if p[2] > 0.55 and dist > 0.3 else 'did not walk / fell'}")
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
