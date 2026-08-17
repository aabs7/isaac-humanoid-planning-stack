"""Walk to a counter, pick a block up with the right hand, carry it to a second counter and
put it down -- one G1, one continuous run, no teleport anywhere.

Three controllers share the robot, which is the whole point of the demo:

    legs    pretrained locomotion policy      g1sim.sim.locomotion
    route   unicycle waypoint controller      g1sim.navigation.waypoint
    arms    Pink IK on task-space targets     g1sim.manipulation.arm_ik

The arm never gets joint angles from this file. It gets a *place in the world* -- "put the
hand's closing pocket at the block" -- and Pink turns that into shoulder, elbow, wrist and
waist targets against the live pelvis pose, which is drifting the whole time because a
humanoid balancing on an RL policy is never quite still. The grasp itself is ordinary
physics: the fingers close on the block and friction does the rest.

Run:
    cd ~/isaac && source .venv/bin/activate
    python isaac_task_planning/demos/demo_table_top_environment.py

Headless (prints the same phase log, no window):
    python isaac_task_planning/demos/demo_table_top_environment.py --headless
"""

import _bootstrap  # noqa: F401

from g1sim.sim.launch import make_parser, launch

parser = make_parser("Pick a block off one counter and place it on another.")
parser.add_argument("--side", choices=("left", "right"), default="right",
                    help="Which arm does the picking (default: right).")
args_cli = parser.parse_args()

simulation_app = launch(args_cli)

# ---- Imports that require the running sim app ----
import numpy as np
import torch

from g1sim.environment.table_top import (
    BLOCK_SIZE, PICK_POS, PLACE_POS, TABLE_HEIGHT, build_table_top_environment, stance_for,
)
from g1sim.manipulation.arm_ik import G1ArmIK
from g1sim.navigation.waypoint import WaypointNavigator
from g1sim.sim.locomotion import CONTROL_HZ, G1LocomotionPolicy

# Pre-grasp standoff along the hand's approach axis (sideways -- see GRASP_APPROACH): far
# enough out that the open jaw clears the block, close enough that the capture is one short
# straight stroke.
PREGRASP_BACK = 0.12
LIFT = 0.12


class Aborted(Exception):
    """The sim window was closed mid-run."""


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sim, scene = build_table_top_environment(spawn_xy=(0.0, 0.0), device=device)

    robot, block = scene["robot"], scene["block"]
    locomotion = G1LocomotionPolicy(robot, sim.device)
    navigator = WaypointNavigator(locomotion)
    arm = G1ArmIK(robot, scene.cfg.robot, sim.device, control_dt=1.0 / CONTROL_HZ)
    # The locomotion policy pins every joint it does not drive; the arm owns these now.
    locomotion.yield_joints(arm.owned_joint_ids)

    sim_dt = sim.get_physics_dt()
    side = args_cli.side
    pick_stance, place_stance = stance_for(PICK_POS, side), stance_for(PLACE_POS, side)

    def block_pos():
        return block.data.root_pos_w.torch[0].cpu().numpy().astype(float)

    # -- one control tick: both controllers write targets, then physics runs ------------
    def tick(command):
        locomotion.step(command)
        arm.step()                       # after locomotion.step, before write_data_to_sim
        for _ in range(locomotion.decimation):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        if not simulation_app.is_running():
            raise Aborted

    # -- phases -------------------------------------------------------------------------
    # While the arm works, the legs are not idle: reaching out over a counter shifts the
    # robot's weight and the balance policy answers by stepping away, tens of centimetres over
    # a few seconds of arm motion -- which is most of the arm's reach. So every manipulation
    # phase runs over a station-keeping base command instead of a stand-still one.
    station = None                       # (stand_xy, face_xy) while manipulating

    def base_command():
        if station is None:
            return locomotion.command()
        command, _, _ = navigator.command_station(*station)
        return command

    def stand(seconds):
        """Hold position -- used to let contacts settle between moves."""
        for _ in range(int(seconds * CONTROL_HZ)):
            tick(base_command())

    def walk_to(goal_xy, timeout=40.0):
        for _ in range(int(timeout * CONTROL_HZ)):
            command, dist, arrived = navigator.command_to(goal_xy)
            tick(command)
            if arrived:
                return True
        return False

    def face(target_xy, timeout=12.0):
        for _ in range(int(timeout * CONTROL_HZ)):
            command, yaw_err, aligned = navigator.command_facing(target_xy)
            tick(command)
            if aligned:
                return True
        return False

    def take_station(stand_xy, face_xy, timeout=20.0):
        """Settle onto the working spot: on the mark, facing the work."""
        for _ in range(int(timeout * CONTROL_HZ)):
            command, dist, on_station = navigator.command_station(stand_xy, face_xy)
            tick(command)
            if on_station:
                return True
        return False

    def reach(timeout=6.0, tol=0.015, settle=0.4):
        """Hold position while the IK converges on whatever target the arm was last given."""
        for _ in range(int(timeout * CONTROL_HZ)):
            tick(base_command())
            if arm.error(side) < tol:
                break
        stand(settle)
        return arm.error(side)

    def report(phase, extra=""):
        x, y, heading = navigator.pose2d()
        print(f"[demo] {phase:<22s} base=({x:+.2f},{y:+.2f}) block={np.round(block_pos(), 3)}"
              f"  {extra}")

    # -- the task ------------------------------------------------------------------------
    arm.relax()                                    # both arms down; the idle arm stays there
    arm.open_hand(side)
    stand(1.5)                                     # let the policy find its feet
    report("start", f"wrist={np.round(arm.wrist_pose(side)[0], 3)}")

    walk_to(pick_stance)
    face(PICK_POS[:2])
    take_station(pick_stance, PICK_POS[:2])
    station = (pick_stance, PICK_POS[:2])
    stand(0.5)
    report("at counter 1", f"reach_deficit={arm.out_of_reach(side):.3f}")

    # Reach in two moves: open hand beside the block, then across into the grasp.
    arm.grasp(side, block_pos(), back=PREGRASP_BACK)
    report("pre-grasp", f"ik_err={reach():.3f}")

    arm.grasp(side, block_pos())
    report("at block", f"ik_err={reach():.3f} "
                       f"pocket_off={np.round(arm.grasp_point(side) - block_pos(), 3)}")

    arm.close_hand(side)
    stand(1.0)
    report("hand closed")

    arm.nudge(side, up=LIFT)
    reach()
    arm.nudge(side, forward=-0.15)                 # back away from the counter, still holding
    reach()
    lifted = block_pos()[2] - (TABLE_HEIGHT + BLOCK_SIZE[2] / 2)
    report("lifted", f"+{lifted:.3f} m")
    if lifted < 0.03:
        print("[demo] FAILED: the block did not come with the hand")
        return finish(locomotion, tick)

    arm.carry(side)
    stand(0.5)
    station = None
    walk_to(place_stance)
    face(PLACE_POS[:2])
    take_station(place_stance, PLACE_POS[:2])
    station = (place_stance, PLACE_POS[:2])
    stand(0.5)
    report("at counter 2", f"carrying={block_pos()[2]:.2f} m up")

    # Set the block down at rest -- not hovering. A 12 cm block dropped even a centimetre
    # onto its 4 cm base topples about as often as it stands.
    arm.grasp(side, PLACE_POS, up=0.005)
    report("over target", f"ik_err={reach(timeout=6.0):.3f}")
    # Ease the fingers apart rather than snapping them open, for the same reason.
    for step in range(25):
        arm.set_hand(side, 1.0 - step / 24.0)
        tick(base_command())
    stand(1.0)
    # Withdraw upward, then back, then relax -- in that order. Pulling back first drags the
    # fingers (which sit 5 cm in front of the pocket) through the block that was just set
    # down, and dropping straight into the relaxed pose from over the counter rakes them
    # across its top on the way past.
    arm.nudge(side, up=0.15)
    reach()
    arm.nudge(side, forward=-0.20)
    reach()
    arm.relax()
    stand(1.0)

    settled = block_pos()
    on_target = float(np.linalg.norm(settled[:2] - np.array(PLACE_POS[:2])))
    # Standing or toppled: the block is 12 cm tall on a 4 cm base, so its own height tells us
    # which, without needing its orientation.
    standing = settled[2] > TABLE_HEIGHT + BLOCK_SIZE[2] / 2 - 0.02
    report("released", f"{on_target:.3f} m from target, "
                       f"{'standing' if standing else 'toppled'}")
    print(f"[demo] {'SUCCEEDED' if (on_target < 0.20 and standing) else 'FAILED'}: "
          f"block at {np.round(settled, 3)}, target {np.round(PLACE_POS, 3)}")

    return finish(locomotion, tick)


def finish(locomotion, tick):
    """Keep the world alive so a GUI run can be looked at; return immediately if headless."""
    if args_cli.headless:
        return
    print("[demo] done -- close the window to exit")
    while simulation_app.is_running():
        tick(locomotion.command())


if __name__ == "__main__":
    try:
        main()
    except Aborted:
        print("[demo] window closed")
    simulation_app.close()
