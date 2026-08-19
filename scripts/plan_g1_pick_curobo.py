import _bootstrap

import torch
from pathlib import Path

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.config_io import load_yaml
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState, Pose

from g1sim.environment.table_top import TABLE1_STANCE, ROBOT_SPAWN_Z, PICK_POS
from g1sim.environment.curobo_table_top_scene import build_curobo_tabletop_scene

def create_g1_pick_planner(
        args,
        scene: Scene,
        stance_xy: tuple[float, float] = TABLE1_STANCE,
        base_z: float = ROBOT_SPAWN_Z,
        tool_frame: str = "right_hand_palm_link",
) -> MotionPlanner:
    # g1_yaml_path = Path(args.g1_yaml_path)
    robot_dict = load_yaml(args.g1_yaml_path)

    # configure end-effector tool frame
    robot_dict["kinematics"]["tool_frames"] = [tool_frame]

    # Fix the floating base and legs at the stance pose
    robot_dict["kinematics"]["lock_joints"] = {
        # Lock legs
        "left_hip_pitch_joint": 0.0,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.0,
        "left_ankle_pitch_joint": 0.0,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": 0.0,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.0,
        "right_ankle_pitch_joint": 0.0,
        "right_ankle_roll_joint": 0.0,
        # Lock left arm (planning for right arm pick)
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.0,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
    }

    config = MotionPlannerCfg.create(
        robot=robot_dict,
        scene_model=scene,
        position_tolerance=0.02,
        orientation_tolerance=0.2,
        optimizer_collision_activation_distance=0.005,
        num_ik_seeds=64,
        num_trajopt_seeds=16
    )

    planner = MotionPlanner(config)
    planner.warmup(enable_graph=False, num_warmup_iterations=2)
    return planner


def plan_pick_motion(args):
    print("=" * 60)
    print("cuRobo Motion Planning for G1 Humanoid Pick")
    print("=" * 60)

    # 1. Instantiate TableTop obstacles
    scene = build_curobo_tabletop_scene()
    print(f"[Scene] Initialized with {len(scene.cuboid)} cuboids and {len(scene.cylinder)} cylinders.")

    # 2. Instantiate MotionPlanner
    planner = create_g1_pick_planner(args, scene, stance_xy=TABLE1_STANCE)
    print(f"[Planner] Active planning joints ({len(planner.joint_names)}): {planner.joint_names}")

    # 3. Start state at rest position
    q_start = JointState.from_position(
        planner.default_joint_state.position.clone().unsqueeze(0),
        joint_names=planner.joint_names,
    )

    # print("\n")
    # print(q_start)
    # print("\n\n")

    bx = planner.joint_names.index("base_j_x")
    by = planner.joint_names.index("base_j_y")
    bz = planner.joint_names.index("base_j_z")
    rsp = planner.joint_names.index("right_shoulder_pitch_joint")
    # lsp = planner.joint_names.index("left_shoulder_pitch_joint")
    relb = planner.joint_names.index("right_elbow_joint")
    # lelb = planner.joint_names.index("left_elbow_joint")

    # q_start.position[0, bx] = TABLE1_STANCE[0]
    # q_start.position[0, by] = TABLE1_STANCE[1]
    # q_start.position[0, bz] = ROBOT_SPAWN_Z

    # print("\n")
    # print(q_start)
    # print("\n\n")

    # 4. Target grasp pose for right palm
    # Rest palm orientation facing +X: [qw, qx, qy, qz] ~= [0.97, -0.16, 0.15, -0.03]
    kin = planner.compute_kinematics(q_start)
    rest_palm_quat = kin.tool_poses.to_dict()["right_hand_palm_link"].quaternion

    # # Offset grasp position slightly behind block center to align palm with block surface
    # target_pos = [PICK_POS[0] - 0.05, PICK_POS[1], PICK_POS[2] + 0.02]
    # target_pos = [PICK_POS[0] - 0.1, PICK_POS[1], PICK_POS[2] + 0.1]
    target_pos = [0.25, 0.0, ROBOT_SPAWN_Z]

    goal_pose = GoalToolPose(
        tool_frames=["right_hand_palm_link"],
        position=torch.tensor([[target_pos]], device="cuda", dtype=torch.float32).view(1, 1, 1, 1, 3),
        quaternion=rest_palm_quat.view(1, 1, 1, 1, 4),
    )

    print(f"[Goal] Pick position: {target_pos}")
    print("[Plan] Solving collision-free trajectory with cuRobo...")

    result = planner.plan_pose(goal_pose, q_start, max_attempts=5)

    if result is not None and result.success.any():
        interpolated = result.get_interpolated_plan()
        n_waypoints = interpolated.position.shape[-2]
        dt = planner.trajopt_solver.config.interpolation_dt
        duration = n_waypoints * dt
        print("✓ Planning succeeded!")
        print(f"  Waypoints: {n_waypoints}")
        print(f"  Trajectory Duration: {duration:.2f} s (dt={dt:.3f} s)")
        print(f"  Final waist & arm positions (rad):")
        for name, val in zip(planner.joint_names[:10], interpolated.position[0, -1, :10]):
            print(f"    - {name:28s}: {val.item():+.4f}")
        return interpolated
    else:
        print("✗ Planning failed. Adjust goal pose, tolerance, or stance.")
        print(f"  Result: {result}")
        return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="cuRobo Motion Planning for G1 Humanoid Pick")
    parser.add_argument("--g1_yaml_path", type=str, default='config/unitree_g1_custom.yml', help="Path to the G1 robot YAML configuration file.")
    args = parser.parse_args()

    plan_pick_motion(args)
