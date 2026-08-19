import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from curobo.types import GoalToolPose, JointState
from curobo.motion_planner import MotionPlanner


def compute_fk_workspace(
    planner: MotionPlanner,
    stance_xy: tuple[float, float] = (0.97, 0.0),
    base_z: float = 0.75,
    tool_frame: str = "right_hand_palm_link",
    n_samples: int = 5000,
) -> np.ndarray:
    """Compute reachable end-effector point cloud via GPU-batched Forward Kinematics."""
    limits = planner.kinematics.get_joint_limits()
    lower = limits.position[0, :]
    upper = limits.position[1, :]

    # Sample random joint angles within limits
    q_rand = lower + torch.rand(n_samples, len(planner.joint_names), device="cuda") * (upper - lower)

    # Lock base to robot stance
    bx = planner.joint_names.index("base_j_x")
    by = planner.joint_names.index("base_j_y")
    bz = planner.joint_names.index("base_j_z")
    q_rand[:, bx] = stance_xy[0]
    q_rand[:, by] = stance_xy[1]
    q_rand[:, bz] = base_z

    # Zero out virtual orientation offsets
    for rot in ["base_j_xtheta", "base_j_ytheta", "base_j_ztheta"]:
        if rot in planner.joint_names:
            q_rand[:, planner.joint_names.index(rot)] = 0.0

    # Batched Forward Kinematics on GPU
    kin = planner.compute_kinematics(JointState.from_position(q_rand, joint_names=planner.joint_names))
    tool_pos = kin.tool_poses.to_dict()[tool_frame].position.squeeze().cpu().numpy()
    return tool_pos


def evaluate_grid_reachability(
    planner: MotionPlanner,
    x_range: tuple[float, float] = (1.0, 1.5),
    y_range: tuple[float, float] = (-0.4, 0.2),
    z_range: tuple[float, float] = (0.75, 1.15),
    resolution: tuple[int, int, int] = (8, 8, 6),
    tool_frame: str = "right_hand_palm_link",
    target_quat: list[float] = [0.97, -0.16, 0.15, -0.03],
    pos_threshold: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate reachability on a 3D grid of Cartesian points using IK solver."""
    xs = np.linspace(x_range[0], x_range[1], resolution[0])
    ys = np.linspace(y_range[0], y_range[1], resolution[1])
    zs = np.linspace(z_range[0], z_range[1], resolution[2])

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    reachable_mask = np.zeros(len(grid), dtype=bool)
    pos_errors = np.zeros(len(grid), dtype=np.float32)

    quat_t = torch.tensor([[[[[target_quat[0], target_quat[1], target_quat[2], target_quat[3]]]]]], device="cuda", dtype=torch.float32)

    print(f"[Reachability] Evaluating {len(grid)} grid points with cuRobo IK...")
    for i, pt in enumerate(grid):
        pos_t = torch.tensor([[[[[pt[0], pt[1], pt[2]]]]]], device="cuda", dtype=torch.float32)
        goal = GoalToolPose(tool_frames=[tool_frame], position=pos_t, quaternion=quat_t)
        res = planner.ik_solver.solve_pose(goal)

        pos_err = res.position_error.item()
        pos_errors[i] = pos_err
        reachable_mask[i] = pos_err <= pos_threshold

    return grid, reachable_mask, pos_errors
