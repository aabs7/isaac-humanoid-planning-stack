"""Pretrained G1 locomotion policy, wrapped as a reusable low-level walking
controller. Import only after the sim app is launched.

The controller turns a base velocity command into leg-joint targets using
IsaacLab's pretrained "agile locomotion" TorchScript policy. This is the
low-level primitive a task planner drives: give it ``[vx, vy, wz]`` (a hip
height is added automatically) and call :meth:`step` each control tick.
"""

from __future__ import annotations

import torch

from isaaclab.utils.assets import retrieve_file_path, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.io.torchscript import load_torchscript_model

# Pretrained G1 policy: 83-d input (4 command + 79 obs) -> 12 leg-joint targets.
POLICY_URL = f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt"
POLICY_OUTPUT_SCALE = 0.25       # leg target = policy_out * scale + default_leg_pos
CONTROL_DECIMATION = 4           # sim @ 200 Hz -> policy @ 50 Hz (as trained)
CONTROL_HZ = 50
NOMINAL_HIP_HEIGHT = 0.72        # standing hip-height command (m)

# Observation joint set the policy was trained on (all 29 body joints, index order).
OBS_JOINT_PATTERNS = [
    ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint",
    ".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "waist_.*_joint",
]
# Joints the policy actually drives (its 12 outputs), index order.
ACTION_JOINT_PATTERNS = [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint"]


class G1LocomotionPolicy:
    """Wraps the pretrained agile-locomotion policy for a spawned G1 articulation."""

    decimation = CONTROL_DECIMATION

    def __init__(self, robot, device):
        self.robot = robot
        self.device = device

        self.policy = load_torchscript_model(retrieve_file_path(POLICY_URL), device=device)
        self.policy.eval()

        # Resolve joint index sets exactly as the training config did (sorted order).
        self.obs_ids, _ = robot.find_joints(OBS_JOINT_PATTERNS)              # 29
        self.leg_ids, leg_names = robot.find_joints(ACTION_JOINT_PATTERNS)   # 12
        self.other_ids = [j for j in range(robot.num_joints) if j not in self.leg_ids]

        self.default_joint_pos = robot.data.default_joint_pos.torch.clone()
        self.default_leg_pos = self.default_joint_pos[:, self.leg_ids].clone()
        # last_action obs term = the policy's own previous 12 (raw) outputs.
        self.last_action = torch.zeros(robot.num_instances, len(self.leg_ids), device=device)

        # Reusable hip-height column for command assembly.
        self._hip = torch.full((robot.num_instances, 1), NOMINAL_HIP_HEIGHT, device=device)
        print(f"[policy] obs joints={len(self.obs_ids)} leg joints={len(self.leg_ids)} {leg_names}")

    def yield_joints(self, joint_ids):
        """Stop holding these joints at their default pose -- another controller owns them.

        :meth:`step` pins every joint it does not drive, which silently overwrites whatever
        an arm controller just wrote. Handing the joints over once at startup is safer than
        depending on which controller writes last.
        """
        owned = set(joint_ids)
        self.other_ids = [j for j in self.other_ids if j not in owned]

    # -- command helpers ---------------------------------------------------
    def command(self, vx=0.0, vy=0.0, wz=0.0, hip_height=None) -> torch.Tensor:
        """Assemble a ``(num_envs, 4)`` command ``[vx, vy, wz, hip_height]``."""
        vel = torch.tensor([[vx, vy, wz]], device=self.device).repeat(self.robot.num_instances, 1)
        return self.command_from_se2(vel, hip_height)

    def command_from_se2(self, se2: torch.Tensor, hip_height=None) -> torch.Tensor:
        """Append the hip-height column to an SE(2) ``[vx, vy, wz]`` tensor."""
        se2 = se2.to(self.device).view(self.robot.num_instances, 3)
        hip = self._hip if hip_height is None else torch.full_like(self._hip, hip_height)
        return torch.cat([se2, hip], dim=-1)

    # -- policy inference --------------------------------------------------
    def _observation(self) -> torch.Tensor:
        d = self.robot.data
        obs_jp = d.joint_pos.torch[:, self.obs_ids] - self.default_joint_pos[:, self.obs_ids]
        obs_jv = (d.joint_vel.torch[:, self.obs_ids] - d.default_joint_vel.torch[:, self.obs_ids]) * 0.1
        return torch.cat([
            d.root_lin_vel_b.torch,       # 3
            d.root_ang_vel_b.torch,       # 3
            d.projected_gravity_b.torch,  # 3
            obs_jp,                        # 29
            obs_jv,                        # 29
            self.last_action,              # 12
        ], dim=-1)                         # -> 79

    def step(self, command: torch.Tensor):
        """One control tick: run the policy for ``command`` (num_envs, 4) and write
        leg targets into the articulation buffers. Hold all other joints at default.
        Caller is responsible for stepping the sim ``decimation`` times afterwards."""
        policy_input = torch.cat([command, self._observation()], dim=-1)  # 4 + 79 = 83
        with torch.no_grad():
            raw = self.policy.forward(policy_input)                       # (n, 12)
        self.last_action = raw
        leg_targets = raw * POLICY_OUTPUT_SCALE + self.default_leg_pos
        self.robot.set_joint_position_target_index(target=leg_targets, joint_ids=self.leg_ids)
        if self.other_ids:
            self.robot.set_joint_position_target_index(
                target=self.default_joint_pos[:, self.other_ids], joint_ids=self.other_ids
            )
