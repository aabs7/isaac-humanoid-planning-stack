"""Arm control for the G1: task-space wrist targets solved by Pink IK.

The locomotion policy owns the legs (:mod:`g1sim.sim.locomotion`); this owns everything
above them. Give it a wrist pose and it produces joint targets for the shoulders, elbows,
wrists and waist by way of IsaacLab's :class:`~isaaclab.controllers.pink_ik.PinkIKController`
-- a differential-IK solver that turns weighted task errors into joint velocities through a
QP. Fingers are not part of the QP: they are position-commanded open or closed.

Import only after the sim app is launched.

Three facts about this setup were expensive to establish and are baked in below:

* **Pink solves on a URDF, not on the USD the sim runs.** IsaacLab ships a matching one for
  this exact robot (:data:`KINEMATICS_URDF`), so no runtime USD->URDF conversion is needed.
  Its link frames carry the ``g1_29dof_with_hand_rev_1_0_`` prefix while its *joint* names do
  not -- the controller cross-references the two sets by name and raises if either side has a
  joint the other lacks, so both name sets here are exact, not patterns.
* **The USD link frames and the URDF link frames agree** (verified numerically at the default
  pose), so a wrist pose measured off the articulation can be handed straight to the solver.
  The pelvis frame is the usual x-forward, y-left, z-up; the robot's *world* orientation at
  spawn is a 90-degree yaw, which is why every target here is expressed relative to the
  pelvis rather than in world axes.
* **A wrist pose is not a grasp pose.** The three-finger hand closes into a pocket in front
  of the palm, ~11 cm out along the wrist's +x and ~2 cm toward the thumb;
  :data:`GRASP_OFFSET` is that pocket, measured by forward-kinematics sweeps of the finger
  chain. :meth:`G1ArmIK.grasp` targets the pocket at the object, not the wrist at the object.

Ordering matters when driving a walking robot: :meth:`G1LocomotionPolicy.step
<g1sim.sim.locomotion.G1LocomotionPolicy.step>` pins every joint it does not drive to its
default pose, which would fight this controller. Hand the joints over once at startup::

    arm = G1ArmIK(robot, scene.cfg.robot, sim.device, control_dt=1 / 50)
    locomotion.yield_joints(arm.owned_joint_ids)
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import torch

import isaaclab.utils.math as math_utils
from isaaclab.controllers.pink_ik import (
    LocalFrameTaskCfg,
    NullSpacePostureTaskCfg,
    PinkIKController,
    PinkIKControllerCfg,
)
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Kinematics-only URDF of this exact robot (29 body DoF + 14 hand DoF), shipped for the
# locomanipulation tasks. Nothing but joints and frames -- no meshes to fetch.
KINEMATICS_URDF = (
    f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/"
    "unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"
)
URDF_LINK_PREFIX = "g1_29dof_with_hand_rev_1_0_"

SIDES = ("left", "right")

# Joints the IK solves for. The waist is included (as in IsaacLab's own G1 locomanipulation
# config): 30 degrees of yaw and 15 of pitch buy roughly 15 cm of forward reach, which is the
# difference between a table object being reachable and not.
ARM_JOINT_PATTERNS = [
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
    ".*_elbow_joint", ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint",
    "waist_.*_joint",
]

# Finger targets for the *right* hand, as (open, closed) radians per joint. Closed values are
# past the point where the fingers meet, so a held object stops them early and the position
# error becomes grip force. The thumb's first joint stays at 0: it swings the thumb *out of*
# the finger plane, and the opposition we want comes from the other two. The left hand's
# joints are mirrored in the USD -- its limits run the other way -- so its closed values are
# these negated (all but the symmetric thumb_0).
FINGER_POSE = {
    "hand_index_0_joint": (0.0, 1.20),
    "hand_index_1_joint": (0.0, 1.30),
    "hand_middle_0_joint": (0.0, 1.20),
    "hand_middle_1_joint": (0.0, 1.30),
    "hand_thumb_0_joint": (0.0, 0.0),
    "hand_thumb_1_joint": (0.0, -1.00),   # limit is -1.047; stay off the stop
    "hand_thumb_2_joint": (0.0, -1.00),
}

# Everything below is authored for the *right* hand and mirrored in y for the left.

# Where a grasped object should sit, in the wrist_yaw frame: the middle of the pocket the
# fingers and thumb close into, measured by sweeping the finger chain through its range.
# Index and middle fingers reach out to x=0.165 and curl in to (0.119, 0.041); the thumb tip
# comes the other way, from (0.065, 0.062) in to (0.104, 0.039). What they trap is this.
GRASP_OFFSET = np.array([0.115, 0.020, 0.0])

# Which way the hand travels to take hold of something. The jaw opens sideways -- toward the
# thumb -- so the hand comes at an object from beside it and closes across, rather than
# driving straight in: the fingers stick out 5 cm past the pocket, and a head-on approach
# knocks the object over with the fingertips before the pocket ever reaches it. Coming in
# from the side, the fingers pass in front of the object and the thumb behind it.
GRASP_APPROACH = np.array([0.0, 1.0, 0.0])

# A neutral "carry" wrist pose in the pelvis frame: elbow bent, hand in front of the hip,
# clear of the legs and of anything the robot walks past -- and comfortably inside
# MAX_REACH, so a carried object is not held at full stretch for the length of a walk.
CARRY_POSE = np.array([0.22, -0.15, 0.08])

# Arms hanging at the sides. The G1's zero pose is not this: it is an L, upper arm down and
# forearm straight out in front, which is what the robot reverts to whenever nothing else is
# asked of it. Hanging means driving the elbow ~75 degrees the other way.
#
# Straight down puts the wrist 0.368 m from the shoulder -- past MAX_REACH, and within a
# centimetre of the arm's full 0.37 m span, where the IK sits on a singularity and jitters.
# So this is the same hanging direction stopped a little short of the lock, which is what a
# relaxed arm looks like anyway: a few degrees of elbow left in it. Slightly outboard of the
# shoulder (y beyond -0.14) so the forearm clears the hip while walking.
RELAX_POSE = np.array([0.05, -0.18, -0.02])

# Fingers pointing at the floor: the wrist's +x axis (out through the fingers) rotated onto
# the pelvis frame's -z. The thumb side stays on +y, so the palm faces in toward the leg.
RELAX_ROT = np.array([[0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0]])

# Shoulder-to-wrist span with the elbow straight is 0.37 m. Targets are clamped inside this,
# short of it: asking for a pose the arm cannot reach does not politely fail, it stretches the
# whole upper body at the target and hands the balance policy a lurch it answers by stepping
# away -- which moves the target further out of reach. That runaway is worth a hard stop.
MAX_REACH = 0.33


def _mirror(v: np.ndarray, side: str) -> np.ndarray:
    """Flip the y component for the left side (poses here are authored right-handed)."""
    return v if side == "right" else v * np.array([1.0, -1.0, 1.0])


def finger_target(joint: str, side: str, closed: float) -> float:
    """One finger joint's commanded angle, ``closed`` running 0 (open) to 1 (shut).

    The left hand is not a copy of the right: its finger joints are authored with the
    opposite sign, so the same positive command that closes the right hand drives the left
    one straight into its lower limit and the hand simply does not shut. ``thumb_0`` is the
    exception -- it is symmetric about zero on both hands.
    """
    open_q, closed_q = FINGER_POSE[joint]
    sign = -1.0 if (side == "left" and "thumb_0" not in joint) else 1.0
    return sign * (open_q + closed * (closed_q - open_q))


def clamp_reach(pos_local, shoulder_local, max_reach: float = MAX_REACH) -> np.ndarray:
    """Pull a wrist target back onto the arm's reach sphere, keeping its direction."""
    pos = np.asarray(pos_local, dtype=np.float64)
    arm = pos - shoulder_local
    span = float(np.linalg.norm(arm))
    if span <= max_reach or span == 0.0:
        return pos
    return shoulder_local + arm * (max_reach / span)


class _Target:
    """What the arm is currently being asked for.

    Three kinds, because a target has to keep meaning the same thing while the robot moves
    under it -- and what "the same thing" is depends on the job:

    ``local``  a pose in the pelvis frame. Rides along with the robot, which is what carrying
               a held object wants.
    ``grasp``  a *world* point for the hand's closing pocket to land on. Re-derived every
               tick from the live pelvis pose and the live wrist orientation, so neither the
               balancing robot stepping backwards nor the IK settling into a slightly rotated
               wrist leaves the fingers closing next to the object instead of around it.
    """

    def __init__(self, kind, pos, rot=None, back=0.0, up=0.0):
        self.kind, self.pos, self.rot, self.back, self.up = kind, pos, rot, back, up


class G1ArmIK:
    """Task-space control of the G1's arms, waist and hands.

    Targets are held until changed, so a caller sets one and then ticks :meth:`step` until
    :meth:`error` reports convergence.
    """

    def __init__(self, robot, robot_cfg, device, *, control_dt: float,
                 urdf_path: str = KINEMATICS_URDF, show_ik_warnings: bool = False):
        self.robot = robot
        self.device = device
        self.dt = control_dt

        joint_names = list(robot.data.joint_names)
        self.arm_ids, arm_names = robot.find_joints(ARM_JOINT_PATTERNS)
        self.hand_ids, hand_names = robot.find_joints(
            [f"{side}_{j}" for side in SIDES for j in FINGER_POSE])

        cfg = PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            urdf_path=urdf_path,
            show_ik_warnings=show_ik_warnings,
            # The solver runs beside a balancing robot whose waist is never exactly where the
            # last solution left it; a limit violation is a reason to clamp, not to give up.
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                # Costs and damping follow IsaacLab's G1 locomanipulation config, which was
                # tuned against this same arm.
                LocalFrameTaskCfg(
                    frame=f"{URDF_LINK_PREFIX}{side}_wrist_yaw_link",
                    base_link_frame_name=f"{URDF_LINK_PREFIX}pelvis",
                    position_cost=8.0, orientation_cost=4.0, lm_damping=75, gain=0.075,
                ) for side in SIDES
            ] + [
                # Keeps the redundant DoF near the default posture instead of wandering into
                # a shoulder-up contortion that happens to satisfy the wrist pose.
                NullSpacePostureTaskCfg(
                    cost=0.05, lm_damping=75, gain=0.075,
                    controlled_frames=[f"{URDF_LINK_PREFIX}{s}_wrist_yaw_link" for s in SIDES],
                    controlled_joints=[f"{s}_shoulder_{a}_joint" for s in SIDES
                                       for a in ("pitch", "roll", "yaw")]
                                      + ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"],
                ),
            ],
            joint_names=arm_names,
            all_joint_names=joint_names,
        )
        self.controller = PinkIKController(
            cfg=cfg, robot_cfg=robot_cfg, device=device, controlled_joint_indices=self.arm_ids)
        self._tasks = dict(zip(SIDES, self.controller.cfg.variable_input_tasks))

        self._pelvis_idx = robot.data.body_names.index("pelvis")
        self._wrist_idx = {s: robot.data.body_names.index(f"{s}_wrist_yaw_link") for s in SIDES}
        # Where each arm hangs from, in the pelvis frame -- measured rather than tabulated,
        # since it is the origin every reachability question is asked about.
        self._shoulder = {
            s: self._local_body_pos(robot.data.body_names.index(f"{s}_shoulder_roll_link"))
            for s in SIDES
        }
        self._hand_cols = {name: i for i, name in enumerate(hand_names)}
        self._hand_target = torch.zeros(robot.num_instances, len(self.hand_ids), device=device)
        for side in SIDES:
            self.open_hand(side)

        # Start by asking for where the arms already are, so the first tick is a no-op.
        self._target = {s: _Target("local", *self.wrist_pose(s)) for s in SIDES}
        print(f"[arm] pink IK on {len(self.arm_ids)} joints + {len(self.hand_ids)} finger joints")

    # -- joint ownership ---------------------------------------------------
    @property
    def owned_joint_ids(self) -> list:
        """Every joint this controller writes -- see the module docstring on handover."""
        return list(self.arm_ids) + list(self.hand_ids)

    # -- frames ------------------------------------------------------------
    def base_pose(self):
        """Pelvis pose in world as ``(position (3,), rotation (3, 3))`` numpy arrays."""
        p = self.robot.data.body_link_pose_w.torch[0, self._pelvis_idx]
        rot = math_utils.matrix_from_quat(p[3:7].unsqueeze(0))[0]
        return p[:3].cpu().numpy().astype(np.float64), rot.cpu().numpy().astype(np.float64)

    def to_local(self, pos_w) -> np.ndarray:
        """A world point in the pelvis frame."""
        base_pos, base_rot = self.base_pose()
        return base_rot.T @ (np.asarray(pos_w, dtype=np.float64) - base_pos)

    def to_world(self, pos_local) -> np.ndarray:
        """A pelvis-frame point in world."""
        base_pos, base_rot = self.base_pose()
        return base_pos + base_rot @ np.asarray(pos_local, dtype=np.float64)

    def _local_body_pos(self, body_idx: int) -> np.ndarray:
        pos_w = self.robot.data.body_link_pose_w.torch[0, body_idx, :3]
        return self.to_local(pos_w.cpu().numpy())

    def wrist_pose(self, side: str):
        """Measured wrist pose in the pelvis frame as ``(position, rotation)``."""
        p = self.robot.data.body_link_pose_w.torch[0, self._wrist_idx[side]]
        rot_w = math_utils.matrix_from_quat(p[3:7].unsqueeze(0))[0].cpu().numpy().astype(np.float64)
        base_pos, base_rot = self.base_pose()
        pos = base_rot.T @ (p[:3].cpu().numpy().astype(np.float64) - base_pos)
        return pos, base_rot.T @ rot_w

    def grasp_point(self, side: str) -> np.ndarray:
        """Where the hand would close, in world -- the wrist pose pushed out by
        :data:`GRASP_OFFSET`. This is the point to compare against an object's position."""
        pos, rot = self.wrist_pose(side)
        return self.to_world(pos + rot @ _mirror(GRASP_OFFSET, side))

    # -- targets -----------------------------------------------------------
    def set_wrist_target(self, side: str, pos_local, rot_local=None):
        """Ask for a wrist pose in the pelvis frame. ``rot_local=None`` keeps the wrist
        aligned with the pelvis, which points the fingers straight ahead."""
        rot = np.eye(3) if rot_local is None else np.asarray(rot_local, dtype=np.float64)
        self._target[side] = _Target("local", np.asarray(pos_local, dtype=np.float64), rot)

    def grasp(self, side: str, object_pos_w, *, back: float = 0.0, up: float = 0.0):
        """Put the hand's closing pocket on a world point, and keep it there.

        ``back`` holds the hand short of it along :data:`GRASP_APPROACH` (a pre-grasp
        standoff, so moving from ``back=0.12`` to ``back=0`` is the capture stroke); ``up``
        holds it above. The point is tracked, not snapshotted -- see :class:`_Target`.
        """
        self._target[side] = _Target("grasp", np.asarray(object_pos_w, dtype=np.float64),
                                     back=back, up=up)

    def nudge(self, side: str, *, forward: float = 0.0, left: float = 0.0, up: float = 0.0):
        """Shift the current target, in pelvis axes, and hold it there relative to the robot.

        Lifting a grasped object is this and nothing else: the object is in the hand, so the
        hand is what moves. It also pins the target to the pelvis, which is what you want
        once something is held -- chasing the object's world position after the hand has hold
        of it is a loop chasing its own tail.
        """
        pos, rot = self.resolve(side)
        self.set_wrist_target(side, pos + np.array([forward, left, up]), rot)

    def carry(self, side: str):
        """Tuck the arm into the neutral carry pose (safe to walk with)."""
        self.set_wrist_target(side, _mirror(CARRY_POSE, side))

    def relax(self, side: str | None = None):
        """Let the arms hang at the sides. Defaults to both.

        The pose to leave the robot in when it is not doing anything with its hands: without
        a target the arms hold the G1's zero pose, forearms stuck out in front like a pair of
        shelf brackets. See :data:`RELAX_POSE` for why this hangs *nearly* straight rather
        than straight.
        """
        for s in (SIDES if side is None else (side,)):
            self.set_wrist_target(s, _mirror(RELAX_POSE, s), RELAX_ROT)

    def resolve(self, side: str):
        """The current target as a concrete, reachable ``(position, rotation)`` in the pelvis
        frame."""
        target = self._target[side]
        if target.kind == "local":
            return self.clamp_to_workspace(side, target.pos), target.rot
        # Aim the wrist so that the *pocket* lands on the point, using the orientation the
        # wrist actually settled into rather than the one that was asked for.
        _, wrist_rot = self.wrist_pose(side)
        pos = self._pocket_goal(side, target) - wrist_rot @ _mirror(GRASP_OFFSET, side)
        return self.clamp_to_workspace(side, pos), np.eye(3)

    def _pocket_goal(self, side: str, target) -> np.ndarray:
        """Where the closing pocket is being asked to go, in the pelvis frame."""
        return (self.to_local(target.pos) - target.back * _mirror(GRASP_APPROACH, side)
                + np.array([0.0, 0.0, target.up]))

    def clamp_to_workspace(self, side: str, pos_local) -> np.ndarray:
        """Pull a pelvis-frame wrist target back inside :data:`MAX_REACH` of the shoulder."""
        return clamp_reach(pos_local, self._shoulder[side])

    def out_of_reach(self, side: str) -> float:
        """How far past the arm's workspace the current target is, in metres (0 if inside)."""
        target = self._target[side]
        pos = (target.pos if target.kind == "local"
               else self._pocket_goal(side, target) - _mirror(GRASP_OFFSET, side))
        return max(0.0, float(np.linalg.norm(pos - self._shoulder[side])) - MAX_REACH)

    def error(self, side: str) -> float:
        """How far the hand is from doing what it was asked, in metres.

        For a grasp target that is the distance from the closing pocket to the point, not
        from the wrist to the wrist target -- the wrist can sit exactly where it was sent
        while the fingers close on empty air beside the object.
        """
        target = self._target[side]
        if target.kind == "local":
            return float(np.linalg.norm(self.wrist_pose(side)[0] - target.pos))
        pos, rot = self.wrist_pose(side)
        return float(np.linalg.norm(
            pos + rot @ _mirror(GRASP_OFFSET, side) - self._pocket_goal(side, target)))

    # -- hand --------------------------------------------------------------
    def set_hand(self, side: str, closed: float):
        """Command the fingers, ``0.0`` open to ``1.0`` closed (values between interpolate)."""
        for joint in FINGER_POSE:
            self._hand_target[:, self._hand_cols[f"{side}_{joint}"]] = \
                finger_target(joint, side, closed)

    def open_hand(self, side: str):
        self.set_hand(side, 0.0)

    def close_hand(self, side: str):
        self.set_hand(side, 1.0)

    # -- control tick ------------------------------------------------------
    def step(self):
        """Solve for the current targets and write joint targets into the articulation.

        Call once per control tick, *after* the locomotion policy's ``step`` and before
        ``scene.write_data_to_sim()``.
        """
        for side, task in self._tasks.items():
            pos, rot = self.resolve(side)
            task.set_target(pin.SE3(rot, pos))

        joint_pos = self.robot.data.joint_pos.torch[0].cpu().numpy().astype(np.float64)
        arm_target = self.controller.compute(joint_pos, self.dt)
        self.robot.set_joint_position_target_index(
            target=arm_target.unsqueeze(0), joint_ids=self.arm_ids)
        self.robot.set_joint_position_target_index(
            target=self._hand_target, joint_ids=self.hand_ids)
