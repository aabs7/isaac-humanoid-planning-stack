"""Point-to-point navigation for the G1: a closed-loop "go to (x, y)" controller
layered on top of :class:`g1sim.locomotion.G1LocomotionPolicy`.

It's a unicycle controller: read the robot's world pose + heading, turn to face
the goal, walk forward, and slow down as it arrives. The heading is derived from
the base orientation via ``quat_apply`` (not assumed to be zero) because the
G1's spawn orientation actually points its forward axis along world -x.

Import only after the sim app is launched (pulls in ``isaaclab.utils.math``).
"""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils


class WaypointNavigator:
    """Emits ``[vx, vy, wz]`` velocity commands that drive the robot to a goal.

    Usage each control tick::

        cmd, dist, arrived = nav.command_to((gx, gy))
        controller.step(cmd)
        # ... step the sim `controller.decimation` times ...
    """

    def __init__(self, controller, *, goal_tol=0.25, vx_max=0.6, wz_max=1.0,
                 kp_lin=1.5, kp_yaw=1.5, face_first=0.6):
        self.c = controller
        self.robot = controller.robot
        self.device = controller.device
        self.goal_tol = goal_tol      # arrival radius (m)
        self.vx_max = vx_max          # max forward speed command (m/s)
        self.wz_max = wz_max          # max turn-rate command (rad/s)
        self.kp_lin = kp_lin          # forward gain (slows within vx_max/kp_lin of goal)
        self.kp_yaw = kp_yaw          # heading gain
        self.face_first = face_first  # |yaw err| (rad) above which we turn in place
        self._fwd = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)

    def pose2d(self):
        """Return the robot base pose in world as ``(x, y, heading)`` (heading rad)."""
        p = self.robot.data.root_pos_w.torch[0, :2]
        q = self.robot.data.root_quat_w.torch[0:1]        # (1, 4), wxyz
        f = math_utils.quat_apply(q, self._fwd)[0]        # robot +x axis in world
        heading = math.atan2(f[1].item(), f[0].item())
        return p[0].item(), p[1].item(), heading

    def command_to(self, goal_xy):
        """Return ``(command, distance, arrived)`` for the current pose and goal.

        ``command`` is a ``(num_envs, 4)`` tensor ready for
        :meth:`G1LocomotionPolicy.step`. When within ``goal_tol`` it is the
        stand-in-place command and ``arrived`` is True."""
        x, y, heading = self.pose2d()
        dx, dy = goal_xy[0] - x, goal_xy[1] - y
        dist = math.hypot(dx, dy)
        if dist < self.goal_tol:
            return self.c.command(), dist, True

        yaw_err = math_utils.wrap_to_pi(torch.tensor(math.atan2(dy, dx) - heading)).item()
        wz = max(-self.wz_max, min(self.wz_max, self.kp_yaw * yaw_err))

        # Turn in place if badly misaligned; otherwise walk forward, throttled by
        # how well we face the goal and slowed as we approach.
        if abs(yaw_err) > self.face_first:
            vx = 0.0
        else:
            vx = min(self.vx_max, self.kp_lin * dist) * math.cos(yaw_err)

        se2 = torch.tensor([[vx, 0.0, wz]], device=self.device)
        return self.c.command_from_se2(se2), dist, False
