"""Point-to-point navigation for the G1: a closed-loop "go to (x, y)" controller
layered on top of :class:`g1sim.sim.locomotion.G1LocomotionPolicy`.

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

    def command_facing(self, target_xy, tol=0.06):
        """Return ``(command, yaw_err, aligned)`` for turning in place to face a point.

        Arriving at a waypoint says nothing about which way the robot ends up pointing --
        walking north to a table that stands to the east leaves it facing the wrong way. Any
        manipulation that follows needs the target in front of the chest, so this is a step
        of its own rather than something ``command_to`` could fold in.
        """
        x, y, heading = self.pose2d()
        yaw_err = math_utils.wrap_to_pi(
            torch.tensor(math.atan2(target_xy[1] - y, target_xy[0] - x) - heading)).item()
        if abs(yaw_err) < tol:
            return self.c.command(), yaw_err, True
        wz = max(-self.wz_max, min(self.wz_max, self.kp_yaw * yaw_err))
        # Keep turning at a usable rate: below ~0.3 rad/s this gait barely rotates at all.
        wz = math.copysign(max(abs(wz), 0.3), wz)
        return self.c.command_from_se2(torch.tensor([[0.0, 0.0, wz]], device=self.device)), yaw_err, False

    def command_station(self, stand_xy, face_xy, pos_tol=0.06, yaw_tol=0.10,
                        v_min=0.35, v_max=0.6):
        """Return ``(command, distance, on_station, )`` for holding a spot while facing a point.

        A manipulating robot has to *stay* put, and a balancing humanoid does not: reaching
        out over a counter shifts its weight and the policy answers by stepping back, tens of
        centimetres over a few seconds of arm motion. So the standing spot gets closed-loop
        control of its own, not a one-off arrival.

        Two things this deliberately does not do. It does not hold a *radius* from the work
        point -- that reads as "get closer" when the robot is off to one side, which walks it
        straight into the counter. And it does not turn to walk: the error is resolved into
        the robot's own frame and corrected with a sideways velocity, so the work stays in
        front of the chest the whole time.

        Corrections are deadbanded *and* floored, which is not the usual pairing. A
        proportional command dies away as the error shrinks, and below roughly 0.3 m/s this
        gait stops translating at all -- it marks time on the spot, indefinitely, while the
        command insists it is walking. So a correction outside the deadband is issued at a
        speed the legs actually honour, and inside it, at nothing.
        """
        x, y, heading = self.pose2d()
        dx, dy = stand_xy[0] - x, stand_xy[1] - y
        dist = math.hypot(dx, dy)
        yaw_err = math_utils.wrap_to_pi(
            torch.tensor(math.atan2(face_xy[1] - y, face_xy[0] - x) - heading)).item()
        if dist < pos_tol and abs(yaw_err) < yaw_tol:
            return self.c.command(), dist, True

        cos_h, sin_h = math.cos(heading), math.sin(heading)
        vx = vy = 0.0
        if dist >= pos_tol:
            speed = min(v_max, max(v_min, self.kp_lin * dist)) / dist
            vx, vy = speed * (cos_h * dx + sin_h * dy), speed * (-sin_h * dx + cos_h * dy)
        wz = 0.0 if abs(yaw_err) < yaw_tol else \
            max(-self.wz_max, min(self.wz_max, self.kp_yaw * yaw_err))
        return self.c.command_from_se2(torch.tensor([[vx, vy, wz]], device=self.device)), dist, False

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
