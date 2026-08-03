"""Record a follow-the-robot video of a task run.

Composites four sources into one 720p frame and writes an H.264 mp4:
  * a big third-person "chase" view from a free camera driven to sit behind/above the
    robot each tick and look at it (the ``record_camera`` in ``ApartmentRecordSceneCfg``),
  * the robot's forward RGB camera, its depth, and the top-down lidar (the same panels
    the sensor GUI shows), and
  * the live occupancy grid + A* plan.

Usage (wired into ``plan_task_g1_apartment.py`` behind ``--video``): construct once with
the running ``sim``/``scene``/``skills``, call :meth:`capture` every few control ticks
(e.g. from the skills ``on_step`` hook), and :meth:`close` at the end.

Needs ``enable_cameras`` and the ``record`` sensor variant. Import only after launch.
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch

from g1sim.sensor_viz import read_sensor_images
from g1sim.task_viz import occupancy_frame

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _rgba_to_rgb(img):
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) if img.shape[-1] == 4 else img


def _label(cell, text):
    """Dark caption strip + text at the top of a panel (in place)."""
    cv2.rectangle(cell, (0, 0), (cell.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(cell, text, (5, 14), _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


class ChaseRecorder:
    def __init__(self, sim, scene, skills, path, *, goal="", fps=15,
                 width=1280, height=720, main_w=960,
                 chase_dist=3.2, chase_height=2.1, look_z=0.8, cam_key="record_camera",
                 cam_margin=0.3, min_dist=0.7):
        self.sim, self.scene, self.skills = sim, scene, skills
        self.goal = goal
        self.W, self.H, self.main_w = width, height, main_w
        self.chase_dist, self.chase_height, self.look_z = chase_dist, chase_height, look_z
        # Camera-collision pull-in: keep the cam `cam_margin` in front of any wall
        # between it and the robot (down to `min_dist`), so the robot stays in frame in
        # tight interiors. Occlusion is tested geometrically against the walls' world
        # bounding boxes (the apartment's colliders are NOT visible to PhysX scene
        # queries here, so a raycast approach silently never hits).
        self.cam_margin, self.min_dist = cam_margin, min_dist
        self.cam = scene[cam_key] if cam_key in scene.sensors else None
        self.device = sim.device
        self._walls = self._build_wall_boxes()
        self.frames = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        import imageio
        self._writer = imageio.get_writer(path, fps=fps, macro_block_size=None,
                                          codec="libx264", quality=8)
        if self.cam is None:
            print("[recorder] warning: no 'record_camera' in scene; main view will be blank")
        self.update_camera()   # aim before the first render

    # -- chase-camera control --------------------------------------------
    def _build_wall_boxes(self):
        """Gather the world-space axis-aligned bounding boxes of every wall mesh once,
        as (mins, maxs) float arrays. Used to pull the chase camera in when a wall is
        between it and the robot. Returns ``None`` if walls can't be found."""
        try:
            import isaaclab.sim as sim_utils
            from pxr import Usd, UsdGeom
            from g1sim.scene import APARTMENT_PRIM
            stage = sim_utils.get_current_stage()
            root = stage.GetPrimAtPath(APARTMENT_PRIM + "/Meshes/wall")
            if not root or not root.IsValid():
                print("[recorder] no wall scope found; chase cam won't pull in")
                return None
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                     [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            mins, maxs = [], []
            for prim in Usd.PrimRange(root):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                if rng.IsEmpty():
                    continue
                mn, mx = rng.GetMin(), rng.GetMax()
                mins.append([mn[0], mn[1], mn[2]])
                maxs.append([mx[0], mx[1], mx[2]])
            if not mins:
                return None
            print(f"[recorder] chase-cam occlusion: {len(mins)} wall boxes")
            return np.asarray(mins), np.asarray(maxs)
        except Exception as e:      # pragma: no cover - defensive
            print(f"[recorder] wall-box build failed ({e}); chase cam won't pull in")
            return None

    def _occluded_t(self, o, u, L):
        """Nearest parameter t in (eps, L] at which the ray o+u*t enters a wall box
        (vectorized slab test over all boxes), or ``None`` if the segment is clear."""
        bmin, bmax = self._walls
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / u                            # (3,); inf where u==0
            t1 = (bmin - o) * inv                    # (N,3)
            t2 = (bmax - o) * inv
            tenter = np.nanmax(np.minimum(t1, t2), axis=1)   # (N,)
            texit = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tenter <= texit) & (texit >= 0) & (tenter > 1e-3) & (tenter <= L)
        return float(tenter[hit].min()) if hit.any() else None

    def _chase_eye(self, target, h):
        """Desired camera position behind+above the robot, pulled IN to just in front of
        the nearest wall between it and the robot so the robot stays visible in tight
        interiors. Full distance when the line of sight is clear."""
        desired = np.array([target[0] - self.chase_dist * math.cos(h),
                            target[1] - self.chase_dist * math.sin(h),
                            self.chase_height])
        d = float(np.linalg.norm(desired - target))
        if self._walls is None or d < 1e-3:
            return desired
        u = (desired - target) / d
        start = 0.25                                  # skip past the robot's own footprint
        t = self._occluded_t(target + u * start, u, d - start)
        if t is None:
            return desired
        allowed = max(self.min_dist, min(d, start + t - self.cam_margin))
        return target + u * allowed

    def update_camera(self):
        """Re-aim the free camera to sit behind + above the robot and look at it,
        pulling in around occluding walls so the robot stays in frame."""
        if self.cam is None:
            return
        x, y, h = self.skills.pose()
        target = np.array([x, y, self.look_z])
        eye = self._chase_eye(target, h)
        try:
            self.cam.set_world_poses_from_view(
                torch.tensor([eye.tolist()], dtype=torch.float32, device=self.device),
                torch.tensor([target.tolist()], dtype=torch.float32, device=self.device))
        except Exception as e:      # pragma: no cover - defensive
            print(f"[recorder] chase-cam pose failed ({e}); disabling main view")
            self.cam = None

    def _main_rgb(self):
        if self.cam is None or self.cam.data.output is None or "rgb" not in self.cam.data.output:
            return None
        return self.cam.data.output["rgb"].torch[0].detach().cpu().numpy()[..., :3].astype(np.uint8)

    # -- frame assembly --------------------------------------------------
    def capture(self):
        """Grab all sources, composite, and append one video frame. Then re-aim the
        chase camera for the next render. Safe to call before sensors are ready."""
        main = self._main_rgb()
        if main is None:
            self.update_camera()
            return
        rgb, depth, lid = read_sensor_images(self.scene)
        panels = [("robot camera", _rgba_to_rgb(rgb)),
                  ("depth", _rgba_to_rgb(depth)),
                  ("lidar (top-down)", _rgba_to_rgb(lid)),
                  ("occupancy + A* plan", occupancy_frame(self.skills))]
        self._writer.append_data(self._compose(main, panels))
        self.frames += 1
        self.update_camera()

    def _compose(self, main_rgb, panels):
        frame = np.zeros((self.H, self.W, 3), np.uint8)
        frame[:, :self.main_w] = cv2.resize(main_rgb, (self.main_w, self.H))
        self._caption(frame)
        side_w = self.W - self.main_w
        n = len(panels)
        for i, (label, img) in enumerate(panels):
            y0 = i * self.H // n
            y1 = (i + 1) * self.H // n
            cell = (cv2.resize(img, (side_w, y1 - y0)) if img is not None
                    else np.full((y1 - y0, side_w, 3), 20, np.uint8))
            cell = np.ascontiguousarray(cell)
            _label(cell, label)
            frame[y0:y1, self.main_w:] = cell
        return frame

    def _caption(self, frame):
        """Translucent top bar on the main view: goal + robot room/held state."""
        x, y, _ = self.skills.pose()
        room = self.skills.smap.room_at(x, y) or "between rooms"
        held = self.skills.held.name if self.skills.held is not None else "nothing"
        line2 = f"in {room}  |  holding: {held}"
        bar = frame[:56, :self.main_w].copy()
        cv2.rectangle(bar, (0, 0), (self.main_w, 56), (0, 0, 0), -1)
        cv2.addWeighted(bar, 0.5, frame[:56, :self.main_w], 0.5, 0, frame[:56, :self.main_w])
        cv2.putText(frame, f"GOAL: {self.goal}"[:78], (10, 22), _FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, line2[:78], (10, 44), _FONT, 0.5, (180, 220, 255), 1, cv2.LINE_AA)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            print(f"[recorder] wrote {self.frames} frames -> {self.path}")
