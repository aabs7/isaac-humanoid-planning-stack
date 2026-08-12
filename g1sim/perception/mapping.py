"""Sensor-based 2D occupancy mapping (no USD ground truth).

Builds a top-down occupancy grid by fusing the robot's sensors, transformed into
the world using the (in sim: ground-truth) sensor pose -- i.e. mapping with known
poses, the mapping half of SLAM. On real hardware the pose would come from SLAM
localization instead; nothing here reads the apartment USD's geometry.

Two sensor sources, both giving world-frame 3D points:
  * depth camera  -> sees walls AND furniture (the obstacles A* must avoid)
  * 3D lidar      -> 360 deg wall returns, fills in beyond the camera's FOV

Points are dropped to a height band (to ignore floor/ceiling) and counted into
grid cells; a cell seen enough times becomes 'occupied'.
"""

from __future__ import annotations

import numpy as np


class OccupancyGridMapper:
    def __init__(self, bounds=(-1.0, -6.0, 16.0, 6.0), res=0.05,
                 occ_band=(0.15, 1.8), hit_thresh=2):
        self.xmin, self.ymin, self.xmax, self.ymax = bounds
        self.res = res
        self.W = int(np.ceil((self.xmax - self.xmin) / res))
        self.H = int(np.ceil((self.ymax - self.ymin) / res))
        self.counts = np.zeros((self.H, self.W), np.int32)
        self.z_lo, self.z_hi = occ_band
        self.hit_thresh = hit_thresh

    # -- coordinate helpers (row i <-> world y, col j <-> world x) ---------
    def world_to_cell(self, x, y):
        return int((y - self.ymin) / self.res), int((x - self.xmin) / self.res)

    def cell_to_world(self, i, j):
        return self.xmin + (j + 0.5) * self.res, self.ymin + (i + 0.5) * self.res

    def in_bounds(self, i, j):
        return 0 <= i < self.H and 0 <= j < self.W

    # -- fusion ------------------------------------------------------------
    def integrate(self, points_world, sensor_xyz=None, min_range=0.5):
        """Add world-frame points (N,3). Points outside the height band, too close
        to the sensor (likely self-hits), or out of grid bounds are dropped."""
        p = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        if p.size == 0:
            return
        p = p[np.isfinite(p).all(axis=1)]
        p = p[(p[:, 2] >= self.z_lo) & (p[:, 2] <= self.z_hi)]
        if sensor_xyz is not None and min_range > 0 and len(p):
            d = np.linalg.norm(p - np.asarray(sensor_xyz, np.float32), axis=1)
            p = p[d > min_range]
        if len(p) == 0:
            return
        j = ((p[:, 0] - self.xmin) / self.res).astype(int)
        i = ((p[:, 1] - self.ymin) / self.res).astype(int)
        ok = (i >= 0) & (i < self.H) & (j >= 0) & (j < self.W)
        np.add.at(self.counts, (i[ok], j[ok]), 1)

    def occupied(self):
        """Boolean grid: True where a cell has been hit at least ``hit_thresh`` times."""
        return self.counts >= self.hit_thresh

    # -- persistence --------------------------------------------------------
    # A map is expensive to build (a walking lap of the apartment) and cheap to store, and
    # several things want one without a simulator: reachability analysis, offline planning,
    # comparing two runs. Note what a saved map does *not* record: a cell with zero hits is
    # "never seen" and "seen and empty" alike -- this grid has no observed/unobserved
    # distinction. Navigation exploits that deliberately (unobserved space is treated as
    # free, which is what makes the optimistic online planner work), but any analysis that
    # cares about where the robot could legally *stand* should pair a loaded map with a
    # second constraint -- a room polygon test, say -- or it will happily pick a stance in
    # a region the lidar never reached.
    def save(self, path: str) -> None:
        np.savez_compressed(
            path, counts=self.counts, res=self.res, hit_thresh=self.hit_thresh,
            bounds=np.array([self.xmin, self.ymin, self.xmax, self.ymax]),
            occ_band=np.array([self.z_lo, self.z_hi]))

    @classmethod
    def load(cls, path: str) -> "OccupancyGridMapper":
        d = np.load(path)
        m = cls(bounds=tuple(float(v) for v in d["bounds"]), res=float(d["res"]),
                occ_band=tuple(float(v) for v in d["occ_band"]),
                hit_thresh=int(d["hit_thresh"]))
        m.counts = d["counts"]
        return m

    def describe(self) -> str:
        occ = self.occupied()
        return (f"OccupancyGrid {self.W}x{self.H} @ {self.res:.3f} m "
                f"({self.xmin:.1f},{self.ymin:.1f})-({self.xmax:.1f},{self.ymax:.1f}): "
                f"{int(occ.sum())} occupied cells, "
                f"{int((self.counts > 0).sum())} cells with any return")

    def save_png(self, path: str, *, free=None) -> None:
        """Render the map to a PNG: black where occupied, white where not.

        ``free`` optionally shades a boolean mask of *traversable* cells -- pass a planner's
        inflated free grid (``plan_path``'s second return, or ``RobotSkills.last_free``) and
        everything outside it is drawn grey, giving the familiar three-tone picture of what
        A* could route through.

        The distinction is worth keeping visible rather than baking in. Inflation is a
        function of the robot's *radius*, applied when planning; a stored map that already
        had it applied would freeze one particular body into the environment, and would be
        wrong for a different robot, or the same robot carrying something wide. So the map
        is the map, and clearance is an overlay the caller opts into.

        cv2 is imported lazily so this module stays cheap to import and free of a rendering
        dependency for anyone who only wants to fuse points.
        """
        import cv2

        img = np.full((self.H, self.W, 3), 255, np.uint8)
        if free is not None:
            img[~free] = (180, 180, 180)
        img[self.occupied()] = (30, 30, 30)
        # Row 0 is ymin, so flip to put north up in the image.
        cv2.imwrite(path, np.flipud(np.ascontiguousarray(img)))
        print(f"[map] saved -> {path}")
