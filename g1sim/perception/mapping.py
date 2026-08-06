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
