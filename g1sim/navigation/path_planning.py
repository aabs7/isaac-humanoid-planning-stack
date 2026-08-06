"""A* grid path planning on an occupancy map, with robot-radius obstacle
inflation and line-of-sight path simplification.

Works purely on the ``OccupancyGridMapper`` grid the robot built from its
sensors -- unknown cells are treated as traversable (optimistic), only sensed
obstacles (inflated by the robot radius) block the plan.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

SQRT2 = 2.0 ** 0.5
_NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def inflate(occupied: np.ndarray, radius_cells: int) -> np.ndarray:
    """Grow obstacles by a disk of ``radius_cells`` so a point-robot plan keeps
    the real robot's body clear of them."""
    if radius_cells <= 0:
        return occupied.copy()
    r = int(radius_cells)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx * xx + yy * yy) <= r * r
    return binary_dilation(occupied, structure=disk)


def nearest_free(free: np.ndarray, cell):
    """Snap ``cell`` to the closest traversable cell (start/goal may land in an
    inflated obstacle or unmapped noise)."""
    i, j = cell
    if free[i, j]:
        return cell
    _, (ii, jj) = distance_transform_edt(~free, return_indices=True)
    return int(ii[i, j]), int(jj[i, j])


def astar(free: np.ndarray, start, goal):
    """8-connected A* on a boolean ``free`` grid. Returns a list of (i, j) cells
    from start to goal, or None if unreachable. Diagonal moves may not cut
    through obstacle corners."""
    H, W = free.shape
    if not (free[start] and free[goal]):
        return None

    def h(c):
        return ((c[0] - goal[0]) ** 2 + (c[1] - goal[1]) ** 2) ** 0.5

    open_heap = [(h(start), 0.0, start)]
    g = {start: 0.0}
    came = {}
    while open_heap:
        _, gc, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if gc > g.get(cur, 1e18):
            continue
        ci, cj = cur
        for di, dj, cost in _NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < H and 0 <= nj < W) or not free[ni, nj]:
                continue
            if di != 0 and dj != 0 and not (free[ci + di, cj] and free[ci, cj + dj]):
                continue  # don't cut obstacle corners
            ng = gc + cost
            if ng < g.get((ni, nj), 1e18):
                g[(ni, nj)] = ng
                came[(ni, nj)] = cur
                heapq.heappush(open_heap, (ng + h((ni, nj)), ng, (ni, nj)))
    return None


def _line_clear(free, a, b):
    """Bresenham line-of-sight test: are all cells between a and b traversable?"""
    (i0, j0), (i1, j1) = a, b
    di, dj = abs(i1 - i0), abs(j1 - j0)
    si = 1 if i1 > i0 else -1
    sj = 1 if j1 > j0 else -1
    err = di - dj
    i, j = i0, j0
    while True:
        if not free[i, j]:
            return False
        if (i, j) == (i1, j1):
            return True
        e2 = 2 * err
        if e2 > -dj:
            err -= dj
            i += si
        if e2 < di:
            err += di
            j += sj


def simplify(free, path):
    """Greedy string-pulling: keep only the cells needed so consecutive kept
    cells still have clear line of sight -- turns a dense grid path into a few
    waypoints."""
    if not path or len(path) < 3:
        return list(path) if path else path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_clear(free, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def plan_path(mapper, start_xy, goal_xy, robot_radius_m=0.35):
    """High-level: inflate the sensed obstacles, snap start/goal to free space,
    A*, simplify. Returns (waypoints_world, free_grid, info) where waypoints are
    (x, y) tuples, or ([], free_grid, info) if no path was found."""
    occ = mapper.occupied()
    free = ~inflate(occ, int(round(robot_radius_m / mapper.res)))

    start = nearest_free(free, mapper.world_to_cell(*start_xy))
    goal = nearest_free(free, mapper.world_to_cell(*goal_xy))
    cells = astar(free, start, goal)
    info = {"start_cell": start, "goal_cell": goal, "n_cells": (len(cells) if cells else 0)}
    if not cells:
        return [], free, info
    waypoints = [mapper.cell_to_world(i, j) for (i, j) in simplify(free, cells)]
    return waypoints, free, info
