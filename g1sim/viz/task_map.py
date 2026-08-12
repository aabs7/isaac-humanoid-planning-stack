"""Live occupancy-map + A* plan visualization for the task entry points (GUI runs).

Shared by both the scripted (``scripts/task_g1_apartment.py``) and LLM-planned
(``scripts/plan_task_g1_apartment.py``) drivers. Kept in its own module (with no top-level
argument parsing) so an entry point can import these helpers *without* re-triggering
another script's ``parse_args()``/``launch()``. All heavy imports (omni.ui, cv2,
numpy) are lazy inside the methods, so importing this module is cheap and safe.
"""

from __future__ import annotations


class MapWindow:
    """Live occupancy-map + A* plan window (GUI runs only), fed from skill state."""

    def __init__(self, title="Task: Occupancy Map + A* plan", width=740, height=560):
        import omni.ui as ui
        self.prov = ui.ByteImageProvider()
        self.win = ui.Window(title, width=width, height=height)
        with self.win.frame:
            ui.ImageWithProvider(self.prov)

    def update(self, skills):
        import cv2
        img = occupancy_frame(skills)     # RGB uint8, or None if map not ready
        if img is None:
            return
        rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        h, w = rgba.shape[:2]
        self.prov.set_bytes_data(list(rgba.tobytes()), [w, h])


def occupancy_frame(skills, scale: int = 2):
    """Render the occupancy grid + A* plan + robot/goal/held markers to an RGB uint8
    image (display-oriented, +y up), or ``None`` if the map isn't built yet. Shared by
    the live GUI window and the video recorder so both draw the same picture."""
    import cv2
    import numpy as np
    mapper = skills.mapper
    if mapper is None or skills.last_free is None:
        return None
    free = skills.last_free
    occ = mapper.occupied()
    img = np.full((mapper.H, mapper.W, 3), 255, np.uint8)
    img[~free] = (180, 180, 180)
    img[occ] = (30, 30, 30)
    img = cv2.resize(img, (mapper.W * scale, mapper.H * scale), interpolation=cv2.INTER_NEAREST)

    def px(x, y):
        i, j = mapper.world_to_cell(x, y)
        return int(j * scale), int(i * scale)

    wps = skills.last_waypoints or []
    if len(wps) >= 2:
        for a, b in zip(wps[:-1], wps[1:]):
            cv2.line(img, px(*a), px(*b), (0, 160, 0), 2)
    rx, ry = skills.xy()
    cv2.circle(img, px(rx, ry), 5, (0, 90, 255), -1)
    if skills.last_goal:
        cv2.drawMarker(img, px(*skills.last_goal), (220, 0, 0), cv2.MARKER_STAR, 16, 2)
    if skills.held is not None:
        cv2.circle(img, px(skills.held.xy[0], skills.held.xy[1]), 4, (200, 0, 200), -1)
    return np.flipud(np.ascontiguousarray(img))


def throttled(fn, every=8):
    """Wrap a per-step callback so it fires only every ``every`` calls."""
    state = {"i": 0}

    def wrapped(s):
        state["i"] += 1
        if state["i"] % every == 0:
            fn(s)
    return wrapped
