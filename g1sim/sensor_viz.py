"""Turn the robot's sensor readings into displayable images (RGBA uint8).

Shared by the sensor teleop script and the video recorder. No top-level argument
parsing (unlike the entry scripts), so it is safe to import from anywhere after the
sim app is up. ``cv2``/``numpy`` only -- no Isaac imports at module load.
"""

from __future__ import annotations

import cv2
import numpy as np

DEPTH_MAX = 5.0     # depth colormap saturation range (m); tuned for indoor contrast
LIDAR_RANGE = 8.0   # half-extent of the top-down lidar window (m)


def rgb_to_rgba(rgb: np.ndarray) -> np.ndarray:
    a = rgb[..., :3].astype(np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_RGB2RGBA)


def depth_to_rgba(depth: np.ndarray) -> np.ndarray:
    d = np.nan_to_num(depth, nan=DEPTH_MAX, posinf=DEPTH_MAX, neginf=DEPTH_MAX)
    d = np.clip(d, 0.0, DEPTH_MAX)
    u = (255.0 * (1.0 - d / DEPTH_MAX)).astype(np.uint8)  # near = warm/bright
    bgr = cv2.applyColorMap(u, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)


def lidar_to_rgba(hits: np.ndarray, origin: np.ndarray, size: int = 320) -> np.ndarray:
    """Top-down, sensor-centered rasterization of the lidar hits (world frame)."""
    img = np.full((size, size, 4), 25, np.uint8)   # dark background
    img[..., 3] = 255
    d = np.linalg.norm(hits - origin, axis=-1)
    m = np.isfinite(d) & (d < LIDAR_RANGE)
    p = hits[m]
    if p.shape[0]:
        px = (((p[:, 0] - origin[0]) / LIDAR_RANGE) * 0.5 + 0.5) * size
        py = (((p[:, 1] - origin[1]) / LIDAR_RANGE) * 0.5 + 0.5) * size
        px = np.clip(px.astype(int), 0, size - 1)
        py = np.clip((size - 1) - py.astype(int), 0, size - 1)  # +y up
        z = p[:, 2]
        zc = np.clip((z - z.min()) / (np.ptp(z) + 1e-6) * 255, 0, 255).astype(np.uint8)
        col = cv2.applyColorMap(zc.reshape(-1, 1), cv2.COLORMAP_VIRIDIS)[:, 0, :]  # BGR
        img[py, px, 0] = col[:, 2]; img[py, px, 1] = col[:, 1]; img[py, px, 2] = col[:, 0]
    c = size // 2                                   # sensor marker (red)
    img[c - 2:c + 3, c - 2:c + 3, :3] = (255, 0, 0)
    return img


def read_sensor_images(scene):
    """Return (rgb, depth, lidar) RGBA images (channels in RGB order), or ``None`` for
    any sensor whose data is not yet ready / not present in the scene."""
    rgb = depth = lid = None
    if "depth_camera" in scene.sensors:
        cam = scene["depth_camera"].data
        if cam.output is not None and "rgb" in cam.output:
            rgb = rgb_to_rgba(cam.output["rgb"].torch[0].detach().cpu().numpy())
        if cam.output is not None and "distance_to_image_plane" in cam.output:
            depth = depth_to_rgba(
                cam.output["distance_to_image_plane"].torch[0].detach().cpu().numpy().squeeze())
    if "lidar" in scene.sensors:
        lidar = scene["lidar"].data
        if lidar.ray_hits_w is not None:
            lid = lidar_to_rgba(lidar.ray_hits_w.torch[0].detach().cpu().numpy(),
                                lidar.pos_w.torch[0].detach().cpu().numpy())
    return rgb, depth, lid
