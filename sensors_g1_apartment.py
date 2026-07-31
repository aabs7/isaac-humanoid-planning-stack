# Teleoperate a G1 (lidar + RGB-D camera) around the apartment while watching
# three live sensor windows in the Isaac Sim GUI:
#   * "G1 Camera (RGB)"   -- the robot's-eye color image
#   * "G1 Camera (Depth)" -- colormapped depth
#   * "G1 Lidar (top-down)"-- the 3D lidar hits, sensor-centered
# The lidar point cloud also draws in the main 3D viewport (debug_vis).
#
# Interactive (GUI + keyboard):
#   cd ~/isaac && source .venv/bin/activate
#   python isaac_task_planning/sensors_g1_apartment.py
#   Keys (focus main window): Up/Down move, Left/Right strafe, Z/X turn, L stop.
#
# Headless snapshot (saves the three images to --outdir and exits):
#   python isaac_task_planning/sensors_g1_apartment.py --headless --snapshot 80

import os

from g1sim.launch import make_parser, launch

parser = make_parser("Teleop a G1 with lidar + RGB-D camera and 3 live views.")
parser.add_argument("--snapshot", type=int, default=0, metavar="N",
                    help="Headless: after N control steps, save the 3 images to "
                         "--outdir and exit.")
parser.add_argument("--outdir", type=str,
                    default="/home/abhish/isaac/isaac_task_planning/sensor_output",
                    help="Where snapshot images are written.")
args = parser.parse_args()

# Camera sensors require the rendering pipeline.
args.enable_cameras = True
simulation_app = launch(args)

# ---- Imports that require the running sim app ----
import cv2
import numpy as np

from g1sim.scene import build_world
from g1sim.locomotion import G1LocomotionPolicy

DEPTH_MAX = 5.0     # depth colormap saturation range (m); tuned for indoor contrast
LIDAR_RANGE = 8.0   # half-extent of the top-down lidar window (m)


# ---------------------------------------------------------------------------
# Turn sensor readings into RGBA uint8 images (H, W, 4)
# ---------------------------------------------------------------------------
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
    """Return (rgb, depth, lidar) RGBA images, or None for any not yet ready."""
    cam = scene["depth_camera"].data
    lidar = scene["lidar"].data
    rgb = depth = lid = None
    if cam.output is not None and "rgb" in cam.output:
        rgb = rgb_to_rgba(cam.output["rgb"].torch[0].detach().cpu().numpy())
    if cam.output is not None and "distance_to_image_plane" in cam.output:
        depth = depth_to_rgba(cam.output["distance_to_image_plane"].torch[0].detach().cpu().numpy().squeeze())
    if lidar.ray_hits_w is not None:
        lid = lidar_to_rgba(lidar.ray_hits_w.torch[0].detach().cpu().numpy(),
                            lidar.pos_w.torch[0].detach().cpu().numpy())
    return rgb, depth, lid


# ---------------------------------------------------------------------------
# Live GUI windows backed by dynamic byte-image providers
# ---------------------------------------------------------------------------
class SensorWindows:
    def __init__(self):
        import omni.ui as ui
        self.ui = ui
        specs = [("G1 Camera (RGB)", 10), ("G1 Camera (Depth)", 360), ("G1 Lidar (top-down)", 710)]
        self.providers, self.windows = {}, []
        for title, x in specs:
            prov = ui.ByteImageProvider()
            win = ui.Window(title, width=330, height=290)
            win.position_x, win.position_y = x, 10
            with win.frame:
                ui.ImageWithProvider(prov)
            self.providers[title] = prov
            self.windows.append(win)

    def update(self, images):
        for title, img in zip(self.providers, images):
            if img is not None:
                h, w = img.shape[:2]
                self.providers[title].set_bytes_data(list(img.tobytes()), [w, h])


def save_snapshot(scene, outdir):
    os.makedirs(outdir, exist_ok=True)
    rgb, depth, lid = read_sensor_images(scene)
    if rgb is not None:
        cv2.imwrite(os.path.join(outdir, "camera_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR))
    if depth is not None:
        cv2.imwrite(os.path.join(outdir, "camera_depth.png"), cv2.cvtColor(depth, cv2.COLOR_RGBA2BGR))
    if lid is not None:
        cv2.imwrite(os.path.join(outdir, "lidar_topdown.png"), cv2.cvtColor(lid, cv2.COLOR_RGBA2BGR))
    print(f"[snapshot] wrote camera_rgb.png, camera_depth.png, lidar_topdown.png -> {outdir}")


# ---------------------------------------------------------------------------
def main():
    sim, scene = build_world(args.spawn, device=args.device, with_sensors=True)
    robot = scene["robot"]
    controller = G1LocomotionPolicy(robot, sim.device)
    print(f"[g1] spawned at {tuple(args.spawn)} with lidar + RGB-D camera")

    # GUI-only: keyboard teleop + three live sensor windows.
    keyboard, windows = None, None
    if not args.snapshot:
        from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg
        keyboard = Se2Keyboard(Se2KeyboardCfg(
            v_x_sensitivity=0.8, v_y_sensitivity=0.5, omega_z_sensitivity=1.0,
            sim_device=str(sim.device)))
        windows = SensorWindows()
        print("[teleop] Arrows=move, Z/X=turn, L=stop. Watch the 3 sensor windows.")

    sim_dt = sim.get_physics_dt()
    ctrl_step = 0

    while simulation_app.is_running():
        command = controller.command_from_se2(keyboard.advance()) if keyboard else controller.command()
        controller.step(command)
        for _ in range(controller.decimation):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        ctrl_step += 1

        # Refresh the live windows at ~camera rate (every 5th control tick = 10 Hz).
        if windows is not None and ctrl_step % 5 == 0:
            windows.update(read_sensor_images(scene))

        if args.snapshot and ctrl_step >= args.snapshot:
            save_snapshot(scene, args.outdir)
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
