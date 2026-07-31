# Optimistic online mapping + A* navigation for the G1 (no pre-mapping spin).
#
# The robot heads straight for the goal treating all *unobserved* space as free
# (optimistic). As it walks, the 3D lidar maps obstacles into a 2D occupancy grid;
# whenever a newly-sensed obstacle blocks the plan, A* re-plans around it. Repeat
# until the goal is reached. A live Isaac Sim window shows the occupancy map with
# the goal, the current A* path, and the robot.
#
#   cd ~/isaac && source .venv/bin/activate
#   python isaac_task_planning/map_and_navigate_g1_apartment.py --goal 3.5 -3.0
#   python isaac_task_planning/map_and_navigate_g1_apartment.py --headless --goal 3.5 -3.0

import os

from g1sim.launch import make_parser, launch

parser = make_parser("Optimistic online-mapping A* navigation for the G1.")
parser.add_argument("--goal", type=float, nargs=2, default=[3.5, -3.0], metavar=("X", "Y"),
                    help="Goal (x, y) in meters (default: behind the living-room coffee table).")
parser.add_argument("--outdir", type=str,
                    default="/home/abhish/isaac/isaac_task_planning/sensor_output")
args = parser.parse_args()

simulation_app = launch(args)

# ---- Imports that require the running sim app ----
import cv2
import numpy as np

from g1sim.scene import build_world, NAV_LIDAR_TARGETS
from g1sim.locomotion import G1LocomotionPolicy
from g1sim.navigation import WaypointNavigator
from g1sim.mapping import OccupancyGridMapper
from g1sim.planning import plan_path

INTEGRATE_EVERY = 3     # control ticks between lidar fusions
REPLAN_EVERY = 30       # control ticks between A* re-plans (~0.6 s) -> reacts to new obstacles
MAP_UPDATE_EVERY = 8    # control ticks between live-window refreshes
GOAL_TOL = 0.35
NAV_TIMEOUT_S = 150.0


def integrate_lidar(scene, mapper):
    lidar = scene["lidar"].data
    if lidar.ray_hits_w is not None:
        mapper.integrate(lidar.ray_hits_w.torch[0].detach().cpu().numpy(),
                         sensor_xyz=lidar.pos_w.torch[0].detach().cpu().numpy())


def render_map(mapper, free, robot_xy, goal_xy, path_world, scale=2):
    """Render the occupancy grid + goal + path + robot to an RGBA image (y-up)."""
    occ = mapper.occupied()
    img = np.full((mapper.H, mapper.W, 3), 255, np.uint8)
    img[~free] = (180, 180, 180)     # robot-radius inflation margin
    img[occ] = (30, 30, 30)          # sensed obstacle
    img = cv2.resize(img, (mapper.W * scale, mapper.H * scale), interpolation=cv2.INTER_NEAREST)

    def px(x, y):
        i, j = mapper.world_to_cell(x, y)
        return int(j * scale), int(i * scale)

    if path_world and len(path_world) >= 2:
        for a, b in zip(path_world[:-1], path_world[1:]):
            cv2.line(img, px(*a), px(*b), (0, 160, 0), 2)
        for w in path_world:
            cv2.circle(img, px(*w), 3, (0, 160, 0), -1)
    cv2.circle(img, px(*robot_xy), 5, (0, 90, 255), -1)                        # robot (blue)
    cv2.drawMarker(img, px(*goal_xy), (220, 0, 0), cv2.MARKER_STAR, 16, 2)     # goal (red)
    img = np.flipud(img)                                                       # +y up
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2RGBA)


class MapWindow:
    """A live Isaac Sim window backed by a dynamic byte-image provider."""
    def __init__(self, title="Occupancy Map + A* plan", width=740, height=560):
        import omni.ui as ui
        self.prov = ui.ByteImageProvider()
        self.win = ui.Window(title, width=width, height=height)
        with self.win.frame:
            ui.ImageWithProvider(self.prov)

    def update(self, rgba):
        h, w = rgba.shape[:2]
        self.prov.set_bytes_data(list(rgba.tobytes()), [w, h])


def save_png(rgba, path):
    cv2.imwrite(path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))


def step(sim, scene, controller, command):
    controller.step(command)
    for _ in range(controller.decimation):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


def main():
    goal_xy = tuple(args.goal)
    # Lidar only (no camera) -> the obstacle sensor for mapping; no enable_cameras.
    sim, scene = build_world(args.spawn, device=args.device, sensors="lidar",
                             lidar_targets=NAV_LIDAR_TARGETS)
    robot = scene["robot"]
    controller = G1LocomotionPolicy(robot, sim.device)
    nav = WaypointNavigator(controller, goal_tol=0.3)
    mapper = OccupancyGridMapper()
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[nav] optimistic start {tuple(args.spawn)} -> goal {goal_xy}")

    # Optimistic first plan on an EMPTY map: a straight shot at the goal.
    start_xy = nav.pose2d()[:2]
    waypoints, free, info = plan_path(mapper, start_xy, goal_xy)
    print(f"[plan] initial (optimistic): {len(waypoints)} waypoints")

    map_window = None if args.headless else MapWindow()

    tick = 0
    max_ticks = int(NAV_TIMEOUT_S * 50)
    n_replans = 0
    done = False
    while simulation_app.is_running() and not done:
        if tick % INTEGRATE_EVERY == 0:
            integrate_lidar(scene, mapper)

        if tick % REPLAN_EVERY == 0:
            wps, free, info = plan_path(mapper, nav.pose2d()[:2], goal_xy)
            if wps:
                waypoints = wps
                n_replans += 1

        px_, py_, _ = nav.pose2d()
        reach = waypoints[-1] if waypoints else goal_xy
        if np.hypot(reach[0] - px_, reach[1] - py_) < GOAL_TOL:
            resid = np.hypot(goal_xy[0] - px_, goal_xy[1] - py_)
            note = "" if resid < GOAL_TOL else f" (closest free point; goal {resid:.2f} m away, in/at obstacle)"
            print(f"[nav] reached ({px_:.2f}, {py_:.2f}) after {n_replans} re-plans{note}")
            done = True

        # Head toward the next waypoint; continuous re-planning handles progress.
        target = waypoints[1] if len(waypoints) > 1 else (waypoints[0] if waypoints else goal_xy)
        command, _, _ = nav.command_to(target)

        if tick % 50 == 0:
            print(f"[nav] at ({px_:.2f}, {py_:.2f}) | {int(mapper.occupied().sum())} occ cells "
                  f"| {len(waypoints)} wpts | {n_replans} replans")
        if map_window is not None and tick % MAP_UPDATE_EVERY == 0:
            map_window.update(render_map(mapper, free, (px_, py_), goal_xy, waypoints))

        if not done:
            step(sim, scene, controller, command)
        tick += 1
        if tick >= max_ticks:
            print("[nav] TIMEOUT (stuck?)")
            break

    save_png(render_map(mapper, free, nav.pose2d()[:2], goal_xy, waypoints),
             os.path.join(args.outdir, "map_result.png"))
    print(f"[map] saved {os.path.join(args.outdir, 'map_result.png')}")

    if not args.headless:   # keep window up, standing at the goal
        while simulation_app.is_running():
            if map_window is not None:
                px_, py_, _ = nav.pose2d()
                map_window.update(render_map(mapper, free, (px_, py_), goal_xy, waypoints))
            step(sim, scene, controller, controller.command())


if __name__ == "__main__":
    main()
    simulation_app.close()
