# Spawn a Unitree G1 humanoid in the kujiale apartment using an IsaacLab
# InteractiveScene. The scene is declared once, up front, in ApartmentSceneCfg --
# so adding sensors later (3D lidar, cameras on the robot) is just a matter of
# adding a RayCasterCfg / CameraCfg field to that config, with no changes to the
# spawn/step plumbing below.
#
# Run with:
#   cd ~/isaac && source .venv/bin/activate
#   python isaac_task_planning/spawn_g1_apartment.py
#
# By default this opens the Isaac Sim (Kit) GUI window. This IsaacLab build is
# headless-by-default and needs '--viz kit' to show a window; the script sets
# that for you on interactive runs.
#
# Optional flags:
#   --spawn X Y   floor spawn location in meters (default: open living-room floor)
#   --headless    run without any GUI
#   --smoke N     step N frames headless, report base pose, then exit
#   --viz ...     override the visualizer explicitly (e.g. --viz none)

import argparse

from isaaclab.app import AppLauncher

# ---- CLI + app launch (must run before any omni/pxr/isaaclab.sim import) ----
parser = argparse.ArgumentParser(description="Spawn Unitree G1 in the apartment.")
parser.add_argument("--spawn", type=float, nargs=2, default=[7.51, 0.08],
                    metavar=("X", "Y"),
                    help="X Y spawn location on the floor (meters). Default: open "
                         "living-room floor (~1.2 m clearance from all furniture).")
parser.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="Smoke test: step N frames, report base pose, then exit "
                         "(0 = run interactively until the window is closed).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# GUI by default for interactive runs: this IsaacLab is headless unless a Kit
# visualizer is explicitly requested. Skip when headless, in smoke mode, or when
# the user already chose a visualizer via --viz/--visualizer.
if not args.headless and not args.smoke and getattr(args, "visualizer", None) is None:
    args.visualizer = ["kit"]

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- Imports that require the running sim app ----
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.unitree import G1_CFG

from pxr import Usd, UsdGeom, UsdPhysics

APARTMENT_USD = "/home/abhish/isaac/InteriorAgent/kujiale_0021/kujiale_0021.usda"
APARTMENT_PRIM = "/World/Apartment"


# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------
@configclass
class ApartmentSceneCfg(InteractiveSceneCfg):
    """The apartment world plus a Unitree G1.

    Everything the robot perceives/acts in lives here. To give the robot a 3D
    lidar or a camera later, add a sensor field next to ``robot`` pointing at
    a body under ``{ENV_REGEX_NS}/Robot/...`` -- e.g.::

        lidar = RayCasterCfg(prim_path="{ENV_REGEX_NS}/Robot/torso_link", ...)
        camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/head_link/front_cam", ...)
    """

    # Apartment: a single shared, static instance (absolute path, not per-env).
    apartment = AssetBaseCfg(
        prim_path=APARTMENT_PRIM,
        spawn=sim_utils.UsdFileCfg(usd_path=APARTMENT_USD),
    )

    # Lighting comes entirely from the apartment USD (a DistantLight "sun", a
    # DomeLight, and per-room fixtures), so we add no lights of our own.

    # Unitree G1 humanoid (per-env namespace so sensors/robots clone cleanly).
    robot: ArticulationCfg = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_apartment_static(stage):
    """The apartment ships with colliders, but every object -- floor, walls and
    ceiling included -- is authored as a *dynamic* rigid body. That is wrong for
    structural geometry and makes the floor/wall triangle-mesh colliders illegal on
    a dynamic body (PhysX errors + convex-hull fallback). Disabling the rigid bodies
    turns their existing colliders into *static* colliders -- exactly what a fixed
    scene needs. No new colliders are added, so there are no duplicate/merge
    conflicts. (For pick-and-place later, re-enable RigidBodyAPI on target objects.)"""
    disabled = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(APARTMENT_PRIM)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            disabled += 1
    print(f"[apartment] disabled {disabled} rigid bodies -> static colliders")


def hide_ceiling(stage):
    """Make the ceiling invisible so the interior is visible from above/outside.
    We hide (not delete) it: the per-room ceiling lights live under the room
    scopes, not under this 'ceiling' scope, so the rooms stay lit. Note this only
    affects rendering -- a ray-cast lidar would still hit the ceiling mesh; use
    prim.SetActive(False) instead if you need it gone from physics too."""
    hidden = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(APARTMENT_PRIM)):
        if prim.GetName() == "ceiling":
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
    print(f"[apartment] hid {hidden} ceiling scope(s) for interior visibility")


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # Hold the default standing pose: PD position targets = default joint positions.
    default_joint_pos = robot.data.default_joint_pos.torch.clone()
    start = robot.data.root_pos_w.torch[0, :3].tolist()

    step = 0
    while simulation_app.is_running():
        robot.set_joint_position_target_index(target=default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1
        if args.smoke and step >= args.smoke:
            final = robot.data.root_pos_w.torch[0, :3].tolist()
            xy_drift = ((final[0] - start[0]) ** 2 + (final[1] - start[1]) ** 2) ** 0.5
            print(f"[smoke] stepped {step} frames")
            print(f"[smoke] base z: {start[2]:.3f} -> {final[2]:.3f} m   "
                  f"xy drift: {xy_drift:.3f} m")
            verdict = "held upright" if final[2] > 0.55 else "collapsed (no balance controller yet)"
            print(f"[smoke] {verdict}")
            break


def main():
    # --- simulation context (Z-up world, gravity on) ---
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 200.0, device=args.device)
    sim = SimulationContext(sim_cfg)
    x, y = args.spawn
    sim.set_camera_view(eye=[x - 4.0, y - 4.0, 2.5], target=[x, y, 0.8])

    # --- build the scene from the config ---
    scene_cfg = ApartmentSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot.init_state.pos = (x, y, 0.74)  # 0.74 m base height above floor
    scene = InteractiveScene(scene_cfg)

    # Fix the apartment's authored physics now that its prims exist on the stage.
    stage = sim_utils.get_current_stage()
    make_apartment_static(stage)
    hide_ceiling(stage)

    # --- start the simulation ---
    sim.reset()
    print(f"[g1] spawned at floor (x={x}, y={y}, z=0.74)")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
