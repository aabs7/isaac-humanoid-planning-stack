"""The apartment world plus a Unitree G1 -- declared once, reused everywhere.

Import only after the sim app is launched (pulls in ``isaaclab.sim``).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.sim import SimulationContext
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from pxr import Usd, UsdGeom, UsdPhysics

APARTMENT_USD = "/home/abhish/isaac/InteriorAgent/kujiale_0021/kujiale_0021.usda"
APARTMENT_PRIM = "/World/Apartment"

# Base height (m) at which the G1 root spawns above a z=0 floor.
ROBOT_SPAWN_Z = 0.75

# Body link the sensors mount on. The G1 has no head link; torso_link is the chest.
SENSOR_MOUNT_LINK = "torso_link"


@configclass
class ApartmentSceneCfg(InteractiveSceneCfg):
    """Apartment (static reference geometry) + a Unitree G1 (29-DOF).

    The robot is the 29-DOF G1 because that is what the pretrained locomotion
    policy in :mod:`g1sim.locomotion` was trained on. Lighting comes entirely
    from the apartment USD (a DistantLight "sun", a DomeLight, and per-room
    fixtures), so we add none of our own.

    To give the robot sensors later, add a field pointing under the robot, e.g.::

        lidar  = RayCasterCfg(prim_path="{ENV_REGEX_NS}/Robot/torso_link", ...)
        camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/head_link/front_cam", ...)
    """

    apartment = AssetBaseCfg(
        prim_path=APARTMENT_PRIM,
        spawn=sim_utils.UsdFileCfg(usd_path=APARTMENT_USD),
    )
    robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ApartmentSensorsSceneCfg(ApartmentSceneCfg):
    """Same world, with a 3D lidar and a depth+RGB camera mounted on the robot's
    torso. Kept as a subclass so the light locomotion demos don't pay the sensor
    (and camera-rendering) cost unless they ask for it.

    Cameras require the app to be launched with ``enable_cameras=True``.
    """

    # 32-channel spinning 3D lidar, raycast against the whole (static) apartment.
    # MultiMesh (not plain RayCaster): plain RayCaster only supports a single mesh
    # prim, whereas the apartment is ~1354 meshes that must be gathered + merged.
    lidar = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + SENSOR_MOUNT_LINK,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)),  # ~head height above torso
        # Raycast against the structural shell (walls + floor). These are low-poly,
        # so the warp mesh builds fast; merging *all* ~1354 apartment meshes (incl.
        # high-poly clutter like books/cups) costs minutes + GBs at startup. Add
        # furniture scopes here later if the robot needs to sense them.
        mesh_prim_paths=[
            APARTMENT_PRIM + "/Meshes/wall",
            APARTMENT_PRIM + "/Meshes/floor",
        ],
        ray_alignment="yaw",                                  # keep scan level as torso pitches
        pattern_cfg=patterns.LidarPatternCfg(
            channels=32,
            vertical_fov_range=(-15.0, 15.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=2.0,                               # 180 pts/ring x 32 = 5760 rays
        ),
        max_distance=15.0,
        debug_vis=True,                                       # draws the point cloud in the viewport
    )

    # Forward-facing RGB-D camera.
    depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + SENSOR_MOUNT_LINK + "/front_camera",
        update_period=0.1,   # 10 Hz
        height=240,
        width=320,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.15, 0.0, 0.1), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )


def make_scene_cfg(spawn_xy, cls=ApartmentSceneCfg, num_envs: int = 1, env_spacing: float = 2.0):
    """Build a scene cfg of type ``cls`` with the robot placed at ``spawn_xy``."""
    cfg = cls(num_envs=num_envs, env_spacing=env_spacing)
    x, y = spawn_xy
    cfg.robot.init_state.pos = (x, y, ROBOT_SPAWN_Z)  # keep the cfg's init orientation
    return cfg


def make_apartment_static(stage):
    """The apartment ships with colliders, but every object -- floor, walls and
    ceiling included -- is authored as a *dynamic* rigid body. That is wrong for
    structural geometry and makes the floor/wall triangle-mesh colliders illegal on
    a dynamic body (PhysX errors + convex-hull fallback). Disabling the rigid bodies
    turns their existing colliders into *static* colliders. No new colliders are
    added, so there are no duplicate/merge conflicts. (For pick-and-place later,
    re-enable RigidBodyAPI on target objects.)"""
    disabled = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(APARTMENT_PRIM)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            disabled += 1
    print(f"[apartment] disabled {disabled} rigid bodies -> static colliders")


def hide_ceiling(stage):
    """Make the ceiling invisible so the interior is visible from above/outside.
    We hide (not delete) it: the per-room ceiling lights live under the room
    scopes, not under this 'ceiling' scope, so the rooms stay lit. This affects
    rendering only -- a ray-cast lidar would still hit the ceiling mesh; use
    ``prim.SetActive(False)`` instead if you need it gone from physics too."""
    hidden = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(APARTMENT_PRIM)):
        if prim.GetName() == "ceiling":
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
    print(f"[apartment] hid {hidden} ceiling scope(s) for interior visibility")


def prepare_apartment(stage):
    """One-shot: apply both apartment fixups after the scene has spawned."""
    make_apartment_static(stage)
    hide_ceiling(stage)


def build_world(spawn_xy, device, dt: float = 1.0 / 200.0, with_sensors: bool = False):
    """Load the environment and robot: create the sim, spawn the apartment + G1,
    apply the apartment fixups, and reset. Returns ``(sim, scene)``; the robot is
    ``scene["robot"]``. With ``with_sensors=True`` the robot also carries a lidar
    and an RGB-D camera (requires launching with ``enable_cameras=True``). This is
    the single entry point every tool in the stack uses to stand up the world."""
    sim = SimulationContext(sim_utils.SimulationCfg(dt=dt, device=device))
    x, y = spawn_xy
    sim.set_camera_view(eye=[x - 3.5, y - 3.5, 2.2], target=[x, y, 0.8])

    cls = ApartmentSensorsSceneCfg if with_sensors else ApartmentSceneCfg
    scene = InteractiveScene(make_scene_cfg(spawn_xy, cls=cls))
    prepare_apartment(sim_utils.get_current_stage())
    sim.reset()
    return sim, scene
