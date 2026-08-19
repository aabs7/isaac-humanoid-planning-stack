"""
A two-table world sized for a G1 to actually pick something up on.

Table1: size=(0.8, 1.2, 0.85), at (1.6, 0.0, 0.425)
Table2: size=(0.8, 1.2, 0.85), at (1.6, 2.5, 0.425)
Mug: cylinder, radius=0.04, height=0.10, at (1.45, 0.30, 0.90)
Block: cuboid, size=(0.04, 0.04, 0.12), at (1.30, -0.14, 0.91)
Ground: Plane at z=0.0

"""

from isaaclab.utils.configclass import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

# TODO: remove this from here, and make a robot config for all the constants
# TODO: share with apartment scene too
G1_DEFAULT_STANDING_POSE = {
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
        "left_shoulder_pitch_joint": 0.20,
        "right_shoulder_pitch_joint": 0.20,
        "left_shoulder_roll_joint": 0.15,
        "right_shoulder_roll_joint": -0.15,
        "left_elbow_joint": 0.00,
        "right_elbow_joint": 0.00,
    }

TABLE_HEIGHT = 0.85
TABLE_SIZE = (0.8, 1.2, TABLE_HEIGHT)
TABLE1_XY = (1.6, 0.0)
TABLE2_XY = (1.6, 2.5)

BLOCK_SIZE = (0.04, 0.04, 0.12)
BLOCK_MASS = 0.2


PICK_POS = (1.30, -0.14, TABLE_HEIGHT + BLOCK_SIZE[2] / 2)
PLACE_POS = (1.30, 2.36, TABLE_HEIGHT + BLOCK_SIZE[2] / 2)

# How the robot lines up on a work point: back far enough that the counter is clear of its
# legs, offset sideways so the working shoulder -- not the sternum -- is the thing pointed at
# the object.
STANDOFF = 0.33
SHOULDER_OFFSET = 0.14


def stance_for(work_pos, side="right"):
    """Where to stand to work on a point, as ``(x, y)``, facing +x.

    Not a pose the layout dictates but one the arm does: much closer and the counter is in
    the robot's legs, much further and the object is outside a reach that is only ~0.37 m
    from the shoulder to begin with.
    """
    return (work_pos[0] - STANDOFF,
            work_pos[1] + (SHOULDER_OFFSET if side == "right" else -SHOULDER_OFFSET))


TABLE1_STANCE = stance_for(PICK_POS)
TABLE2_STANCE = stance_for(PLACE_POS)

ROBOT_SPAWN_Z = 0.75

# The stage authors no physics material, so contacts default to mu ~= 0.5 -- marginal for a
# three-finger pinch on a smooth block, and enough to let it slide off the counter on contact.
GRASP_FRICTION = sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8,
                                                restitution=0.0)


def _table_cfg(name, xy):
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=GRASP_FRICTION,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.25, 0.1)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(xy[0], xy[1], TABLE_HEIGHT / 2)),
    )


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):

    # Ground and lighting
    ground = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000),
    )

    table1 = _table_cfg("Table1", TABLE1_XY)   # holds the block
    table2 = _table_cfg("Table2", TABLE2_XY)   # the destination

    # The block to move: upright, light, and grippy (see the module docstring).
    block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Block",
        spawn=sim_utils.CuboidCfg(
            size=BLOCK_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=BLOCK_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=GRASP_FRICTION,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PICK_POS),
    )

    # A second object, not part of the task: something for a grasp to miss, and a check that
    # the arm's approach clears its neighbours.
    mug = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Mug",
        spawn=sim_utils.CylinderCfg(
            radius=0.04, height=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=GRASP_FRICTION,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.45, 0.30, TABLE_HEIGHT + 0.05)),
    )

    robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot",
                                 init_state=G1_29DOF_CFG.init_state.replace(joint_pos=G1_DEFAULT_STANDING_POSE))


def build_table_top_environment(spawn_xy, device, dt: float = 1 / 200.0):
    # simulation context and set camera view
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(-0.6, 1.6, 2.0), target=(1.4, 1.0, 0.8))

    # build the scene
    scene_cfg = TableTopSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot.init_state.pos = (spawn_xy[0], spawn_xy[1], ROBOT_SPAWN_Z)

    scene = InteractiveScene(scene_cfg)

    sim.reset()

    return sim, scene
