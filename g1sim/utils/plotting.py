import numpy as np
import matplotlib.pyplot as plt
from curobo.config_io import load_yaml

import time
import numpy as np
import viser
import yourdfpy
from viser.extras import ViserUrdf

from curobo.types import GoalToolPose, JointState
from curobo.scene import Scene


def visualize_reachability_viser(
    grid_points: np.ndarray,
    reachable_mask: np.ndarray,
    scene: Scene = None,
    urdf_path: str = "assets/g1/g1_29dof_with_hand_rev_1_0.urdf",
    g1_yaml_path: str = "config/unitree_g1_custom.yml",
    stance_xy: tuple[float, float] = (0.97, 0.0),
    base_z: float = 0.75,
    pick_pos: tuple[float, float, float] = (1.30, -0.14, 0.91),
    fk_cloud: np.ndarray = None,
    port: int = 8080,
):
    """Launch an interactive Viser 3D Web Visualizer."""
    server = viser.ViserServer(port=port)
    print(f"\n" + "=" * 60)
    print(f"Viser Interactive Visualizer Running at: http://localhost:{port}")
    print(f"=" * 60 + "\n")

    # 1. Add Robot Stance Frame & 3D URDF Mesh
    server.scene.add_frame(
        name="/robot",
        position=(stance_xy[0], stance_xy[1], base_z),
        show_axes=True,
        axes_length=0.2,
        axes_radius=0.005,
    )
    try:
        urdf = yourdfpy.URDF.load(urdf_path)
        viser_urdf = ViserUrdf(server, urdf_or_path=urdf, root_node_name="/robot")
        robot_dict = load_yaml(g1_yaml_path)
        csp = robot_dict["kinematics"]["cspace"]
        default_cfg = {name: val for name, val in zip(csp["joint_names"], csp["default_joint_position"])
                       if not name.startswith("base_j_")}
        # Ready pose with arms raised slightly above table
        viser_urdf.update_cfg(default_cfg)
    except Exception as e:
        print(f"[Viser] Note: Could not load URDF meshes ({e}), showing frame marker instead.")

    # 2. Add Scene Obstacles (Tables, Ground, Mug)
    if scene is not None:
        if hasattr(scene, "cuboid"):
            for cuboid in scene.cuboid:
                is_ground = "ground" in cuboid.name.lower()
                server.scene.add_box(
                    name=f"/scene/{cuboid.name}",
                    dimensions=tuple(cuboid.dims),
                    position=tuple(cuboid.pose[:3]),
                    color=(180, 140, 100) if not is_ground else (100, 100, 100),
                    opacity=0.5 if not is_ground else 0.2,
                )
        if hasattr(scene, "cylinder"):
            for cyl in scene.cylinder:
                server.scene.add_cylinder(
                    name=f"/scene/{cyl.name}",
                    radius=cyl.radius,
                    height=cyl.height,
                    position=tuple(cyl.pose[:3]),
                    color=(200, 50, 50),
                    opacity=0.8,
                )

    # 3. Add Pick Target Marker (Gold Sphere)
    server.scene.add_icosphere(
        name="/targets/pick_target",
        radius=0.03,
        position=tuple(pick_pos),
        color=(255, 215, 0),
    )

    # 4. Add Reachable (Green) and Unreachable (Red) Point Clouds
    reachable_pts = grid_points[reachable_mask].astype(np.float32)
    unreachable_pts = grid_points[~reachable_mask].astype(np.float32)

    if len(reachable_pts) > 0:
        server.scene.add_point_cloud(
            name="/reachability/reachable_points",
            points=reachable_pts,
            colors=np.array([[30, 220, 30]] * len(reachable_pts), dtype=np.uint8),
            point_size=0.01,
        )
    if len(unreachable_pts) > 0:
        server.scene.add_point_cloud(
            name="/reachability/unreachable_points",
            points=unreachable_pts,
            colors=np.array([[220, 30, 30]] * len(unreachable_pts), dtype=np.uint8),
            point_size=0.01,
        )

    # 5. Add FK Continuous Reachable Volume Cloud (Cyan)
    if fk_cloud is not None:
        server.scene.add_point_cloud(
            name="/reachability/fk_workspace_volume",
            points=fk_cloud.astype(np.float32),
            colors=np.array([[0, 180, 255]] * len(fk_cloud), dtype=np.uint8),
            point_size=0.004,
        )

    # Keep server open
    print("Press Ctrl+C in terminal to stop the Viser visualizer...")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping Viser server.")
        server.stop()
