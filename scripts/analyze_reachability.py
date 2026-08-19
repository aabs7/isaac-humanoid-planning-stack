import _bootstrap
import argparse
from curobo.config_io import load_yaml
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

from g1sim.environment.curobo_table_top_scene import build_curobo_tabletop_scene
from g1sim.environment.table_top import TABLE1_STANCE, ROBOT_SPAWN_Z, PICK_POS
from g1sim.utils.utility import compute_fk_workspace, evaluate_grid_reachability
from g1sim.utils.plotting import visualize_reachability_viser


def main():
    parser = argparse.ArgumentParser(description="Analyze G1 Reachability with Interactive Viser 3D Plot")
    parser.add_argument("--g1_yaml_path", type=str, default="config/unitree_g1_custom.yml")
    parser.add_argument("--urdf_path", type=str, default="assets/g1/g1_29dof_with_hand_rev_1_0.urdf")
    parser.add_argument("--n_fk_samples", type=int, default=10000)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    # 1. Instantiate TableTop Scene
    scene = build_curobo_tabletop_scene()
    print(f"[Scene] Loaded {len(scene.cuboid)} cuboids and {len(scene.cylinder)} cylinders.")

    # 2. Instantiate cuRobo Motion Planner with Scene
    robot_dict = load_yaml(args.g1_yaml_path)
    robot_dict["kinematics"]["tool_frames"] = ["right_hand_palm_link"]

    cfg = MotionPlannerCfg.create(
        robot=robot_dict,
        scene_model=scene,
        position_tolerance=0.03,
        orientation_tolerance=0.3,
        num_ik_seeds=32,
    )
    planner = MotionPlanner(cfg)

    # 3. Compute FK Reachable Volume
    print(f"[1/3] Sampling {args.n_fk_samples} FK points on GPU...")
    fk_cloud = compute_fk_workspace(
        planner,
        stance_xy=TABLE1_STANCE,
        base_z=ROBOT_SPAWN_Z,
        tool_frame="right_hand_palm_link",
        n_samples=args.n_fk_samples,
    )

    # 4. Evaluate Grid Points with Collision-Aware IK
    print("[2/3] Evaluating 3D grid against TableTop scene obstacles...")
    # forward facing palm orientation
    grid, reachable_mask, _ = evaluate_grid_reachability(
        planner,
        x_range=(1.05, 1.55),
        y_range=(-0.35, 0.15),
        z_range=(0.80, 1.15),
        resolution=(8, 8, 6),
    )
    reachable_count = reachable_mask.sum()
    total_count = len(reachable_mask)
    print(f"  Reachability Score: {reachable_count}/{total_count} ({reachable_count/total_count*100:.1f}%)")

    # 5. Launch Interactive Viser Visualizer
    print("[3/3] Launching interactive Viser 3D visualizer...")
    visualize_reachability_viser(
        grid_points=grid,
        reachable_mask=reachable_mask,
        scene=scene,
        urdf_path=args.urdf_path,
        g1_yaml_path=args.g1_yaml_path,
        stance_xy=TABLE1_STANCE,
        base_z=ROBOT_SPAWN_Z,
        pick_pos=PICK_POS,
        fk_cloud=fk_cloud,
        port=args.port,
    )


if __name__ == "__main__":
    main()
