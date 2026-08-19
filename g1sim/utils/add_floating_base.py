import sys
from curobo.config_io import load_yaml, write_yaml


def add_floating_base(yaml_path: str, root_link: str = "pelvis"):
    data = load_yaml(yaml_path)
    k = data["kinematics"]

    k["extra_links"] = {
        "base_link_x": {
            "parent_link_name": "base_link", "link_name": "base_link_x", "joint_name": "base_j_x",
            "joint_type": "X_PRISM", "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_limits": [-10.0, 10.0], "joint_velocity_limits": [-1.0, 1.0],
        },
        "base_link_y": {
            "parent_link_name": "base_link_x", "link_name": "base_link_y", "joint_name": "base_j_y",
            "joint_type": "Y_PRISM", "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_limits": [-10.0, 10.0], "joint_velocity_limits": [-1.0, 1.0],
        },
        "base_link_z": {
            "parent_link_name": "base_link_y", "link_name": "base_link_z", "joint_name": "base_j_z",
            "joint_type": "Z_PRISM", "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_limits": [-10.0, 10.0], "joint_velocity_limits": [-1.0, 1.0],
        },
        "base_link_xtheta": {
            "parent_link_name": "base_link_z", "link_name": "base_link_xtheta", "joint_name": "base_j_xtheta",
            "joint_type": "X_ROT", "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_limits": [-16.0, 16.0], "joint_velocity_limits": [-0.5, 0.5],
        },
        "base_link_ytheta": {
            "parent_link_name": "base_link_xtheta", "link_name": "base_link_ytheta", "joint_name": "base_j_ytheta",
            "joint_type": "Y_ROT", "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_limits": [-16.0, 16.0], "joint_velocity_limits": [-0.5, 0.5],
        },
        "base_link_ztheta": {
            "parent_link_name": "base_link_ytheta", "child_link_name": root_link, "link_name": "base_link_ztheta",
            "joint_name": "base_j_ztheta", "joint_type": "Z_ROT", "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
            "joint_limits": [-16.0, 16.0], "joint_velocity_limits": [-0.5, 0.5],
        },
    }

    base_joints = ["base_j_x", "base_j_y", "base_j_z", "base_j_xtheta", "base_j_ytheta", "base_j_ztheta"]
    csp = k["cspace"]
    csp["joint_names"] = base_joints + csp["joint_names"]
    csp["default_joint_position"] = [0.0] * 6 + csp["default_joint_position"]
    csp["cspace_distance_weight"] = [1.0] * 6 + csp["cspace_distance_weight"]
    csp["acceleration_scale"] = [1.0] * 6 + csp["acceleration_scale"]
    csp["jerk_scale"] = [1.0] * 6 + csp["jerk_scale"]
    csp["max_acceleration"] = [10.0] * 6 + csp["max_acceleration"]
    csp["max_jerk"] = [500.0] * 6 + csp["max_jerk"]

    write_yaml(data, yaml_path)
    print(f"Added floating base to {yaml_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, default="config/unitree_g1_custom.yml", help="Path to the robot YAML file")
    parser.add_argument("--root_link", type=str, default="pelvis", help="Name of the root link for the floating base")
    args = parser.parse_args()
    add_floating_base(args.yaml_path, root_link=args.root_link)
