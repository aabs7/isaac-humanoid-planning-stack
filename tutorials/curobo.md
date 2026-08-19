To generate collision spheres for the humanoid robot.
```
python -m curobo.examples.getting_started.build_robot_model \
        --urdf isaac_task_planning/assets/g1/g1_29dof_with_hand_rev_1_0.urdf \
        --asset-path isaac_task_planning/assets/g1 \
        --tool-frames right_hand_palm_link \
        --output isaac_task_planning/config/unitree_g1_custom.yml \
        --visualize --viz-port 8080
```

When you build robot configuration consisting collision information saved in yaml like above, you have to understand whether the robot is fixed-base or mobile.
- Fixed base (e.g., Franka, UR10) is bolted to a table or the floor. Its base link is physically at (0, 0, 0). You can use build_robot_model to generate yaml file, and the planner works immediately.
- Mobile base (e.g., Humanoid, mobile robot) URDF only defines the internal joints (hips, knees, spine, shoulders). A URDF does not have joints connecting pelvis to the world because robot walks/stands in free space. For cuRobo to plan motions with robot standing at an arbitrary world coordinate, it needs 6 virtual degrees of freedom (base_j_x, base_j_y, base_j_z, base_j_xtheta, base_j_ytheta, base_j_ztheta) inserted between base_link and the root body (pelvis). Therefore, you have to add these to your config file.
For that, you have to run the following:

```shell
python -m g1sim.utils.add_floating_base --yaml-path "config/unitree_g1_custom.yml" --base "pelvis"
```

Once you generate `unitree_g1_custom.yml`, you can pass it directly to `MotionPlannerCfg`:
```
config = MotionPlannerCfg.create(
    robot="isaac_task_planning/config/unitree_g1_custom.yml",
    scene_model=table_top_scene,
    position_tolerance=0.02,
    orientation_tolerance=0.2,
)
planner = MotionPlanner(config)
```
