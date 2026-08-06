"""What the robot knows about the world -- two complementary models.

    mapping        metric: a 2D occupancy grid fused from lidar/depth returns
    semantic_map   symbolic: rooms and objects as a 3D scene graph with 'on' edges

Both are sim-free. ``semantic_map`` is currently built from the apartment USD
(ground truth); Phase 1.2 replaces that source with on-robot detection while keeping
the query API identical.
"""
