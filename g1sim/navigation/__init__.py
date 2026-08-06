"""Getting the robot from A to B.

    path_planning   A* over the occupancy grid, with robot-radius inflation
    waypoint        closed-loop unicycle controller to an (x, y) goal

``path_planning`` is pure numpy/scipy and sim-free; ``waypoint`` reads the robot's
pose through ``isaaclab`` and must be imported after the app is launched.
"""
