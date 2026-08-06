"""Debug visualization: in-GUI windows, image conversion, and video capture.

    sensors    RGB/depth/lidar frames -> displayable RGBA
    task_map   the occupancy-grid + goal + path window
    recorder   chase-camera video recording of a run

Sim-free at import (numpy/cv2 only); the recorder reaches into the running stage
lazily, when it is actually used.
"""
