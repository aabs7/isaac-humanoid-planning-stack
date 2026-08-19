# curobo imports
from curobo.scene import Scene, Cuboid, Cylinder

from .table_top import (
    TABLE_HEIGHT,
    TABLE_SIZE,
    TABLE1_XY,
    TABLE2_XY,
)

def build_curobo_tabletop_scene() -> Scene:
    return Scene(
        cuboid=[
            # Table1
            Cuboid(
                name="Table1",
                dims=[TABLE_SIZE[0], TABLE_SIZE[1], TABLE_SIZE[2]],
                pose=[TABLE1_XY[0], TABLE1_XY[1], TABLE_HEIGHT / 2, 1.0, 0.0, 0.0, 0.0],
            ),
            # Table2
            Cuboid(
                name="Table2",
                dims=[TABLE_SIZE[0], TABLE_SIZE[1], TABLE_SIZE[2]],
                pose=[TABLE2_XY[0], TABLE2_XY[1], TABLE_HEIGHT / 2, 1.0, 0.0, 0.0, 0.0],
            ),
            # Ground
            Cuboid(
                name="Ground",
                dims=[10.0, 10.0, 0.01],
                pose=[0.0, 0.0, -0.005, 1.0, 0.0, 0.0, 0.0],
            ),
        ],
        cylinder=[
            Cylinder(
                name="Mug",
                radius=0.04,
                height=0.10,
                pose=[1.45, 0.30, TABLE_HEIGHT + 0.05, 1.0, 0.0, 0.0, 0.0],
            ),
        ],
    )
