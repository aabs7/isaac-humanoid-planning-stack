"""The skill seam -- the composable verbs a planner calls.

    types    SkillResult + the reach geometry, shared by both implementations
    mock     sim-free virtual robot, for planner development and tests
    robot    the real thing: locomotion, mapping, A* and USD prim manipulation

``types`` and ``mock`` are sim-free; ``robot`` imports ``isaaclab`` and ``pxr`` at
module level, so import it only after the app is launched. Nothing is re-exported
here for exactly that reason -- a bare ``import g1sim.skills`` must stay cheap.
"""
