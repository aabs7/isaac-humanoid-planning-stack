"""The skill seam -- the composable verbs a planner calls.

    types    SkillResult + the reach geometry, shared by both implementations
    mock     sim-free virtual robot, for planner development and tests
    robot    the real thing: locomotion, mapping, A* nav
    grasp    *how* an object is held -- a strategy `robot` delegates to, so a real
             physical grasp can sit beside the magic one and be selected per run

``types`` and ``mock`` are sim-free; ``robot`` and ``grasp`` import ``isaaclab`` and
``pxr`` at module level, so import them only after the app is launched. Nothing is
re-exported here for exactly that reason -- a bare ``import g1sim.skills`` must stay cheap.
"""
