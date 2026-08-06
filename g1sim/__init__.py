"""Reusable G1-in-apartment stack, layered around the *skill seam*.

The organizing principle (see claude.md): a planner speaks only in skills, and
everything sim-specific lives below that line, so the same planner can eventually
drive real hardware by swapping only the skill implementations and the localization
source. The package layout makes that boundary visible:

    task/         above the seam -- LLM task planner + the model client
    skills/       THE SEAM -- the verbs a planner calls (types, mock, robot)
    perception/   the robot's world model -- occupancy grid + semantic scene graph
    navigation/   getting from A to B -- A* path planning + waypoint controller
    sim/          Isaac-only foundation -- app launcher, scene, locomotion policy
    viz/          debug windows, image conversion, video recording

**Import-time Isaac dependency.** These are importable anywhere, with no simulator:
``task.*``, ``skills.types``, ``skills.mock``, ``perception.*``,
``navigation.path_planning``, ``viz.*``. These pull in ``isaaclab``/``pxr`` at import
and must only be imported *after* the app is launched: ``sim.scene``,
``sim.locomotion``, ``navigation.waypoint``, ``skills.robot``. (``sim.launch`` is the
exception -- it is what starts the app, so it is safe beforehand.)

Submodules are deliberately NOT imported here: doing so would drag Isaac into every
sim-free consumer. Entry-point scripts import what they need, after ``launch()``.
"""
