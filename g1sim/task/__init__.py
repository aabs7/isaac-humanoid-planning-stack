"""Task planning -- the layer *above* the skill seam.

Turns a goal into skill calls. Two interchangeable planner families live here:

* ``llm_based`` -- a ReAct loop over a local LLM (Phase 1.1).
* ``symbolic`` -- search over a symbolic state, backed by ``third_party/railroad``.

Both are sim-free by construction: a planner drives an *interface* (``.smap``,
``.xy()``, ``.held`` + the verb methods), never Isaac, so the identical planner runs
against ``skills.mock`` in tests and ``skills.robot`` in the simulator.
"""
