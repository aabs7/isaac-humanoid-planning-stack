"""Task planning -- the layer *above* the skill seam.

Turns a natural-language goal into skill calls (``planner``) using a local LLM
(``llm``). Sim-free by construction: the planner drives an interface (``.smap``,
``.xy()``, ``.held`` + the verb methods), never Isaac, so the identical planner runs
against ``skills.mock`` in tests and ``skills.robot`` in the simulator.
"""
