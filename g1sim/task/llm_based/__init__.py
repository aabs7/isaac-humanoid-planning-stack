"""LLM-based task planning (Phase 1.1).

A ReAct loop that asks a local LLM (``llm``) for one skill call at a time and
executes it through the skill seam (``planner``). Kept as one of two interchangeable
planner families under :mod:`g1sim.task`; see :mod:`g1sim.task.symbolic` for the
search-based alternative.
"""
