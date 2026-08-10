"""Symbolic task planning -- the railroad-backed planner (work in progress).

Drives the same skill seam as :mod:`g1sim.task.llm_based`, but chooses actions by
search over a symbolic state rather than by asking a language model. The planner
itself is vendored in ``third_party/railroad``; this package holds only the adapters
that express *our* environment (semantic map, skills, navigation costs) in the terms
that planner expects.
"""
