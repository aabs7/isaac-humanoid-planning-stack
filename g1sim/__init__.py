"""Reusable G1-in-apartment simulation stack.

Import order matters: :mod:`g1sim.launch` is safe to import before the Isaac Sim
app exists, but :mod:`g1sim.scene` and :mod:`g1sim.locomotion` pull in
``isaaclab.sim`` and therefore must only be imported *after* the app has been
launched. So this package intentionally does NOT import its submodules here --
entry-point scripts import them explicitly after ``launch()``.
"""
