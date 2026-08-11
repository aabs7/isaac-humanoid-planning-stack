from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import List

from railroad.core import Fluent as F, get_action_by_name
from railroad.planner import MCTSPlanner

# Cost charged to a branch the relaxation proves cannot reach the goal. railroad's default
# is None, which clamps the heuristic to 0 and thereby makes dead ends look *better* than
# reachable states -- actively drawing the search into them. Any value that dominates
# typical plan costs works; ours are tens of seconds, so 1e4 is comfortably clear.
DEAD_END_PENALTY = 1e4
MCTS_ITERATIONS = 10000
MCTS_EXPLORATION_C = 20.0
STALL_LIMIT = 4


@dataclass
class PlanOutcome:
    """What happened over one plan-act run."""
    ok: bool
    reason: str
    dispatches: List[str] = field(default_factory=list)
    plan_time_s: float = 0.0
    total_time_s: float = 0.0

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        head = f"[{'OK ' if self.ok else 'FAIL'}] {self.reason}"
        body = "".join(f"\n  {i + 1}. {n}" for i, n in enumerate(self.dispatches))
        return (f"{head}{body}\n  ({len(self.dispatches)} dispatches, "
                f"{self.plan_time_s:.1f}s planning, {self.total_time_s:.1f}s total)")


def solve(env, goal, *, max_dispatches: int = 30, mcts_iterations: int = MCTS_ITERATIONS,
          c: float = MCTS_EXPLORATION_C, dead_end_penalty: float = DEAD_END_PENALTY,
          stall_limit: int = STALL_LIMIT, verbose: bool = True, on_dispatch=None,
          **planner_kwargs) -> PlanOutcome:
    """Drive ``env`` until ``goal`` holds, one MCTS-chosen action at a time.

    Args:
        env: a :class:`~g1sim.task.symbolic.environment.G1Environment`.
        goal: a railroad goal expression, e.g. ``F("at cup_0003 tea_table_0001")`` or
            ``F("holding g1 cup_0003") & ~F("at cup_0003 dining_table_0001")``.
        max_dispatches: give up after this many actions (a robot that keeps failing will
            exhaust its options long before this, but it bounds pathological loops).
        mcts_iterations: search budget per dispatch; see :data:`MCTS_ITERATIONS`.
        c: MCTS exploration constant; see :data:`MCTS_EXPLORATION_C`.
        stall_limit: stop after this many consecutive dispatches that leave the world
            unchanged; see :data:`STALL_LIMIT`.
        on_dispatch: called ``(step, action_name, result)`` twice per action -- once before
            it executes with ``result=None``, and once after with the
            :class:`~g1sim.skills.types.SkillResult` (``None`` if the verb had no bound
            skill). Two calls, because an overlay needs to show "running..." and then the
            outcome; this mirrors the LLM planner's ``on_action``. Keeps the loop sim-free
            while still letting a caller drive a video overlay or a GUI.
        planner_kwargs: passed through to ``MCTSPlanner`` (e.g. ``prune_cheapest_m``).

    Raises:
        ValueError: if the goal names symbols the domain does not have.
    """
    env.validate_goal(goal)
    started = _time.perf_counter()
    outcome = PlanOutcome(ok=False, reason="max dispatches reached")
    world, stalled = _world_key(env), 0

    for _ in range(max_dispatches):
        if goal.evaluate(env.state.fluents):
            # Distinguish "achieved it" from "it was already true", because the second is
            # usually a bad goal rather than a finished task. An LLM handed a task whose
            # premise does not hold ("take the plate off the dining table" when no plate is
            # there) will produce a condition that happens to hold already, and reporting
            # that as success is a silent wrong answer.
            outcome.ok = True
            outcome.reason = ("goal satisfied" if outcome.dispatches else
                              "goal was already satisfied before acting -- nothing to do")
            break

        actions = env.get_actions()
        if not actions:
            outcome.reason = "no applicable actions"
            break

        plan_started = _time.perf_counter()
        planner = MCTSPlanner(actions, dead_end_penalty=dead_end_penalty,
                              **planner_kwargs)
        name = planner(env.state, goal, max_iterations=mcts_iterations, c=c)
        outcome.plan_time_s += _time.perf_counter() - plan_started

        if name == "NONE":
            outcome.reason = "planner found no action toward the goal"
            break

        if verbose:
            print(f"[plan] t={env.state.time:6.1f}s  -> {name}")
        outcome.dispatches.append(name)
        step = len(outcome.dispatches)
        if on_dispatch is not None:
            on_dispatch(step, name, None)
        env.last_result = None
        env.act(get_action_by_name(actions, name))
        if on_dispatch is not None:
            on_dispatch(step, name, env.last_result)

        moved_on = _world_key(env)
        stalled = 0 if moved_on != world else stalled + 1
        world = moved_on
        if stalled >= stall_limit:
            outcome.reason = (f"stalled: {stalled} dispatches without moving anything "
                              f"(the goal is probably unreachable)")
            break

    outcome.total_time_s = _time.perf_counter() - started
    if verbose:
        print(outcome)
    return outcome


def _world_key(env) -> frozenset:
    """The part of the state that constitutes progress: where the objects are and what is
    in the hand.

    Deliberately excludes the robot's own position and the ``failed-*`` bans. Both change
    on every wandering step, which is exactly why a whole-state comparison cannot detect a
    wander, and neither is an end in itself -- a run that only ever moves the robot and
    accumulates bans has achieved nothing.
    """
    objects = env.domain.objects
    return frozenset(f for f in env.state.fluents
                     if f.name in ("holding", "hand-full")
                     or (f.name == "at" and f.args and f.args[0] in objects))


# -- goal helpers -----------------------------------------------------------
# Thin, but they keep railroad's reserved predicate names ("at", "holding") in one place
# instead of spelled out at every call site, where a typo silently produces a goal that
# can never be satisfied.

def object_at(obj: str, location: str):
    """Goal: ``obj`` is resting at ``location`` (a surface or a room)."""
    return F(f"at {obj} {location}")


def holding(obj: str, robot: str = "g1"):
    """Goal: the robot is carrying ``obj``."""
    return F(f"holding {robot} {obj}")
