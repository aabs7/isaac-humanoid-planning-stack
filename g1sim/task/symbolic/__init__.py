"""Symbolic task planning -- the railroad-backed planner.

Drives the same skill seam as :mod:`g1sim.task.llm_based`, but chooses actions by MCTS
over a symbolic state instead of by asking a language model. The planner itself is
vendored in ``third_party/railroad``; this package holds only the adapters that express
*our* world in the terms it expects.

    domain       SemanticMap -> railroad symbols (locations, objects, fluents)
    costs        move-duration models (Euclidean by default, A* on request)
    operators    the move/pick/place operator set
    skills       SkillBridge: one railroad action -> one g1sim.skills call
    environment  G1Environment: the seam, and the closed observation loop
    planner      solve(): the plan-act loop, plus goal helpers
    goals        translate(): plain English -> goal fluents, via a local LLM

Everything here is sim-free and importable without Isaac: it drives an *interface*, so
the identical planner runs against ``skills.mock`` in tests and ``skills.robot`` in the
simulator. It does require ``railroad`` to be installed (see the project README).

Typical use, with the goal written by hand::

    from g1sim.task.symbolic import G1Environment, SemanticDomain, object_at, solve

    domain = SemanticDomain.for_task(smap, ["cup_0003"], ["tea_table_0001"])
    env = G1Environment(skills, domain)
    solve(env, object_at("cup_0003", "tea_table_0001"))

...or stated in English, with a local LLM translating it once, up front::

    from g1sim.task.symbolic import translate
    from g1sim.task.llm_based.llm import OllamaChat

    task = translate(OllamaChat(), smap, "put the kitchen cup on the coffee table")
    domain = SemanticDomain.for_task(smap, task.objects, task.locations)
    solve(G1Environment(skills, domain), task.goal)
"""

from g1sim.task.symbolic.costs import AStarMoveTime, euclidean_move_time
from g1sim.task.symbolic.domain import SemanticDomain
from g1sim.task.symbolic.environment import G1Environment
from g1sim.task.symbolic.goals import (GoalTranslationError, GoalUnsatisfiable,
                                       TranslatedGoal, describe_symbols, translate)
from g1sim.task.symbolic.operators import build_operators
from g1sim.task.symbolic.planner import PlanOutcome, holding, object_at, solve
from g1sim.task.symbolic.skills import SkillBridge

__all__ = ["AStarMoveTime", "euclidean_move_time", "SemanticDomain", "G1Environment",
           "GoalTranslationError", "GoalUnsatisfiable", "TranslatedGoal",
           "describe_symbols", "translate", "build_operators", "PlanOutcome", "holding",
           "object_at", "solve", "SkillBridge"]
