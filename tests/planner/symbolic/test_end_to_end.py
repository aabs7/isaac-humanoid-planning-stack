"""The whole loop: MCTS plans, the mock robot executes, the map ends up right.

These are the tests that would catch a regression in the *integration* rather than in any
one piece -- the operator set being unsolvable, the observation loop contradicting the
effects, the search budget being too small for the branching factor.
"""

import pytest
from railroad.core import Fluent as F

from g1sim.skills.types import SkillResult
from g1sim.task.symbolic import G1Environment, holding, object_at, solve

pytestmark = pytest.mark.symbolic


def test_fetch_a_cup_across_rooms(sym_env, env, smap):
    """The canonical task: the kitchen cup ends up on the livingroom table."""
    out = solve(sym_env, object_at("cup_0001", "dining_table_0000"), verbose=False)

    assert out.ok, out.reason
    assert smap.get("cup_0001").supported_by == "dining_table_0000"
    assert smap.get("cup_0001").room == "livingroom"
    assert env.held is None


def test_the_plan_is_the_obvious_four_actions(sym_env):
    """move -> pick -> move -> place, with nothing extra. A longer plan still satisfies
    the goal, so this is really a check that the cost model and heuristic agree with
    common sense."""
    out = solve(sym_env, object_at("cup_0001", "dining_table_0000"), verbose=False)

    assert out.ok
    assert [n.split()[0] for n in out.dispatches] == ["move", "pick", "move", "place"]
    assert out.dispatches[1] == "pick g1 counter_0000 cup_0001"
    assert out.dispatches[3] == "place g1 dining_table_0000 cup_0001"


def test_a_holding_goal_stops_once_carried(sym_env, env):
    out = solve(sym_env, holding("cup_0001"), verbose=False)

    assert out.ok
    assert env.held is not None and env.held.name == "cup_0001"
    assert [n.split()[0] for n in out.dispatches] == ["move", "pick"]


def test_an_already_satisfied_goal_dispatches_nothing_and_says_so(sym_env):
    """Succeeding without acting is reported distinctly: it usually means the goal was
    mis-stated (an LLM given a task whose premise is false will produce a condition that
    already holds), and calling that a completed task hides the mistake."""
    out = solve(sym_env, object_at("cup_0000", "dining_table_0000"), verbose=False)

    assert out.ok
    assert out.dispatches == []
    assert "already satisfied" in out.reason


def test_a_negated_goal_clears_a_surface(sym_env, smap):
    """`~F(at ...)` is a legitimate goal: get the cup off the table, anywhere else."""
    out = solve(sym_env, ~F("at cup_0000 dining_table_0000"), verbose=False)

    assert out.ok
    assert smap.get("cup_0000").supported_by != "dining_table_0000"


class BrokenGrasp:
    """`pick` fails for one named object and works for everything else -- a marginal
    grasp, which is the failure our skills actually produce."""

    def __init__(self, inner, doomed):
        self._inner = inner
        self._doomed = doomed
        self.attempts = 0

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def pick(self, obj):
        if obj == self._doomed:
            self.attempts += 1
            return SkillResult(False, "pick", f"{obj} is out of reach")
        return self._inner.pick(obj)


def test_an_unachievable_goal_terminates_instead_of_looping(env, domain):
    """Two guards working together. The `failed-*` ban stops the identical failing pick
    from being re-dispatched forever; the stall detector stops the *consequence* of the
    ban, which is a planner whose every branch is now a dead end returning an arbitrary
    move and touring the apartment until max_dispatches."""
    skills = BrokenGrasp(env, doomed="cup_0001")
    sym_env = G1Environment(skills, domain, verbose=False)

    out = solve(sym_env, object_at("cup_0001", "dining_table_0000"),
                max_dispatches=30, verbose=False)

    assert not out.ok
    assert skills.attempts == 1, "the banned pick must not be retried"
    assert out.reason != "max dispatches reached", (
        f"the loop must detect futility, not just run out: {out}")
    assert len(out.dispatches) < 30


def test_a_wander_is_caught_even_when_no_state_ever_repeats(env, domain):
    """The exact failure the first Isaac run produced, and the reason progress is measured
    on the world rather than on the whole state.

    Every dispatch there banned a new target or left the robot somewhere new, so no two
    states were equal and a repeated-state check never fired -- the robot walked for four
    minutes. Here every move fails too, so each dispatch adds a fresh `failed-move` and
    changes the robot's position: all states distinct, nothing accomplished."""
    class NothingWorks:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, item):
            return getattr(self._inner, item)

        def goto_object(self, obj, **kw):
            return SkillResult(False, "goto_object", "timeout after 90s")

        def goto_room(self, room, **kw):
            return SkillResult(False, "goto_room", "timeout after 90s")

    sym_env = G1Environment(NothingWorks(env), domain, verbose=False)
    out = solve(sym_env, object_at("cup_0001", "dining_table_0000"),
                max_dispatches=40, verbose=False)

    assert not out.ok
    assert "stalled" in out.reason, out.reason
    assert len(out.dispatches) <= 6, f"gave up too slowly: {out}"
    # Distinct states throughout: this is what defeats a repeated-state check.
    assert len({f"{d}" for d in out.dispatches}) == len(out.dispatches)


def test_a_second_object_is_still_reachable_after_a_failure(env, domain):
    """A ban is per (robot, object): one bad grasp must not poison the whole run."""
    skills = BrokenGrasp(env, doomed="cup_0001")
    sym_env = G1Environment(skills, domain, verbose=False)

    solve(sym_env, object_at("cup_0001", "bed_0000"), max_dispatches=8, verbose=False)
    out = solve(sym_env, object_at("cup_0000", "bed_0000"), max_dispatches=12,
                verbose=False)

    assert out.ok, out.reason
    assert env.smap.get("cup_0000").supported_by == "bed_0000"


def test_goal_with_an_unknown_symbol_raises_before_planning(sym_env):
    with pytest.raises(ValueError, match="not in the domain"):
        solve(sym_env, object_at("cup_0001", "no_such_place_0000"), verbose=False)


def test_on_dispatch_fires_before_and_after_each_action(sym_env):
    """The overlay contract: two calls per action, the first with result=None ("running")
    and the second carrying the real SkillResult. Recording a run depends on it, and a
    mistake here only shows up under --video."""
    calls = []
    solve(sym_env, holding("cup_0001"), verbose=False,
          on_dispatch=lambda step, name, result: calls.append((step, name, result)))

    assert len(calls) == 4, calls                      # 2 actions x 2 calls
    for (s1, n1, r1), (s2, n2, r2) in zip(calls[::2], calls[1::2]):
        assert (s1, n1) == (s2, n2), "the pair must describe the same action"
        assert r1 is None, "the first call announces the action, before it runs"
        assert r2 is not None and r2.ok, "the second carries its outcome"
    assert [c[0] for c in calls] == [1, 1, 2, 2]


def test_on_dispatch_reports_a_failure(env, domain):
    """A failed skill must reach the overlay as a failure, not as a silent "running..."."""
    skills = BrokenGrasp(env, doomed="cup_0001")
    sym_env = G1Environment(skills, domain, verbose=False)
    calls = []
    solve(sym_env, holding("cup_0001"), max_dispatches=6, verbose=False,
          on_dispatch=lambda step, name, result: calls.append((name, result)))

    picks = [(n, r) for n, r in calls if n.startswith("pick") and r is not None]
    assert picks and not picks[0][1].ok
    assert "out of reach" in picks[0][1].detail


@pytest.mark.symbolic
def test_a_task_scoped_domain_solves_what_the_full_domain_could_not(smap, env):
    """Regression for the failure the first natural-language sim run produced.

    "put the kitchen cup on the living room table" translated correctly, then the planner
    oscillated between two adjacent chairs and never set out for the cup. It was not a
    budget problem -- 120k iterations still chose the chair -- but a heuristic one: with
    `move` unconstrained between all 49 locations, the relaxed plan is nearly the same
    length from every location, so cost dominates and the nearest furniture wins. Scoping
    the domain to the task removes the decoys.
    """
    from g1sim.task.symbolic import SemanticDomain

    domain = SemanticDomain.for_task(smap, ["cup_0001"], ["dining_table_0000"])
    sym_env = G1Environment(env, domain, verbose=False)

    # The decoys are gone, which is the actual fix.
    assert "chair_0000" not in domain.locations
    assert len(sym_env.get_actions()) < 100

    out = solve(sym_env, object_at("cup_0001", "dining_table_0000"), verbose=False)
    assert out.ok, out.reason
    assert [n.split()[0] for n in out.dispatches] == ["move", "pick", "move", "place"]
    assert smap.get("cup_0001").supported_by == "dining_table_0000"
