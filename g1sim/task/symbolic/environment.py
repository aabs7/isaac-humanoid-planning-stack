"""The railroad ``Environment`` for the G1 in an apartment.

This is the seam. Above it, railroad's MCTS planner sees a flat symbolic world and picks
one grounded action at a time. Below it, ``g1sim.skills`` walks, grasps and sets things
down. :class:`G1Environment` owns the translation in both directions and nothing else --
it holds no geometry (that is :mod:`~g1sim.task.symbolic.domain`), no duration model
(:mod:`~g1sim.task.symbolic.costs`) and no robot code (:mod:`~g1sim.task.symbolic.skills`).

It subclasses ``ObjectSearchEnvironment`` rather than ``SymbolicEnvironment`` even though
we do not yet have a ``search`` operator. That base is inert without one -- its revelation
machinery keys on ``searched`` fluents that never appear -- while its ``{robot}_loc``
handling and move/place hygiene filters are useful immediately, and it is the class the
partially-known-world phase will need. Inheriting it now costs nothing and saves a
migration later.

Because the same object is both the railroad environment *and* the thing our skills
mutate, this class runs a genuine closed loop: :meth:`observe` re-reads object locations
out of the semantic map after every skill instead of trusting that the effects fired.
That is what makes failure tractable without any failure algebra -- a pick that slipped
leaves the map unchanged, so the re-read restores ``at cup dining_table`` by itself.

**The known gap.** Failure is handled *reactively*: a ``failed-*`` fluent bans the target
and the planner routes around it. The planner cannot *anticipate* failure, because a
duration/probability model would have to say which grasps are risky and we have no such
model yet. When we do -- per-object grasp priors from geometry, or learned -- the right
move is to give pick/place a probabilistic effect branch and resolve it in
``resolve_probabilistic_effect`` from the real ``SkillResult``. That is the mechanism
railroad is built around, and it would let MCTS prefer a reachable mug over a marginal
one. Until the model exists it would only add uniform noise, so it is deliberately not
here.
"""

from __future__ import annotations

from typing import List, Optional, Set

from railroad.core import Fluent as F, Operator, State
from railroad.environment import ObjectSearchEnvironment

from g1sim.task.symbolic.costs import euclidean_move_time
from g1sim.task.symbolic.domain import SemanticDomain
from g1sim.task.symbolic.operators import build_operators
from g1sim.task.symbolic.skills import SkillBridge

# Verbs SkillBridge knows how to execute. Any operator whose name is not here falls back
# to railroad's SymbolicSkill, i.e. it is *simulated* rather than executed -- which is
# occasionally useful (a wait/no-op) but silent, so keep the mapping explicit.
EXECUTED_VERBS = ("move", "pick", "place")


class G1Environment(ObjectSearchEnvironment):
    """Binds a skills object (mock or real) and a :class:`SemanticDomain` into a
    railroad environment the MCTS planner can drive.

    Args:
        skills: ``MockSkills`` or ``RobotSkills`` -- anything exposing ``smap``, ``xy()``,
            ``held`` and ``goto_room``/``goto_object``/``pick``/``place``.
        domain: the symbol universe, from :meth:`SemanticDomain.build`.
        move_time: duration model ``(robot, from, to) -> seconds``. Defaults to
            :func:`~g1sim.task.symbolic.costs.euclidean_move_time`; pass an
            :class:`~g1sim.task.symbolic.costs.AStarMoveTime` for wall-aware costs.
        verbose: echo each skill result and state repair.
    """

    def __init__(self, skills, domain: SemanticDomain, *, move_time=None,
                 seed: Optional[int] = None, verbose: bool = True) -> None:
        # Subclass state must exist before super().__init__, which calls
        # define_operators() and may build the initial-effects skill.
        self.skills = skills
        self.domain = domain
        self.verbose = verbose
        # The most recent SkillResult, so the plan-act loop can report an action's outcome
        # without reaching into the skill that produced it (act() has already discarded it).
        self.last_result = None
        self._move_time = move_time if move_time is not None else euclidean_move_time(domain)

        initial = domain.initial_fluents(skills.smap, skills.xy(), held=skills.held)
        super().__init__(
            state=State(0.0, initial),
            objects_by_type=domain.objects_by_type(),
            skill_overrides={verb: SkillBridge for verb in EXECUTED_VERBS},
            seed=seed,
        )

    # -- railroad hooks ---------------------------------------------------
    def define_operators(self) -> List[Operator]:
        return build_operators(self._move_time)

    def runtime_mutated_predicates(self) -> Set[str]:
        """Predicates we write outside operator effects, so grounding keeps them dynamic.

        Getting this wrong is silent: grounding treats any predicate no effect touches as
        static, evaluates it once against the initial facts and *compiles the precondition
        away*. The ``failed-*`` guards are the sharp case -- no effect ever produces one,
        so without this declaration they would be judged static, the guards would be
        stripped, and a failing action would be re-dispatched forever.
        """
        return super().runtime_mutated_predicates() | {
            "holding", "hand-full", "failed-move", "failed-pick", "failed-place",
        }

    # -- observation ------------------------------------------------------
    def observe(self) -> None:
        """Re-read the world out of the semantic map.

        Replaces every object-location, ``holding`` and ``hand-full`` fluent with what the
        map currently says. Robot *status* (``free``, ``just-picked``, and the robot's own
        ``at``) is left to the effect machinery: those are bookkeeping, not observations,
        and the map has no opinion on them.
        """
        observed = self.domain.world_fluents(self.skills.smap, held=self.skills.held)
        kept = {f for f in self._fluents if not self._is_world_predicate(f)}
        self._fluents = kept | observed
        self._warn_about_lost_objects(observed)

    def _warn_about_lost_objects(self, observed) -> None:
        """Say something when a known object has no location symbol.

        It can happen: ``place`` puts an object down at a point, and if that point falls
        outside every room polygon (a doorway, say) the map's room reassignment finds no
        room and the object ends up with neither a support edge nor a room. Symbolically it
        has *vanished* -- no ``at`` fluent -- so any goal naming it becomes unreachable and
        the run ends in the cycle detector with no clue why. One line of warning turns that
        into a diagnosis.
        """
        located = {f.args[0] for f in observed if f.name == "at"}
        held = {self.skills.held.name} if self.skills.held is not None else set()
        lost = set(self.domain.objects) - located - held
        for name in sorted(lost):
            o = self.skills.smap.get(name)
            where = f"({o.xy[0]:.2f}, {o.xy[1]:.2f})" if o is not None else "unknown"
            self.log(f"[warn] {name} has no location symbol -- at {where}, room="
                     f"{getattr(o, 'room', None)}. It is invisible to the planner.")

    def _is_world_predicate(self, fluent) -> bool:
        if fluent.name in ("holding", "hand-full"):
            return True
        return (fluent.name == "at" and bool(fluent.args)
                and fluent.args[0] in self.domain.objects)

    # -- failure ----------------------------------------------------------
    def on_skill_failed(self, verb: str, robot: str, target: str, result) -> None:
        """Repair the state after a skill reported failure, and ban the target.

        Two repairs are needed because the ``time=0`` effects already fired:

        * The robot was marked busy and nothing will free it, so the ``act()`` loop would
          spin until every skill is done and then return a state in which no operator is
          applicable. Re-assert ``free``.
        * A failed *move* deleted ``at ?r ?from`` without ever adding ``at ?r ?to``,
          leaving the robot at no location at all -- which makes every operator
          inapplicable, since all three require ``at ?r ?loc``. Ask the robot where it
          actually ended up.
        """
        self._fluents.add(F(f"free {robot}"))
        if verb == "move":
            self._relocate_robot(robot)
        self._fluents.add(F(f"failed-{verb} {robot} {target}"))
        self.log(f"[fail] {verb} -> banning '{target}' for {robot}: {result.detail}")

    def _relocate_robot(self, robot: str) -> None:
        """Set the robot's symbolic location from its real pose (nearest location)."""
        where = self.domain.nearest_location(self.skills.xy())
        self._fluents = {f for f in self._fluents
                         if not (f.name == "at" and f.args and f.args[0] == robot)}
        self._fluents.add(F(f"at {robot} {where}"))
        self.log(f"[fail] {robot} is actually at {where}")

    # -- goals ------------------------------------------------------------
    def validate_goal(self, goal) -> None:
        """Raise if the goal names a symbol this domain does not have.

        Worth being strict about: a goal referring to a nonexistent symbol is not
        *unsatisfiable* in any visible way, it is simply never achieved, so MCTS wanders
        among equally-dead branches and the run looks like a planner bug rather than a
        typo. This is the same guard the LLM planner gets from grounding its arguments
        against the map before executing them.
        """
        known = set(self.domain.locations) | set(self.domain.objects) | {self.domain.robot}
        unknown = {arg for literal in goal.get_all_literals()
                   for arg in literal.args if arg not in known}
        if unknown:
            raise ValueError(
                f"goal names symbols not in the domain: {sorted(unknown)}. An object must "
                f"be in SemanticDomain.objects (small enough to be pickable, and included "
                f"in build()'s `objects` if you restricted it); a location must be a room "
                f"or a support surface.")

    # -- misc -------------------------------------------------------------
    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
