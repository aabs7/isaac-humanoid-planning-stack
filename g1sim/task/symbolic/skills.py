"""The execution bridge: one railroad action becomes one call into ``g1sim.skills``.

railroad models an action as a set of timed :class:`Effect`s that *will* fire. Our
skills are blocking calls that return a :class:`~g1sim.skills.types.SkillResult` which
may say no. :class:`SkillBridge` is where those two views are reconciled.

It satisfies railroad's ``ActiveSkill`` protocol by subclassing ``SymbolicSkill`` (which
already implements the effect-queue bookkeeping correctly) and inserting the real robot
between the immediate effects and the completion effects:

1. ``advance`` first lets the base class apply the ``time=0`` effects -- the ones that
   mark the robot busy and, for a pick, optimistically take the object off its surface.
2. It then dispatches the actual skill **once**, blocking while ``RobotSkills`` steps
   Isaac (or returning instantly for ``MockSkills``).
3. It calls ``env.observe()``, so what the planner believes about the world is re-read
   from the semantic map rather than assumed.
4. On success the base class applies the completion effects. On failure they are
   discarded and ``env.on_skill_failed`` repairs the state instead.

**Dispatch happens inside ``advance``, and ``advance`` is not only called by ``act()``.**
``Environment.state`` advances every active skill to the current time before assembling
the state, so *reading* the state would dispatch a robot action. That is safe here only
because a bridge skill always completes within the single ``act()`` call that created it:
it is non-interruptible, and its completion effects fire on the second advance, after
which ``act()`` drops it from the active list. A future step-wise or interruptible variant
would have to move dispatch out of ``advance`` to keep that property.

**On the clock.** ``time_to_next_event`` returns the *planned* duration, so railroad's
simulated time advances by what the cost model predicted, not by how long the robot
actually took (recorded in :attr:`wall_seconds` for diagnostics). With one robot the
clock only accumulates plan cost, so this is harmless. It would not be with two: making
it exact needs the polling pattern -- step-wise skills plus a ``loop_callback_fn`` that
ticks the sim -- which is also what interruptible moves would require.
"""

from __future__ import annotations

import time as _time

from railroad.environment import SymbolicSkill

from g1sim.skills.types import PICK_RADIUS, PLACE_RADIUS, SkillResult

# How close to press when the destination is a support surface. A robot radius, which is
# the limit A*'s obstacle inflation allows anyway, and the same decomposition PICK_RADIUS is
# derived from in g1sim/skills/types.py: robot radius plus the object's inset from the
# surface edge. Deliberately an absolute number rather than a fraction of PICK_RADIUS --
# it is a property of the robot's body and the planner's inflation, so raising the arm's
# reach must not push the robot's approach further out from the furniture.
SURFACE_REACH = 0.35


class SkillBridge(SymbolicSkill):
    """Executes ``move`` / ``pick`` / ``place`` against a live skills object."""

    def __init__(self, action, start_time: float, env) -> None:
        super().__init__(action=action, start_time=start_time)
        self.env = env
        parts = action.name.split()
        self.verb = parts[0]
        self.robot = parts[1]
        self.args = parts[2:]
        self.result = None
        self.wall_seconds: float | None = None
        self._dispatched = False

    # -- ActiveSkill ------------------------------------------------------
    def advance(self, time: float, env) -> None:
        # Immediate effects first: the robot must be marked busy before it moves.
        # Idempotent -- on later calls nothing is still due at start_time.
        super().advance(self._start_time, env)

        if not self._dispatched:
            self._dispatched = True
            started = _time.perf_counter()
            self.result = self._run(env.skills, env.domain)
            self.wall_seconds = _time.perf_counter() - started
            env.last_result = self.result
            env.log(f"[{self.verb}] {self.result}  ({self.wall_seconds:.1f}s wall)")
            # Closed loop: believe the map, not the effects.
            env.observe()
            if not self.result:
                self._upcoming_effects = []      # completion effects never happened
                env.on_skill_failed(self.verb, self.robot, self.failure_target,
                                    self.result)
                return

        super().advance(time, env)

    # -- dispatch ---------------------------------------------------------
    @property
    def failure_target(self) -> str:
        """The argument a failure is attributed to: the destination of a move, the
        object of a pick, the location of a place."""
        if self.verb == "move":
            return self.args[1]          # move ?r ?from ?to
        if self.verb == "pick":
            return self.args[1]          # pick ?r ?loc ?obj
        return self.args[0]              # place ?r ?loc ?obj

    def _run(self, skills, domain):
        if self.verb == "move":
            return self._goto(skills, domain, self.args[1])
        if self.verb == "pick":
            return self._pick(skills, domain, self.args[1])
        if self.verb == "place":
            return self._place(skills, domain, self.args[0], self.args[1])
        raise ValueError(f"no skill bound to action verb {self.verb!r}")

    def _goto(self, skills, domain, dest: str, *, reach: float = SURFACE_REACH):
        """A room symbol is a floor position; a surface symbol is a thing to walk up to
        and face.

        Surfaces are approached to ``SURFACE_REACH`` rather than the default
        ``PICK_RADIUS``, because a move to a surface is almost always a move to grasp
        something *on* it. Stopping at a full ``PICK_RADIUS`` from the table leaves nothing
        for the cup's own inset from the table edge -- which is exactly how the first sim
        run failed: it arrived 0.80 m from a cabinet (the whole budget, as PICK_RADIUS was
        0.80 then) and found the cup 0.95 m away.

        A move that stops short of that tighter goal is still counted as a success if the
        surface is within ``PICK_RADIUS``: pressing closer is an optimisation, and
        furniture the robot cannot quite reach around is not a reason to declare the
        location unreachable and ban it.
        """
        if dest in domain.rooms:
            return skills.goto_room(dest)

        res = skills.goto_object(dest, reach=reach)
        if not res:
            surface = skills.smap.get(dest)
            if surface is not None and surface.xy_dist(*skills.xy()) <= PICK_RADIUS:
                self.env.log(f"[move] stopped {res.data.get('distance', float('nan')):.2f} m "
                             f"from {dest} (wanted {reach:.2f}), still within reach")
                return SkillResult(True, "goto_object", f"within reach of {dest}",
                                   data=dict(res.data))
        return res

    def _pick(self, skills, domain, obj: str):
        """Pick, after closing any last few centimetres of reach.

        ``at ?r ?loc`` means the robot is within reach of the *surface*; a particular
        item sitting on it can still be marginally further away (this is the
        ``PICK_RADIUS`` argument in ``g1sim/skills/types.py`` -- required reach is the
        robot radius plus however far in from the edge the object sits). Re-approaching
        the object itself is a sub-metre creep, not navigation, so doing it here rather
        than making the planner model it keeps the symbolic problem honest and small.

        The target here is the *object*, so the approach asks for ``PICK_RADIUS``, not the
        tight surface standoff: pressing to a robot radius of a teacup is neither possible
        nor necessary.
        """
        o = skills.smap.get(obj)
        if o is not None and o.xy_dist(*skills.xy()) > PICK_RADIUS:
            self.env.log(f"[pick] {obj} out of reach, closing in")
            self._goto(skills, domain, obj, reach=PICK_RADIUS)
        return skills.pick(obj)

    def _place(self, skills, domain, loc: str, obj: str):
        """Place at the current location, re-approaching it first if the robot has
        drifted out of reach (same argument as :meth:`_pick`)."""
        surface = skills.smap.get(loc)
        if surface is not None and surface.xy_dist(*skills.xy()) > PLACE_RADIUS:
            self.env.log(f"[place] {loc} out of reach, closing in")
            self._goto(skills, domain, loc)
        return skills.place(loc)
