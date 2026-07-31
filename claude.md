I have isaacsim installed. You can run isaacsim by doing the following:

```script
cd ~/isaac
source .venv/bin/activate
isaacsim
```

I have an apartment usd file in the folder: `/home/abhish/isaac/InteriorAgent/kujiale_0021/kujiale_0021.usda`

# Goal
Develop a task planning framework for a humanoid robot (Unitree G1) that goes to
places in the environment, picks things up from one place, places them in another,
and so on. The eventual target is **task planning for a humanoid in any household
environment, on a real robot**.

# Progress so far (working in sim)

The mobility + sensing + mapping + planning + navigation half is done. Reusable
core lives in the `g1sim/` package; thin entry-point scripts drive it.

- **Spawn + locomotion** — G1 (29-DOF) spawned in the apartment; walks via a
  pretrained "agile locomotion" policy. Command is `[vx, vy, wz, hip_height]`.
  (`g1sim/locomotion.py`, `spawn_g1_apartment.py`, `teleop_g1_apartment.py`)
- **Sensors** — torso-mounted 3D lidar + forward RGB-D camera, with 3 live
  in-GUI viewports (RGB, depth, lidar). (`g1sim/scene.py`, `sensors_g1_apartment.py`)
- **Point-to-point navigation** — closed-loop unicycle controller to `(x, y)`.
  (`g1sim/navigation.py`, `navigate_g1_apartment.py`)
- **Mapping + A\* obstacle avoidance** — 2D occupancy grid built from the robot's
  own sensors (SLAM's mapping half; ground-truth pose used as localization stand-in,
  NOT the apartment USD geometry), A* planning with robot-radius inflation, and
  **optimistic online navigation**: head straight for the goal treating unobserved
  space as free, map with the lidar while walking, re-plan around newly-sensed
  obstacles, plus a live in-GUI occupancy-map/goal/path window.
  (`g1sim/mapping.py`, `g1sim/planning.py`, `map_and_navigate_g1_apartment.py`)

Key learnings are recorded in the project memory (`g1-apartment-stack`).

# Roadmap to the goal

## Phase 0 - Semantic map builder from USD & skills test
Get an end-to-end task working with *stubs* first, then deepen each piece.
1. **Semantic map builder**: The USD of the apartment should be used to create a semantic map ("mug @ table @ livingroom @ (x, y)), ("livingroom @ (x, y)). I don't know the exact data-structure for this but semantic information should be generated for the environment using USD so that skills can be checked. Later on, in other phases, we use perception module to build this.
2. **Skill API (`g1sim/skills.py`)** — expose composable verbs the planner calls: `goto(x, y)` (have it), `scan()`, `pick(object)`, `place(location)`; each returns success/failure. The modular `g1sim/` design already sets this up. For now, pick and place can be 'magic' pick where it just attaches the object to the arm.
3. **Pick & place in sim** — re-enable `RigidBodyAPI` on the target object; start with a scripted/attach-based grasp to prove the loop before investing in real grasping. IsaacLab ships a **G1 loco-manipulation pick_place env** using the same agile lower-body policy + upper-body IK — mine it.

## Phase 1 — Close the full task loop in sim (build on the current stack)
Get an end-to-end task working with *stubs* first, then deepen each piece.
1. **Semantic perception** — run open-vocabulary detection (GroundingDINO+SAM, Detic,
   or a VLM) on the RGB-D camera and back-project detections into the map as *labeled
   object poses*, turning the occupancy grid into a **semantic map** ("mug @ (x, y)").
2. **Task planner** — an **LLM/VLM planner** a natural-language goal + semantic-map state, emits a skill sequence, re-plans on
   failure. Best fit for open-ended household tasks. Or a **PDDL** planner but somehow we have to make this fit open-ended household tasks.

## Phase 2 — Generalize to any household
5. **Many environments** — use Infinigen / InteriorAgent (procedural scene generation,
   already in `~/isaac`) to test across many houses. Make the pipeline
   **environment-agnostic**: `scene.py` currently hardcodes kujiale room scopes,
   spawn points, and `NAV_LIDAR_TARGETS`; the lidar should sense obstacles generically
   (all collidable geometry / PhysX colliders) with no per-house lists. Measure task
   success rate across generated houses.

## Phase 3 — Sim-to-real (the long pole)
6. **Localization** — replace ground-truth pose with real **SLAM** (lidar or
   visual-inertial); add odometry/sensor noise in sim first so mapping+nav degrade
   gracefully before hardware.
7. **Policy transfer** — domain-randomize dynamics/sensors; deploy locomotion +
   manipulation on the real Unitree G1 via its SDK; onboard compute, latency, safety.


# Immediate next step:
Phase 0: with a stub grasp — define the skill API and run *one* end-to-end task (e.g., "go to kitchen table → pick object → bring to living room") with a placeholder grasp. Of course the task depends on objects present in the environment so take a look at the semantic map.


# Hard Problems (not to tackle right now)
- Dexterous grasping of arbitrary objects.
- Robust open-world perception.
- Humanoid sim-to-real locomotion (bipeds are far less forgiving than quadrupeds).
- Whole-body loco-manipulation (walking while carrying).

# Architectural principle
Keep skills modular and the planner environment-agnostic, so the *same* planner runs
in sim and on hardware — only the low-level skill implementations and the localization
source swap out. The `g1sim/` layering already respects this; protect it.
