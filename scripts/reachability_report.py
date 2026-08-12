"""Which objects in this apartment could the robot actually stand close enough to grasp?

No simulator. Reads a semantic map and an occupancy grid the robot built for itself
(``scripts/build_occupancy_map.py``) and answers, per object, whether there is a place the
robot can stand from which its shoulder comes within arm's reach of the object.

This is the question ``PICK_RADIUS`` does not answer. That constant governs how close the
*base* parks to an object's 2D footprint, and it was chosen from A*'s obstacle inflation --
the comment above it says so. A grasp is decided by the *shoulder*, 1.02 m up, in 3D. The
probe measured a cup at 0.698 m of footprint distance -- comfortably inside a 1.00 m budget
-- whose shoulder distance was 0.693 m against a ~0.70 m arm, i.e. reachable by 7 mm.

Free space comes from the sensed grid, paired with a room-polygon test. Both are needed:
the grid knows where furniture is but treats never-scanned cells as empty (it has no
observed/unobserved distinction), while the room polygons exclude everything outside the
apartment and inside walls but know nothing about furniture. Neither alone is sound.

    python isaac_task_planning/scripts/reachability_report.py
    python isaac_task_planning/scripts/reachability_report.py --arm-reach 0.60 --verbose
"""

import argparse
import collections

# Make g1sim importable when this file is run directly (see scripts/_bootstrap.py).
import _bootstrap  # noqa: F401

from g1sim.perception.mapping import OccupancyGridMapper
from g1sim.perception.semantic_map import SemanticMap
from g1sim.skills.reach import (ARM_REACH, best_stance, in_a_room, occupancy_free)
from g1sim.skills.types import PICK_RADIUS

# A three-finger hand roughly 6-8 cm across. An object is only a candidate if its smallest
# dimension fits between the fingers and its largest is not unwieldy. Checked before reach
# because it is pure arithmetic and, in this apartment, rules out more objects than reach
# does: plates are 1.7-2.0 cm discs with nothing to pinch, vases are over 8 cm wide.
HAND_OPENING = 0.08
MIN_GRASPABLE = 0.010
MAX_LONGEST = 0.30


def hand_fits(o) -> tuple:
    """(fits, why-not). Judged on the sorted extents, not on max_dim alone."""
    dims = sorted(o.size)
    if dims[0] < MIN_GRASPABLE:
        return False, f"only {dims[0] * 1000:.0f} mm thick -- nothing to pinch"
    if dims[0] > HAND_OPENING:
        return False, f"{dims[0] * 100:.0f} cm across at its narrowest; the hand opens {HAND_OPENING * 100:.0f} cm"
    if dims[2] > MAX_LONGEST:
        return False, f"{dims[2] * 100:.0f} cm long -- unwieldy even if pinchable"
    return True, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map-json", default="sensor_output/semantic_map.json")
    ap.add_argument("--grid", default="sensor_output/occupancy_map.npz",
                    help="Occupancy grid from scripts/build_occupancy_map.py. Without it the "
                         "report falls back to room polygons alone, which is far too "
                         "optimistic -- it will place stances inside solid furniture.")
    ap.add_argument("--arm-reach", type=float, default=ARM_REACH)
    ap.add_argument("--robot-radius", type=float, default=0.35)
    ap.add_argument("--verbose", action="store_true", help="List every object, not a summary.")
    args = ap.parse_args()

    smap = SemanticMap.load(args.map_json)
    print(smap.describe())

    try:
        mapper = OccupancyGridMapper.load(args.grid)
        print(mapper.describe())
        grid_free = occupancy_free(mapper, robot_radius=args.robot_radius)
        room_free = in_a_room(smap)

        def free(x, y):
            return room_free(x, y) and grid_free(x, y)
        source = f"sensed grid ({args.grid}) + room polygons"
    except FileNotFoundError:
        free = in_a_room(smap)
        source = "room polygons ONLY -- run build_occupancy_map.py for a real answer"

    print(f"\nfree space: {source}")
    print(f"arm reach:  {args.arm_reach:.2f} m from a shoulder 1.02 m up")

    # ---- filter 1: does the hand fit? -------------------------------------
    candidates, rejected = [], []
    for o in smap.small_objects():
        fits, why = hand_fits(o)
        (candidates if fits else rejected).append((o, why))

    print(f"\n=== HAND FIT ===")
    print(f"  {len(candidates)} of {len(candidates) + len(rejected)} pickable objects fit "
          f"a {HAND_OPENING * 100:.0f} cm three-finger hand")
    by_reason = collections.Counter(w.split(" --")[0].split(";")[0] for _, w in rejected)
    for reason, n in by_reason.most_common(5):
        print(f"    {n:3d}  {reason}")

    # ---- filter 2: is there a stance that reaches it? ---------------------
    reachable, unreachable, nowhere = [], [], []
    for o, _ in candidates:
        r = best_stance(smap, o, arm_reach=args.arm_reach, free_xy=free)
        if r.stance is None:
            nowhere.append((o, r))
        elif r.reachable:
            reachable.append((o, r))
        else:
            unreachable.append((o, r))

    print(f"\n=== REACH (of the {len(candidates)} that fit the hand) ===")
    print(f"  {len(reachable):3d}  reachable")
    print(f"  {len(unreachable):3d}  too far in from the edge of whatever they rest on")
    print(f"  {len(nowhere):3d}  no legal stance anywhere around them")

    if reachable:
        print(f"\n=== GRASPABLE ({len(reachable)}) ===")
        for o, r in sorted(reachable, key=lambda t: t[1].required_reach):
            room = o.room or "?"
            on = f" on {o.supported_by}" if o.supported_by else ""
            print(f"  {o.name:<24} {room:<11} needs {r.required_reach:.2f} m "
                  f"from ({r.stance[0]:6.2f},{r.stance[1]:6.2f}){on}")

    if args.verbose and unreachable:
        print(f"\n=== OUT OF REACH ({len(unreachable)}) ===")
        for o, r in sorted(unreachable, key=lambda t: t[1].required_reach):
            print(f"  {o.name:<24} needs {r.required_reach:.2f} m  ({r.reason})")

    # ---- what PICK_RADIUS would have said ---------------------------------
    print(f"\n=== WHY THIS IS NOT PICK_RADIUS ===")
    admitted = sum(1 for o, r in unreachable + nowhere
                   if r.stance is not None and o.xy_dist(*r.stance) <= PICK_RADIUS)
    print(f"  PICK_RADIUS = {PICK_RADIUS:.2f} m would admit {admitted} object(s) that the arm "
          f"cannot actually reach from the same stance.")
    print(f"  Those are the tasks a planner would accept, walk to, and fail.")


if __name__ == "__main__":
    main()
