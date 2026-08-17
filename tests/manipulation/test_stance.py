"""Where the robot stands decides whether the arm can reach at all.

The G1's arm spans ~0.37 m from shoulder to wrist, so a stance is not a matter of taste: a
few centimetres too far back and the target is outside the workspace, whereupon the reach
clamp fires and the hand stops short of the object. These guard the two things
``stance_for`` has to get right -- the standoff, and putting the *working shoulder* rather
than the sternum in line with the work.
"""

import math

import numpy as np
import pytest

from g1sim.environment.table_top import (
    BLOCK_SIZE, SHOULDER_OFFSET, STANDOFF, TABLE_HEIGHT, PICK_POS, stance_for,
)
from g1sim.manipulation.arm_ik import GRASP_APPROACH, GRASP_OFFSET, MAX_REACH, _mirror

PELVIS_Z = 0.78        # standing hip height
SHOULDER_RISE = 0.29   # shoulder above the pelvis, measured off the spawned robot


def wrist_target(work_pos, side, counter_height=TABLE_HEIGHT, back=0.0):
    """Where the wrist has to be, and how far that is from the shoulder, to put the hand's
    pocket on a point on a counter -- the question that decides whether a stance works.

    Returns ``(distance_from_shoulder, )``. The robot faces +x, so pelvis axes and world
    axes line up and the grasp offset can be applied directly.
    """
    stance = stance_for(work_pos, side)
    shoulder = np.array([stance[0], work_pos[1], PELVIS_Z + SHOULDER_RISE])
    work = np.array([work_pos[0], work_pos[1], counter_height + BLOCK_SIZE[2] / 2])
    pocket = work - back * _mirror(GRASP_APPROACH, side)
    return float(np.linalg.norm(pocket - _mirror(GRASP_OFFSET, side) - shoulder))


@pytest.mark.parametrize("side, expected_sign", [("right", +1.0), ("left", -1.0)])
def test_the_working_shoulder_lines_up_with_the_work(side, expected_sign):
    x, y = stance_for(PICK_POS, side)
    assert math.copysign(1.0, y - PICK_POS[1]) == expected_sign
    shoulder_y = y - expected_sign * SHOULDER_OFFSET
    assert shoulder_y == pytest.approx(PICK_POS[1]), "the object is not in front of the arm"
    assert PICK_POS[0] - x == pytest.approx(STANDOFF)


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("back", [0.0, 0.12])
def test_both_the_pre_grasp_and_the_grasp_are_inside_the_arm_s_reach(side, back):
    """The whole reason the standoff is 0.33 and not, say, 0.5. Note this asks about the
    *wrist*: the object is further away than the arm can reach and always will be -- the
    hand covers the last 0.115 m."""
    span = wrist_target(PICK_POS, side, back=back)
    assert span < MAX_REACH, f"wrist target {span:.3f} m out, arm clamps at {MAX_REACH}"


def test_the_stance_keeps_the_robot_clear_of_the_counter():
    """A stance that reaches beautifully but stands inside the counter is worth nothing: the
    legs foul it on the way in and the balance policy spends the whole reach recovering."""
    counter_near_face = PICK_POS[0] - 0.10            # block sits 0.10 m in from the edge
    x, _ = stance_for(PICK_POS)
    assert counter_near_face - x > 0.20, "less than 20 cm between pelvis and counter"


def test_a_dining_table_would_put_the_same_grasp_out_of_reach():
    """Why the counters are 0.85 m and not the 0.75 m a table usually is: 10 cm lower puts
    the object 0.30 m below the shoulder, and an arm spending that much of itself reaching
    down has none left to reach forward."""
    assert wrist_target(PICK_POS, "right", counter_height=0.75) > MAX_REACH
