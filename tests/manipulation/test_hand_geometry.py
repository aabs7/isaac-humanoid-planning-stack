"""The hand is not symmetric, and neither of the ways it is asymmetric is visible in a log.

Both bugs guarded here produced a run that looked entirely healthy -- the base walked to the
counter, the IK reported converging on its target to within a centimetre, the fingers were
commanded shut -- and ended with the block knocked over or untouched:

* The grasp pocket was measured on the *right* hand but mirrored as if it had been authored
  for the left, so every target was 4 cm off across the jaw. On a 4 cm block that is the
  whole object.
* The two hands' finger joints are authored with opposite signs in the USD. The right hand's
  closing command drives the left hand into its lower limit, where it sits open.

The third test covers the reach clamp, which exists because an out-of-reach target does not
fail politely: the solver stretches the whole upper body toward it, the balance policy
answers the lurch by stepping away, and the target ends up further out of reach than it
started.
"""

import numpy as np
import pytest

from g1sim.manipulation.arm_ik import (
    FINGER_POSE, GRASP_APPROACH, GRASP_OFFSET, MAX_REACH, _mirror, clamp_reach, finger_target,
)

# Position limits as the articulation reports them, right hand then left. Read off the
# spawned G1 (`Simulation Joint Information`); the mirroring is in the asset, not a choice.
JOINT_LIMITS = {
    "right": {
        "hand_index_0_joint": (0.0, 1.571), "hand_index_1_joint": (0.0, 1.745),
        "hand_middle_0_joint": (0.0, 1.571), "hand_middle_1_joint": (0.0, 1.745),
        "hand_thumb_0_joint": (-1.047, 1.047), "hand_thumb_1_joint": (-1.047, 0.724),
        "hand_thumb_2_joint": (-1.745, 0.0),
    },
    "left": {
        "hand_index_0_joint": (-1.571, 0.0), "hand_index_1_joint": (-1.745, 0.0),
        "hand_middle_0_joint": (-1.571, 0.0), "hand_middle_1_joint": (-1.745, 0.0),
        "hand_thumb_0_joint": (-1.047, 1.047), "hand_thumb_1_joint": (-0.724, 1.047),
        "hand_thumb_2_joint": (0.0, 1.745),
    },
}


@pytest.mark.parametrize("side, thumb_sign", [("right", +1.0), ("left", -1.0)])
def test_the_pocket_and_the_approach_are_on_the_thumb_side(side, thumb_sign):
    """The jaw closes toward the thumb, so the pocket must sit on that side of the wrist and
    the hand must travel that way to take hold of something. Mirror them the wrong way round
    and the hand reaches to the far side of the object and closes on air."""
    assert np.sign(_mirror(GRASP_OFFSET, side)[1]) == thumb_sign
    assert np.sign(_mirror(GRASP_APPROACH, side)[1]) == thumb_sign


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("joint", list(FINGER_POSE))
def test_commanded_finger_angles_stay_inside_that_hand_s_limits(side, joint):
    low, high = JOINT_LIMITS[side][joint]
    for closed in (0.0, 0.5, 1.0):
        angle = finger_target(joint, side, closed)
        assert low - 1e-6 <= angle <= high + 1e-6, (
            f"{side} {joint} commanded {angle:.3f}, outside [{low}, {high}] -- this hand "
            f"cannot close")


@pytest.mark.parametrize("side", ["right", "left"])
def test_closing_actually_moves_the_fingers(side):
    """A sign error shows up as limit violations *or* as a hand that never leaves open."""
    moved = [abs(finger_target(j, side, 1.0) - finger_target(j, side, 0.0))
             for j in FINGER_POSE]
    assert sum(m > 0.5 for m in moved) >= 4, "fewer than four finger joints actually close"


SHOULDER = np.array([0.0, -0.14, 0.29])     # right shoulder in the pelvis frame


def test_a_target_inside_the_workspace_is_left_alone():
    target = SHOULDER + np.array([0.2, 0.0, -0.1])
    assert np.allclose(clamp_reach(target, SHOULDER), target)


def test_a_target_past_the_workspace_is_pulled_onto_the_sphere():
    direction = np.array([0.6, 0.2, -0.3])
    target = SHOULDER + direction
    clamped = clamp_reach(target, SHOULDER)
    assert np.linalg.norm(clamped - SHOULDER) == pytest.approx(MAX_REACH)
    # Same bearing from the shoulder -- the arm still points at what it was asked for.
    unit = lambda v: v / np.linalg.norm(v)
    assert np.allclose(unit(clamped - SHOULDER), unit(direction))
