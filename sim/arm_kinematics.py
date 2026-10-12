"""Forward and inverse kinematics for the 5-DOF arm.

Joint order is (base_yaw, shoulder, elbow, wrist_pitch, wrist_roll).  Five
joints cannot reach an arbitrary 6-DOF pose, but pinning the approach to
straight down leaves a planar 2R solve, which is closed form.

The planar solve works in the arm's frame after the yaw: +x radially outward
from the column, +z up, pitch measured from +z towards +x.
"""

import math

import numpy as np

from . import board_layout as L

TWO_PI = 2 * math.pi


def wrap(a):
    """Fold an angle into (-pi, pi]."""
    return (a + math.pi) % TWO_PI - math.pi


def forward(q):
    """Tool-centre pose for joint vector `q`.

    Returns (xyz, jaw_angle, pitch): the jaw-opening axis heading in world
    coordinates, and the approach angle from +z, where pi points straight down.
    """
    q1, q2, q3, q4, q5 = q
    a2, a3, a4 = q2, q2 + q3, q2 + q3 + q4

    r = L.L_UPPER * math.sin(a2) + L.L_FORE * math.sin(a3) + L.L_TOOL * math.sin(a4)
    z = (L.SHOULDER_Z + L.L_UPPER * math.cos(a2) + L.L_FORE * math.cos(a3)
         + L.L_TOOL * math.cos(a4))

    xyz = np.array([L.ARM_BASE[0] + r * math.cos(q1),
                    L.ARM_BASE[1] + r * math.sin(q1),
                    L.ARM_BASE[2] + z])
    return xyz, wrap(math.pi + q1 - q5), wrap(a4)


def elbow_height(q):
    """World z of the elbow joint, used to prefer an elbow-up solution."""
    return L.ARM_BASE[2] + L.SHOULDER_Z + L.L_UPPER * math.cos(q[1])


class Unreachable(ValueError):
    """Raised when a requested tool pose lies outside the arm's workspace."""


def inverse(x, y, z, jaw_angle=0.0):
    """Joint vector putting the tool centre at (x, y, z), pointing down.

    `jaw_angle` is the world heading of the jaw-opening axis; 0 closes the jaws
    along the file direction.  Raises Unreachable outside the workspace.
    """
    dx = x - L.ARM_BASE[0]
    dy = y - L.ARM_BASE[1]
    q1 = math.atan2(dy, dx)

    # Wrist centre in the vertical plane, relative to the shoulder.
    dr = math.hypot(dx, dy)
    dz = (z - L.ARM_BASE[2] + L.L_TOOL) - L.SHOULDER_Z
    dist = math.hypot(dr, dz)
    if dist > L.MAX_REACH - 1e-6:
        raise Unreachable(
            f"({x:.3f}, {y:.3f}, {z:.3f}) needs a {dist:.3f} m span, "
            f"arm reaches {L.MAX_REACH:.3f} m")
    if dist < L.MIN_REACH + 1e-6:
        raise Unreachable(
            f"({x:.3f}, {y:.3f}, {z:.3f}) is {dist:.3f} m from the shoulder, "
            f"inside the {L.MIN_REACH:.3f} m dead zone")

    cos_e = (dist ** 2 - L.L_UPPER ** 2 - L.L_FORE ** 2) / (2 * L.L_UPPER * L.L_FORE)
    elbow = math.acos(max(-1.0, min(1.0, cos_e)))
    phi = math.atan2(dr, dz)

    solutions = []
    for q3 in (elbow, -elbow):
        q2 = phi - math.atan2(L.L_FORE * math.sin(q3),
                              L.L_UPPER + L.L_FORE * math.cos(q3))
        q4 = wrap(math.pi - (q2 + q3))
        # The jaw axis is symmetric under a half turn, so either roll works;
        # take whichever sits further from the roll limit.
        base_roll = wrap(q1 + math.pi - jaw_angle)
        for q5 in sorted((base_roll, wrap(base_roll + math.pi)),
                         key=lambda v: abs(v)):
            q = np.array([wrap(q1), wrap(q2), wrap(q3), q4, q5])
            if _within_limits(q):
                solutions.append(q)
                break

    if not solutions:
        raise Unreachable(
            f"({x:.3f}, {y:.3f}, {z:.3f}) with jaw {math.degrees(jaw_angle):.0f} deg "
            f"needs a joint outside its limit")

    # Elbow up keeps the forearm clear of the pieces standing on the board.
    return max(solutions, key=elbow_height)


def _within_limits(q, margin=1e-3):
    for value, name in zip(q, L.JOINT_NAMES):
        lo, hi = L.JOINT_RANGE[name]
        if not (lo + margin <= value <= hi - margin):
            return False
    return True


def reachable(x, y, z=None, jaw_angle=0.0):
    """True if `inverse` can solve for this point."""
    try:
        inverse(x, y, L.GRASP_Z if z is None else z, jaw_angle)
        return True
    except Unreachable:
        return False
