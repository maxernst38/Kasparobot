"""Trajectory control and pick-and-place for the 5-DOF arm.

Position actuators on gravity-compensated links, so an interpolated joint
target tracks to well under a millimetre.  Heights and poses above are all
expressed at the tool centre point between the jaws.
"""

import math

import numpy as np

import mujoco

from . import arm_kinematics as K
from . import board_layout as L

TRANSIT_Z = 0.13      # tool height while crossing the board
HOVER_Z = 0.075       # tool height directly above a square before descending
RELEASE_Z = L.GRASP_Z + 0.0005   # piece base effectively resting on the board
SETTLE_TOL = 5e-4     # rad; ~0.3 mm of tool error at full extension


class ArmController:
    """Drives the arm in a model built by `sim.scene_builder`."""

    def __init__(self, model, data, on_step=None, speed=0.45):
        self.model = model
        self.data = data
        self.on_step = on_step          # called after every physics step
        self.speed = speed              # metres per second of tool travel

        name2id = lambda t, n: mujoco.mj_name2id(model, t, n)
        self.jnt_qpos = np.array(
            [model.jnt_qposadr[name2id(mujoco.mjtObj.mjOBJ_JOINT, n)]
             for n in L.JOINT_NAMES])
        self.act = np.array(
            [name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in L.JOINT_NAMES])
        self.finger_act = np.array(
            [name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, n)
             for n in ("finger_left", "finger_right")])
        self.tool_site = name2id(mujoco.mjtObj.mjOBJ_SITE, "tool")
        self.palm_body = name2id(mujoco.mjtObj.mjOBJ_BODY, "palm")
        self._held = None               # name of the welded piece, if any

    @property
    def q(self):
        return self.data.qpos[self.jnt_qpos].copy()

    @property
    def tool_xyz(self):
        return self.data.site_xpos[self.tool_site].copy()

    @property
    def held(self):
        return self._held

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            if self.on_step is not None:
                self.on_step()

    def settle(self, seconds=0.3):
        self.step(int(seconds / self.model.opt.timestep))

    def goto_joint(self, q_target, duration, converge=True):
        """Interpolate the actuator targets with a cosine profile.

        Position actuators lag their target, so hold the final target until the
        joints arrive; otherwise the tool ends up centimetres short.
        """
        q_target = np.asarray(q_target, dtype=float)
        q_start = self.data.ctrl[self.act].copy()
        steps = max(2, int(duration / self.model.opt.timestep))
        for i in range(1, steps + 1):
            s = 0.5 * (1 - math.cos(math.pi * i / steps))
            self.data.ctrl[self.act] = q_start + s * (q_target - q_start)
            self.step()
        if converge:
            self.converge(q_target)

    def converge(self, q_target, tol=SETTLE_TOL, timeout=1.5):
        """Hold a joint target until the arm has settled onto it."""
        for _ in range(int(timeout / self.model.opt.timestep)):
            if np.max(np.abs(self.q - q_target)) < tol:
                return True
            self.step()
        return False

    def goto_tool(self, x, y, z, jaw=0.0, duration=None):
        """Move the tool centre to a point, approaching straight down."""
        q = K.inverse(x, y, z, jaw)
        if duration is None:
            travel = np.linalg.norm(self.tool_xyz - np.array([x, y, z]))
            duration = max(0.25, travel / self.speed)
        self.goto_joint(q, duration)
        return q

    # Park pose.  Every link stays behind the board, and the camera looks
    # across from the opposite side, so a parked arm never covers a square.
    HOME_TOOL = (0.35, L.ARM_BASE[1] - 0.10, 0.28)

    def home(self, duration=1.2):
        """Park clear of the board."""
        self.set_gripper(L.GRIP_OPEN, settle=False)
        q = K.inverse(*self.HOME_TOOL, jaw_angle=0.0)
        self.goto_joint(q, duration)
        return q

    def set_gripper(self, opening, settle=True, duration=0.25):
        self.data.ctrl[self.finger_act] = opening
        if settle:
            self.step(int(duration / self.model.opt.timestep))

    def open_gripper(self):
        self.set_gripper(L.GRIP_OPEN)

    def close_gripper(self):
        self.set_gripper(L.GRIP_CLOSED, duration=0.35)

    def _weld_id(self, body_name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                 f"grasp_{body_name}")

    def attach(self, body_name):
        """Weld a piece to the palm at its current relative pose.

        Friction alone survives a careful lift, but a game is hundreds of them
        and one slip corrupts the board.  The weld freezes whatever pose the
        jaws actually achieved, so a bad grasp still shows as a misplaced piece.
        """
        eq = self._weld_id(body_name)
        if eq < 0:
            raise KeyError(f"no weld defined for body {body_name!r}")
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        # relpose = pose of body1 (palm) expressed in body2 (piece) frame
        p_palm, q_palm = self.data.xpos[self.palm_body], self.data.xquat[self.palm_body]
        p_piece, q_piece = self.data.xpos[bid], self.data.xquat[bid]
        neg = np.zeros(4)
        mujoco.mju_negQuat(neg, q_piece)
        dp = np.zeros(3)
        mujoco.mju_rotVecQuat(dp, p_palm - p_piece, neg)
        dq = np.zeros(4)
        mujoco.mju_mulQuat(dq, neg, q_palm)

        self.model.eq_data[eq, 0:3] = 0.0
        self.model.eq_data[eq, 3:6] = dp
        self.model.eq_data[eq, 6:10] = dq
        self.model.eq_data[eq, 10] = 1.0
        self.data.eq_active[eq] = 1
        self._held = body_name

    def detach(self):
        if self._held is None:
            return
        self.data.eq_active[self._weld_id(self._held)] = 0
        self._held = None

    def pick(self, x, y, body_name, jaw=0.0):
        """Open, descend onto the piece's stem, close, weld and lift."""
        self.open_gripper()
        self.goto_tool(x, y, TRANSIT_Z, jaw)
        self.goto_tool(x, y, HOVER_Z, jaw, duration=0.35)
        self.goto_tool(x, y, L.GRASP_Z, jaw, duration=0.45)
        self.close_gripper()
        self.attach(body_name)
        self.goto_tool(x, y, TRANSIT_Z, jaw, duration=0.5)

    def place(self, x, y, jaw=0.0):
        """Carry to the target, set the piece down, release and retreat."""
        self.goto_tool(x, y, TRANSIT_Z, jaw)
        self.goto_tool(x, y, HOVER_Z, jaw, duration=0.35)
        self.goto_tool(x, y, RELEASE_Z, jaw, duration=0.45)
        self.detach()
        self.settle(0.1)          # let the piece take its own weight first
        self.open_gripper()
        self.goto_tool(x, y, TRANSIT_Z, jaw, duration=0.5)

    def transfer(self, src, dst, body_name, jaw_src=0.0, jaw_dst=0.0):
        self.pick(src[0], src[1], body_name, jaw_src)
        self.place(dst[0], dst[1], jaw_dst)


def clearest_jaw_angle(x, y, occupied):
    """Pick the jaw heading with the most room around a square.

    `occupied(x, y)` reports whether a piece stands near a point.  The jaws
    need room on two opposite sides, so headings are scored in pairs.
    """
    best, best_score = 0.0, -1
    for angle in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        dx, dy = math.cos(angle) * L.SQUARE, math.sin(angle) * L.SQUARE
        score = 2 - occupied(x + dx, y + dy) - occupied(x - dx, y - dy)
        # A diagonal neighbour is further away, so prefer it on a tie.
        score = score * 2 + (1 if abs(math.sin(2 * angle)) > 0.5 else 0)
        if score > best_score:
            best, best_score = angle, score
    return best
