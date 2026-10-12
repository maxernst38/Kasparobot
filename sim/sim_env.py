"""The simulated chess cell: board, 5-DOF arm, camera.

    sim = ChessSim(viewer=True)
    sim.apply_move("e4")
    img = sim.capture()

The arm parks clear of the board before every capture, so a frame always shows
a static position.
"""

import time

import chess
import cv2
import numpy as np

import mujoco

from . import arm_kinematics as K
from . import board_layout as L
from . import scene_builder
from .arm_controller import ArmController, clearest_jaw_angle
from .move_executor import REMOVE, SHIFT, SPAWN, plan_move

SETTLE_RADIUS = 0.022     # how close a piece must be to count as "on" a point
UPRIGHT_TOL = np.cos(np.deg2rad(30))


class OutOfPieces(RuntimeError):
    """The scene has no spare body left for what the position asks for."""


class ChessSim:
    def __init__(self, fen=None, viewer=False, realtime=False, speed=0.45,
                 markers=False, xml_path=None, verbose=True):
        """`markers` adds a coloured type disc to each piece's crown, which is
        what `MarkerPerception` reads.  Off by default: the set renders plain.
        """
        self.markers = markers
        if xml_path is None:
            xml_path = scene_builder.write_scene(markers=markers)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.verbose = verbose
        self.realtime = realtime

        self._viewer = None
        self._renderers = {}
        self._rec = None
        self._cam_view = None
        self._cam_every = 0
        self._cam_n = 0
        self._last_sync = 0.0
        if viewer:
            import mujoco.viewer as mj_viewer   # binding `mujoco` here would
            self._viewer = mj_viewer.launch_passive(   # shadow the module
                self.model, self.data, show_left_ui=False, show_right_ui=False)

        self.arm = ArmController(self.model, self.data, on_step=self._on_step,
                                 speed=speed)

        self._bid = {}
        self._qadr = {}
        self._pool = {}
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name or self.model.body_jntnum[i] != 1:
                continue
            jadr = self.model.body_jntadr[i]
            if self.model.jnt_type[jadr] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            self._bid[name] = i
            self._qadr[name] = self.model.jnt_qposadr[jadr]
            parts = name.split("_")
            key = (parts[1], parts[2]) if parts[0] == "stock" else (parts[0], parts[1])
            self._pool.setdefault(key, []).append(name)
        for key in self._pool:                 # board pieces before spares
            self._pool[key].sort(key=lambda n: (n.startswith("stock_"), n))

        # Original render groups, so a hidden body can be restored exactly.
        self._geom_group0 = self.model.geom_group.copy()

        self.board = chess.Board()
        self.piece_at = {}        # square name -> body name
        self.off_board = {}       # body name -> (x, y)
        self._free_slots = []
        self.reset(fen)

    def attach_camera_view(self, view, fps=8):
        """Refresh `view` while the arm moves, at roughly `fps` sim-fps."""
        self._cam_view = view
        self._cam_every = max(1, round(1 / (fps * self.model.opt.timestep)))
        self._cam_n = 0
        return view

    def _on_step(self):
        if self._cam_view is not None:
            self._cam_n += 1
            if self._cam_n % self._cam_every == 0:
                # Raw frame, arm included: that is what the camera sees.
                self._cam_view.update(overlay=False, save=False)
        if self._rec is not None:
            writer, every, cam, size, n = self._rec
            if n % every == 0:
                frame = self.capture(size[0], size[1], camera=cam, park=False)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self._rec[4] = n + 1
        if self._viewer is not None:
            now = time.perf_counter()
            if now - self._last_sync > 1 / 60:
                self._viewer.sync()
                self._last_sync = now
            if self.realtime:
                lag = self.data.time - (now - self._t0)
                if lag > 0:
                    time.sleep(min(lag, 0.01))

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def start_recording(self, path, fps=30, camera="oblique", size=(960, 720)):
        """Record to `path` while the arm works."""
        # Spelled both ways across OpenCV 4.x and 5.x.
        make_fourcc = getattr(cv2.VideoWriter, "fourcc", None)
        if make_fourcc is None:
            make_fourcc = cv2.VideoWriter_fourcc
        fourcc = make_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not writer.isOpened():
            raise RuntimeError(f"could not open {path} for writing")
        every = max(1, round(1 / (fps * self.model.opt.timestep)))
        self._rec = [writer, every, camera, size, 0]
        return self

    def stop_recording(self):
        if self._rec is not None:
            self._rec[0].release()
            self._rec = None

    def close(self):
        self.stop_recording()
        if self._cam_view is not None:
            self._cam_view.close()
            self._cam_view = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def body_xy(self, body):
        return self.data.xpos[self._bid[body]][:2].copy()

    def _place_body(self, body, x, y, z=0.0004):
        adr = self._qadr[body]
        self.data.qpos[adr:adr + 7] = [x, y, z, 1, 0, 0, 0]
        dof = self.model.jnt_dofadr[self.model.body_jntadr[self._bid[body]]]
        self.data.qvel[dof:dof + 6] = 0

    HIDDEN_GROUP = 4          # MuJoCo renders groups 0-2 by default

    def set_body_visible(self, body, visible=True):
        """Show or hide a body in renders, without touching its physics.

        From a side view the off-board parking spots sit between the lens and
        the board, so parking a piece aside is not enough to keep it out of
        frame.
        """
        bid = self._bid[body] if body in self._bid else mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        start = self.model.body_geomadr[bid]
        for g in range(start, start + self.model.body_geomnum[bid]):
            original = int(self._geom_group0[g])
            if original >= 3:            # collision-only geoms stay hidden
                continue
            self.model.geom_group[g] = original if visible else self.HIDDEN_GROUP

    def set_arm_visible(self, visible=True):
        """Hide the robot itself, for board-only training images."""
        for name in ("arm_base", "link1", "link2", "link3", "link4", "palm",
                     "finger_left", "finger_right"):
            self.set_body_visible(name, visible)

    def reset(self, fen=None, stash=False):
        """Put every piece where `fen` says, and park the arm.

        `stash` puts unused pieces off the table edge rather than in the
        discard trays and hides them.  Promotion needs the trays, so it is only
        for rendering stills.
        """
        self.board = chess.Board(fen) if fen else chess.Board()
        mujoco.mj_resetData(self.model, self.data)
        for eq in range(self.model.neq):
            self.data.eq_active[eq] = 0
        self.arm._held = None

        pool = {k: list(v) for k, v in self._pool.items()}
        self.piece_at, self.off_board = {}, {}
        for square, piece in sorted(self.board.piece_map().items()):
            color = "w" if piece.color == chess.WHITE else "b"
            key = (color, piece.symbol().lower())
            if not pool.get(key):
                raise OutOfPieces(
                    f"position needs another {color}{key[1]} but the scene only "
                    f"has {len(self._pool.get(key, []))}")
            body = pool[key].pop(0)
            name = chess.square_name(square)
            self.piece_at[name] = body
            self._place_body(body, *L.square_to_xy(name))

        self._free_slots = list(L.capture_slots())
        stash = list(L.stash_slots()) if stash else None
        for key, leftover in pool.items():
            for body in leftover:
                if stash is not None:
                    xy = stash.pop(0)
                elif body.startswith("stock_"):
                    xy = L.stock_slot(*key)
                else:
                    xy = self._free_slots.pop(0)
                self.off_board[body] = xy
                self._place_body(body, *xy)

        # Parking alone leaves a piece in shot from a side view, so hide it too.
        for body in self._bid:
            self.set_body_visible(body, not stash or body in self.piece_at.values())

        q_home = K.inverse(*ArmController.HOME_TOOL, jaw_angle=0.0)
        self.data.qpos[self.arm.jnt_qpos] = q_home
        self.data.ctrl[self.arm.act] = q_home
        self.data.ctrl[self.arm.finger_act] = L.GRIP_OPEN
        mujoco.mj_forward(self.model, self.data)
        self._t0 = time.perf_counter()
        self.arm.settle(0.4)
        return self

    def occupied(self, x, y, ignore=()):
        """True if a piece is standing near this point on the board."""
        for body in list(self.piece_at.values()):
            if body in ignore:
                continue
            if np.hypot(*(self.body_xy(body) - np.array([x, y]))) < SETTLE_RADIUS:
                return True
        return False

    def ground_truth(self):
        """Read the board from the physics state.

        What is actually standing there, which is not the same as what the
        rules say: a toppled or misplaced piece shows up here and nowhere else.
        """
        out = {}
        for body, i in self._bid.items():
            pos = self.data.xpos[i]
            upright = self.data.xmat[i].reshape(3, 3)[2, 2] > UPRIGHT_TOL
            square = L.xy_to_square(pos[0], pos[1])
            if square is None or not upright or pos[2] > 0.02:
                continue
            parts = body.split("_")
            color, kind = ((parts[1], parts[2]) if parts[0] == "stock"
                           else (parts[0], parts[1]))
            out[square] = kind.upper() if color == "w" else kind
        return out

    def ground_truth_fen(self):
        return piece_map_to_board_fen(self.ground_truth())

    def consistent(self, tol=0.012):
        """Compare the physical layout against the rules' view of the game.

        Returns `(ok, problems)`.
        """
        problems = []
        expected = {chess.square_name(s): p.symbol()
                    for s, p in self.board.piece_map().items()}
        actual = self.ground_truth()
        for square in sorted(set(expected) | set(actual)):
            if expected.get(square) != actual.get(square):
                problems.append(
                    f"{square}: rules say {expected.get(square, '-')}, "
                    f"board shows {actual.get(square, '-')}")
        for square, body in self.piece_at.items():
            off = np.linalg.norm(self.body_xy(body) - np.array(L.square_to_xy(square)))
            if off > tol:
                problems.append(f"{body} is {off * 1000:.1f} mm off the centre of {square}")
        return (not problems), problems

    def move_camera(self, azimuth=None, elevation=None, distance=None,
                    target=None, refit=True):
        """Re-aim the floating camera around the board.

        `elevation` is degrees above the board plane (90 looks straight down),
        `azimuth` degrees around it from +x, `distance` metres to `target`.

        `refit` recomputes the field of view for the new pose; without it a
        lower or closer camera pushes the board off the edge of the frame.
        """
        pos, quat = L.camera_pose(azimuth, elevation, distance, target)
        self.data.mocap_pos[0] = pos
        self.data.mocap_quat[0] = quat
        if refit:
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                    L.CAMERA_NAME)
            self.model.cam_fovy[cid] = L.camera_fovy(azimuth, elevation,
                                                     distance, target)
        mujoco.mj_forward(self.model, self.data)
        return pos, quat

    @property
    def camera_fovy(self):
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                L.CAMERA_NAME)
        return float(self.model.cam_fovy[cid])

    def place_camera(self, pos, quat):
        """Put the camera at an explicit pose."""
        self.data.mocap_pos[0] = pos
        self.data.mocap_quat[0] = quat
        mujoco.mj_forward(self.model, self.data)

    @property
    def camera_pos(self):
        return self.data.mocap_pos[0].copy()

    def capture(self, width=None, height=None, camera=None, park=True):
        """Render a frame.  Parks the arm first so it never occludes the board."""
        camera = L.CAMERA_NAME if camera is None else camera
        if park and camera == L.CAMERA_NAME:
            self.park()
        w = width or L.CAM_RES
        h = height or w
        renderer = self._renderers.get((w, h))
        if renderer is None:
            renderer = self._renderers[(w, h)] = mujoco.Renderer(
                self.model, height=h, width=w)
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def park(self):
        if np.max(np.abs(self.arm.q - K.inverse(*ArmController.HOME_TOOL))) > 1e-3:
            self.arm.home()

    def _jaw_for(self, x, y, ignore=()):
        return clearest_jaw_angle(x, y, lambda px, py: self.occupied(px, py, ignore))

    def _take_slot(self):
        if not self._free_slots:
            raise OutOfPieces("every discard slot beside the board is full")
        return self._free_slots.pop(0)

    def _transfer(self, body, dst_xy, ignore_src=True):
        src = self.body_xy(body)
        jaw_src = self._jaw_for(*src, ignore=(body,) if ignore_src else ())
        jaw_dst = self._jaw_for(*dst_xy, ignore=(body,))
        self.arm.transfer(src, dst_xy, body, jaw_src, jaw_dst)

    def parse(self, text):
        """Accept UCI ('e2e4') or SAN ('Nf3', 'O-O', 'a8=Q')."""
        try:
            return chess.Move.from_uci(text)
        except ValueError:
            return self.board.parse_san(text)

    def apply_move(self, move, check=True):
        """Play `move` physically, then on the rules board."""
        if isinstance(move, str):
            move = self.parse(move)
        if move not in self.board.legal_moves:
            raise ValueError(f"{move} is not legal in {self.board.fen()}")

        san = self.board.san(move)
        for step in plan_move(self.board, move):
            self._log(f"    {step}")
            if step.kind == REMOVE:
                body = self.piece_at.pop(step.src)
                slot = self._take_slot()
                self._transfer(body, slot)
                self.off_board[body] = slot
            elif step.kind == SHIFT:
                body = self.piece_at.pop(step.src)
                self._transfer(body, L.square_to_xy(step.dst))
                self.piece_at[step.dst] = body
            elif step.kind == SPAWN:
                color = "w" if step.piece.isupper() else "b"
                body = f"stock_{color}_{step.piece.lower()}"
                if body not in self.off_board:
                    raise OutOfPieces(
                        f"no spare {step.piece} left to promote to; the scene "
                        f"carries one of each per colour")
                self.off_board.pop(body)
                self._transfer(body, L.square_to_xy(step.dst))
                self.piece_at[step.dst] = body

        self.board.push(move)
        self.park()
        self.arm.settle(0.2)
        if check:
            ok, problems = self.consistent()
            if not ok:
                self._log("    ! board no longer matches the game:")
                for p in problems:
                    self._log(f"      {p}")
        return san


def piece_map_to_board_fen(piece_map):
    """`{'e4': 'P', ...}` -> the placement field of a FEN."""
    rows = []
    for rank in range(8, 0, -1):
        row, empty = "", 0
        for file in L.FILES:
            piece = piece_map.get(f"{file}{rank}")
            if piece is None:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            row += piece
        rows.append(row + (str(empty) if empty else ""))
    return "/".join(rows)
