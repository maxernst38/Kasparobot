"""Turning a camera frame into a board position.

Pieces are 48-68 mm tall, so under an oblique camera a crown is displaced from
the square it stands on by up to a square and a half.  Detections are therefore
back-projected at the piece's own height, not at z=0.
"""

import math

import cv2
import numpy as np

from . import board_layout as L

# OpenCV HSV: H 0-179, S/V 0-255.  The board peaks at S=156, markers sit near
# S=220, so saturation alone separates them.
MARKER_S_MIN = 170
MARKER_V_MIN = 110
HUE_TOL = 9
# The smallest marker covers only a few dozen pixels at 1.3 m and an angle.
MIN_BLOB_AREA = 16
BODY_S_MAX = 75             # piece bodies are nearly grey, the board is not
BODY_V_SPLIT = 110          # white bodies sit near 200+, black near 40


def _hue_of(rgba):
    bgr = np.uint8([[[rgba[2] * 255, rgba[1] * 255, rgba[0] * 255]]])
    return int(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0, 0])


MARKER_HUE = {t: _hue_of(c) for t, c in L.MARKER_RGBA.items()}


class BoardCamera:
    """Projection between the board and a camera at an arbitrary pose."""

    def __init__(self, pos=None, rot=None, fovy=None, resolution=None):
        if pos is None or rot is None:
            p, quat = L.camera_pose()
            pos = p if pos is None else pos
            rot = _quat_to_mat(quat) if rot is None else rot
        self.pos = np.asarray(pos, dtype=float)
        self.rot = np.asarray(rot, dtype=float).reshape(3, 3)
        self.fovy = L.CAM_FOVY if fovy is None else float(fovy)
        self.res = L.CAM_RES if resolution is None else int(resolution)
        self.f = (self.res / 2) / math.tan(math.radians(self.fovy) / 2)

    @classmethod
    def from_sim(cls, sim, resolution=None):
        """Read the live pose out of the model, wherever the rig has been put."""
        import mujoco

        cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                L.CAMERA_NAME)
        return cls(pos=sim.data.cam_xpos[cid].copy(),
                   rot=sim.data.cam_xmat[cid].reshape(3, 3).copy(),
                   fovy=float(sim.model.cam_fovy[cid]),
                   resolution=resolution or L.CAM_RES)

    def resized(self, resolution):
        return BoardCamera(self.pos, self.rot, self.fovy, resolution)

    def project(self, x, y, z=0.0):
        """Board point -> pixel.  MuJoCo cameras look down -z with +y up."""
        rel = np.array([x, y, z], dtype=float) - self.pos
        local = self.rot.T @ rel
        depth = -local[2]
        if depth <= 1e-9:
            return (float("nan"), float("nan"))
        return (self.res / 2 + self.f * local[0] / depth,
                self.res / 2 - self.f * local[1] / depth)

    def unproject(self, u, v, z=0.0):
        """Pixel -> the point at height `z` whose image is that pixel."""
        ray_cam = np.array([(u - self.res / 2) / self.f,
                            -(v - self.res / 2) / self.f, -1.0])
        ray = self.rot @ ray_cam
        if abs(ray[2]) < 1e-12:
            return (float("nan"), float("nan"))
        t = (z - self.pos[2]) / ray[2]
        if t <= 0:
            return (float("nan"), float("nan"))
        p = self.pos + t * ray
        return (float(p[0]), float(p[1]))

    def homography(self, z=0.0):
        """3x3 mapping board metres at height `z` to pixels."""
        m = self.rot.T
        a = np.column_stack([m[:, 0], m[:, 1], z * m[:, 2] - m @ self.pos])
        k = np.array([[self.f, 0.0, -self.res / 2],
                      [0.0, -self.f, -self.res / 2],
                      [0.0, 0.0, -1.0]])
        return k @ a

    def square_corners(self, square, z=0.0):
        cx, cy = L.square_to_xy(square)
        h = L.SQUARE / 2
        return np.array([self.project(cx + sx * h, cy + sy * h, z)
                         for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

    def visible(self, x, y, z=0.0, margin=0):
        u, v = self.project(x, y, z)
        return (not math.isnan(u) and margin <= u < self.res - margin
                and margin <= v < self.res - margin)


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Detection:
    __slots__ = ("kind", "color", "u", "v", "area", "x", "y", "square")

    def __init__(self, kind, color, u, v, area, x, y, square):
        self.kind, self.color = kind, color
        self.u, self.v, self.area = u, v, area
        self.x, self.y, self.square = x, y, square

    @property
    def symbol(self):
        return self.kind.upper() if self.color == "w" else self.kind

    @property
    def radius_px(self):
        """Marker radius in pixels, from the measured blob rather than the
        geometry, so foreshortening is already accounted for."""
        return max(1.5, math.sqrt(self.area / math.pi))

    def __repr__(self):
        return (f"<{self.symbol} on {self.square} at ({self.u:.0f},{self.v:.0f})"
                f" area={self.area}>")


def _disc_values(hsv, u, v, radius):
    """Near-grey brightness values in a small disc of the image."""
    r = int(math.ceil(radius))
    u0, v0 = int(round(u)), int(round(v))
    y0, y1 = max(0, v0 - r), min(hsv.shape[0], v0 + r + 1)
    x0, x1 = max(0, u0 - r), min(hsv.shape[1], u0 + r + 1)
    if y1 <= y0 or x1 <= x0:
        return []
    patch = hsv[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = np.hypot(xx - u, yy - v) <= radius
    grey = inside & (patch[..., 1] < BODY_S_MAX)
    return list(patch[..., 2][grey])


def _ring_values(hsv, u, v, marker_px):
    r_out = int(marker_px * 1.9) + 5
    u0, v0 = int(round(u)), int(round(v))
    y0, y1 = max(0, v0 - r_out), min(hsv.shape[0], v0 + r_out + 1)
    x0, x1 = max(0, u0 - r_out), min(hsv.shape[1], u0 + r_out + 1)
    if y1 <= y0 or x1 <= x0:
        return []
    patch = hsv[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - u, yy - v)
    ring = (d > marker_px + 1.0) & (d <= r_out)
    grey = ring & (patch[..., 1] < BODY_S_MAX)
    if not grey.any():
        grey = ring
    return list(patch[..., 2][grey])


class MarkerPerception:
    """Finds pieces by the coloured type disc on the crown of each one."""

    requires_markers = True

    def __init__(self, camera=None):
        self.camera = camera or BoardCamera()

    def detect(self, rgb):
        cam = self.camera
        if rgb.shape[0] != cam.res:
            cam = cam.resized(rgb.shape[0])
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue = hsv[..., 0].astype(np.int16)
        saturated = (hsv[..., 1] >= MARKER_S_MIN) & (hsv[..., 2] >= MARKER_V_MIN)

        found = []
        for kind, want in MARKER_HUE.items():
            dh = np.abs(hue - want)
            dh = np.minimum(dh, 180 - dh)
            mask = (saturated & (dh <= HUE_TOL)).astype(np.uint8)
            n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < MIN_BLOB_AREA:
                    continue
                u, v = centroids[i]
                x, y = cam.unproject(u, v, L.PIECE_HEIGHT[kind])
                square = L.xy_to_square(x, y)
                if square is None:
                    continue          # a spare piece parked beside the board
                det = Detection(kind, None, u, v, area, x, y, square)
                det.color = self._side(hsv, det, cam)
                found.append(det)
        return found

    @staticmethod
    def _side(hsv, det, cam):
        """White or black, from the piece's own body.

        Seen obliquely the body is below the crown in the image rather than
        around it, so sample along the crown-to-base segment.  Near-grey pixels
        only: the board is strongly coloured, the pieces are not.
        """
        base_u, base_v = cam.project(det.x, det.y, 0.0)
        du, dv = base_u - det.u, base_v - det.v
        span = math.hypot(du, dv)
        probe = max(1.5, det.radius_px * 0.55)

        values = []
        if span > 5.0:
            for t in np.linspace(0.55, 0.95, 9):
                values += _disc_values(hsv, det.u + du * t, det.v + dv * t, probe)
        if not values:          # degenerate: camera almost overhead the piece
            values = _ring_values(hsv, det.u, det.v, det.radius_px)
        if not values:
            return "w"
        return "w" if float(np.median(values)) > BODY_V_SPLIT else "b"

    def read(self, rgb):
        """``{'e4': 'P', ...}`` for everything standing on the board."""
        best = {}
        for det in self.detect(rgb):
            if det.square not in best or det.area > best[det.square].area:
                best[det.square] = det
        return {sq: d.symbol for sq, d in best.items()}

    def read_fen(self, rgb):
        from .sim_env import piece_map_to_board_fen
        return piece_map_to_board_fen(self.read(rgb))


def annotate(rgb, detections, camera=None, grid=True):
    """Draw the square grid and the detections onto a frame.

    The grid comes from the camera model rather than the image, so the two
    drifting apart is visible.
    """
    cam = camera or BoardCamera(resolution=rgb.shape[0])
    if cam.res != rgb.shape[0]:
        cam = cam.resized(rgb.shape[0])
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    k = rgb.shape[0] / 640.0
    if grid:
        for f in L.FILES:
            for r in L.RANKS:
                poly = cam.square_corners(f + r)
                if np.isnan(poly).any():
                    continue
                cv2.polylines(out, [poly.astype(np.int32)], True, (0, 0, 0), 1,
                              cv2.LINE_AA)
    for d in detections:
        centre = (int(d.u), int(d.v))
        radius = max(5, int(d.radius_px * 1.7))
        cv2.circle(out, centre, radius, (0, 0, 255), max(1, int(2 * k)))
        label = f"{d.symbol}{d.square}"
        origin = (centre[0] - int(18 * k), centre[1] - radius - int(5 * k))
        for colour, thick in (((0, 0, 0), int(3 * k) + 2),
                              ((255, 255, 255), max(1, int(k)))):
            cv2.putText(out, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45 * k, colour, thick, cv2.LINE_AA)
    return out


def grid_homography(camera):
    """Image pixels -> the 0-8 grid units `core.detection` works in.

    Computed from the known camera pose rather than fitted from grid lines:
    `core.detection.detect_board` bins Hough lines into near-horizontal and
    near-vertical, which finds nothing on an oblique frame.
    """
    src, dst = [], []
    for j in range(9):
        for i in range(9):
            src.append(camera.project((j - 4) * L.SQUARE, (4 - i) * L.SQUARE, 0.0))
            dst.append([j, i])
    homography, _ = cv2.findHomography(np.array(src, dtype=np.float32),
                                       np.array(dst, dtype=np.float32))
    return homography


class YoloPerception:
    """Reads the board through `core.ChessBoard`: its weights, class numbering
    and FEN assembly, with the homography supplied by the simulator."""

    requires_markers = False

    def __init__(self, weights, camera=None, conf=0.25, orientation=0):
        from core.ChessBoard import ChessBoard

        self.chess_board = ChessBoard(weights=weights)
        self.camera = camera or BoardCamera()
        self.conf = conf
        self.orientation = orientation
        self._homography = None
        self._pose = None

    def _grid(self, cam):
        pose = (tuple(cam.pos), tuple(cam.rot.ravel()), cam.fovy, cam.res)
        if self._homography is None or pose != self._pose:
            self._homography = grid_homography(cam)
            self._pose = pose
        return self._homography

    def _camera_for(self, rgb):
        cam = self.camera
        return cam if rgb.shape[0] == cam.res else cam.resized(rgb.shape[0])

    def read(self, rgb):
        """``{'e4': 'P', ...}`` using the project's detector and mapping."""
        from core.detection.map_pieces_to_board import map_detections_to_squares

        cam = self._camera_for(rgb)
        pieces = self.chess_board.detect_pieces(rgb, conf=self.conf)
        squares = map_detections_to_squares(pieces, rgb, self._grid(cam),
                                            board_orientation=self.orientation)
        lookup = self.chess_board.class_to_piece
        return {square: lookup[cls] for square, cls in squares.items()
                if lookup.get(cls) not in (None, "?")}

    def read_fen(self, rgb):
        from .sim_env import piece_map_to_board_fen
        return piece_map_to_board_fen(self.read(rgb))

    def detect(self, rgb):
        """Detections in the shape `annotate` wants.

        Reports where `warp_point` actually lands rather than the centre of the
        square it rounded to, so the remaining margin stays visible.
        """
        from core.detection.map_pieces_to_board import warp_point

        cam = self._camera_for(rgb)
        homography = self._grid(cam)
        height, width = rgb.shape[:2]
        lookup = self.chess_board.class_to_piece

        found = []
        for cls, (cx, cy, _bw, bh) in self.chess_board.detect_pieces(
                rgb, conf=self.conf):
            symbol = lookup.get(cls)
            if symbol in (None, "?"):
                continue
            u = cx * width
            v = cy * height + bh * height / 2      # foot of the box
            gx, gy = warp_point((u, v), homography)
            x = (gx / 100.0 - 4.0) * L.SQUARE
            y = (4.0 - gy / 100.0) * L.SQUARE
            square = L.xy_to_square(x, y)
            if square is None:
                continue
            found.append(Detection(symbol.lower(),
                                   "w" if symbol.isupper() else "b",
                                   u, v, 260, x, y, square))
        return found
