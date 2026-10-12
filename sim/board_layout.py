"""Every dimension of the simulated scene, in metres and radians.

World frame: +x runs files a->h, +y runs ranks 1->8, +z is up, and z=0 is the
playing surface.  The board is centred on the origin and the arm stands on
White's side.
"""

import math

import numpy as np

SQUARE = 0.057                # edge length of one square (tournament size)
BOARD_HALF = 4 * SQUARE       # 0.228, half the playing area
BORDER = 0.024                # frame around the playing area, carries the
                              # rank/file coordinates
PLATE_HALF = BOARD_HALF + BORDER
PLATE_THICK = 0.01
TABLE_HALF_X, TABLE_HALF_Y = 0.75, 0.65
TABLE_THICK = 0.74    # a plinth down to the floor, not a floating slab

FILES = "abcdefgh"
RANKS = "12345678"

PIECE_BASE_R = 0.018          # radius of the felt base
NECK_R = 0.007                # radius of the graspable stem
NECK_Z0, NECK_Z1 = 0.012, 0.030
GRASP_Z = 0.5 * (NECK_Z0 + NECK_Z1)   # height the tool centre aims for

# per-type total height (top of the sculpted head)
PIECE_HEIGHT = {"p": 0.048, "n": 0.054, "b": 0.055, "r": 0.050,
                "q": 0.056, "k": 0.068}
TALLEST_PIECE = max(PIECE_HEIGHT.values())

# visual top marker: identifies the piece type to the camera
MARKER_RGBA = {
    "p": (0.90, 0.12, 0.12, 1.0),   # red
    "n": (0.96, 0.55, 0.05, 1.0),   # orange
    "b": (0.30, 0.95, 0.15, 1.0),   # green
    "r": (0.10, 0.35, 0.95, 1.0),   # blue
    "q": (0.85, 0.10, 0.80, 1.0),   # magenta
    "k": (0.10, 0.85, 0.85, 1.0),   # cyan
}
MARKER_R = {"p": 0.0065, "n": 0.0075, "b": 0.0070, "r": 0.0095,
            "q": 0.0105, "k": 0.0080}

WHITE_RGBA = (0.92, 0.89, 0.82, 1.0)
BLACK_RGBA = (0.11, 0.11, 0.13, 1.0)

ARM_BASE = (0.0, -0.36, 0.0)  # where the pedestal meets the table
PEDESTAL_H = 0.02
COLUMN_H = 0.30
SHOULDER_Z = PEDESTAL_H + COLUMN_H    # 0.32, height of joint 2 above ARM_BASE
L_UPPER = 0.40                # joint2 -> joint3
L_FORE = 0.37                 # joint3 -> joint4/5 (wrist)
L_TOOL = 0.10                 # wrist -> tool centre point (between the jaws)
MAX_REACH = L_UPPER + L_FORE
MIN_REACH = abs(L_UPPER - L_FORE)

JOINT_NAMES = ("base_yaw", "shoulder", "elbow", "wrist_pitch", "wrist_roll")
JOINT_RANGE = {                       # radians
    "base_yaw": (-2.9, 2.9),
    "shoulder": (-2.2, 2.2),
    "elbow": (-2.8, 2.8),
    "wrist_pitch": (-2.6, 2.6),
    "wrist_roll": (-3.0, 3.0),
}
FINGER_RANGE = (0.0, 0.022)    # 0 = closed on a 7 mm stem, 0.022 = wide open
FINGER_X = 0.010               # pad offset from the tool axis when fully closed
GRIP_CLOSED = 0.0
GRIP_OPEN = 0.018

CAPTURE_X = (0.30, 0.345, -0.30, -0.345)      # four columns of discard slots
CAPTURE_Y = tuple(-3.5 * SQUARE + SQUARE * i for i in range(8))
# Behind the arm: beside the board they would stand in a side camera's
# foreground.
STOCK_Y = -0.47
STOCK_X = tuple(-3.5 * SQUARE + SQUARE * i for i in range(8))

# Far corners of the table, out of frame at any camera pose used here.
STASH_X = (0.60, 0.67, -0.60, -0.67)
STASH_Y = tuple(-0.52 + 0.07 * i for i in range(16))

# The camera floats on a mocap body, aimed from a spherical position.
# Elevation 90 would look straight down.  The default azimuth puts it opposite
# the arm, so the robot is never between the lens and the squares.
CAMERA_NAME = "boardcam"
CAM_AZIMUTH = 0.0            # degrees in the xy plane, measured from +x
CAM_ELEVATION = 52.0          # degrees above the board plane
CAM_DISTANCE = 0.95           # metres from CAM_TARGET to the lens
CAM_TARGET = (0.0, 0.0, 0.02)
CAM_MARGIN = 1.07             # slack around the board when fitting the lens
CAM_RES = 640                 # square render


def camera_frame(azimuth=None, elevation=None, distance=None, target=None):
    """Lens position and camera axes for a spherical aim."""
    import numpy as np

    az = math.radians(CAM_AZIMUTH if azimuth is None else azimuth)
    el = math.radians(CAM_ELEVATION if elevation is None else elevation)
    dist = CAM_DISTANCE if distance is None else distance
    tgt = np.array(CAM_TARGET if target is None else target, dtype=float)

    offset = np.array([math.cos(el) * math.cos(az),
                       math.cos(el) * math.sin(az),
                       math.sin(el)])
    pos = tgt + dist * offset

    z_axis = offset / np.linalg.norm(offset)          # points back at the lens
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(z_axis @ up)) > 0.999:               # looking straight down
        up = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return pos, np.column_stack([x_axis, y_axis, z_axis])


def camera_pose(azimuth=None, elevation=None, distance=None, target=None):
    """Lens position and orientation quaternion.

    MuJoCo cameras look down their own -z with +y up.
    """
    pos, axes = camera_frame(azimuth, elevation, distance, target)
    return pos, _mat_to_quat(axes)


def _mat_to_quat(m):
    import numpy as np

    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s,
             (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s,
             (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
             0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return np.array(q) / np.linalg.norm(q)


def camera_fovy(azimuth=None, elevation=None, distance=None, target=None,
                margin=None):
    """Vertical field of view that just contains the board and its border.

    Computed rather than tuned, so re-aiming cannot crop a corner off.
    """
    import numpy as np

    pos, axes = camera_frame(azimuth, elevation, distance, target)
    x_axis, y_axis, z_axis = axes[:, 0], axes[:, 1], axes[:, 2]

    worst = 0.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            for z in (0.0, TALLEST_PIECE):
                rel = np.array([sx * PLATE_HALF, sy * PLATE_HALF, z]) - pos
                depth = -float(rel @ z_axis)
                if depth <= 1e-6:
                    continue
                worst = max(worst, abs(float(rel @ x_axis)) / depth,
                            abs(float(rel @ y_axis)) / depth)
    slack = CAM_MARGIN if margin is None else margin
    return math.degrees(2 * math.atan(worst * slack))


CAM_FOVY = camera_fovy()


def square_to_xy(square):
    """'e4' -> (x, y) of the centre of that square on the playing surface."""
    f = FILES.index(square[0])
    r = RANKS.index(square[1])
    return ((f - 3.5) * SQUARE, (r - 3.5) * SQUARE)


def xy_to_square(x, y):
    """Nearest square to a world point, or None if it is off the board."""
    f = int(math.floor(x / SQUARE + 4))
    r = int(math.floor(y / SQUARE + 4))
    if 0 <= f < 8 and 0 <= r < 8:
        return FILES[f] + RANKS[r]
    return None


def stash_slots():
    """Parking spots outside the camera frame entirely."""
    return [(x, y) for x in STASH_X for y in STASH_Y]


def capture_slots():
    """Discard slots, ordered so the inner columns fill first."""
    return [(x, y) for x in CAPTURE_X for y in CAPTURE_Y]


def stock_slot(color, piece_type):
    """Where a spare promotion piece waits.  color is 'w' or 'b'."""
    order = "qrbn"
    i = order.index(piece_type)
    return (STOCK_X[i] if color == "w" else STOCK_X[4 + i], STOCK_Y)


def reach_of(x, y, z=GRASP_Z):
    """Span the 2R linkage needs to put the tool at (x, y, z) pointing down.

    Compare against MAX_REACH to test the workspace.
    """
    r = math.hypot(x - ARM_BASE[0], y - ARM_BASE[1])
    dz = (z + L_TOOL) - (ARM_BASE[2] + SHOULDER_Z)
    return math.hypot(r, dz)
