"""Generates the piece meshes.

Every piece but the knight is a surface of revolution, as on a real lathe: a
few control points, smoothed into a profile and revolved.  The rest - the
rook's crenellations, the queen's crown, the king's cross, the knight's head -
is added on top.

Visual only; collision stays on the primitives in `sim.scene_builder`.

Dimensions are millimetres, converted on the way out.
"""

import math
import pathlib

import numpy as np

from . import board_layout as L

MESH_DIR = pathlib.Path(__file__).parent / "assets" / "meshes"
SEGMENTS = 40          # radial divisions
SMOOTHING = 3          # spline samples between control points

# Control points: (radius, height) from the base up.  A profile may double back
# on itself, which is how the rook gets its hollow top.
PROFILES = {
    "p": [(0, 0), (13, 0), (17, 1), (18, 3.5), (17, 6), (14, 8), (11.5, 10),
          (9.5, 12.5), (8, 16), (7.2, 20), (7, 24), (8, 27), (10.5, 29),
          (11.8, 30.5), (10.5, 32), (8, 32.8), (8.6, 34.5), (9.2, 38),
          (8.6, 42), (6.5, 45.5), (3.5, 47.4), (0, 48)],
    "r": [(0, 0), (13, 0), (17, 1), (18, 3.5), (17, 6), (14.5, 8), (12, 10.5),
          (10.5, 14), (10.2, 20), (10.5, 26), (11.5, 30), (12.8, 33),
          (13.2, 35), (12.6, 37), (12.4, 39), (13.5, 41), (13.5, 46),
          # across the rim and down into the hollow top
          (10.5, 46), (10, 42.5), (9.5, 41), (0, 41)],
    "b": [(0, 0), (13, 0), (17, 1), (18, 3.5), (17, 6), (14, 8), (11.5, 10),
          (9.5, 13), (8, 17), (7.2, 21), (7.2, 25), (8.5, 28), (10.5, 30),
          (11.8, 31.5), (10.2, 33), (7.8, 34), (9, 36), (9.6, 40),
          (8.6, 44.5), (6, 48), (3.4, 50), (3.2, 51.5), (4.2, 52.5),
          (3.4, 53.5), (1.8, 54.6), (0, 55)],
    "n": [(0, 0), (13, 0), (17, 1), (18, 3.5), (17, 6), (14, 8), (11.5, 10.5),
          (10, 14), (9.6, 19), (10.4, 23), (11.6, 25.5), (12.4, 27),
          (11.2, 28.5), (9.4, 29.2), (0, 29.2)],
    "q": [(0, 0), (13.5, 0), (17.5, 1), (18, 3.5), (17, 6), (14.5, 8),
          (12, 10.5), (10, 14), (8.4, 18), (7.6, 22), (7.6, 26), (9, 29.5),
          (11, 32), (12.4, 33.5), (10.8, 35), (8.6, 36), (10, 38.5),
          (11.4, 42), (11, 46), (9.2, 48.5), (10.8, 50), (12.2, 51.5),
          (12.2, 53), (9, 53), (8.6, 51.5), (0, 51.5)],
    "k": [(0, 0), (13.5, 0), (17.5, 1), (18, 3.5), (17, 6), (14.5, 8),
          (12, 10.5), (10, 14), (8.4, 18), (7.6, 22), (7.6, 26), (9, 29.5),
          (11, 32), (12.4, 33.5), (10.8, 35), (8.6, 36), (10, 38.5),
          (11.2, 42), (10.6, 46), (8.8, 49), (10.4, 50.5), (11.6, 52),
          (11.6, 53.5), (8.6, 53.5), (8.2, 52), (0, 52)],
}

# Knight head: a side-on silhouette, extruded across and tapered at the cheeks.
KNIGHT_HEAD = [
    (-9.0, 26.0), (-10.5, 31.0), (-10.0, 36.0), (-7.0, 40.0), (-2.0, 42.5),
    (3.0, 43.5), (7.5, 45.0), (10.5, 48.0), (11.5, 52.0), (9.0, 54.0),
    (4.5, 53.0), (1.0, 50.0), (-3.0, 48.5), (-8.0, 47.0), (-11.5, 44.0),
    (-13.0, 39.0), (-12.5, 33.0), (-11.0, 27.5),
]
KNIGHT_HALF_WIDTH = 7.0
KNIGHT_TAPER = 0.55        # cheeks narrower than the centre


def _catmull_rom(points, samples):
    """Smooth a control polygon so the turned profile has no visible corners."""
    pts = np.asarray(points, dtype=float)
    pad = np.vstack([pts[0], pts, pts[-1], pts[-1]])
    out = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = pad[i], pad[i + 1], pad[i + 2], pad[i + 3]
        for s in range(samples):
            t = s / samples
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                              + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(pts[-1])
    return np.asarray(out)


def _lathe(profile, segments=SEGMENTS):
    """Revolve a (radius, height) profile into vertices, normals and faces."""
    prof = np.asarray(profile, dtype=float)
    n = len(prof)

    # Profile-plane normal: rotate the tangent a quarter turn outward.
    tang = np.gradient(prof, axis=0)
    norm2 = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    lengths = np.linalg.norm(norm2, axis=1, keepdims=True)
    norm2 = norm2 / np.where(lengths < 1e-9, 1.0, lengths)

    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)

    verts = np.empty((n * segments, 3))
    norms = np.empty((n * segments, 3))
    for i in range(n):
        r, z = prof[i]
        nr, nz = norm2[i]
        verts[i * segments:(i + 1) * segments] = np.stack(
            [r * cos, r * sin, np.full(segments, z)], axis=1)
        nvec = np.stack([nr * cos, nr * sin, np.full(segments, nz)], axis=1)
        mag = np.linalg.norm(nvec, axis=1, keepdims=True)
        norms[i * segments:(i + 1) * segments] = nvec / np.where(mag < 1e-9, 1.0, mag)

    faces = []
    for i in range(n - 1):
        for j in range(segments):
            k = (j + 1) % segments
            a = i * segments + j
            b = i * segments + k
            c = (i + 1) * segments + k
            d = (i + 1) * segments + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    return verts, norms, np.asarray(faces, dtype=int)


def _prism(polygon, half_width, taper=1.0):
    """Extrude a side-on polygon across y, narrowing at the outer faces."""
    poly = np.asarray(polygon, dtype=float)
    m = len(poly)
    rings = [(-half_width, taper), (-half_width * 0.75, 1.0),
             (half_width * 0.75, 1.0), (half_width, taper)]
    verts, norms = [], []
    centre = poly.mean(axis=0)
    for y, scale in rings:
        for px, pz in poly:
            sx = centre[0] + (px - centre[0]) * scale
            sz = centre[1] + (pz - centre[1]) * scale
            verts.append([sx, y, sz])
    verts = np.asarray(verts)

    faces = []
    for ring in range(len(rings) - 1):
        for i in range(m):
            j = (i + 1) % m
            a = ring * m + i
            b = ring * m + j
            c = (ring + 1) * m + j
            d = (ring + 1) * m + i
            faces.append((a, b, c))
            faces.append((a, c, d))
    # Close both ends with a fan from the centroid.
    for ring, flip in ((0, True), (len(rings) - 1, False)):
        base = len(verts)
        mid = verts[ring * m:(ring + 1) * m].mean(axis=0)
        verts = np.vstack([verts, mid])
        for i in range(m):
            j = (i + 1) % m
            tri = (base, ring * m + j, ring * m + i)
            faces.append(tri if flip else (tri[0], tri[2], tri[1]))
    faces = np.asarray(faces, dtype=int)
    norms = _vertex_normals(verts, faces)
    return verts, norms, faces


def _block(cx, cy, cz, hx, hy, hz, yaw=0.0):
    """An axis-aligned box, optionally spun about z."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                      for sz in (-1, 1)], dtype=float)
    local = signs * np.array([hx, hy, hz])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    verts = local @ rot.T + np.array([cx, cy, cz])
    # Corner order above is (x, y, z) with z fastest.
    faces = np.array([
        (0, 1, 3), (0, 3, 2), (4, 7, 5), (4, 6, 7),
        (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
        (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)], dtype=int)
    return verts, _vertex_normals(verts, faces), faces


def _vertex_normals(verts, faces):
    norms = np.zeros_like(verts)
    tri = verts[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for i in range(3):
        np.add.at(norms, faces[:, i], face_n)
    mag = np.linalg.norm(norms, axis=1, keepdims=True)
    return norms / np.where(mag < 1e-9, 1.0, mag)


def _merge(parts):
    verts, norms, faces, offset = [], [], [], 0
    for v, n, f in parts:
        verts.append(v)
        norms.append(n)
        faces.append(f + offset)
        offset += len(v)
    return np.vstack(verts), np.vstack(norms), np.vstack(faces)


def _crown(count, radius, z0, z1, hx, hy, phase=0.0):
    return [_block(radius * math.cos(phase + 2 * math.pi * i / count),
                   radius * math.sin(phase + 2 * math.pi * i / count),
                   (z0 + z1) / 2, hx, hy, (z1 - z0) / 2,
                   yaw=phase + 2 * math.pi * i / count)
            for i in range(count)]


def build(kind):
    """Vertices, normals and faces for one piece type, in metres."""
    parts = [_lathe(_catmull_rom(PROFILES[kind], SMOOTHING))]
    if kind == "r":
        parts += _crown(6, 11.8, 46.0, 50.0, 2.6, 4.0)
    elif kind == "q":
        parts += _crown(7, 11.0, 52.5, 56.0, 1.7, 2.6)
    elif kind == "k":
        parts += [_block(0, 0, 60.0, 1.9, 1.9, 8.0),
                  _block(0, 0, 62.0, 5.4, 1.8, 1.9)]
    elif kind == "n":
        parts.append(_prism(KNIGHT_HEAD, KNIGHT_HALF_WIDTH, KNIGHT_TAPER))
    verts, norms, faces = _merge(parts)
    return verts / 1000.0, norms, faces


def write_obj(kind, path=None):
    verts, norms, faces = build(kind)
    path = pathlib.Path(path or MESH_DIR / f"{kind}.obj")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# kasparobot {kind}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in norms]
    lines += [f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}"
              for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")
    return path


def write_all(directory=MESH_DIR):
    directory = pathlib.Path(directory)
    return {k: write_obj(k, directory / f"{k}.obj") for k in PROFILES}


if __name__ == "__main__":
    for kind, path in write_all().items():
        v, n, f = build(kind)
        top = v[:, 2].max() * 1000
        print(f"  {kind}: {len(v):5d} verts {len(f):5d} tris  top {top:5.1f} mm"
              f"  (board_layout says {L.PIECE_HEIGHT[kind] * 1000:.0f})")
