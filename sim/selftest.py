"""End-to-end checks for the simulated cell.

    python -m sim.selftest            # everything
    python -m sim.selftest --quick    # skip the slow physical checks

Each check prints a measured number rather than just a pass mark, so a
regression shows up as a number drifting rather than a silent pass.
"""

import argparse
import math
import random
import sys
import time

import chess
import numpy as np

import mujoco

from . import arm_kinematics as K
from . import board_layout as L
from . import dataset as DS
from .move_executor import REMOVE, SHIFT, SPAWN, plan_move
from .perception import MarkerPerception, BoardCamera

PASS, FAIL = "  ok  ", " FAIL "


class Results:
    def __init__(self):
        self.failures = []

    def check(self, name, ok, detail=""):
        print(f"[{PASS if ok else FAIL}] {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)
        return ok


def check_kinematics(r):
    """IK must agree with MuJoCo's forward kinematics everywhere the arm has
    to reach: all 64 squares, every discard slot, every promotion spare."""
    from . import scene_builder
    model = mujoco.MjModel.from_xml_path(str(scene_builder.write_scene()))
    data = mujoco.MjData(model)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool")
    adr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
           for n in L.JOINT_NAMES]

    targets = [L.square_to_xy(f + rk) for f in L.FILES for rk in L.RANKS]
    targets += list(L.capture_slots())
    targets += [L.stock_slot(c, p) for c in "wb" for p in "qrbn"]

    worst_pos = worst_dir = worst_jaw = 0.0
    unreachable = []
    for x, y in targets:
        for jaw in (0.0, math.pi / 4, math.pi / 2):
            for z in (L.GRASP_Z, 0.075, 0.13):
                try:
                    q = K.inverse(x, y, z, jaw)
                except K.Unreachable as exc:
                    unreachable.append(f"({x:.3f},{y:.3f},{z:.3f}): {exc}")
                    continue
                data.qpos[adr] = q
                mujoco.mj_forward(model, data)
                rot = data.site_xmat[site].reshape(3, 3)
                worst_pos = max(worst_pos, float(np.linalg.norm(
                    data.site_xpos[site] - np.array([x, y, z]))))
                worst_dir = max(worst_dir, float(np.linalg.norm(
                    rot[:, 2] - np.array([0.0, 0.0, -1.0]))))
                err = abs(K.wrap(math.atan2(rot[1, 0], rot[0, 0]) - jaw))
                worst_jaw = max(worst_jaw, min(err, abs(math.pi - err)))

    n = len(targets) * 9
    r.check("inverse kinematics reach every target", not unreachable,
            f"{n} poses, {len(unreachable)} unreachable")
    for u in unreachable[:3]:
        print(f"         {u}")
    r.check("IK position matches MuJoCo FK", worst_pos < 1e-6,
            f"worst {worst_pos * 1e6:.3f} um")
    r.check("approach axis is exactly vertical", worst_dir < 1e-9,
            f"worst {worst_dir:.1e}")
    r.check("jaw heading matches request", worst_jaw < 1e-9,
            f"worst {math.degrees(worst_jaw):.2e} deg")


def check_camera(r, sim):
    """The analytic projection must agree with what the renderer draws."""
    cam = BoardCamera.from_sim(sim)
    worst_rt = worst_h = 0.0
    for z in (0.0, L.GRASP_Z, L.TALLEST_PIECE):
        homography = cam.homography(z)
        for f in L.FILES:
            for rk in L.RANKS:
                x, y = L.square_to_xy(f + rk)
                u, v = cam.project(x, y, z)
                bx, by = cam.unproject(u, v, z)
                worst_rt = max(worst_rt, abs(bx - x), abs(by - y))
                h = homography @ np.array([x, y, 1.0])
                worst_h = max(worst_h, float(np.hypot(*(h[:2] / h[2] - (u, v)))))
    r.check("project / unproject round-trips", worst_rt < 1e-9,
            f"worst {worst_rt:.1e} m")
    r.check("homography agrees with the projection", worst_h < 1e-6,
            f"worst {worst_h:.1e} px")

    corners = [cam.project(sx * L.PLATE_HALF, sy * L.PLATE_HALF, 0.0)
               for sx in (-1, 1) for sy in (-1, 1)]
    inside = all(0 <= u < cam.res and 0 <= v < cam.res for u, v in corners)
    r.check("the whole board is inside the frame", inside,
            "all four plate corners project into the image")

    # Record how far a height-blind back-projection would slip, so a
    # regression in the height handling is obvious.
    slip = max(
        math.hypot(*(np.array(cam.unproject(*cam.project(*L.square_to_xy(f + rk),
                                                         L.PIECE_HEIGHT["k"]), 0.0))
                     - np.array(L.square_to_xy(f + rk))))
        for f in L.FILES for rk in L.RANKS)
    print(f"         (height-blind back-projection would slip "
          f"{slip * 1000:.0f} mm = {slip / L.SQUARE:.2f} squares)")

    # The rendered checker must line up with the logical squares: sample the
    # middle of every empty square and confirm the colour matches its parity.
    sim.reset("8/8/8/8/8/8/8/8 w - - 0 1")
    img = sim.capture()
    wrong = 0
    for fi, f in enumerate(L.FILES):
        for ri, rk in enumerate(L.RANKS):
            poly = cam.square_corners(f + rk)
            centre = poly.mean(axis=0)
            # A square far from the camera covers fewer pixels, so size the
            # sample from its own projected footprint.
            radius = max(2.0, 0.3 * min(np.hypot(*(c - centre)) for c in poly))
            yy, xx = np.mgrid[int(centre[1] - radius):int(centre[1] + radius) + 1,
                              int(centre[0] - radius):int(centre[0] + radius) + 1]
            patch = img[np.clip(yy, 0, img.shape[0] - 1),
                        np.clip(xx, 0, img.shape[1] - 1)]
            # Both square colours are greenish; brightness separates them.
            dark = patch.mean() < 130
            if dark != ((fi + ri) % 2 == 0):       # a1 is a dark square
                wrong += 1
    r.check("rendered board aligns with the square grid", wrong == 0,
            f"{64 - wrong}/64 squares correct")


def check_rules(r):
    """Every special move must expand into the right physical steps."""
    cases = [
        ("quiet", chess.Board(), "e2e4", [(SHIFT, "e2", "e4")]),
        ("capture",
         chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
         "e4d5", [(REMOVE, "d5", None), (SHIFT, "e4", "d5")]),
        ("en passant",
         chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"),
         "e5f6", [(REMOVE, "f5", None), (SHIFT, "e5", "f6")]),
        ("castle king side", chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
         "e1g1", [(SHIFT, "e1", "g1"), (SHIFT, "h1", "f1")]),
        ("castle queen side", chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"),
         "e1c1", [(SHIFT, "e1", "c1"), (SHIFT, "a1", "d1")]),
        ("promotion", chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1"),
         "a7a8q", [(REMOVE, "a7", None), (SPAWN, None, "a8")]),
        ("capture promotion", chess.Board("1n6/P7/8/8/8/8/8/K6k w - - 0 1"),
         "a7b8n", [(REMOVE, "b8", None), (REMOVE, "a7", None), (SPAWN, None, "b8")]),
    ]
    bad = []
    for label, board, uci, expected in cases:
        move = chess.Move.from_uci(uci)
        got = [(s.kind, s.src, s.dst) for s in plan_move(board, move)]
        if got != expected:
            bad.append(f"{label}: {got} != {expected}")
    r.check("special moves expand correctly", not bad,
            f"{len(cases) - len(bad)}/{len(cases)} cases")
    for b in bad:
        print(f"         {b}")


def check_physical(r, sim):
    """Actually move pieces and measure where they end up."""
    sim.reset()
    moves = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O", "Nf6", "Bxf7+", "Kxf7"]
    t0 = time.time()
    for mv in moves:
        sim.apply_move(mv, check=False)
    elapsed = time.time() - t0

    ok, problems = sim.consistent()
    r.check("board still matches the game after 10 half-moves", ok,
            f"{len(problems)} discrepancies, {elapsed / len(moves):.1f}s per move")
    for p in problems[:4]:
        print(f"         {p}")

    errs = [float(np.linalg.norm(sim.body_xy(b) - np.array(L.square_to_xy(sq))))
            for sq, b in sim.piece_at.items()]
    r.check("pieces land on the centre of their square", max(errs) < 0.004,
            f"mean {np.mean(errs) * 1000:.2f} mm, worst {max(errs) * 1000:.2f} mm")

    tilt = []
    for body in sim.piece_at.values():
        rot = sim.data.xmat[sim._bid[body]].reshape(3, 3)
        tilt.append(math.degrees(math.acos(min(1.0, max(-1.0, rot[2, 2])))))
    r.check("no piece was left leaning", max(tilt) < 5.0,
            f"worst {max(tilt):.2f} deg from upright")


def check_perception(r, sim, n_positions=10, seed=7):
    """Read sampled positions back off the camera."""
    reader = MarkerPerception(BoardCamera.from_sim(sim))
    rng = random.Random(seed)
    board = chess.Board()
    fens = [board.fen()]
    for i in range(n_positions * 8):
        if board.is_game_over():
            board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))
        if i % 8 == 7:
            fens.append(board.fen())
    fens.append("r6r/8/8/3qk3/3QK3/8/8/R6R w - - 0 1")   # all four corners

    compared = wrong = 0
    examples = []
    for fen in fens:
        sim.reset(fen)
        got = reader.read(sim.capture())
        want = {chess.square_name(s): p.symbol()
                for s, p in sim.board.piece_map().items()}
        for square in set(got) | set(want):
            compared += 1
            if got.get(square) != want.get(square):
                wrong += 1
                examples.append(f"{square}: truth {want.get(square)} read {got.get(square)}")
    r.check("camera reads the position correctly", wrong == 0,
            f"{compared - wrong}/{compared} squares over {len(fens)} positions")
    for e in examples[:4]:
        print(f"         {e}")


def check_dataset(r, tmpdir, n=4, seed=5):
    """Generate a small dataset and check the labels really describe it."""
    import csv
    import pathlib

    from .perception import BoardCamera
    from .sim_env import ChessSim

    view = DS.ViewConfig()
    out = pathlib.Path(DS.generate(tmpdir, n_train=n, n_val=0, res=416,
                                   seed=seed, view=view, progress_every=0))

    rows = list(csv.DictReader(open(out / "labels.csv")))
    frames = list(csv.DictReader(open(out / "frames.csv")))
    bad = 0
    for row in rows:
        x0, y0 = float(row["x_min"]), float(row["y_min"])
        x1, y1 = float(row["x_max"]), float(row["y_max"])
        w, h = int(row["width"]), int(row["height"])
        if not (0 <= x0 < x1 <= w - 1 and 0 <= y0 < y1 <= h - 1):
            bad += 1
        elif int(row["class_id"]) not in DS.CLASS_TO_PIECE or int(row["class_id"]) == 0:
            bad += 1
        elif DS.CLASS_TO_PIECE[int(row["class_id"])] != row["piece"]:
            bad += 1
    r.check("labels.csv rows are well formed", bad == 0 and bool(rows),
            f"{len(rows)} boxes over {len(frames)} frames, {bad} bad")
    r.check("every frame is accounted for", len(frames) == n,
            f"{len(frames)} frames, piece counts "
            f"{[f['pieces_labelled'] for f in frames]}")

    # The CSV and the YOLO files must describe the same boxes.
    per_image = {}
    for row in rows:
        per_image.setdefault(row["image"], []).append(row)
    mismatch = 0
    for image, group in per_image.items():
        stem = pathlib.Path(image).stem
        lines = [l for l in (out / "train" / "labels" / f"{stem}.txt")
                 .read_text().split("\n") if l.strip()]
        if len(lines) != len(group):
            mismatch += 1
    r.check("YOLO labels match the CSV", mismatch == 0,
            f"{len(per_image)} images compared")

    # Positions must vary, including boards that are far from full.
    counts = sorted(int(f["pieces_on_board"]) for f in frames)
    r.check("positions vary in how full the board is", counts[0] < counts[-1],
            f"pieces per frame {counts}")

    # Whatever segmentation finds must lie inside the projected bounds.
    sim = ChessSim(markers=False, verbose=False)
    seg = DS.Segmenter(sim, 416)
    rng = random.Random(seed)
    outside = 0
    checked = 0
    tight = []
    margins = []
    for _ in range(3):
        sim.reset(DS.sample_position(rng), stash=True)
        DS.aim_side_view(sim, rng, view)
        sim.arm.settle(0.2)
        project = DS.camera_matrix(sim, L.CAMERA_NAME, 416)
        boxes = seg.boxes(sim.piece_at.values(), min_pixels=40)
        for body, (x0, y0, x1, y1, _px) in boxes.items():
            box = DS.projected_box(sim, body, project, 416)
            if box is None:
                continue
            checked += 1
            if not (x0 >= box[0] - 2 and y0 >= box[1] - 2
                    and x1 <= box[2] + 2 and y1 <= box[3] + 2):
                outside += 1
            area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
            tight.append(((x1 - x0) * (y1 - y0)) / area)
        cam = BoardCamera.from_sim(sim, 416)
        corners = [cam.project(sx * L.PLATE_HALF, sy * L.PLATE_HALF, z)
                   for sx in (-1, 1) for sy in (-1, 1)
                   for z in (0.0, L.TALLEST_PIECE)]
        margins.append(min(min(u, v, 416 - u, 416 - v) for u, v in corners))
    sim.close()
    r.check("visible boxes sit inside the projected bounds", outside == 0,
            f"{checked} boxes cross-checked, median {np.median(tight):.2f} of "
            f"the projected area")
    r.check("the board stays inside the frame as the camera moves",
            min(margins) > 0,
            f"worst margin {min(margins):.0f} px over the jitter range")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true",
                   help="skip the physical pick-and-place and dataset checks")
    args = p.parse_args(argv)

    import tempfile
    r = Results()
    print("kinematics")
    check_kinematics(r)

    from .sim_env import ChessSim
    # Markers on: the perception checks exist to exercise the marker detector.
    sim = ChessSim(verbose=False, markers=True)
    print("\ncamera")
    check_camera(r, sim)
    print("\nchess rules")
    check_rules(r)
    print("\nperception")
    check_perception(r, sim, n_positions=4 if args.quick else 10)
    if not args.quick:
        print("\nphysics")
        check_physical(r, sim)
        print("\ndataset")
        with tempfile.TemporaryDirectory() as tmp:
            check_dataset(r, tmp)
    sim.close()

    print()
    if r.failures:
        print(f"{len(r.failures)} check(s) failed: " + ", ".join(r.failures))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
