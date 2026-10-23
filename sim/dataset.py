"""Render a labelled dataset of board images for training a detector.

    python -m sim.dataset --out dataset_sim --train 400 --val 80 --preview

Produces, under --out:

    labels.csv              one row per visible piece, with its bounding box
    frames.csv              one row per image: size, camera pose, piece count
    {train,val}/images/*.png
    {train,val}/labels/*.txt    the same boxes in YOLO format
    data.yaml

Boxes come from a segmentation render rather than from projecting each piece's
bounding volume, so they are tight and a piece hidden behind another is dropped
instead of labelled.
"""

import argparse
import csv
import math
import pathlib
import random
from dataclasses import dataclass

import chess
import cv2
import numpy as np

import mujoco

from . import board_layout as L

# Matches ChessBoard.class_to_piece, so a model trained here drops straight
# into the existing pipeline.
CLASS_TO_PIECE = {0: '?', 1: 'b', 2: 'k', 3: 'n', 4: 'p', 5: 'q', 6: 'r',
                  7: 'B', 8: 'K', 9: 'N', 10: 'P', 11: 'Q', 12: 'R'}
PIECE_TO_CLASS = {v: k for k, v in CLASS_TO_PIECE.items() if k}

# Geoms in this group and above are collision-only and never rendered.
HIDDEN_GROUP = 3

PIECE_CAPS = {"p": 8, "r": 2, "n": 2, "b": 2, "q": 1, "k": 1}


@dataclass
class ViewConfig:
    """How far the camera may wander between frames.

    Azimuth 0 and 180 look along the ranks; those are the angles where neither
    the arm nor its column comes between the lens and the board.
    """
    azimuth: float = 0.0
    azimuth_jitter: float = 12.0
    both_sides: bool = True          # also sample the opposite side, 180 away
    elevation: float = 42.0
    elevation_jitter: float = 9.0
    distance: float = 0.95
    distance_jitter: float = 0.07
    target_jitter: float = 0.02
    light_jitter: bool = True


def aim_side_view(sim, rng, view=None):
    """Nudge the camera, keeping it an oblique side view.  Returns the pose."""
    view = view or ViewConfig()
    base = view.azimuth
    if view.both_sides and rng.random() < 0.5:
        base += 180.0
    azimuth = base + rng.uniform(-view.azimuth_jitter, view.azimuth_jitter)
    elevation = view.elevation + rng.uniform(-view.elevation_jitter,
                                             view.elevation_jitter)
    distance = view.distance + rng.uniform(-view.distance_jitter,
                                           view.distance_jitter)
    target = (rng.uniform(-view.target_jitter, view.target_jitter),
              rng.uniform(-view.target_jitter, view.target_jitter), 0.02)
    sim.move_camera(azimuth, elevation, distance, target)   # refits the fovy
    if view.light_jitter:
        for i in range(sim.model.nlight):
            sim.model.light_pos[i] = [rng.uniform(-0.7, 0.7),
                                      rng.uniform(-0.1, 0.9),
                                      rng.uniform(1.0, 1.7)]
        mujoco.mj_forward(sim.model, sim.data)
    return {"azimuth": azimuth, "elevation": elevation, "distance": distance,
            "fovy": sim.camera_fovy}


def random_game_position(rng, max_plies=70):
    """A position from a random game: realistic counts and clustering."""
    board = chess.Board()
    for _ in range(rng.randint(0, max_plies)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board.fen()


def random_sparse_position(rng, min_pieces=2, max_pieces=24):
    """A handful of pieces scattered on an otherwise empty board.

    Not a legal position.  Game positions alone are mostly crowded back ranks;
    these cover the sparse end.
    """
    from .sim_env import piece_map_to_board_fen

    wanted = rng.randint(min_pieces, max_pieces)
    pool = []
    for colour in "wb":
        for kind, cap in PIECE_CAPS.items():
            pool += [(colour, kind)] * cap
    rng.shuffle(pool)

    squares = [f + r for f in L.FILES for r in L.RANKS]
    rng.shuffle(squares)
    placement, used = {}, 0
    for colour, kind in pool:
        if used >= wanted:
            break
        for i, square in enumerate(squares):
            # Pawns never stand on the first or last rank.
            if kind == "p" and square[1] in "18":
                continue
            squares.pop(i)
            placement[square] = kind.upper() if colour == "w" else kind
            used += 1
            break
    return piece_map_to_board_fen(placement) + " w - - 0 1"


def sample_position(rng, sparse_fraction=0.35):
    if rng.random() < sparse_fraction:
        return random_sparse_position(rng)
    return random_game_position(rng)


def camera_matrix(sim, camera, res):
    """Pinhole projection for an arbitrary camera pose."""
    cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    pos = sim.data.cam_xpos[cid].copy()
    rot = sim.data.cam_xmat[cid].reshape(3, 3).copy()
    f = (res / 2) / math.tan(math.radians(sim.model.cam_fovy[cid]) / 2)

    def project(points):
        local = (np.asarray(points, dtype=float) - pos) @ rot    # R^T @ (p - c)
        depth = -local[:, 2]
        u = res / 2 + f * local[:, 0] / depth
        v = res / 2 - f * local[:, 1] / depth
        return np.stack([u, v], axis=1), depth

    return project


def projected_box(sim, body, project, res):
    """Image box covering the visible geoms of `body`, ignoring occlusion.

    An independent cross-check on the segmentation boxes, which must fall
    inside this.
    """
    bid = sim._bid[body]
    corners = []
    for g in range(sim.model.body_geomadr[bid],
                   sim.model.body_geomadr[bid] + sim.model.body_geomnum[bid]):
        if sim.model.geom_group[g] >= HIDDEN_GROUP:
            continue
        centre, half = sim.model.geom_aabb[g][:3], sim.model.geom_aabb[g][3:]
        gpos, gmat = sim.data.geom_xpos[g], sim.data.geom_xmat[g].reshape(3, 3)
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                          for sz in (-1, 1)], dtype=float)
        corners.append(gpos + (centre + signs * half) @ gmat.T)
    if not corners:
        return None
    uv, _ = project(np.concatenate(corners))
    u0, v0 = uv.min(axis=0)
    u1, v1 = uv.max(axis=0)
    return (max(u0, 0.0), max(v0, 0.0), min(u1, res - 1.0), min(v1, res - 1.0))


class Segmenter:
    """Per-piece pixel masks, for occlusion-aware boxes."""

    def __init__(self, sim, res):
        self.sim = sim
        self.res = res
        self.renderer = mujoco.Renderer(sim.model, height=res, width=res)
        self.renderer.enable_segmentation_rendering()
        self.geom_body = sim.model.geom_bodyid.copy()

    def boxes(self, bodies, min_pixels):
        """`{body: (x0, y0, x1, y1, pixels)}` for everything actually seen."""
        self.renderer.update_scene(self.sim.data, camera=L.CAMERA_NAME)
        seg = self.renderer.render()
        geom_ids = seg[..., 0]

        valid = geom_ids >= 0
        body_map = np.full(geom_ids.shape, -1, dtype=np.int64)
        body_map[valid] = self.geom_body[geom_ids[valid]]

        out = {}
        for body in bodies:
            bid = self.sim._bid[body]
            ys, xs = np.nonzero(body_map == bid)
            if len(xs) < min_pixels:
                continue
            out[body] = (float(xs.min()), float(ys.min()),
                         float(xs.max()), float(ys.max()), int(len(xs)))
        return out


def piece_symbol(body):
    parts = body.split("_")
    colour, kind = ((parts[1], parts[2]) if parts[0] == "stock"
                    else (parts[0], parts[1]))
    return (kind.upper() if colour == "w" else kind), colour, kind


LABEL_FIELDS = ["split", "image", "width", "height", "piece", "colour",
                "kind", "class_id", "square", "x_min", "y_min", "x_max",
                "y_max", "pixels"]
FRAME_FIELDS = ["split", "image", "width", "height", "azimuth", "elevation",
                "distance", "fovy", "pieces_on_board", "pieces_labelled", "fen"]


def generate(out_dir, n_train=400, n_val=80, res=640, seed=0, view=None,
             sparse_fraction=0.35, min_pixels=48, settle=0.25, yolo=True,
             hide_arm=False, progress_every=25):
    from .sim_env import ChessSim

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    view = view or ViewConfig()

    sim = ChessSim(markers=False, verbose=False)
    if hide_arm:
        sim.set_arm_visible(False)
    segmenter = Segmenter(sim, res)

    label_rows, frame_rows = [], []
    for split, count in (("train", n_train), ("val", n_val)):
        if count <= 0:
            continue
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        if yolo:
            (out / split / "labels").mkdir(parents=True, exist_ok=True)

        for i in range(count):
            fen = sample_position(rng, sparse_fraction)
            sim.reset(fen, stash=True)
            if hide_arm:
                sim.set_arm_visible(False)      # reset restores visibility
            pose = aim_side_view(sim, rng, view)
            sim.arm.settle(settle)

            image = sim.capture(res, res, park=False)
            name = f"{split}_{i:05d}"
            rel = f"{split}/images/{name}.png"
            cv2.imwrite(str(out / rel), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            boxes = segmenter.boxes(sim.piece_at.values(), min_pixels)
            yolo_lines = []
            for square, body in sorted(sim.piece_at.items()):
                if body not in boxes:
                    continue                 # hidden behind another piece
                x0, y0, x1, y1, pixels = boxes[body]
                symbol, colour, kind = piece_symbol(body)
                label_rows.append({
                    "split": split, "image": rel, "width": res, "height": res,
                    "piece": symbol, "colour": colour, "kind": kind,
                    "class_id": PIECE_TO_CLASS[symbol], "square": square,
                    "x_min": round(x0, 1), "y_min": round(y0, 1),
                    "x_max": round(x1, 1), "y_max": round(y1, 1),
                    "pixels": pixels})
                yolo_lines.append(
                    f"{PIECE_TO_CLASS[symbol]} "
                    f"{(x0 + x1) / 2 / res:.6f} {(y0 + y1) / 2 / res:.6f} "
                    f"{(x1 - x0) / res:.6f} {(y1 - y0) / res:.6f}")

            if yolo:
                (out / split / "labels" / f"{name}.txt").write_text(
                    "\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
            frame_rows.append({
                "split": split, "image": rel, "width": res, "height": res,
                "azimuth": round(pose["azimuth"], 2),
                "elevation": round(pose["elevation"], 2),
                "distance": round(pose["distance"], 4),
                "fovy": round(pose["fovy"], 3),
                "pieces_on_board": len(sim.piece_at),
                "pieces_labelled": len(yolo_lines),
                "fen": fen.split(" ")[0]})

            if progress_every and (i + 1) % progress_every == 0:
                print(f"  {split}: {i + 1}/{count}")

    _write_csv(out / "labels.csv", LABEL_FIELDS, label_rows)
    _write_csv(out / "frames.csv", FRAME_FIELDS, frame_rows)
    if yolo:
        # '?' is a YAML indicator character and would not parse unquoted.
        names = "\n".join(f'  {i}: "{CLASS_TO_PIECE[i]}"' for i in range(13))
        (out / "data.yaml").write_text(
            f"path: {out.resolve()}\ntrain: train/images\n"
            f"val: {'val/images' if n_val > 0 else 'train/images'}\n\n"
            f"names:\n{names}\n")
    sim.close()

    hidden = sum(r["pieces_on_board"] - r["pieces_labelled"] for r in frame_rows)
    print(f"{len(frame_rows)} images, {len(label_rows)} boxes, "
          f"{hidden} pieces dropped as hidden or off-frame")
    return out


def _write_csv(path, fields, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def preview(out_dir, image=None, dst=None):
    """Draw a frame's boxes back onto it, straight from labels.csv."""
    out = pathlib.Path(out_dir)
    rows = list(csv.DictReader(open(out / "labels.csv")))
    if not rows:
        raise SystemExit("labels.csv is empty")
    image = image or rows[0]["image"]
    img = cv2.imread(str(out / image))
    for row in rows:
        if row["image"] != image:
            continue
        p0 = (int(float(row["x_min"])), int(float(row["y_min"])))
        p1 = (int(float(row["x_max"])), int(float(row["y_max"])))
        cv2.rectangle(img, p0, p1, (0, 0, 255), 1)
        cv2.putText(img, row["piece"], (p0[0], p0[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    dst = dst or (out / f"preview_{pathlib.Path(image).stem}.png")
    cv2.imwrite(str(dst), img)
    return dst


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="dataset_sim",
                   help="directory to write the dataset into")
    p.add_argument("--train", type=int, default=400,
                   help="number of training frames")
    p.add_argument("--val", type=int, default=80,
                   help="number of validation frames; 0 writes no val split")
    p.add_argument("--img-size", type=int, default=640,
                   help="rendered frame size in pixels, square")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for positions, camera jitter and lighting")
    p.add_argument("--sparse-fraction", type=float, default=0.35,
                   help="share of frames that are scattered pieces rather than "
                        "positions from a game (0 disables)")
    p.add_argument("--azimuth", type=float, default=0.0,
                   help="base side-view azimuth; 0 and 180 are the side views")
    p.add_argument("--azimuth-jitter", type=float, default=12.0,
                   help="degrees of random wobble either side of --azimuth")
    p.add_argument("--sides", choices=("one", "both"), default="both",
                   help="sample only the --azimuth side, or that side and the "
                        "one 180 degrees opposite")
    p.add_argument("--elevation", type=float, default=42.0,
                   help="degrees above the board plane")
    p.add_argument("--elevation-jitter", type=float, default=9.0,
                   help="degrees of random wobble either side of --elevation")
    p.add_argument("--distance", type=float, default=0.95,
                   help="metres from the board centre to the lens")
    p.add_argument("--distance-jitter", type=float, default=0.07,
                   help="metres of random wobble either side of --distance")
    p.add_argument("--hide-arm", action="store_true",
                   help="leave the robot out of the frames; by default it is "
                        "in shot, as it would be through the real camera")
    p.add_argument("--no-yolo", action="store_true",
                   help="write only the CSVs, no YOLO .txt files or data.yaml")
    p.add_argument("--preview", action="store_true",
                   help="also write one image with its boxes drawn on")
    args = p.parse_args(argv)

    view = ViewConfig(azimuth=args.azimuth, azimuth_jitter=args.azimuth_jitter,
                      both_sides=args.sides == "both", elevation=args.elevation,
                      elevation_jitter=args.elevation_jitter,
                      distance=args.distance,
                      distance_jitter=args.distance_jitter)
    out = generate(args.out, args.train, args.val, args.img_size, args.seed, view,
                   sparse_fraction=args.sparse_fraction,
                   yolo=not args.no_yolo, hide_arm=args.hide_arm)
    print("labels:", out / "labels.csv")
    print("frames:", out / "frames.csv")
    if args.preview:
        print("preview:", preview(out))


if __name__ == "__main__":
    main()
