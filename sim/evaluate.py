"""Score a trained detector the way the robot actually uses it.

    python -m sim.evaluate --weights runs/.../weights/best.pt --positions 100

Detection metrics say how many boxes landed in the right place, not whether
the board was read correctly - and one square wrong out of 32 is a wrong move.
This renders fresh positions, turns the detections into a FEN, and compares
against what is physically on the board.

Boxes become squares by unprojecting the foot of the box onto the board plane.
That point is the near rim of the piece's base rather than its centre, so it is
pushed back out by one base radius first; --no-base-correction skips that.
"""

import argparse
import collections
import math
import random

import numpy as np

from . import board_layout as L
from . import dataset as DS
from .perception import BoardCamera


def box_to_board_point(box, cam, correct_base=True):
    """Bottom-centre of a detection box -> a point on the board plane."""
    x0, y0, x1, y1 = box
    u, v = (x0 + x1) / 2.0, y1
    x, y = cam.unproject(u, v, 0.0)
    if math.isnan(x):
        return None
    if correct_base:
        # The lowest visible point of a piece is the rim of its base nearest
        # the camera, so step back along the view direction to the centre.
        away = np.array([x, y]) - cam.pos[:2]
        norm = np.linalg.norm(away)
        if norm > 1e-9:
            x, y = np.array([x, y]) + (away / norm) * L.PIECE_BASE_R
    return float(x), float(y)


def read_board(result, cam, conf=0.25, correct_base=True):
    """A YOLO result -> `{'e4': 'P', ...}`, keeping the best box per square."""
    best = {}
    boxes = result.boxes
    if boxes is None:
        return {}
    for cls, score, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(),
                                boxes.xyxy.tolist()):
        if score < conf:
            continue
        symbol = DS.CLASS_TO_PIECE.get(int(cls))
        if not symbol or symbol == "?":
            continue
        point = box_to_board_point(xyxy, cam, correct_base)
        if point is None:
            continue
        square = L.xy_to_square(*point)
        if square is None:
            continue
        if square not in best or score > best[square][1]:
            best[square] = (symbol, score, point)
    return best


# The view the robot works from.  Evaluating elsewhere measures generalisation
# it is never asked for.
DEPLOY_VIEW = DS.ViewConfig(azimuth=0.0, azimuth_jitter=12.0, both_sides=False)


def evaluate(weights, positions=100, res=640, seed=12345, conf=0.25,
             sparse_fraction=0.35, view=None, correct_base=True,
             device=None, verbose=True, model=None, sim=None):
    from ultralytics import YOLO

    from .sim_env import ChessSim

    own_sim = sim is None
    model = model if model is not None else YOLO(weights)
    sim = sim if sim is not None else ChessSim(markers=False, verbose=False)
    rng = random.Random(seed)
    view = view or DEPLOY_VIEW

    squares = correct = wrong_type = missed = spurious = 0
    exact = 0
    offsets = []
    confusion = collections.Counter()
    per_class = collections.defaultdict(lambda: [0, 0])     # symbol -> [ok, total]

    for i in range(positions):
        sim.reset(DS.sample_position(rng, sparse_fraction), stash=True)
        DS.aim_side_view(sim, rng, view)
        sim.arm.settle(0.2)

        image = sim.capture(res, res, park=False)
        cam = BoardCamera.from_sim(sim, res)
        result = model.predict(image[..., ::-1], verbose=False, conf=conf,
                               device=device)[0]
        read = read_board(result, cam, conf, correct_base)

        truth = {sq: DS.piece_symbol(body)[0]
                 for sq, body in sim.piece_at.items()}
        position_ok = True
        for square in set(truth) | set(read):
            squares += 1
            want = truth.get(square)
            got = read.get(square, (None,))[0]
            if want == got:
                correct += 1
                if want:
                    per_class[want][0] += 1
                    per_class[want][1] += 1
            else:
                position_ok = False
                if want is None:
                    spurious += 1
                elif got is None:
                    missed += 1
                    per_class[want][1] += 1
                else:
                    wrong_type += 1
                    per_class[want][1] += 1
                    confusion[(want, got)] += 1
            if want is not None and got is not None and square in read:
                centre = np.array(L.square_to_xy(square))
                offsets.append(float(np.linalg.norm(read[square][2] - centre)))
        exact += position_ok
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{positions} positions")

    if own_sim:
        sim.close()
    return {
        "positions": positions, "squares": squares, "correct": correct,
        "wrong_type": wrong_type, "missed": missed, "spurious": spurious,
        "exact_positions": exact,
        "offsets": offsets, "confusion": confusion, "per_class": dict(per_class),
    }


def report(stats):
    sq, n = stats["squares"], stats["positions"]
    acc = 100.0 * stats["correct"] / max(1, sq)
    print(f"\npositions            : {n}")
    print(f"squares compared     : {sq}")
    print(f"square accuracy      : {acc:.3f}%  "
          f"({stats['correct']}/{sq})")
    print(f"  wrong piece type   : {stats['wrong_type']}")
    print(f"  missed (no box)    : {stats['missed']}")
    print(f"  spurious (no piece): {stats['spurious']}")
    print(f"whole position right : {stats['exact_positions']}/{n}"
          f"  ({100.0 * stats['exact_positions'] / max(1, n):.1f}%)"
          "   <- what the robot needs")
    if stats["offsets"]:
        off = np.array(stats["offsets"]) * 1000
        print(f"localisation error   : median {np.median(off):.1f} mm, "
              f"90th pct {np.percentile(off, 90):.1f} mm, max {off.max():.1f} mm"
              f"   (square is {L.SQUARE * 1000:.0f} mm)")
    if stats["confusion"]:
        print("most confused pairs  :")
        for (want, got), count in stats["confusion"].most_common(6):
            print(f"    {want} read as {got}: {count}")
    weak = sorted(((ok / max(1, tot), sym, ok, tot)
                   for sym, (ok, tot) in stats["per_class"].items()))
    print("weakest classes      : " + ", ".join(
        f"{sym} {100 * r:.1f}% ({ok}/{tot})" for r, sym, ok, tot in weak[:5]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="path to best.pt")
    p.add_argument("--positions", type=int, default=100,
                   help="how many fresh positions to render and read")
    p.add_argument("--img-size", type=int, default=640,
                   help="rendered frame size in pixels, square")
    p.add_argument("--seed", type=int, default=12345,
                   help="keep this different from the dataset seed so the "
                        "positions are ones the model has not seen")
    p.add_argument("--conf", type=float, default=0.25,
                   help="minimum detection confidence to accept")
    p.add_argument("--azimuth", type=float, default=DEPLOY_VIEW.azimuth,
                   help="side-view azimuth to evaluate at (default 0, the "
                        "angle the robot works from)")
    p.add_argument("--azimuth-jitter", type=float,
                   default=DEPLOY_VIEW.azimuth_jitter,
                   help="degrees of random wobble either side of --azimuth")
    p.add_argument("--sides", choices=("one", "both"), default="one",
                   help="sample only the --azimuth side, or that side and the "
                        "one 180 degrees opposite")
    p.add_argument("--device", default=None, help="e.g. cpu or 0")
    args = p.parse_args(argv)

    view = DS.ViewConfig(azimuth=args.azimuth,
                         azimuth_jitter=args.azimuth_jitter,
                         both_sides=args.sides == "both")
    stats = evaluate(args.weights, args.positions, args.img_size, args.seed,
                     args.conf, view=view, device=args.device)
    print(f"evaluated at azimuth {args.azimuth:g} +/- {args.azimuth_jitter:g} deg")
    report(stats)


if __name__ == "__main__":
    main()
