#!/usr/bin/env python3
"""Play chess in the MuJoCo cell: the arm physically moves every piece.

    python main.py --weights runs/.../weights/best.pt
    python main.py --weights ... --mode human        # you play White
    python main.py --weights ... --camera-view       # watch the camera feed
    python main.py --weights ... --video game.mp4    # record it

Every game opens the MuJoCo window, runs at wall-clock speed, starts from the
opening position and stops after 20 half-moves.  Each turn the position is read
back off the camera with the trained detector and played from what was seen, so
weights are required.

Rendering needs a GL backend: MUJOCO_GL=glfw under WSLg, MUJOCO_GL=egl on a
headless box with an NVIDIA driver.
"""

import argparse
import random
import sys

import chess

from core.stockfish_path import available, open_engine


#: A game runs to this many half-moves unless it ends sooner.
MAX_PLIES = 20


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("stockfish", "human", "random"),
                   default="stockfish", help="who chooses the moves")
    p.add_argument("--depth", type=int, default=8, help="Stockfish search depth")
    p.add_argument("--video", default=None, help="record an mp4 to this path")
    p.add_argument("--weights", required=True,
                   help="trained detector weights, e.g. runs/.../weights/best.pt")
    p.add_argument("--camera-azimuth", type=float, default=None, metavar="DEG",
                   help="where the camera stands, in degrees around the board "
                        "from +x: 90 is over Black (default), 0 and 180 are "
                        "side views, 270 is over White")
    p.add_argument("--camera-elevation", type=float, default=None, metavar="DEG",
                   help="degrees above the board plane; 90 looks straight down")
    p.add_argument("--camera-distance", type=float, default=None, metavar="M",
                   help="metres from the board centre to the lens")
    p.add_argument("--camera-view", action="store_true",
                   help="show what the board camera sees: live while the arm "
                        "moves, and with the detector's reading drawn on after "
                        "each move")
    p.add_argument("--camera-homography", action="store_true",
                   help="add the flattened board view beside the camera view, "
                        "with a red dot and label where each detection was "
                        "mapped")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for random move choice and fallbacks")
    return p


def pick_move(args, sim, engine, rng, reader=None):
    """Choose a move for the side to play."""
    legal = list(sim.board.legal_moves)
    if args.mode == "random":
        return rng.choice(legal)
    if args.mode == "human" and sim.board.turn == chess.WHITE:
        while True:
            text = input(f"your move ({sim.board.fen()})> ").strip()
            if text in ("quit", "exit"):
                return None
            try:
                move = sim.parse(text)
            except ValueError as exc:
                print(f"  {exc}")
                continue
            if move in legal:
                return move
            print("  not legal here")
    if engine is None:
        return rng.choice(legal)

    fen = perceived_fen(sim, reader)
    if fen is None:
        print("  camera could not read the board; falling back to game state")
        fen = sim.board.fen()
    engine.set_fen_position(fen)
    best = engine.get_best_move()
    if best is None:
        return None
    move = chess.Move.from_uci(best)
    return move if move in legal else rng.choice(legal)


def make_reader(sim, args):
    """The trained detector, reading the board through core.ChessBoard."""
    from sim.perception import BoardCamera, YoloPerception

    return YoloPerception(args.weights, BoardCamera.from_sim(sim))


def perceived_fen(sim, reader):
    """Full FEN from the camera image plus the fields an image cannot show.

    Side to move, castling rights and the en-passant square are not visible in
    any photograph, so they come from the game record.
    """
    placement = reader.read_fen(sim.capture())
    tail = sim.board.fen().split(" ", 1)[1]
    candidate = f"{placement} {tail}"
    try:
        chess.Board(candidate)
    except ValueError:
        return None
    return candidate


def main(argv=None):
    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)

    engine = None
    if args.mode in ("stockfish", "human"):
        if available():
            engine = open_engine(depth=args.depth)
        else:
            print("Stockfish not found - playing random legal moves instead.\n"
                  "  Install it, or set $STOCKFISH_PATH (see the README).\n")
            if args.mode == "stockfish":
                args.mode = "random"

    from sim.sim_env import ChessSim

    sim = ChessSim(viewer=True, realtime=True, verbose=True)
    if args.video:
        sim.start_recording(args.video)

    if any(v is not None for v in (args.camera_azimuth, args.camera_elevation,
                                   args.camera_distance)):
        sim.move_camera(args.camera_azimuth, args.camera_elevation,
                        args.camera_distance)

    reader = make_reader(sim, args)

    view = None
    if args.camera_view:
        from sim.camera_view import CameraView
        view = CameraView(sim, reader=reader,
                          homography=args.camera_homography)
        sim.attach_camera_view(view)
        view.update(tag="start")

    print(sim.board.unicode(invert_color=True, borders=False))
    try:
        for ply in range(MAX_PLIES):
            if sim.board.is_game_over():
                print(f"\ngame over: {sim.board.result()} "
                      f"({sim.board.outcome().termination.name.lower()})")
                break
            move = pick_move(args, sim, engine, rng, reader)
            if move is None:
                break
            side = "White" if sim.board.turn == chess.WHITE else "Black"
            print(f"\n[{ply + 1}] {side}: {sim.board.san(move)}")
            sim.apply_move(move)
            if view is not None:
                view.update(tag=f"{ply + 1:03d}_{move.uci()}")
            print(sim.board.unicode(invert_color=True, borders=False))
            seen = perceived_fen(sim, reader)
            ok = seen is not None and seen.split()[0] == sim.board.board_fen()
            print(f"    camera agrees with the game: {ok}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        ok, problems = sim.consistent()
        print(f"\nfinal board matches the game: {ok}")
        for p in problems:
            print("   ", p)
        print("final FEN:", sim.board.fen())
        if args.video:
            sim.stop_recording()
            print("video written to", args.video)
        sim.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
