#!/usr/bin/env python3
"""Read a chess position out of a photograph and ask Stockfish for a move.

    python get_stockfish_response.py board.jpg
    python get_stockfish_response.py board.jpg --orientation 0

The board detection finds the grid lines, so the photograph needs to be roughly
top-down and square-on; see sim.perception for the simulator's version, which
takes the homography from a known camera pose instead.
"""

import argparse
import sys

from core.ChessBoard import ChessBoard
from core.stockfish_path import StockfishMissing, open_engine


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="photograph of a chess board")
    p.add_argument("--weights", default=None,
                   help="detector weights (default: model/Kasparobot.pt)")
    p.add_argument("--depth", type=int, default=20,
                   help="Stockfish search depth")
    p.add_argument("--orientation", type=int, default=90,
                   choices=(0, 90, 180, 270),
                   help="how the board is turned in the photograph, in degrees")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    board = ChessBoard(weights=args.weights)
    fen = board.update_from_image(args.image, board_orientation=args.orientation)
    print("position:", fen)

    try:
        engine = open_engine(depth=args.depth)
    except StockfishMissing as exc:
        print(exc)
        return 2
    engine.set_fen_position(fen)
    print("best move:", engine.get_best_move())
    return 0


if __name__ == "__main__":
    sys.exit(main())
