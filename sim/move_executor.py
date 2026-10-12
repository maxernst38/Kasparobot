"""Turns a chess move into the physical transfers the arm performs.

The arm can only move the piece on square A to square B, or to a tray.
Captures, en passant, castling and promotion each become several of those, in
an order that never asks the arm to put two pieces on one square.

Needs python-chess but not MuJoCo, so the rule handling is testable alone.
"""

from dataclasses import dataclass
from typing import List, Optional

import chess

REMOVE = "remove"    # take the piece on `src` off the board, into a tray
SHIFT = "shift"      # carry the piece on `src` to `dst`
SPAWN = "spawn"      # fetch a spare `piece` from stock and put it on `dst`


@dataclass
class Step:
    """One pick-and-place the arm will carry out."""
    kind: str
    src: Optional[str] = None       # square name, e.g. 'e2'
    dst: Optional[str] = None
    piece: Optional[str] = None     # FEN letter, e.g. 'Q' - only for SPAWN
    why: str = ""

    def __str__(self):
        if self.kind == REMOVE:
            return f"remove {self.src} ({self.why})"
        if self.kind == SPAWN:
            return f"spawn {self.piece} on {self.dst} ({self.why})"
        return f"{self.src} -> {self.dst}" + (f" ({self.why})" if self.why else "")


def plan_move(board: chess.Board, move: chess.Move) -> List[Step]:
    """Physical steps for `move`, given the position before it is played.

    A square is always vacated before anything is placed on it.
    """
    steps: List[Step] = []
    src = chess.square_name(move.from_square)
    dst = chess.square_name(move.to_square)

    if board.is_en_passant(move):
        # The captured pawn is beside the destination, not on it.
        captured = chess.square(chess.square_file(move.to_square),
                                chess.square_rank(move.from_square))
        steps.append(Step(REMOVE, src=chess.square_name(captured),
                          why="en passant"))
    elif board.is_capture(move):
        steps.append(Step(REMOVE, src=dst, why="captured"))

    if board.is_castling(move):
        king_file = chess.square_file(move.to_square)
        rank = chess.square_rank(move.from_square)
        if king_file > chess.square_file(move.from_square):      # king side
            rook_from, rook_to = 7, 5
            why = "castles king side"
        else:                                                     # queen side
            rook_from, rook_to = 0, 3
            why = "castles queen side"
        steps.append(Step(SHIFT, src=src, dst=dst, why=why))
        steps.append(Step(SHIFT, src=chess.square_name(chess.square(rook_from, rank)),
                          dst=chess.square_name(chess.square(rook_to, rank)),
                          why="rook"))
        return steps

    if move.promotion:
        letter = chess.piece_symbol(move.promotion)
        letter = letter.upper() if board.turn == chess.WHITE else letter
        steps.append(Step(REMOVE, src=src, why="pawn promotes"))
        steps.append(Step(SPAWN, dst=dst, piece=letter, why="promoted piece"))
        return steps

    steps.append(Step(SHIFT, src=src, dst=dst))
    return steps


def describe(steps: List[Step]) -> str:
    return "; ".join(str(s) for s in steps)
