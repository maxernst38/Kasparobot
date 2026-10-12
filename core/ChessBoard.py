import pathlib

import chess
from ultralytics import YOLO

from core.detection.detect_board import *
from core.detection.map_pieces_to_board import *


class ChessBoard:
    """Reads a chess position out of an image.

    Detects the pieces with YOLO, finds the board's homography, maps each
    detection onto a square and assembles a FEN.
    """

    DEFAULT_WEIGHTS = "model/Kasparobot.pt"

    CLASS_TO_PIECE = {
        0: '?', 1: 'b', 2: 'k', 3: 'n', 4: 'p', 5: 'q',
        6: 'r', 7: 'B', 8: 'K', 9: 'N', 10: 'P', 11: 'Q', 12: 'R',
    }

    def __init__(self, weights=None):
        self.board = chess.Board()
        self.class_to_piece = dict(self.CLASS_TO_PIECE)

        # Loaded on first use, then kept.
        self._weights = weights or self.DEFAULT_WEIGHTS
        self._detector = None

    def set_fen(self, fen):
        self.board.set_fen(fen)

    def get_fen(self):
        return self.board.fen()

    @property
    def detector(self):
        """The YOLO model, loaded once and kept.

        Rebuilding it per call costs about a second a frame.
        """
        if self._detector is None:
            if not pathlib.Path(self._weights).exists():
                raise FileNotFoundError(
                    f"no detector weights at {self._weights!r}. They are not "
                    f"kept in the repository - see model/README.md to fetch a "
                    f"release or train your own.")
            self._detector = YOLO(self._weights)
        return self._detector

    def detect_pieces(self, image, conf=0.25):
        """Detections as (class_id, (cx, cy, w, h)) normalised to the image.

        `image` may be a path or an RGB array.
        """
        source = image
        if hasattr(image, "shape"):
            source = image[..., ::-1]          # ultralytics expects BGR arrays

        results = self.detector.predict(source, conf=conf, verbose=False)

        pieces = []
        for det in results[0].boxes:
            cls = int(det.cls.item())
            coords = det.xywhn.tolist()[0]
            pieces.append((cls, coords))

        return pieces

    def detect_board(self, image):
        """Homography from the board's grid lines.

        Only works on a roughly top-down, axis-aligned view; see
        `sim.perception.grid_homography` for the simulator's version.
        """
        mask = create_mask(image)
        intersections = find_intersections(mask)
        grid = sort_grid_geometrically(intersections, row_thresh=20)
        grid, H, H_inv = compute_homography_from_grid(grid)

        return grid, H, H_inv

    def square_map_to_fen(self, square_map, class_to_piece):
        board = [['' for _ in range(8)] for _ in range(8)]

        for square, class_id in square_map.items():
            file = ord(square[0]) - ord('a')
            rank = 8 - int(square[1])
            board[rank][file] = class_to_piece[class_id]

        fen_rows = []
        for row in board:
            fen_row = ''
            empty = 0
            for square in row:
                if square == '':
                    empty += 1
                else:
                    if empty > 0:
                        fen_row += str(empty)
                        empty = 0
                    fen_row += square
            if empty > 0:
                fen_row += str(empty)
            fen_rows.append(fen_row)

        return '/'.join(fen_rows)

    def update_from_image(self, image, board_orientation=90):
        """Read `image` and set the board to the position found in it."""
        pieces = self.detect_pieces(image)

        grid, H, H_inv = self.detect_board(image)

        square_map = map_detections_to_squares(
            pieces, image, H, board_orientation=board_orientation)

        fen = self.square_map_to_fen(square_map, self.class_to_piece)

        self.set_fen(fen)
        return fen
