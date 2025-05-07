import chess
import chess.svg
import cairosvg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
from ultralytics import YOLO
from stockfish import Stockfish

from detect_board import *
from map_pieces_to_board import *

class ChessBoard:
    def __init__(self, game_style="pvp-C", players=["Player 1", "Player 2"]):
        self.game_style = game_style

        self.board = chess.Board()
        self.board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

        self.stockfish = Stockfish(path="/home/maxernst/CHESS_TRACKER/stockfish-ubuntu-x86-64-avx2", depth=20)

        self.class_to_piece = CLASS_TO_PIECE = {
            0:  '?', 1:  'b', 2:  'k', 3:  'n', 4:  'p', 5:  'q', 6:  'r', 7:  'B', 8:  'K', 9:  'N', 10: 'P', 11: 'Q', 12: 'R',
        }
    
    # Utilities to interact with the chess board
    def print(self):
        print(self.board)

    def set_fen(self, fen):
        self.board.set_fen(fen)
    
    def get_fen(self):
        return self.board.fen()
    
    def uci_to_san(self, uci_move):
        try:
            move = chess.Move.from_uci(uci_move)
            san_move = self.board.san(move)
            return san_move
        except ValueError:
            print(f"Invalid UCI move: {uci_move}")
            return None
    
    def get_turn(self):
        if self.board.turn:
            return "White"
        else:
            return "Black"
    
    def get_legal_moves(self):
        legal_moves = []
        for move in self.board.legal_moves:
            san_move = self.board.san(move)
            legal_moves.append(san_move)
        return legal_moves
    
    def is_game_over(self):
        if self.board.is_checkmate():
            print("Checkmate!")
            return True
        elif self.board.is_stalemate():
            print("Stalemate!")
            return True
        elif self.board.is_insufficient_material():
            print("Insufficient material!")
            return True
        elif self.board.is_seventyfive_moves():
            print("Draw by 75-move rule!")
            return True
        elif self.board.is_fivefold_repetition():
            print("Draw by fivefold repetition!")
            return True
        return False

    def is_check(self):
        if self.board.is_check():
            print("Check!")
            return True
        return False

    def move(self, instruction):
        try:
            self.board.push_san(instruction)
        except:
            print(f"Invalid move: {instruction}")
            print(f"Valid moves are: {self.get_legal_moves()}")
            self.board.turn = not self.board.turn

    def new_game(self):
        self.board.reset()
        self.board.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    # Utilities to interact with AI chess engine
    def get_best_move(self):
        self.stockfish.set_fen_position(self.board.fen())
        best_move = self.stockfish.get_best_move()
        return best_move

    def stockfish_move(self):
        best_move = self.get_best_move()
        self.move(best_move)
    
    # Functions to enable AI vision of 
    def detect_pieces(self, image):
        model = YOLO("runs/train/custom_model/weights/best.pt")

        results = model(image)

        detections = results[0].boxes

        pieces = []
        for det in detections:
            cls = int(det.cls.item())
            coords = det.xywhn.tolist()[0]
            pieces.append((cls, coords))

        return pieces
    
    def detect_board(self, image):
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
    
    def update_from_image(self, image):
        pieces = self.detect_pieces(image)

        grid, H, H_inv = self.detect_board(image)

        square_map = map_detections_to_squares(pieces, image, H)

        fen = self.square_map_to_fen(square_map, self.class_to_piece)
        
        self.set_fen(fen)
        return fen

    def show(self):
        svg_data = chess.svg.board(board=self.board)

        png_bytes = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))

        image_stream = io.BytesIO(png_bytes)
        img = mpimg.imread(image_stream, format='png')

        plt.imshow(img)
        plt.axis('off')
        plt.show()