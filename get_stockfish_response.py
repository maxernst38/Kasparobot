import sys

from core.ChessBoard import ChessBoard
from core.stockfish_path import open_engine

DEFAULT_IMAGE = "dataset/test/images/IMG_0170_JPG.rf.7d0b4b3a0712b93745e320d32c1b7e65.jpg"

def main():
    image = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    board = ChessBoard()
    sf = open_engine(depth=20)
    fen = board.update_from_image(image)
    sf.set_fen_position(fen)
    print(sf.get_fen_position())
    print(sf.get_best_move())

if __name__ == "__main__":
    main()