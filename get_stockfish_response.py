from core.ChessBoard import ChessBoard
from stockfish import Stockfish

def main():
    board = ChessBoard()
    sf = Stockfish(path="/home/maxernst/CHESS_TRACKER/stockfish-ubuntu-x86-64-avx2", depth=20)
    fen = board.update_from_image("dataset/test/images/IMG_0170_JPG.rf.7d0b4b3a0712b93745e320d32c1b7e65.jpg")
    sf.set_fen_position(fen)
    print(sf.get_fen_position())
    print(sf.get_best_move())

if __name__ == "__main__":
    main()