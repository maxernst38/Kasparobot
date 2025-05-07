from stockfish import Stockfish

stockfish = Stockfish(path="/home/maxernst/CHESS_TRACKER/stockfish-ubuntu-x86-64-avx2", depth=20)

stockfish.set_position(["e2e4", "e7e5"])
print(stockfish.get_best_move())