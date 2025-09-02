from ChessBoard import ChessBoard

def main():
    board = ChessBoard()
    fen = board.update_from_image("dataset/test/images/IMG_0169_JPG.rf.149a812d43870ec909dd9fb6cd5ad96b.jpg")
    board.show()
    print("FEN:", fen)

if __name__ == "__main__":
    main()