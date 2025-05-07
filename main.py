from ChessBoard import ChessBoard

def display_help():
    """Displays the list of available commands."""
    print("Commands:")
    print("  - 'move <move>' to make a move (e.g., 'move Nf3')")
    print("  - 'reset' to start a new game")
    print("  - 'exit' to quit the program")
    print("  - 'get legal moves' to see legal moves")
    print("  - 'show' to display the current board")
    print("  - 'get best move' to get the best move from the AI")

def handle_common_commands(board, command):
    """Handles common commands like 'help', 'reset', 'exit', and 'get legal moves'."""
    if command.lower() == "help":
        display_help()
        return True

    if command.lower() == "get legal moves":
        legal_moves = board.get_legal_moves()
        print("Legal Moves:", legal_moves)
        return True

    if command.lower() == "show":
        board.show()
        return True
    
    if command.lower() == "get best move":
        best_move = board.get_best_move()
        best_move_san = board.uci_to_san(best_move)
        print("Best Move:", best_move_san)
        return True
    
    if command.lower() == "reset":
        board.new_game()
        print("New game started.")
        return True

    if command.lower() == "exit":
        print("Exiting the game.")
        exit()

    return False

def ai_game():
    """Handles the AI vs Player game mode."""
    board = ChessBoard()
    player_color = input("Enter your color (white/black): ").strip().lower()

    turn = "Player" if player_color == "white" else "AI"
    
    while not board.is_game_over():
        board.is_check()
        
        color = board.get_turn()
        print(f"Current Turn: {color}")

        if turn == "Player":
            move = input("Command (Help for Instructions): ").strip()

            if handle_common_commands(board, move):
                continue

            move = move.replace("move ", "")
            board.move(move)
            turn = "AI"
        
        else:
            board.stockfish_move()
            turn = "Player"
        
        board.show()

def pvp_game():
    """Handles the Player vs Player game mode."""
    board = ChessBoard()

    while not board.is_game_over():
        move = input("Command (Help for Instructions): ").strip()

        if handle_common_commands(board, move):
            continue

        print(f"Current Turn: {board.get_turn()}")
        move = move.replace("move ", "")
        board.move(move)
        board.show()

def main():
    """Main entry point for the program."""
    print("Welcome to the Chess Game!")
    game_style = input("Enter Game Style (pvp, ai):").strip().lower()

    if game_style == "ai":
        ai_game()
    elif game_style == "pvp":
        pvp_game()
    else:
        print("Invalid game style. Please choose 'pvp' or 'ai'.")

if __name__ == "__main__":
    main()
