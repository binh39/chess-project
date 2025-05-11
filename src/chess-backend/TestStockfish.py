import chess
import chess.engine

stockfish_path = "D:\ADMIN\Documents\Code\ChessProject\chess-project\src\chess-backend\stockfish17.exe"

# Khởi tạo bàn cờ
board = chess.Board()

# Khởi tạo engine
engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
engine.configure({
    "UCI_LimitStrength": True,
    "UCI_Elo": 1500
})

while not board.is_game_over():
    print(board)
    print()

    # Stockfish chọn nước đi
    result = engine.play(board, chess.engine.Limit(time=0.1))  # hoặc depth=15
    move = result.move
    
    print("Stockfish chọn:", move)

    # Thực hiện nước đi
    board.push(move)
    move_str = move.uci()
    print(move_str)
    print(type(move_str))

# Kết thúc
print(board)
print("Kết quả:", board.result())

engine.quit()