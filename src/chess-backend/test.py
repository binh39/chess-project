import chess

board = chess.Board()

# Lặp đi lặp lại chuỗi các nước đi để tái hiện vị trí giống nhau
moves = ["Nf3", "Nf6", "Ng1", "Ng8"]  # Mã nhảy ra rồi quay lại

for _ in range(6):  # Lặp lại chuỗi 3 lần
    for move in moves:
        board.push_san(move)
        print(board)

print(board)
print("Is draw by threefold repetition?", board.can_claim_threefold_repetition())
