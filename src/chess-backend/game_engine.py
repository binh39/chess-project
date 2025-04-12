import chess
from models import GameState, Piece

board = chess.Board()

def initialize_board():
    global board
    board = chess.Board() 

def get_game_state():
    pieces = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Tìm tất cả nước đi hợp lệ bắt đầu từ ô này
            legal_moves = [
                chess.square_name(m.to_square)
                for m in board.legal_moves
                if m.from_square == square
            ]

            pieces.append(Piece(
                square=chess.square_name(square),
                piece_type=piece.symbol().lower() if piece.color == chess.BLACK else piece.symbol().upper(),
                color="black" if piece.color == chess.BLACK else "white",
                possible_moves=legal_moves 
            ))

    return GameState(
        pieces=pieces,
        turn="white" if board.turn == chess.WHITE else "black",
        is_check=board.is_check(),
        is_checkmate=board.is_checkmate(),
    )

def promote_pawn(from_square, to_square, piece_type):
    # Chuyển từ định dạng UCI (ví dụ: "e7e8") sang tọa độ của ô
    move = chess.Move.from_uci(from_square + to_square)

    # Kiểm tra quân tốt có di chuyển hợp lệ
    if move.from_square is None or move.to_square is None or board.piece_at(move.from_square) is None:
        return False

    piece = board.piece_at(move.from_square)
    
    # Nếu quân tại ô bắt đầu không phải là quân tốt (pawn), trả về False
    if piece.piece_type != chess.PAWN:
        return False

    # Thực hiện di chuyển quân tốt
    board.push(move)

    # Kiểm tra xem quân tốt có đến hàng cuối không:
    # - Quân trắng phong hậu nếu đến hàng 8 (rank index = 7)
    # - Quân đen phong hậu nếu đến hàng 1 (rank index = 0)
    if (piece.color == chess.WHITE and move.to_square // 8 == 7) or (piece.color == chess.BLACK and move.to_square // 8 == 0):
        # Phong hậu thành quân cờ mới
        piece_mapping = {
            'R': chess.ROOK,
            'N': chess.KNIGHT,
            'B': chess.BISHOP,
            'Q': chess.QUEEN
        }
        
        # Kiểm tra và lấy quân cờ phong hậu
        new_piece = piece_mapping.get(piece_type.upper())
        if new_piece is None:
            return False  # Nếu không có quân cờ hợp lệ, trả về False
        
        # Cập nhật quân cờ tại vị trí mới
        board.set_piece_at(move.to_square, chess.Piece(new_piece, piece.color))
        
        return True

    return False

def make_move(from_square, to_square, piece_type=None):
    if from_square == to_square:
        return False  # Bỏ qua nếu không di chuyển
    
    try:
        if piece_type:  # Nếu có phong hậu, gọi hàm phong hậu
            success = promote_pawn(from_square, to_square, piece_type)
            if success:
                return True
        else:
            move = chess.Move.from_uci(from_square + to_square)
            if move in board.legal_moves:
                board.push(move)
                return True
            else:
                print(from_square + to_square)
    except:
        return False
    return False


def is_valid_move(from_square: str, to_square: str) -> bool:
    move = chess.Move.from_uci(from_square + to_square)

    return move in board.legal_moves

from models import PieceModel, Position
from typing import List

def calculate_valid_moves(square: str):
    """
    Trả về danh sách các nước đi hợp lệ từ ô được chọn (square).
    :param square: vị trí của quân cờ, ví dụ 'e2'
    :return: danh sách ô đích hợp lệ, ví dụ ['e3', 'e4']
    """
    from_square = chess.parse_square(square)
    valid_moves = []

    for move in board.legal_moves:
        if move.from_square == from_square:
            valid_moves.append(chess.square_name(move.to_square))

    return valid_moves
