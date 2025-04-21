import chess
import random
import math
import time
from models import GameState, Piece
from core.chess_base import ChessEnv  # Lớp môi trường cờ
from core.model import ChessNet
from training.utils import load_predict_model
import torch
import numpy as np
from core.mcts import MCTS


env = ChessEnv()
env.reset()

board = env.chess_board
MAX_TIME = 2
action_dim = 4864
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChessNet()
model = load_predict_model(r"model_checkpoint\best_model.pth", model)
model.to(device)
model.eval()

def initialize_board():
    global board
    state = env.reset()
    board = env.chess_board

# --- MCTS Implementation ---
class MCTSNode:
    def __init__(self, move=None, parent=None):
        self.move = move  # Nước đi dẫn đến node này
        self.parent = parent
        self.children = []  # Các node con (các nước đi tiếp theo)
        self.wins = 0  # Số lần thắng
        self.visits = 0  # Số lần node được thăm
        self.untried_moves = None  # Các nước đi chưa thử

    def uct(self):
        # Tính giá trị UCT (Upper Confidence Bound for Trees)
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = math.sqrt(2 * math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self):
        # Chọn node con có giá trị UCT cao nhất
        return max(self.children, key=lambda c: c.uct())

    def add_child(self, move):
        # Thêm một node con với nước đi mới
        child = MCTSNode(move=move, parent=self)
        self.children.append(child)
        return child

    def update(self, result):
        # Cập nhật số lần thắng và thăm
        self.visits += 1
        self.wins += result

def mcts_search(board, max_time=10):
    root = MCTSNode()
    root.untried_moves = list(board.legal_moves)

    print("Thinking... (max 10s)")
    count = 0
    start_time = time.time()
    while time.time() - start_time < max_time:
        count = count + 1
        node = root
        temp_board = board.copy()

        # 1. Selection
        while node.untried_moves == [] and node.children != []:
            node = node.select_child()
            temp_board.push(node.move)

        # 2. Expansion
        if node.untried_moves:
            move = node.untried_moves.pop(0)
            temp_board.push(move)
            node = node.add_child(move)

        # 3. Simulation
        result = simulate_random_game(temp_board)

        # 4. Backpropagation
        while node is not None:
            node.update(result)
            node = node.parent

    print("Lặp ", count, " bước.")
    # Chọn nước đi có số lần thăm cao nhất
    if not root.children:  # Nếu không có nước đi nào được thử
        return random.choice(list(board.legal_moves))
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move

def simulate_random_game(board):
    # Mô phỏng một ván đấu ngẫu nhiên
    temp_board = board.copy()
    while not temp_board.is_game_over():
        legal_moves = list(temp_board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        temp_board.push(move)
    # Trả về kết quả: 1 nếu bot thắng, 0 nếu hòa, -1 nếu bot thua
    if temp_board.is_checkmate():
        return 1 if temp_board.turn == chess.BLACK else -1
    return 0  # Hòa

def evaluate_board(board):
    # Hàm đánh giá bàn cờ đơn giản
    if board.is_checkmate():
        return 1000 if board.turn == chess.BLACK else -1000  # Chiếu hết: điểm cao
    if board.is_check():
        return 50 if board.turn == chess.BLACK else -50  # Chiếu: điểm trung bình
    # Đếm số quân cờ trên bàn để đánh giá vị thế, điểm tương ứng với mỗi quân
    material_score = 0
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            material_score += value if piece.color == chess.WHITE else -value
    return material_score

def choose_best_promotion(from_square, to_square):
    # Chọn quân phong hậu tối ưu
    move = chess.Move.from_uci(from_square + to_square)
    if move.from_square is None or move.to_square is None or board.piece_at(move.from_square) is None:
        return 'q'  # Mặc định chọn Queen nếu có lỗi

    piece = board.piece_at(move.from_square)
    if piece.piece_type != chess.PAWN:
        return 'q'  # Mặc định chọn Queen nếu không phải pawn

    # Danh sách các quân có thể phong hậu
    promotion_options = [
        (chess.QUEEN, 'q'),
        (chess.ROOK, 'r'),
        (chess.BISHOP, 'b'),
        (chess.KNIGHT, 'n')
    ]

    best_score = float('-inf')
    best_promotion = 'q'  # Mặc định là Queen

    # Thử từng loại quân phong hậu và đánh giá
    for promotion_piece, promotion_letter in promotion_options:
        temp_board = board.copy()
        temp_board.push(move)
        if (piece.color == chess.WHITE and move.to_square // 8 == 7) or (piece.color == chess.BLACK and move.to_square // 8 == 0):
            temp_board.set_piece_at(move.to_square, chess.Piece(promotion_piece, piece.color))
        score = evaluate_board(temp_board)
        # Nếu bot là trắng, tối đa hóa điểm; nếu bot là đen, tối thiểu hóa điểm
        adjusted_score = score if piece.color == chess.WHITE else -score
        if adjusted_score > best_score:
            best_score = adjusted_score
            best_promotion = promotion_letter

    return best_promotion

# def bot_move():
#     global board
#     try:
#         if board.is_game_over():
#             print("Game over, no move to make")
#             return False, None
#         move = mcts_search(board, max_time=MAX_TIME)
#         piece = board.piece_at(move.from_square)
#         if piece is None:
#             print(" Invalid move: from_square has no piece.")
#             return False, None
#         is_promotion = (
#             piece.piece_type == chess.PAWN and
#             ((piece.color == chess.WHITE and move.to_square // 8 == 7) or
#              (piece.color == chess.BLACK and move.to_square // 8 == 0))
#         )
#         if is_promotion:
#             promotion = choose_best_promotion(chess.square_name(move.from_square), chess.square_name(move.to_square))
#             if promotion not in ['q', 'r', 'b', 'n']:
#                 print(f"Invalid promotion {promotion}, defaulting to Queen")
#                 promotion = 'q'
#             move = chess.Move(move.from_square, move.to_square, promotion=chess.Piece.from_symbol(promotion.upper()))
#         board.push(move)
#         return True, move.uci()
#     except Exception as e:
#         print(f"Error in bot_move: {e}")
#         raise

move_count = 0
def bot_move():
    global board
    state = env._observation()
    legal_moves = list(env.chess_board.legal_moves)
    mcts = MCTS(
        neural_net=model,
        converter=env.chess_coords,
        env=env,
        simulations=50,  # Số lượt mô phỏng cho mỗi nước đi
        max_depth=30,     # Độ sâu tối đa cho mỗi mô phỏng
        device=device,
        num_processes=8,  # Số process cho parallel search
        use_model=True    # Sử dụng model để dự đoán nước đi
    )
    try:
        if board.is_game_over():
            print("Game over, no move to make")
            return False, None
        pi = mcts.run(env.chess_board)
        # Chọn nước đi dựa trên policy từ MCTS
        env._update_legal_actions()
        valid_moves = env.legal_actions
        pi_valid = pi * valid_moves
        
        if np.sum(pi_valid) > 0:
            
            action = np.argmax(pi_valid)
        else:
            action = np.random.choice(np.where(valid_moves)[0])

        # Thực hiện nước đi
        move_uci = env.chess_coords.index_to_move(action)
        
        env.step(action)
        return True, move_uci.uci()
    except Exception as e:
        print(f"Error in bot_move: {e}")
        raise


# --- Game ---
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
    action = env.chess_coords.move_to_index(move)
    env.step(action)

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
                action = env.chess_coords.move_to_index(move)
                env.step(action)
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
