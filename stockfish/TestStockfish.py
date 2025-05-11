import chess
import chess.engine

import chess
import random
import math
import time
from models import GameState, Piece
import chess.engine
import numpy as np
import tensorflow as tf
import sys
from chess_zero.agent.player_chess import ChessPlayer
from chess_zero.config import Config, PlayWithHumanConfig
from chess_zero.env.chess_env import ChessEnv
from chess_zero.agent.model_chess import ChessModel
from chess_zero.lib.model_helper import load_best_model_weight
from chess_zero.config import ResourceConfig

src = ResourceConfig()
model_weight = src.model_best_weight_path

default_config = Config("normal")
PlayWithHumanConfig().update_play_config(default_config.play)
me_player = None
env = ChessEnv().reset()


with tf.device('/GPU:0'):
    model = ChessModel(default_config)
    model.build()
    model.model.summary()
    if not load_best_model_weight(model):
        raise RuntimeError("Best model not found!")
    
    import h5py
    with h5py.File(model_weight, 'r') as f:
        print(list(f.keys()))  # Kiểm tra các nhóm trong file
    if not load_best_model_weight(model):
        raise RuntimeError("Best model not found!")

board = env.board

def initialize_board():
    global board
    global env
    env.reset()
    board = env.board

def get_player(config):
    
    return ChessPlayer(config, model.get_pipes(config.play.search_threads))



def info(depth, move, score):
    print(f"info score cp {int(score*100)} depth {depth} pv {move}")
    sys.stdout.flush()
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

stockfish_path = "D:\Mon_Hoc\Ky_2_24-25\AI\chess-project\stockfish\stockfish17.exe"

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

# Kết thúc
print(board)
print("Kết quả:", board.result())

engine.quit()