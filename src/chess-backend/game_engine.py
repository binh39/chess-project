import chess
import random
import math
import time
from models import GameState, Piece
import chess.engine
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

def mcts_search(board, max_time=5):
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

move_count = 0
def bot_move():
    global board
    global me_player
    global engine_path
    global engine
    
    if not me_player:
        me_player = get_player(default_config)
    try:
        action = me_player.action(env, False)
        print(action)
        env.step(action)
    except (chess.IllegalMoveError, ValueError) as e:
        print("Phát hiện hòa cờ hoặc nước đi không hợp lệ:", e)
        env.board.clear_stack()  # nếu cần xóa stack để tránh lỗi tiếp
        env._done = True
        env._winner = 0 #Hòa
        return False, action

    return True, action

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
        is_draw = board.is_stalemate() or
                board.is_insufficient_material() or
                board.is_fifty_moves() or
                board.can_claim_threefold_repetition()
    )

def promote_pawn(from_square, to_square, piece_type):
    global board
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
    global board
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

import concurrent.futures
stockfish_path = r"D:\ADMIN\Documents\Code\ChessProject\chess-project\src\chess-backend\stockfish17.exe"

def create_chuppy():
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({
        "UCI_LimitStrength": True,
        "UCI_Elo": 1500
    })
    return engine
import traceback
from stockfish import Stockfish

def stock_fish_move():
    global board
    try:
        stockfish = Stockfish(path=stockfish_path)
        stockfish.update_engine_parameters({
        "UCI_LimitStrength": True,
        "UCI_Elo": 2500
    })
        stockfish.set_fen_position(board.fen())
        action = stockfish.get_best_move()

        try:
            print(action)
            env.step(action)
        except (chess.IllegalMoveError, ValueError) as e:
            print("Phát hiện hòa cờ hoặc nước đi không hợp lệ:", e)
            env.board.clear_stack()  # nếu cần xóa stack để tránh lỗi tiếp
            env._done = True
            env._winner = 0 #Hòa
            return False, action

        return True, action
    except Exception as e:
        traceback.print_exc()
        return False, None

