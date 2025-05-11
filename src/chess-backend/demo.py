import chess
import chess.engine

import chess
import random
import math
import time
from models import GameState, Piece

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
    model.model.load_weights(model_weight)
    model.model.summary()
    if not load_best_model_weight(model):
        raise RuntimeError("Best model not found!")
    import h5py

    with h5py.File(model_weight, 'r') as f:
        print(list(f.keys()))  # Kiểm tra các nhóm trong file
    if not load_best_model_weight(model):
        raise RuntimeError("Best model not found!")

board = env.board

def get_player(config):
    
    return ChessPlayer(config, model.get_pipes(config.play.search_threads))



def info(depth, move, score):
    print(f"info score cp {int(score*100)} depth {depth} pv {move}")
    sys.stdout.flush()

env.reset()
# Khởi tạo engine
engine_path = "D:\App Install\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
mcts_win_count = 0
mcts_lose_count =0

# MCTS chơi trắng, Stockfish chơi đen
mcts_color = chess.WHITE
board = env.board
res = np.array([])
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

for i in range (10):
    env.reset()
    board = env.board
    x = 0
    while not board.is_game_over(claim_draw=True):
        
        if board.turn == mcts_color:
            # MCTS chọn nước đi
            if not me_player:
                me_player = get_player(default_config)
            try:
                action = me_player.action(env, False)
                print(action)
                env.step(action)
                x += 1
                print(x)
            except (chess.IllegalMoveError, ValueError) as e:
                print("Phát hiện hòa cờ hoặc nước đi không hợp lệ:", e)
                env.board.clear_stack()  # nếu cần xóa stack để tránh lỗi tiếp
                env._done = True
                env._winner = 0 #Hòa
                break
            print("MCTS chọn:", action)
        else:
            # Stockfish chọn nước đi
            with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
                engine.configure({
                    "UCI_LimitStrength": True,  # Bật giới hạn sức mạnh
                    "UCI_Elo": 2800             # Chỉnh mức ELO (mặc định Stockfish mạnh hơn 3200)
                })
                result = engine.play(board, chess.engine.Limit(time=10))
            move = result.move.uci()
            env.step(move)
            print("Stockfish chọn:", move)
        print(board)

        # Thực hiện nước đi

    # Kết thúc
    print(board)
    print("Kết quả:", board.result(claim_draw=True))
    np.append(res, board.result(claim_draw=True))
    result_str = board.result(claim_draw=True)
    if result_str == "1-0":
        mcts_win_count += 1
    elif result_str == "0-1":
        mcts_lose_count += 1

print(mcts_win_count, mcts_lose_count)