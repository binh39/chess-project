from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import  PieceModel, Position, MoveRequest,  SquareRequest
from game_engine import make_move, get_game_state, initialize_board, is_valid_move, calculate_valid_moves, bot_move
import threading
from fastapi.concurrency import run_in_threadpool

board_lock = threading.Lock()

app = FastAPI()

# Cho phép React frontend gọi API này
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/state")
def get_state():
    with board_lock:
        return get_game_state()

@app.post("/move")
def play_move(move: MoveRequest):
    with board_lock:
        success = make_move(move.from_square, move.to_square)
    return {"success": success, "state": get_game_state()}

@app.post("/bot_move")
def play_bot_move():
    with board_lock:
        success, move = bot_move()
    return {"success": success, "move": move, "state": get_game_state()}

@app.post("/restart")
def restart_game():
    with board_lock:
        initialize_board()
    return get_game_state()

@app.post("/valid_move")
def valid_move(req: MoveRequest):
    with board_lock:
        result = is_valid_move(req.from_square, req.to_square)
    return {"valid": result}

@app.post("/valid_move")
def valid_move(req: SquareRequest):
    with board_lock:
        valid_moves = calculate_valid_moves(req.square)
    return {"moves": valid_moves}