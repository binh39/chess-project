from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import  PieceModel, Position, MoveRequest,  SquareRequest
from game_engine import make_move, get_game_state, initialize_board, is_valid_move, calculate_valid_moves

app = FastAPI()

# Cho phép React frontend gọi API này
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/state")
def get_state():
    return get_game_state()

@app.post("/move")
def play_move(move: MoveRequest):
    success = make_move(move.from_square, move.to_square)
    return {"success": success, "state": get_game_state()}

@app.post("/restart")
def restart_game():
    initialize_board()
    return get_game_state()

@app.post("/valid_move")
def valid_move(req: MoveRequest):
    result = is_valid_move(req.from_square, req.to_square)
    return {"valid": result}

@app.post("/valid_move")
def valid_move(req: SquareRequest):
    valid_moves = calculate_valid_moves(req.square)
    return {"moves": valid_moves}