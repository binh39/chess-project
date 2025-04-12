from pydantic import BaseModel
from typing import List, Optional

class Position(BaseModel):
    x: int
    y: int

class PieceModel(BaseModel):
    type: str
    team: int
    position: Position
    isKing: Optional[bool] = False
    hasMoved: Optional[bool] = False
    enPassant: Optional[bool] = False

class SquareRequest(BaseModel):
    square: str  # ví dụ: "e2"


class MoveRequest(BaseModel):
    from_square: str  # e.g. "e2"
    to_square: str    # e.g. "e4"

class Piece(BaseModel):
    square: str
    piece_type: str
    color: str
    possible_moves: list[str] = []

class GameState(BaseModel):
    pieces: list[Piece]
    turn: str
    is_check: bool
    is_checkmate: bool
