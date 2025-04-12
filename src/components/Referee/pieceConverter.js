import { Piece } from "../../models/Piece";
import { Pawn } from "../../models/Pawn";
import { Position } from "../../models/Position";
import { PieceType, TeamType } from "../../Types";

//  "e2" => Position(x: 4, y: 1)
function convertSquareToPosition(square) {
  const file = square.charCodeAt(0) - 97; // a=0, b=1, ...
  const rank = parseInt(square[1], 10) - 1; // "1" => 0
  return new Position(file, rank);
}

// "P"/"p" → "pawn", "R"/"r" → "rook", ...
function mapPieceType(letter) {
  const lower = letter.toLowerCase();
  switch (lower) {
    case "p": return PieceType.PAWN;
    case "r": return PieceType.ROOK;
    case "n": return PieceType.KNIGHT;
    case "b": return PieceType.BISHOP;
    case "q": return PieceType.QUEEN;
    case "k": return PieceType.KING;
    default: return null;
  }
}

// "white"/"black" → TeamType
function mapTeam(color) {
  return color === "white" ? TeamType.OUR : TeamType.OPPONENT;
}

//  Convert tất cả pieces từ backend về class instance
export function convertBackendPieces(piecesFromAPI) {
  return piecesFromAPI.map((p) => {
    const position = convertSquareToPosition(p.square);
    const type = mapPieceType(p.piece_type);
    const team = mapTeam(p.color);
    const hasMoved = false;         // bạn có thể sửa nếu backend gửi về
    const possibleMoves = Array.isArray(p.possible_moves)
      ? p.possible_moves.map(convertSquareToPosition)
      : [];

    if (type === PieceType.PAWN) {
      return new Pawn(position, team, hasMoved, false, possibleMoves); // Pawn(pos, team, hasMoved, enPassant, moves)
    } else {
      return new Piece(position, type, team, hasMoved, possibleMoves); // Piece(pos, type, team, hasMoved, moves)
    }
  });
}
