import { TeamType, PieceType } from "../Types";
import { Position } from "./Position";

export class Piece {
  constructor(position, type, team, hasMoved, possibleMoves = []) {
    this.image = `assets/images/${type}_${team}.png`;
    this.position = position;
    this.type = type;
    this.team = team;
    this.possibleMoves = possibleMoves;
    this.hasMoved = hasMoved;
  }

  get isPawn() {
    return this.type === PieceType.PAWN;
  }

  get isRook() {
    return this.type === PieceType.ROOK;
  }

  get isKnight() {
    return this.type === PieceType.KNIGHT;
  }

  get isBishop() {
    return this.type === PieceType.BISHOP;
  }

  get isKing() {
    return this.type === PieceType.KING;
  }

  get isQueen() {
    return this.type === PieceType.QUEEN;
  }

  samePiecePosition(otherPiece) {
    return this.position.samePosition(otherPiece.position);
  }

  samePosition(otherPosition) {
    return this.position.samePosition(otherPosition);
  }

  clone() {
    return new Piece(
      this.position.clone(),
      this.type,
      this.team,
      this.hasMoved,
      this.possibleMoves?.map((m) => m.clone())
    );
  }
}
