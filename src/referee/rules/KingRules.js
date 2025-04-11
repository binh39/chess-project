import { Piece, Position } from "../../models/index.js";
import {
  tileIsEmptyOrOccupiedByOpponent,
  tileIsOccupied,
  tileIsOccupiedByOpponent,
} from "./GeneralRules.js";

export const kingMove = (
  initialPosition,
  desiredPosition,
  team,
  boardState
) => {
  for (let i = 1; i < 2; i++) {
    const multiplierX =
      desiredPosition.x < initialPosition.x
        ? -1
        : desiredPosition.x > initialPosition.x
        ? 1
        : 0;
    const multiplierY =
      desiredPosition.y < initialPosition.y
        ? -1
        : desiredPosition.y > initialPosition.y
        ? 1
        : 0;

    const passedPosition = new Position(
      initialPosition.x + i * multiplierX,
      initialPosition.y + i * multiplierY
    );

    if (passedPosition.samePosition(desiredPosition)) {
      if (tileIsEmptyOrOccupiedByOpponent(passedPosition, boardState, team)) {
        return true;
      }
    } else {
      if (tileIsOccupied(passedPosition, boardState)) {
        break;
      }
    }
  }
  return false;
};

export const getPossibleKingMoves = (king, boardstate) => {
  const possibleMoves = [];
  const directions = [
    [0, 1],
    [0, -1],
    [-1, 0],
    [1, 0],
    [1, 1],
    [1, -1],
    [-1, -1],
    [-1, 1],
  ];

  for (const [dx, dy] of directions) {
    const x = king.position.x + dx;
    const y = king.position.y + dy;

    if (x < 0 || x > 7 || y < 0 || y > 7) continue;

    const destination = new Position(x, y);

    if (!tileIsOccupied(destination, boardstate)) {
      possibleMoves.push(destination);
    } else if (tileIsOccupiedByOpponent(destination, boardstate, king.team)) {
      possibleMoves.push(destination);
    }
  }

  return possibleMoves;
};

export const getCastlingMoves = (king, boardstate) => {
  const possibleMoves = [];

  if (king.hasMoved) return possibleMoves;

  const rooks = boardstate.filter(
    (p) => p.isRook && p.team === king.team && !p.hasMoved
  );

  for (const rook of rooks) {
    const direction = rook.position.x - king.position.x > 0 ? 1 : -1;
    const adjacentPosition = king.position.clone();
    adjacentPosition.x += direction;

    if (!rook.possibleMoves?.some((m) => m.samePosition(adjacentPosition)))
      continue;

    const concerningTiles = rook.possibleMoves.filter(
      (m) => m.y === king.position.y
    );

    const enemyPieces = boardstate.filter((p) => p.team !== king.team);

    let valid = true;

    for (const enemy of enemyPieces) {
      if (!enemy.possibleMoves) continue;

      for (const move of enemy.possibleMoves) {
        if (concerningTiles.some((t) => t.samePosition(move))) {
          valid = false;
          break;
        }
      }

      if (!valid) break;
    }

    if (!valid) continue;

    possibleMoves.push(rook.position.clone());
  }

  return possibleMoves;
};
