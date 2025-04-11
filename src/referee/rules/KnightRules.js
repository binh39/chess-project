import { Piece, Position } from "../../models/index.js";
import { tileIsEmptyOrOccupiedByOpponent } from "./GeneralRules.js";

export const knightMove = (
  initialPosition,
  desiredPosition,
  team,
  boardState
) => {
  for (let i = -1; i < 2; i += 2) {
    for (let j = -1; j < 2; j += 2) {
      // Top/Bottom movement
      if (
        desiredPosition.y - initialPosition.y === 2 * i &&
        desiredPosition.x - initialPosition.x === j
      ) {
        if (
          tileIsEmptyOrOccupiedByOpponent(desiredPosition, boardState, team)
        ) {
          return true;
        }
      }

      // Right/Left movement
      if (
        desiredPosition.x - initialPosition.x === 2 * i &&
        desiredPosition.y - initialPosition.y === j
      ) {
        if (
          tileIsEmptyOrOccupiedByOpponent(desiredPosition, boardState, team)
        ) {
          return true;
        }
      }
    }
  }
  return false;
};

export const getPossibleKnightMoves = (knight, boardstate) => {
  const possibleMoves = [];

  for (let i = -1; i < 2; i += 2) {
    for (let j = -1; j < 2; j += 2) {
      const verticalMove = new Position(
        knight.position.x + j,
        knight.position.y + i * 2
      );
      const horizontalMove = new Position(
        knight.position.x + i * 2,
        knight.position.y + j
      );

      if (
        tileIsEmptyOrOccupiedByOpponent(verticalMove, boardstate, knight.team)
      ) {
        possibleMoves.push(verticalMove);
      }

      if (
        tileIsEmptyOrOccupiedByOpponent(horizontalMove, boardstate, knight.team)
      ) {
        possibleMoves.push(horizontalMove);
      }
    }
  }

  return possibleMoves;
};
