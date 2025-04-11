import { TeamType } from "../../Types";
import { Piece, Position } from "../../models";
import {
  tileIsEmptyOrOccupiedByOpponent,
  tileIsOccupied,
  tileIsOccupiedByOpponent,
} from "./GeneralRules";

export const queenMove = (
  initialPosition,
  desiredPosition,
  team,
  boardState
) => {
  for (let i = 1; i < 8; i++) {
    // Diagonal hoặc thẳng
    let multiplierX =
      desiredPosition.x < initialPosition.x
        ? -1
        : desiredPosition.x > initialPosition.x
        ? 1
        : 0;
    let multiplierY =
      desiredPosition.y < initialPosition.y
        ? -1
        : desiredPosition.y > initialPosition.y
        ? 1
        : 0;

    let passedPosition = new Position(
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

export const getPossibleQueenMoves = (queen, boardstate) => {
  const possibleMoves = [];

  const directions = [
    { x: 0, y: 1 }, // top
    { x: 0, y: -1 }, // bottom
    { x: -1, y: 0 }, // left
    { x: 1, y: 0 }, // right
    { x: 1, y: 1 }, // top-right
    { x: 1, y: -1 }, // bottom-right
    { x: -1, y: -1 }, // bottom-left
    { x: -1, y: 1 }, // top-left
  ];

  for (const dir of directions) {
    for (let i = 1; i < 8; i++) {
      const destination = new Position(
        queen.position.x + i * dir.x,
        queen.position.y + i * dir.y
      );

      if (!tileIsOccupied(destination, boardstate)) {
        possibleMoves.push(destination);
      } else if (
        tileIsOccupiedByOpponent(destination, boardstate, queen.team)
      ) {
        possibleMoves.push(destination);
        break;
      } else {
        break;
      }
    }
  }

  return possibleMoves;
};
