import { Piece, Position } from "../../models/index.js";
import { TeamType } from "../../Types.js";
import {
  tileIsEmptyOrOccupiedByOpponent,
  tileIsOccupied,
  tileIsOccupiedByOpponent,
} from "./GeneralRules.js";

export const bishopMove = (
  initialPosition,
  desiredPosition,
  team,
  boardState
) => {
  for (let i = 1; i < 8; i++) {
    // Up right
    if (
      desiredPosition.x > initialPosition.x &&
      desiredPosition.y > initialPosition.y
    ) {
      const passedPosition = new Position(
        initialPosition.x + i,
        initialPosition.y + i
      );
      if (passedPosition.samePosition(desiredPosition)) {
        if (tileIsEmptyOrOccupiedByOpponent(passedPosition, boardState, team))
          return true;
      } else if (tileIsOccupied(passedPosition, boardState)) break;
    }

    // Bottom right
    if (
      desiredPosition.x > initialPosition.x &&
      desiredPosition.y < initialPosition.y
    ) {
      const passedPosition = new Position(
        initialPosition.x + i,
        initialPosition.y - i
      );
      if (passedPosition.samePosition(desiredPosition)) {
        if (tileIsEmptyOrOccupiedByOpponent(passedPosition, boardState, team))
          return true;
      } else if (tileIsOccupied(passedPosition, boardState)) break;
    }

    // Bottom left
    if (
      desiredPosition.x < initialPosition.x &&
      desiredPosition.y < initialPosition.y
    ) {
      const passedPosition = new Position(
        initialPosition.x - i,
        initialPosition.y - i
      );
      if (passedPosition.samePosition(desiredPosition)) {
        if (tileIsEmptyOrOccupiedByOpponent(passedPosition, boardState, team))
          return true;
      } else if (tileIsOccupied(passedPosition, boardState)) break;
    }

    // Top left
    if (
      desiredPosition.x < initialPosition.x &&
      desiredPosition.y > initialPosition.y
    ) {
      const passedPosition = new Position(
        initialPosition.x - i,
        initialPosition.y + i
      );
      if (passedPosition.samePosition(desiredPosition)) {
        if (tileIsEmptyOrOccupiedByOpponent(passedPosition, boardState, team))
          return true;
      } else if (tileIsOccupied(passedPosition, boardState)) break;
    }
  }
  return false;
};

export const getPossibleBishopMoves = (bishop, boardState) => {
  const possibleMoves = [];

  // Up right
  for (let i = 1; i < 8; i++) {
    const destination = new Position(
      bishop.position.x + i,
      bishop.position.y + i
    );
    if (!tileIsOccupied(destination, boardState)) {
      possibleMoves.push(destination);
    } else if (tileIsOccupiedByOpponent(destination, boardState, bishop.team)) {
      possibleMoves.push(destination);
      break;
    } else {
      break;
    }
  }

  // Bottom right
  for (let i = 1; i < 8; i++) {
    const destination = new Position(
      bishop.position.x + i,
      bishop.position.y - i
    );
    if (!tileIsOccupied(destination, boardState)) {
      possibleMoves.push(destination);
    } else if (tileIsOccupiedByOpponent(destination, boardState, bishop.team)) {
      possibleMoves.push(destination);
      break;
    } else {
      break;
    }
  }

  // Bottom left
  for (let i = 1; i < 8; i++) {
    const destination = new Position(
      bishop.position.x - i,
      bishop.position.y - i
    );
    if (!tileIsOccupied(destination, boardState)) {
      possibleMoves.push(destination);
    } else if (tileIsOccupiedByOpponent(destination, boardState, bishop.team)) {
      possibleMoves.push(destination);
      break;
    } else {
      break;
    }
  }

  // Top left
  for (let i = 1; i < 8; i++) {
    const destination = new Position(
      bishop.position.x - i,
      bishop.position.y + i
    );
    if (!tileIsOccupied(destination, boardState)) {
      possibleMoves.push(destination);
    } else if (tileIsOccupiedByOpponent(destination, boardState, bishop.team)) {
      possibleMoves.push(destination);
      break;
    } else {
      break;
    }
  }

  return possibleMoves;
};
