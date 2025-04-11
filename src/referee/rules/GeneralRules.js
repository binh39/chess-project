import { Piece, Position } from "../../models/index.js";
import { TeamType } from "../../Types.js";

export const tileIsOccupied = (position, boardState) => {
  const piece = boardState.find((p) => p.samePosition(position));
  return !!piece;
};

export const tileIsOccupiedByOpponent = (position, boardState, team) => {
  const piece = boardState.find(
    (p) => p.samePosition(position) && p.team !== team
  );
  return !!piece;
};

export const tileIsEmptyOrOccupiedByOpponent = (position, boardState, team) => {
  return (
    !tileIsOccupied(position, boardState) ||
    tileIsOccupiedByOpponent(position, boardState, team)
  );
};
