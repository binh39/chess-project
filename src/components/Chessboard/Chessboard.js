import { useEffect, useRef, useState } from "react";
import React from "react";
import "./Chessboard.css";
import Tile from "../Tile/Tile";
import { VERTICAL_AXIS, HORIZONTAL_AXIS, GRID_SIZE } from "../../Constants";
import { Position } from "../../models";

export default function Chessboard({ playMove, pieces, isStart, lastMove }) {
  const [activePiece, setActivePiece] = useState(null);
  const [grabPosition, setGrabPosition] = useState(new Position(-1, -1));
  const chessboardRef = useRef(null);

  useEffect(() => {
    if (!isStart) {
      setActivePiece(null);
    }
  }, [isStart]);

  function grabPiece(e) {
    if (isStart) {
      const element = e.target;
      const chessboard = chessboardRef.current;
      if (
        element.classList.contains("chess-piece") &&
        chessboard &&
        activePiece == element
      ) {
        setActivePiece(null);
        return;
      } else if (
        element.classList.contains("chess-piece") &&
        chessboard &&
        activePiece != element
      ) {
        const clickX = e.clientX - chessboard.offsetLeft;
        const clickY = e.clientY - chessboard.offsetTop - 800;
        const grabX = Math.floor(clickX / GRID_SIZE);
        const grabY = Math.abs(Math.ceil(clickY / GRID_SIZE));
        setGrabPosition(new Position(grabX, grabY));
        setActivePiece(element);
        return;
      } else if (
        !element ||
        (!element.classList.contains("chess-piece") &&
          !element.classList.contains("tile-highlight"))
      ) {
        setActivePiece(null);
        return;
      }

      if (activePiece && chessboard) {
        const clickX = e.clientX - chessboard.offsetLeft;
        const clickY = e.clientY - chessboard.offsetTop - 800;
        const x = Math.floor(clickX / GRID_SIZE);
        const y = Math.abs(Math.ceil(clickY / GRID_SIZE));

        const currentPiece = pieces.find((p) => p.samePosition(grabPosition));
        if (currentPiece) {
          const succes = playMove(currentPiece.clone(), new Position(x, y));

          if (!succes) {
            activePiece.style.position = "relative";
            activePiece.style.removeProperty("top");
            activePiece.style.removeProperty("left");
          }
        }
        return;
      }
    } else {
      setActivePiece(null);
    }
  }

  const highlightLastMove = (i, j) => {
    if (lastMove) {
      const from = lastMove.from;
      const to = lastMove.to;
  
      // Chuyển đổi từ 'a1' thành (x, y)
      const fromX = from.charCodeAt(0) - 97; // 'a' = 97
      const fromY = parseInt(from[1]) - 1;
  
      // Chuyển đổi từ 'h8' thành (x, y)
      const toX = to.charCodeAt(0) - 97;
      const toY = parseInt(to[1]) - 1;
  
      // Kiểm tra xem (i, j) có phải là ô 'from' hoặc 'to' không
      return (i === fromX && j === fromY) || (i === toX && j === toY);
    }
    return false;
  };
  

  const board = [];

  for (let j = VERTICAL_AXIS.length - 1; j >= 0; j--) {
    for (let i = 0; i < HORIZONTAL_AXIS.length; i++) {
      const number = j + i + 2;
      const piece = pieces.find((p) => p.samePosition(new Position(i, j)));
      const image = piece ? piece.image : undefined;

      const currentPiece =
        activePiece !== null
          ? pieces.find((p) => p.samePosition(grabPosition))
          : undefined;

      const highlight = currentPiece?.possibleMoves
        ? currentPiece.possibleMoves.some((p) =>
            p.samePosition(new Position(i, j))
          )
        : false;

        board.push(
          <Tile
            key={`${j},${i}`}
            image={image}
            number={number}
            highlight={highlight} 
            highlightLastMove={highlightLastMove(i, j)}
          />
        );
    }
  }

  return React.createElement(
    "div",
    {
      id: "chessboard",
      ref: chessboardRef,
      onClick: grabPiece,
    },
    board
  );
}
