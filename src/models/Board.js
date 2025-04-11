import {
    getPossibleBishopMoves,
    getPossibleKingMoves,
    getPossibleKnightMoves,
    getPossiblePawnMoves,
    getPossibleQueenMoves,
    getPossibleRookMoves,
    getCastlingMoves,
  } from "../referee/rules";
  import { PieceType, TeamType } from "../Types";
  import { Pawn } from "./Pawn";
  import { Piece } from "./Piece";
  import { Position } from "./Position";
  
  export class Board {
    constructor(pieces, totalTurns) {
      this.pieces = pieces;
      this.totalTurns = totalTurns;
      this.winningTeam = undefined;
    }
  
    get currentTeam() {
      return this.totalTurns % 2 === 0 ? TeamType.OPPONENT : TeamType.OUR;
    }
  
    calculateAllMoves() {
      for (const piece of this.pieces) {
        piece.possibleMoves = this.getValidMoves(piece, this.pieces);
      }
  
      for (const king of this.pieces.filter((p) => p.isKing)) {
        if (king.possibleMoves === undefined) continue;
  
        king.possibleMoves = [
          ...king.possibleMoves,
          ...getCastlingMoves(king, this.pieces),
        ];
      }
  
      this.checkCurrentTeamMoves();
  
      for (const piece of this.pieces.filter(
        (p) => p.team !== this.currentTeam
      )) {
        piece.possibleMoves = [];
      }
  
      const hasMovesLeft = this.pieces
        .filter((p) => p.team === this.currentTeam)
        .some((p) => p.possibleMoves && p.possibleMoves.length > 0);
  
      if (!hasMovesLeft) {
        this.winningTeam =
          this.currentTeam === TeamType.OUR ? TeamType.OPPONENT : TeamType.OUR;
      }
    }
  
    checkCurrentTeamMoves() {
      for (const piece of this.pieces.filter(
        (p) => p.team === this.currentTeam
      )) {
        if (!piece.possibleMoves) continue;
  
        for (const move of piece.possibleMoves) {
          const simulatedBoard = this.clone();
  
          simulatedBoard.pieces = simulatedBoard.pieces.filter(
            (p) => !p.samePosition(move)
          );
  
          const clonedPiece = simulatedBoard.pieces.find((p) =>
            p.samePiecePosition(piece)
          );
          if (!clonedPiece) continue;
  
          clonedPiece.position = move.clone();
  
          const clonedKing = simulatedBoard.pieces.find(
            (p) => p.isKing && p.team === simulatedBoard.currentTeam
          );
          if (!clonedKing) continue;
  
          for (const enemy of simulatedBoard.pieces.filter(
            (p) => p.team !== simulatedBoard.currentTeam
          )) {
            enemy.possibleMoves = simulatedBoard.getValidMoves(
              enemy,
              simulatedBoard.pieces
            );
  
            const kingThreat = enemy.possibleMoves.some((m) => {
              if (enemy.isPawn) {
                return m.x !== enemy.position.x &&
                  m.samePosition(clonedKing.position);
              } else {
                return m.samePosition(clonedKing.position);
              }
            });
  
            if (kingThreat) {
              piece.possibleMoves = piece.possibleMoves.filter(
                (m) => !m.samePosition(move)
              );
            }
          }
        }
      }
    }
  
    getValidMoves(piece, boardState) {
      switch (piece.type) {
        case PieceType.PAWN:
          return getPossiblePawnMoves(piece, boardState);
        case PieceType.KNIGHT:
          return getPossibleKnightMoves(piece, boardState);
        case PieceType.BISHOP:
          return getPossibleBishopMoves(piece, boardState);
        case PieceType.ROOK:
          return getPossibleRookMoves(piece, boardState);
        case PieceType.QUEEN:
          return getPossibleQueenMoves(piece, boardState);
        case PieceType.KING:
          return getPossibleKingMoves(piece, boardState);
        default:
          return [];
      }
    }
  
    playMove(enPassantMove, validMove, playedPiece, destination) {
      const pawnDirection = playedPiece.team === TeamType.OUR ? 1 : -1;
      const destinationPiece = this.pieces.find((p) =>
        p.samePosition(destination)
      );
  
      if (
        playedPiece.isKing &&
        destinationPiece?.isRook &&
        destinationPiece.team === playedPiece.team
      ) {
        const direction =
          destinationPiece.position.x - playedPiece.position.x > 0 ? 1 : -1;
        const newKingX = playedPiece.position.x + direction * 2;
  
        this.pieces = this.pieces.map((p) => {
          if (p.samePiecePosition(playedPiece)) {
            p.position.x = newKingX;
          } else if (p.samePiecePosition(destinationPiece)) {
            p.position.x = newKingX - direction;
          }
          return p;
        });
  
        this.calculateAllMoves();
        return true;
      }
  
      if (enPassantMove) {
        this.pieces = this.pieces.reduce((results, piece) => {
          if (piece.samePiecePosition(playedPiece)) {
            if (piece.isPawn) piece.enPassant = false;
            piece.position = destination.clone();
            piece.hasMoved = true;
            results.push(piece);
          } else if (
            !piece.samePosition(
              new Position(destination.x, destination.y - pawnDirection)
            )
          ) {
            if (piece.isPawn) piece.enPassant = false;
            results.push(piece);
          }
          return results;
        }, []);
        this.calculateAllMoves();
      } else if (validMove) {
        this.pieces = this.pieces.reduce((results, piece) => {
          if (piece.samePiecePosition(playedPiece)) {
            if (piece.isPawn) {
              piece.enPassant =
                Math.abs(playedPiece.position.y - destination.y) === 2;
            }
            piece.position = destination.clone();
            piece.hasMoved = true;
            results.push(piece);
          } else if (!piece.samePosition(destination)) {
            if (piece.isPawn) piece.enPassant = false;
            results.push(piece);
          }
          return results;
        }, []);
        this.calculateAllMoves();
      } else {
        return false;
      }
  
      return true;
    }
  
    clone() {
      return new Board(
        this.pieces.map((p) => p.clone()),
        this.totalTurns
      );
    }
  }
  