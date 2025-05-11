import { useEffect, useRef, useState } from "react";
import { PieceType, TeamType } from "../../Types";
import Chessboard from "../Chessboard/Chessboard";
import { Howl } from "howler";
import { convertBackendPieces } from "../Referee/pieceConverter";

const moveSound = new Howl({ src: ["/sounds/move-self.mp3"] });
const checkmateSound = new Howl({ src: ["/sounds/move-check.mp3"] });

export default function Referee({
  isPlayerVsBot,
  isBotVsBot,
  setThinkingTime,
  setCurrentTurn,
  currentTurn,
  isRestarting,
  restart,
  setRestart,
  isStart,
  setIsStart,
}) {
  const [lastMove, setLastMove] = useState(null);

  const [boardState, setBoardState] = useState(null);
  const [promotionData, setPromotionData] = useState(null);
  const modalRef = useRef(null);
  const checkmateModalRef = useRef(null); // chiếu hết cờ
  const drawModalRef = useRef(null); // Hòa cờ

  const controllerRef = useRef(null);

  useEffect(() => {
    const fetchBoardState = async () => {
      try {
        const res = await fetch("http://localhost:8000/state");
        const data = await res.json();
        const converted = convertBackendPieces(data.pieces);
        setBoardState({ ...data, pieces: converted });
      } catch (err) {
        console.error("Failed to fetch /state:", err);
      }
    };
    fetchBoardState();
  }, []);

  useEffect(() => {
    if (restart) {
      restartGame();
    }
    setRestart(false);
    setIsStart(false);
  }, [restart]);

  useEffect(() => {
    if (boardState && !isRestarting && isStart) {
      setCurrentTurn(boardState.turn);
      // Nếu là chế độ Bot VS Bot, bot tự động chơi cho cả hai bên
      if (isBotVsBot && !boardState.is_checkmate) {
        if (boardState.turn === "black") {
          playStockFishMove();
        } else {
          playBotMove();
        }
      }
      // Nếu là chế độ Player VS Bot và đến lượt bot (đen)
      else if (
        isPlayerVsBot &&
        boardState.turn === "black" &&
        !boardState.is_checkmate
      ) {
        playBotMove();
      }
    }
  }, [boardState, isPlayerVsBot, isBotVsBot, setCurrentTurn, isRestarting]);

  useEffect(() => {
    if (controllerRef.current && isPlayerVsBot && boardState.turn === "black") {
      controllerRef.current.abort(); // Hủy fetch cũ khi chuyển mode
    }
  }, [isPlayerVsBot]);

  useEffect(() => {
    setThinkingTime(0);
    let time = 0;
    const interval = setInterval(() => {
      time += 1;
      setThinkingTime(time);
    }, 1000);

    return () => clearInterval(interval);
  }, [currentTurn]);

  async function playStockFishMove() {
    // Khởi tạo controller mới mỗi lần gọi
    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      const res = await fetch("http://localhost:8000/stockfish_move", {
        method: "POST",
        signal: controller.signal, // dùng tín hiệu để có thể hủy fetch
      });

      const data = await res.json();
      if (data.success) {
        const convertedPieces = convertBackendPieces(data.state.pieces);
        setBoardState({ ...data.state, pieces: convertedPieces });
        moveSound.play();

        const fromSquare = data.move.substring(0, 2);
        const toSquare = data.move.substring(2, 4);
        setLastMove({ from: fromSquare, to: toSquare });

        if (data.state.is_checkmate) {
          checkmateModalRef.current?.classList.remove("hidden");
          checkmateSound.play();
        } else if (data.state.is_draw) {
          drawModalRef.current?.classList.remove("hidden");
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        console.log("Bot move bị huỷ do restart.");
      } else {
        console.error("Lỗi playBotMove:", err);
      }
    } finally {
    }
  }

  async function playBotMove() {
    // Khởi tạo controller mới mỗi lần gọi
    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      const res = await fetch("http://localhost:8000/bot_move", {
        method: "POST",
        signal: controller.signal, // dùng tín hiệu để có thể hủy fetch
      });

      const data = await res.json();
      if (data.success) {
        const convertedPieces = convertBackendPieces(data.state.pieces);
        setBoardState({ ...data.state, pieces: convertedPieces });
        moveSound.play();

        const fromSquare = data.move.substring(0, 2);
        const toSquare = data.move.substring(2, 4);
        setLastMove({ from: fromSquare, to: toSquare });

        if (data.state.is_checkmate) {
          checkmateModalRef.current?.classList.remove("hidden");
          checkmateSound.play();
        } else if (data.state.is_draw) {
          drawModalRef.current?.classList.remove("hidden");
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        console.log("Bot move bị huỷ do restart.");
      } else {
        console.error("Lỗi playBotMove:", err);
      }
    } finally {
    }
  }

  async function playMove(playedPiece, destination) {
    const from = playedPiece.position;
    const to = destination;

    const from_square = String.fromCharCode(97 + from.x) + (from.y + 1);
    const to_square = String.fromCharCode(97 + to.x) + (to.y + 1);

    const promotionRank = playedPiece.team === TeamType.OUR ? 7 : 0;

    // Nếu là người chơi (trắng) và cần phong hậu, hiển thị modal
    if (
      playedPiece.type === PieceType.PAWN &&
      to.y === promotionRank &&
      !isBotVsBot
    ) {
      setPromotionData({
        from: from_square,
        to: to_square,
        color: playedPiece.team === TeamType.OUR ? "w" : "b",
      });
      modalRef.current?.classList.remove("hidden");
      return;
    }

    const move = { from: from_square, to: to_square };
    setLastMove(move);
    // Di chuyển bình thường
    await sendMove(from_square, to_square);
  }

  // Hàm dùng chung để gửi move
  async function sendMove(from_square, to_square_with_promotion = null) {
    const body = {
      from_square,
      to_square:
        to_square_with_promotion ?? from_square + to_square_with_promotion,
    };

    const res = await fetch("http://localhost:8000/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await res.json();
    if (data.success) {
      const convertedPieces = convertBackendPieces(data.state.pieces);
      setBoardState({ ...data.state, pieces: convertedPieces });
      modalRef.current?.classList.add("hidden");
      moveSound.play();

      if (data.state.is_checkmate) {
        checkmateModalRef.current?.classList.remove("hidden");
        checkmateSound.play();
      } else if (data.state.is_draw) {
        drawModalRef.current?.classList.remove("hidden");
      }
    }
  }

  // Gửi phong hậu lên backend
  async function promotePawn(pieceType) {
    if (!promotionData) return;

    //Ưu tiên xác định dựa vào current turn
    const promotionLetter = {
      [PieceType.QUEEN]: "q",
      [PieceType.ROOK]: "r",
      [PieceType.BISHOP]: "b",
      [PieceType.KNIGHT]: "n",
    }[pieceType];

    const move = {
      from_square: promotionData.from,
      to_square: promotionData.to + promotionLetter, // ví dụ: e7e8Q
    };
    console.log(move);
    try {
      const res = await fetch("http://localhost:8000/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(move),
      });

      const data = await res.json();

      if (data.success) {
        const converted = convertBackendPieces(data.state.pieces);
        setBoardState({ ...data.state, pieces: converted });

        modalRef.current?.classList.add("hidden");
        moveSound.play();

        if (data.state.is_checkmate) {
          checkmateModalRef.current?.classList.remove("hidden");
          checkmateSound.play();
        } else if (data.state.is_draw) {
          drawModalRef.current?.classList.remove("hidden");
        }
      } else {
        console.error("Promotion failed:", data.message || data);
      }
    } catch (err) {
      console.error("Promotion error:", err);
    }
  }

  async function restartGame() {
    try {
      setLastMove(null);
      // Hủy bot đang chạy
      if (controllerRef.current) {
        controllerRef.current.abort(); // Hủy bot nếu đang chạy
      }
      const res = await fetch("http://localhost:8000/restart", {
        method: "POST",
      });

      const data = await res.json();

      if (data.pieces) {
        const converted = convertBackendPieces(data.pieces);
        setBoardState({ ...data, pieces: converted });

        // Ẩn modal checkmate nếu có
        checkmateModalRef.current?.classList.add("hidden");
        drawModalRef.current?.classList.add("hidden");
      } else {
        console.error("Lỗi reset game: Không có dữ liệu bàn cờ");
      }
    } catch (err) {
      console.error("Lỗi gọi /restart:", err);
    }
  }

  if (!boardState) return <div>Đang tải bàn cờ...</div>;

  return (
    <>
      {/* Modal phong hậu */}
      <div className="modal hidden" ref={modalRef}>
        <div className="modal-body">
          <img
            onClick={() => promotePawn(PieceType.ROOK)}
            src={`/assets/images/rook_${promotionData?.color}.png`}
          />
          <img
            onClick={() => promotePawn(PieceType.BISHOP)}
            src={`/assets/images/bishop_${promotionData?.color}.png`}
          />
          <img
            onClick={() => promotePawn(PieceType.KNIGHT)}
            src={`/assets/images/knight_${promotionData?.color}.png`}
          />
          <img
            onClick={() => promotePawn(PieceType.QUEEN)}
            src={`/assets/images/queen_${promotionData?.color}.png`}
          />
        </div>
      </div>

      {/* Modal endgame */}
      <div className="modal hidden" ref={checkmateModalRef}>
        <div className="modal-body">
          <div className="checkmate-body">
            <span>Tie!</span>
            <button onClick={restartGame}>Play again</button>
          </div>
        </div>
      </div>

      <div className="modal hidden" ref={drawModalRef}>
        <div className="modal-body">
          <div className="checkmate-body">
            <span>
              The winning team is{" "}
              {boardState.turn === "white" ? "black" : "white"}!
            </span>
            <button onClick={restartGame}>Play again</button>
          </div>
        </div>
      </div>

      <Chessboard
        playMove={playMove}
        pieces={boardState.pieces}
        isStart={isStart}
        lastMove={lastMove}
      />
    </>
  );
}
