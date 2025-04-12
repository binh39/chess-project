import { useEffect, useRef, useState } from "react";
import { PieceType, TeamType } from "../../Types";
import Chessboard from "../Chessboard/Chessboard";
import { Howl } from "howler";
import { convertBackendPieces } from "../Referee/pieceConverter";

const moveSound = new Howl({ src: ["/sounds/move-self.mp3"] });
const captureSound = new Howl({ src: ["/sounds/capture.mp3"] });
const checkmateSound = new Howl({ src: ["/sounds/move-check.mp3"] });

export default function Referee() {
  const [boardState, setBoardState] = useState(null);
  const [promotionData, setPromotionData] = useState(null);
  const modalRef = useRef(null);
  const checkmateModalRef = useRef(null);

  useEffect(() => {
    fetch("http://localhost:8000/state")
      .then(res => res.json())
      .then(data => {
        const converted = convertBackendPieces(data.pieces);
        setBoardState({ ...data, pieces: converted });
      });
  }, []);

  useEffect(() => {
    if (boardState) {
      console.log(boardState.pieces);
    }
  }, [boardState]);

  async function playMove(playedPiece, destination) {
    const from = playedPiece.position;
    const to = destination;

    const from_square = String.fromCharCode(97 + from.x) + (from.y + 1);
    const to_square = String.fromCharCode(97 + to.x) + (to.y + 1);

    const promotionRank = playedPiece.team === TeamType.OUR ? 7 : 0;

    // Nếu là tốt và đến hàng phong hậu
    if (playedPiece.type === PieceType.PAWN && to.y === promotionRank) {
      setPromotionData({
        from: from_square,
        to: to_square,
        color: playedPiece.team === TeamType.OUR ? "w" : "b"
      });
      modalRef.current?.classList.remove("hidden");
      return; // chờ người dùng chọn quân phong hậu
    }

    // Di chuyển bình thường
    await sendMove(from_square, to_square);
  }

  // Hàm dùng chung để gửi move
  async function sendMove(from_square, to_square_with_promotion = null) {
    const body = {
      from_square,
      to_square: to_square_with_promotion ?? from_square + to_square_with_promotion
    };

    const res = await fetch("http://localhost:8000/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
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
      [PieceType.BISHOP]:"b",
      [PieceType.KNIGHT]: "n",
    }[pieceType];
  
    const move = {
      from_square: promotionData.from,
      to_square: promotionData.to + promotionLetter // ví dụ: e7e8Q
    };
    console.log(move);
    try {
      const res = await fetch("http://localhost:8000/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(move)
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
      const res = await fetch("http://localhost:8000/restart", {
        method: "POST"
      });
  
      const data = await res.json();
  
      if (data.pieces) {
        const converted = convertBackendPieces(data.pieces);
        setBoardState({ ...data, pieces: converted });
  
        // Ẩn modal checkmate nếu có
        checkmateModalRef.current?.classList.add("hidden");
      } else {
        console.error("Lỗi reset game: Không có dữ liệu bàn cờ");
      }
    } catch (err) {
      console.error("🔥 Lỗi gọi /restart:", err);
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

      {/* Modal checkmate */}
      <div className="modal hidden" ref={checkmateModalRef}>
        <div className="modal-body">
          <div className="checkmate-body">
            <span>
              The winning team is {boardState.turn === "white" ? "black" : "white"}!
            </span>
            <button onClick={restartGame}>Play again</button>
          </div>
        </div>
      </div>

      <Chessboard playMove={playMove} pieces={boardState.pieces} />
    </>
  );
}
