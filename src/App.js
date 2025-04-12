import "./App.css";
import Referee from "./components/Referee/Referee";
import { useState } from "react";

function App() {
  const [isPlayerVsBot, setIsPlayerVsBot] = useState(false);
  const [isBotVsBot, setIsBotVsBot] = useState(false);
  const [thinkingTime, setThinkingTime] = useState(null);
  const [currentTurn, setCurrentTurn] = useState(null);
  const [isStart, setIsStart] = useState(false);
  const [isRestarting, setIsRestarting] = useState(false);
  const [resetTrigger, setResetTrigger] = useState(0);

  const startPlayerVsBot = () => {
    setIsPlayerVsBot(true);
    setIsBotVsBot(false);
    setThinkingTime(null);
    setIsStart(true);
    console.log("Player VS Bot True");
  };

  const startBotVsBot = () => {
    setIsBotVsBot(true);
    setIsPlayerVsBot(false);
    setThinkingTime(null);
    setIsStart(true);
    console.log("Bot VS Bot True");
  };

  const restartGame = async () => {
    setIsRestarting(true);
    try {
      const res = await fetch("http://localhost:8000/restart", {
        method: "POST",
      });
      const data = await res.json();
      if (data.pieces) {
        setIsPlayerVsBot(false);
        setIsBotVsBot(false);
        setThinkingTime(null);
        setCurrentTurn(null);
        setIsStart(false);
        setIsRestarting(false);
        setResetTrigger((prev) => prev + 1);
      } else {
        console.error("Lỗi reset game: Không có dữ liệu bàn cờ");
      }
    } catch (err) {
      console.error("🔥 Lỗi gọi /restart:", err);
      setIsRestarting(false);
    }
  };
  return (
    <div id="mid">
      <div id="app">
        <Referee
          isPlayerVsBot={isPlayerVsBot}
          isBotVsBot={isBotVsBot}
          setThinkingTime={setThinkingTime}
          setCurrentTurn={setCurrentTurn}
          isRestarting={isRestarting}
          resetTrigger={resetTrigger}
        />
      </div>
      <div id="menu">
        <div className="Status">
          {isStart
            ? currentTurn === "white"
              ? "White thinking..."
              : "Black thinking..."
            : "Welcome"}
          {thinkingTime !== null && (isBotVsBot || isPlayerVsBot)
            ? thinkingTime
            : null}
        </div>
        <button className="buttonBase buttonPlay" onClick={startBotVsBot}>
          Bot VS Bot
        </button>
        <button className="buttonBase buttonPlay" onClick={startPlayerVsBot}>
          Player VS Bot
        </button>
        <button className="buttonBase buttonRestart" onClick={restartGame}>
          Restart
        </button>
      </div>
    </div>
  );
}

export default App;
