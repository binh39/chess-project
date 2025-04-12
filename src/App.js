import "./App.css";
import Referee from "./components/Referee/Referee";

function App() {
  return (
    <div id="mid">
      <div id="app">
        <Referee />
      </div>
      <div id="menu">
        <button className="buttonPlay">Bot VS Bot</button>
        <button className="buttonPlay">Player VS Bot</button>
        <button className="buttonRestart">Restart</button>
      </div>
    </div>
  );
}

export default App;
