import "./Tile.css";

export default function Tile({ number, image, highlight, highlightLastMove }) {
  const className = [
    "tile",
    number % 2 === 0 && "black-tile",
    number % 2 !== 0 && "white-tile",
    highlight && "tile-highlight",
    highlightLastMove && "tile-highlight-last-move",
    image && "chess-piece-tile"
  ].filter(Boolean).join(" ");

  return (
    <div className={className}>
      {image && (
        <div
          style={{ backgroundImage: `url(${image})` }}
          className="chess-piece"
        ></div>
      )}
    </div>
  );
}
