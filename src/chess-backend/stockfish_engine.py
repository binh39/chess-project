import subprocess
import time
import chess
import chess.engine  # Import for Limit compatibility


class StockfishEngine:
    def __init__(self, path_to_engine):
        """
        Initialize the Stockfish engine using direct subprocess management
        instead of the python-chess built-in functionality.
        
        Args:
            path_to_engine: Path to the Stockfish executable
        """
        self.path = path_to_engine
        self.process = None
        self.config = {}
        
    def start(self):
        """Start the Stockfish engine process."""
        # Using subprocess.Popen directly instead of asyncio
        self.process = subprocess.Popen(
            self.path,
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Initialize UCI mode
        self._send_command("uci")
        self._wait_for_response("uciok")
        self._send_command("isready")
        self._wait_for_response("readyok")
        
        # Apply any stored configuration
        self._apply_config()
    
    def _send_command(self, command):
        """Send a command to the Stockfish engine."""
        if self.process is None:
            raise Exception("Stockfish process not started")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
    
    def _wait_for_response(self, wait_for=None, timeout=5.0):
        """
        Wait for a specific response from the engine.
        
        Args:
            wait_for: String to wait for in the engine output
            timeout: Maximum time to wait in seconds
        
        Returns:
            List of response lines
        """
        if self.process is None:
            raise Exception("Stockfish process not started")
            
        start_time = time.time()
        responses = []
        
        while time.time() - start_time < timeout:
            line = self.process.stdout.readline().strip()
            if line:
                responses.append(line)
                if wait_for and wait_for in line:
                    break
                
        return responses
    
    def configure(self, options):
        """
        Configure the engine with the given options.
        
        Args:
            options: Dictionary of option name to value
        """
        # Store options for later use if engine restarts
        self.config.update(options)
        
        # If engine is already running, apply the options now
        if self.process is not None:
            self._apply_config()
    
    def _apply_config(self):
        """Apply the stored configuration to the engine."""
        for name, value in self.config.items():
            if isinstance(value, bool):
                value = "true" if value else "false"
            self._send_command(f"setoption name {name} value {value}")
        
        # Wait for engine to process options
        self._send_command("isready")
        self._wait_for_response("readyok")
    
    def set_skill_level(self, skill_level):
        """
        Set the skill level of the engine (0-20).
        
        Args:
            skill_level: Skill level between 0 (weakest) and 20 (strongest)
        """
        skill_level = max(0, min(20, skill_level))
        self.configure({
            "Skill Level": skill_level,
            # These are commonly modified with skill level
            "UCI_LimitStrength": True,
            "UCI_Elo": 1500 + (100 * skill_level)  # Scale ELO with skill level
        })
    
    def set_fen_position(self, fen):
        """
        Set the board position using FEN notation.
        
        Args:
            fen: FEN string representing the board position
        """
        self._send_command(f"position fen {fen}")
    
    def _extract_limit_values(self, limit):
        """
        Extract limit values from either a chess.engine.Limit object or a dictionary.
        
        Args:
            limit: Either a chess.engine.Limit object or a dictionary with limit values
            
        Returns:
            Dictionary with time and/or depth values
        """
        result = {}
        
        # Handle chess.engine.Limit objects
        if isinstance(limit, chess.engine.Limit):
            if limit.time is not None:
                result['time'] = limit.time
            if limit.depth is not None:
                result['depth'] = limit.depth
            if not result and limit.nodes is not None:
                # Convert nodes limit to approximate time limit
                result['time'] = limit.nodes / 1000000  # Rough approximation
        # Handle dictionary-like objects
        elif hasattr(limit, 'get') or hasattr(limit, '__getitem__'):
            if 'time' in limit:
                result['time'] = limit['time']
            if 'depth' in limit:
                result['depth'] = limit['depth']
        
        # Default time limit if nothing specified
        if not result:
            result['time'] = 1.0
            
        return result
    
    def get_best_move(self, board=None, time_limit=1.0, depth=None):
        """
        Get the best move for the given board position.
        
        Args:
            board: A chess.Board instance (optional if set_fen_position was called)
            time_limit: Time to think in seconds (ignored if depth is provided)
            depth: Search depth (optional)
        
        Returns:
            The best move as a chess.Move object, or None if no move found
        """
        # Set up the position if a board is provided
        if board is not None:
            self._send_command("position fen " + board.fen())
        
        # Tell engine to find the best move with a specific time limit or depth
        if depth is not None:
            self._send_command(f"go depth {depth}")
        else:
            self._send_command(f"go movetime {int(time_limit * 1000)}")
        
        # Parse the output to find the best move
        best_move = None
        timeout = (depth * 2.0) if depth is not None else (time_limit + 1.0)
        for line in self._wait_for_response("bestmove", timeout=timeout):
            if line.startswith("bestmove"):
                move_str = line.split()[1]
                if move_str == "(none)":
                    return None
                try:
                    best_move = chess.Move.from_uci(move_str)
                    break
                except ValueError:
                    continue
        
        return best_move
    
    def analyse(self, board, limit):
        """
        Analyze the position with the given time or depth limit.
        Similar to the chess.engine.SimpleEngine.analyse method.
        
        Args:
            board: A chess.Board instance
            limit: A chess.engine.Limit object or dictionary with 'time' or 'depth' keys
        
        Returns:
            Dictionary with analysis information
        """
        # Extract limit values
        limit_values = self._extract_limit_values(limit)
        
        # Set up the position
        self._send_command("position fen " + board.fen())
        
        # Configure the analysis
        if 'depth' in limit_values:
            self._send_command(f"go depth {limit_values['depth']}")
            timeout = limit_values['depth'] * 2.0  # Allow more time for deep searches
        else:
            time_ms = int(limit_values.get('time', 1.0) * 1000)
            self._send_command(f"go movetime {time_ms}")
            timeout = limit_values.get('time', 1.0) + 1.0
        
        # Collect analysis data
        analysis = {
            'score': None,
            'pv': [],
            'depth': 0,
            'nodes': 0,
            'time': 0,
            'multipv': []
        }
        
        for line in self._wait_for_response("bestmove", timeout=timeout):
            # Parse info lines for scores, PV, etc.
            if line.startswith("info") and "score" in line:
                parts = line.split()
                
                # Get depth
                if "depth" in line:
                    depth_idx = parts.index("depth")
                    if depth_idx + 1 < len(parts):
                        analysis['depth'] = int(parts[depth_idx + 1])
                
                # Get score
                if "score" in line:
                    score_idx = parts.index("score")
                    if score_idx + 2 < len(parts):
                        score_type = parts[score_idx + 1]
                        score_value = parts[score_idx + 2]
                        
                        if score_type == "cp":
                            analysis['score'] = int(score_value) / 100.0
                        elif score_type == "mate":
                            analysis['score'] = f"mate {score_value}"
                
                # Get PV (principal variation)
                if "pv" in line:
                    pv_idx = parts.index("pv")
                    if pv_idx + 1 < len(parts):
                        pv_moves = []
                        for i in range(pv_idx + 1, len(parts)):
                            try:
                                move = chess.Move.from_uci(parts[i])
                                pv_moves.append(move)
                            except ValueError:
                                pass
                        analysis['pv'] = pv_moves
        
        return analysis
    
    def play(self, board, limit):
        """
        Play a move in the given position with the specified time limit.
        Compatible with chess.engine.SimpleEngine.play method.
        
        Args:
            board: A chess.Board instance
            limit: A chess.engine.Limit object or dictionary with 'time' or 'depth' keys
        
        Returns:
            An object with a 'move' attribute containing the chosen move
        """
        # Extract limit values
        limit_values = self._extract_limit_values(limit)
        
        # Get the best move
        if 'depth' in limit_values:
            move = self.get_best_move(board, depth=limit_values['depth'])
        else:
            move = self.get_best_move(board, time_limit=limit_values.get('time', 1.0))
        
        # Create a result object compatible with chess.engine.PlayResult
        class Result:
            def __init__(self, move):
                self.move = move
                self.ponder = None  # We don't implement pondering
                self.info = {}
        
        return Result(move)
    
    def quit(self):
        """Quit the Stockfish engine."""
        if self.process is not None:
            self._send_command("quit")
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            finally:
                self.process = None
                
    # Add context manager support for 'with' statement
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.quit()