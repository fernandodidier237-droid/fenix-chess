"""
Stockfish UCI integration for AlphaZero Chess.

Provides:
  - Automated discovery of Stockfish binary
  - Configurable ELO and search limits
  - Head-to-head evaluation vs the neural network
"""

import subprocess
import shutil
import time
import logging
from typing import Optional
import chess
import chess.engine

import config

logger = logging.getLogger(__name__)


def find_stockfish() -> Optional[str]:
    """
    Locate the Stockfish binary on the system.
    Tries common installation paths.
    """
    # Common paths on Linux / macOS / Windows (WSL)
    candidates = [
        config.STOCKFISH_PATH,
        "/usr/bin/stockfish",
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
        "stockfish",
        "/snap/bin/stockfish",
    ]
    # Check if user provided a path in config
    if config.STOCKFISH_PATH and shutil.which(config.STOCKFISH_PATH):
        return shutil.which(config.STOCKFISH_PATH)

    for path in candidates:
        found = shutil.which(path)
        if found:
            logger.info("Found Stockfish at %s", found)
            return found

    # Last resort: try on PATH
    found = shutil.which("stockfish")
    if found:
        return found

    logger.warning("Stockfish binary not found. Please install Stockfish 18.")
    return None


class StockfishEvaluator:
    """
    Wrapper around Stockfish for evaluation games against the neural network.

    Usage:
        sf = StockfishEvaluator(elo=2800)
        result = sf.play_game(model, board, time_limit=0.1)
    """

    def __init__(
        self,
        engine_path: Optional[str] = None,
        elo: int = config.STOCKFISH_ELO,
        threads: int = config.STOCKFISH_THREADS,
        hash_size_mb: int = config.STOCKFISH_HASH_SIZE_MB,
        time_per_move_ms: int = 100,
    ):
        self.engine_path = engine_path or find_stockfish()
        if self.engine_path is None:
            raise RuntimeError(
                "Stockfish binary not found. Please install Stockfish and "
                "set config.STOCKFISH_PATH."
            )
        self.elo = elo
        self.threads = threads
        self.hash_size_mb = hash_size_mb
        self.time_per_move_ms = time_per_move_ms
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def _resolve_elo(self, desired_elo: int) -> int:
        """Clamp ELO to a supported range for the UCI_Elo option."""
        return max(1350, min(3190, desired_elo))

    def start(self):
        """Open the Stockfish engine process and configure it."""
        if self._engine is not None:
            return
        self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        # Configure
        resolved_elo = self._resolve_elo(self.elo)
        self._engine.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": resolved_elo,
            "Threads": self.threads,
            "Hash": self.hash_size_mb,
        })
        logger.info(
            "Stockfish started — ELO=%d, threads=%d, hash=%dMB",
            resolved_elo, self.threads, self.hash_size_mb,
        )

    def stop(self):
        """Terminate the Stockfish engine process."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None
            logger.info("Stockfish stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def get_best_move(
        self, board: chess.Board, time_ms: Optional[int] = None
    ) -> chess.Move:
        """
        Query Stockfish for the best move in a position.

        Args:
            board: current position.
            time_ms: time limit for the search (ms). Default: self.time_per_move_ms.

        Returns:
            Best move according to Stockfish.
        """
        if self._engine is None:
            self.start()
        t_ms = time_ms or self.time_per_move_ms
        result = self._engine.play(board, chess.engine.Limit(time=t_ms / 1000.0))
        return result.move

    def evaluate_position(self, board: chess.Board, time_ms: int = 500) -> float:
        """
        Get Stockfish's evaluation (in centipawns) for the current position.
        Positive = favourable for the side to move.
        """
        if self._engine is None:
            self.start()
        info = self._engine.analyse(
            board, chess.engine.Limit(time=time_ms / 1000.0),
        )
        score = info.get("score")
        if score is None:
            return 0.0
        # Convert to centipawns from the perspective of the side to move
        if score.is_mate():
            mate_in = score.mate()
            if mate_in is None:
                return 0.0
            # Mate in N → very high score
            sign = 1 if mate_in > 0 else -1
            return sign * 10000 / abs(mate_in)
        return score.relative.score() or 0.0


    def play_game(
        self,
        model: "NeuralNetwork",
        board: chess.Board,
        time_per_move_ms: int = 100,
        mcts_sims: int = 400,
        sf_plays_white: bool = False,
        temperature: float = 0.0,
    ) -> int:
        """
        Play a full game between the neural-network (using MCTS) and Stockfish.

        Args:
            model: the neural network.
            board: starting position.
            time_per_move_ms: Stockfish's time per move.
            mcts_sims: MCTS simulations per move for the neural network.
            sf_plays_white: if True, Stockfish plays white.
            temperature: MCTS temperature (0 = deterministic).

        Returns:
            Result from white's perspective:
              +1 = white wins, -1 = black wins, 0 = draw.
        """
        from mcts import MCTS

        mcts = MCTS(
            model=model,
            num_simulations=mcts_sims,
            c_puct=config.C_PUCT,
            temperature=max(temperature, config.TEMPERATURE_TAU),
        )

        self.start()
        move_count = 0
        while not board.is_game_over():
            move_count += 1
            nn_turn = (board.turn == chess.WHITE and not sf_plays_white) or \
                       (board.turn == chess.BLACK and sf_plays_white)

            if nn_turn:
                # Neural network move via MCTS
                root = mcts.search(board, add_root_noise=False)
                temp = mcts.compute_temperature(move_count)
                best_move, _ = root.get_visit_probs(temperature=temp)
                board.push(best_move)
            else:
                # Stockfish move
                sf_move = self.get_best_move(board, time_ms=time_per_move_ms)
                board.push(sf_move)

        outcome = board.outcome()
        if outcome is None:
            return 0
        if outcome.winner is None:
            return 0
        return 1.0 if outcome.winner == chess.WHITE else -1.0


# ── API compatibility alias ────────────────────────────────────────────
# The user's example code uses `from stockfish_integration import Stockfish`
Stockfish = StockfishEvaluator
