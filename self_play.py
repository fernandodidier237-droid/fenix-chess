"""
Self-play game generation for AlphaZero.

Each self-play game is played entirely by MCTS guided by the current neural
network. Training examples (state, policy_target, result) are collected from
every move of every game.
"""

import logging
import numpy as np
import chess
from typing import List, Tuple
from tqdm import tqdm

import config
import chess_env
from mcts import MCTS

logger = logging.getLogger(__name__)

# Type alias for a training example
TrainingExample = Tuple[np.ndarray, np.ndarray, float]
# (board_tensor_8x8x119, policy_target_4672, game_outcome_from_perspective)


class SelfPlayWorker:
    """
    Generates self-play games using MCTS guided by the neural network.

    Each game:
      1. MCTS is run from the current position (with Dirichlet noise at the root).
      2. A move is sampled from the MCTS visit-distribution (temperature τ).
      3. The training example (board → policy_target) is stored.
      4. After the game ends, each example receives the game result as value target.
    """

    def __init__(
        self,
        model: "NeuralNetwork",
        mcts_sims: int = config.NUM_MCTS_SIMS,
        c_puct: float = config.C_PUCT,
        temp_threshold: int = config.TEMPERATURE_THRESHOLD,
        device: str = None,
    ):
        self.model = model
        self.mcts_sims = mcts_sims
        self.c_puct = c_puct
        self.temp_threshold = temp_threshold
        self.device = device or config.INFERENCE_DEVICE

    def play_game(self, temperature: float = config.TEMPERATURE) -> List[TrainingExample]:
        """
        Play a single self-play game.

        Returns:
            List of (board_tensor, policy_target, result) training examples.
            result is from the perspective of the *player who made the move*.
        """
        board = chess.Board()
        examples: List[TrainingExample] = []

        # We'll re-use the MCTS instance; instantiate per game to avoid stale caches
        mcts = MCTS(
            model=self.model,
            num_simulations=self.mcts_sims,
            c_puct=self.c_puct,
            temperature=temperature,
            temperature_threshold=self.temp_threshold,
            device=self.device,
        )

        move_count = 0
        while not board.is_game_over():
            move_count += 1

            # ── MCTS search ──────────────────────────────────────────
            root = mcts.search(board, add_root_noise=True)

            # ── Get move probabilities from visit counts ─────────────
            temp = mcts.compute_temperature(move_count)
            selected_move, visit_probs = root.get_visit_probs(temperature=temp)

            # ── Build training example ───────────────────────────────
            # Encode board in canonical form (always from white's persp.)
            board_tensor = chess_env.encode_board_canonical(board)

            # Build full 4672-dimensional policy target
            # We place the visit probabilities at the indices corresponding
            # to the legal moves in the order they appear in root.children
            policy_target = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)
            for i, child in enumerate(root.children):
                move_idx = chess_env.move_to_index(child.action)
                policy_target[move_idx] = visit_probs[i]

            examples.append((board_tensor, policy_target, 0.0))  # result filled later

            # ── Make the move ────────────────────────────────────────
            board.push(selected_move)

        # ── Determine game outcome ──────────────────────────────────────
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            result = 0.0  # Draw
        elif outcome.winner == chess.WHITE:
            result = 1.0
        else:
            result = -1.0

        # Assign result to each training example from the player's perspective
        # The player who moved is the one who had the turn when the board was
        # recorded. Since we stored in canonical form (white's perspective),
        # we need to track which side made each move.
        # Simpler approach: replay the game and track perspective.
        # But we already know: canonical encoding is from white's perspective,
        # so the result is from white's perspective.
        # However, each training example should be from the *current player's*
        # perspective at the time of the move. Since encode_board_canonical
        # always flips to white, the perspective is always white.
        # So for moves by white: result = result, for moves by black: result = -result.
        # But wait — in canonical form, we always flip. So the stored tensor
        # is always from white's perspective. The result should then be from
        # the perspective of the player who moved.
        #
        # Let's re-think: encode_board_canonical always returns a tensor
        # from white's perspective (white pieces = current player). So when
        # black is to move, we flip the board. This means the training example
        # is always from the "current player's" perspective, which is white
        # after canonicalization. So the result should be from white's perspective.
        #
        # But the player who made the move might not be white! When black moves,
        # we flip the board so it looks like white's position. The result should
        # then be negated for black's moves.
        #
        # Actually no — the standard practice in AlphaZero is to store each
        # example from the perspective of the player who made the move.
        # Since encode_board_canonical always shows the board from white's
        # perspective, for a move made by black, we flip the board, so the
        # stored tensor is from white's perspective but the player was black.
        # The target value should be: if black won → +1 (from black's perspective),
        # but the board shows white's view, so we need to negate.
        #
        # This is getting confused. Let me re-think.
        #
        # Standard approach from the AlphaZero paper:
        # - Each training example: (state, π, z) where z ∈ {-1, 0, +1} is the
        #   game outcome from the perspective of the player who made the move.
        # - The state is always from the current player's perspective.
        # - Since we flip the board to canonical form, the "current player"
        #   in the stored tensor is always the player who is about to move.
        #
        # So:
        # - If white is to move and white wins → z = +1
        # - If black is to move and black wins → z = +1
        # - If white is to move and black wins → z = -1
        #
        # This means: z = result * (1 if the player who moved is white else -1)
        # But when using encode_board_canonical, we always flip so the tensor is
        # from the current player's perspective. So z is always just the result
        # from the current player's perspective at move time.
        #
        # The simplest correct approach: store the board in the original
        # orientation (without canonical flip), and store the result from
        # the perspective of the current player. During training, we can
        # flip randomly or train directly.
        #
        # Let me use a simpler approach:
        # - Store examples WITHOUT canonical flip
        # - Each example stores the result from the current player's perspective

        # Re-examine: we used encode_board_canonical which flips to white's
        # perspective. So for black's moves, the board is flipped. The result
        # should be from the perspective of the player who moved.
        #
        # White's perspective result: +1 if white wins
        # For a move by white: z = result_from_white_perspective
        # For a move by black: z = -result_from_white_perspective
        # (because black's perspective is opposite white's)

        # Let's re-write the examples with correct results.
        # Actually, the simplest fix: don't use canonical form. Instead store
        # the raw board with result from the mover's perspective.

        # Let me re-do this properly:

        board = chess.Board()
        examples = []
        move_count = 0

        while not board.is_game_over():
            move_count += 1
            root = mcts.search(board, add_root_noise=True)
            temp = mcts.compute_temperature(move_count)
            selected_move, visit_probs = root.get_visit_probs(temperature=temp)

            # Encode WITHOUT canonical flip — we'll handle perspective during value assignment
            board_tensor = chess_env.encode_board(board)  # current player's perspective

            policy_target = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)
            for i, child in enumerate(root.children):
                move_idx = chess_env.move_to_index(child.action)
                policy_target[move_idx] = visit_probs[i]

            # Store perspective: True if white to move
            white_to_move = board.turn == chess.WHITE
            examples.append((board_tensor, policy_target, 0.0, white_to_move))

            board.push(selected_move)

        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            result = 0.0
        elif outcome.winner == chess.WHITE:
            result = 1.0
        else:
            result = -1.0

        # Assign result from the mover's perspective
        final_examples = []
        for tensor, policy, _, white_moved in examples:
            # If white moved, result is from white's perspective → z = result
            # If black moved, result is from black's perspective → z = -result
            z = result if white_moved else -result
            final_examples.append((tensor, policy, z))

        return final_examples

    def play_batch(
        self, num_games: int, desc: str = "Self-play"
    ) -> List[TrainingExample]:
        """
        Play num_games self-play games and return all training examples.

        Returns:
            Flattened list of (board_tensor, policy_target, result).
        """
        all_examples: List[TrainingExample] = []
        for _ in tqdm(range(num_games), desc=desc):
            try:
                game_examples = self.play_game()
                all_examples.extend(game_examples)
            except Exception as e:
                logger.error("Self-play game failed: %s", e)
                continue
        logger.info(
            "Generated %d training examples from %d self-play games.",
            len(all_examples), num_games,
        )
        return all_examples
