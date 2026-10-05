"""
Monte Carlo Tree Search (MCTS) for AlphaZero Chess.

Implements the search described in the AlphaZero paper:
  - Node selection via PUCT
  - Leaf evaluation by neural network
  - Dirichlet noise at root for exploration
  - Temperature-based move selection from visit counts
"""

import math
import numpy as np
import chess
from typing import List, Optional, Tuple

import config
import chess_env

# torch is imported lazily inside methods that need it (search, _get_priors)


class MCTSNode:
    """A single node in the MCTS tree."""

    __slots__ = (
        "state_fen", "parent", "prior", "action", "action_index",
        "children", "visit_count", "total_value", "is_expanded",
        "is_terminal", "reward",
    )

    def __init__(self, state_fen: str, parent: Optional["MCTSNode"] = None,
                 prior: float = 0.0, action: chess.Move = None,
                 action_index: int = -1):
        self.state_fen = state_fen
        self.parent = parent
        self.prior = prior
        self.action = action
        self.action_index = action_index  # index in parent's action list
        self.children: List["MCTSNode"] = []
        self.visit_count = 0
        self.total_value = 0.0
        self.is_expanded = False
        self.is_terminal = False
        self.reward = 0.0

    @property
    def value(self) -> float:
        """Mean value estimate."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        """PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))."""
        if self.visit_count == 0:
            return float("inf")
        exploration = (
            c_puct
            * self.prior
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        return self.value + exploration

    def best_child(self, c_puct: float) -> "MCTSNode":
        """Select the child with the highest PUCT score."""
        return max(self.children, key=lambda c: c.ucb_score(c_puct))

    def expand(self, actions: List[chess.Move], priors: List[float],
               fen: str = ""):
        """Expand leaf node with children for given actions and priors."""
        self.is_expanded = True
        for i, (move, prior) in enumerate(zip(actions, priors)):
            # Generate FEN string for the child state lazily or pass it
            child = MCTSNode(
                state_fen=fen,
                parent=self,
                prior=prior,
                action=move,
                action_index=i,
            )
            self.children.append(child)

    def add_dirichlet_noise(self, alpha: float, epsilon: float):
        """Mix Dirichlet noise into the priors of all children."""
        if not self.children:
            return
        noise = np.random.dirichlet([alpha] * len(self.children))
        for child, n in zip(self.children, noise):
            child.prior = (1.0 - epsilon) * child.prior + epsilon * n

    def get_visit_probs(self, temperature: float = 1.0) -> Tuple[chess.Move, np.ndarray]:
        """
        Get move probabilities from visit counts using temperature.
        Returns (selected_move, probs_array).
        """
        visits = np.array([c.visit_count for c in self.children], dtype=np.float64)

        if temperature < 0.05:  # near-deterministic: argmax
            # Deterministic: argmax
            idx = int(np.argmax(visits))
            probs = np.zeros(len(visits))
            probs[idx] = 1.0
        else:
            visits = visits ** (1.0 / temperature)
            probs = visits / visits.sum()

        # Sample from the distribution
        idx = np.random.choice(len(probs), p=probs)
        return self.children[idx].action, probs

    def __repr__(self) -> str:
        return (f"MCTSNode(visits={self.visit_count}, "
                f"value={self.value:.3f}, prior={self.prior:.3f}, "
                f"expanded={self.is_expanded}, terminal={self.is_terminal})")


class MCTS:
    """
    Monte Carlo Tree Search orchestration.

    The search() method returns the root node with populated visit counts.
    """

    def __init__(
        self,
        model: "NeuralNetwork",
        num_simulations: int = config.NUM_MCTS_SIMS,
        c_puct: float = config.C_PUCT,
        dirichlet_alpha: float = config.DIRICHLET_ALPHA,
        dirichlet_epsilon: float = config.DIRICHLET_EPSILON,
        temperature: float = config.TEMPERATURE,
        temperature_threshold: int = config.TEMPERATURE_THRESHOLD,
        device: str = None,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.device = device or config.INFERENCE_DEVICE
        self.model.to(self.device)
        self.model.eval()

    def search(
        self,
        board: chess.Board,
        add_root_noise: bool = True,
        return_probs: bool = False,
        deadline: Optional[float] = None,
        stop_flag=None,
    ) -> MCTSNode:
        """
        Run MCTS from the given board position.

        Args:
            board: current chess position.
            add_root_noise: if True, add Dirichlet noise to root's priors.
            return_probs: if True, return the root with action probabilities
                          from the network (before search) for diagnostics.
            deadline: monotonic timestamp; search stops when reached.
            stop_flag: callable returning True to abort the search.

        Returns:
            root node with fully populated children visit counts.
        """
        # Build root
        root = MCTSNode(state_fen=board.fen())

        # Evaluate root with neural network
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            root.is_terminal = True
            outcome = board.outcome()
            if outcome is None:
                root.reward = 0.0
            elif outcome.winner is None:
                root.reward = 0.0
            elif outcome.winner == board.turn:
                root.reward = 1.0  # root is terminal: reward from root's perspective
            else:
                root.reward = -1.0
            return root

        import torch
        encoded = chess_env.encode_board(board)
        tensor = torch.FloatTensor(encoded).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(tensor)

        policy_logits = policy_logits.squeeze(0)  # (4672,)
        value = value.item()

        # Compute priors over legal moves
        priors = self._get_priors(policy_logits, legal_moves)

        # Expand root
        root.expand(legal_moves, priors, fen=board.fen())

        # Add Dirichlet noise to root
        if add_root_noise:
            root.add_dirichlet_noise(self.dirichlet_alpha, self.dirichlet_epsilon)

        # Run simulations
        import time as _time
        for sim_idx in range(self.num_simulations):
            if sim_idx & 15 == 0:
                if deadline is not None and _time.monotonic() >= deadline:
                    break
                if stop_flag is not None and stop_flag():
                    break
            node = root
            sim_board = board.copy()
            path = [node]

            # ── SELECT ──
            while node.is_expanded and not node.is_terminal:
                node = node.best_child(self.c_puct)
                sim_board.push(node.action)
                path.append(node)

            # ── EXPAND & EVALUATE ──
            if not node.is_terminal:
                outcome = sim_board.outcome()
                if outcome is not None:
                    node.is_terminal = True
                    if outcome.winner is None:
                        node.reward = 0.0
                    elif outcome.winner == sim_board.turn:
                        node.reward = 1.0  # current player (who would move) wins
                    else:
                        node.reward = -1.0  # current player loses
                else:
                    # Expand this leaf
                    child_moves = list(sim_board.legal_moves)
                    encoded_leaf = chess_env.encode_board(sim_board)
                    leaf_tensor = torch.FloatTensor(encoded_leaf).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        leaf_logits, leaf_value = self.model(leaf_tensor)

                    leaf_logits = leaf_logits.squeeze(0)
                    child_priors = self._get_priors(leaf_logits, child_moves)
                    node.expand(child_moves, child_priors, fen=sim_board.fen())

                    node.reward = leaf_value.item()

            # ── BACKUP ──
            backup_value = node.reward
            for n in reversed(path):
                n.visit_count += 1
                n.total_value += backup_value
                backup_value = -backup_value  # flip for opponent's perspective

        return root

    def _get_priors(
        self, policy_logits: "torch.Tensor", legal_moves: List[chess.Move]
    ) -> List[float]:
        """
        Extract and softmax the logits for legal moves only.
        """
        indices = [chess_env.move_to_index(m) for m in legal_moves]
        legal_logits = policy_logits[indices]
        import torch.nn.functional as F
        probs = F.softmax(legal_logits, dim=0).cpu().numpy()
        return probs.tolist()

    def compute_temperature(self, move_count: int) -> float:
        """
        Return temperature based on move number.
        τ = 1 for first TEMPERATURE_THRESHOLD moves, τ → 0 thereafter.
        """
        if move_count < self.temperature_threshold:
            return self.temperature
        return config.TEMPERATURE_TAU
