"""
Fénix Coach — orchestrates the full training pipeline.

Pipeline per iteration:
  1. Self-play: generate training examples using current model + MCTS
  2. Train: update neural network on the collected examples
  3. Evaluate: play against Stockfish; save checkpoint if improved

TensorBoard logging is used throughout.
"""

import os
import time
import logging
import json
from typing import List, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from neural_network import NeuralNetwork
from self_play import SelfPlayWorker, TrainingExample
from stockfish_integration import StockfishEvaluator, find_stockfish

logger = logging.getLogger(__name__)


class Coach:
    """
    Coordinates self-play, neural network training, and evaluation against
    Stockfish.
    """

    def __init__(
        self,
        model: NeuralNetwork,
        mcts_sims: int = config.NUM_MCTS_SIMS,
        num_self_play_games: int = config.NUM_SELF_PLAY_GAMES,
        num_eval_games: int = config.NUM_EVAL_GAMES,
        batch_size: int = config.BATCH_SIZE,
        lr: float = config.INIT_LR,
        num_iterations: int = config.NUM_ITERATIONS,
        eval_interval: int = config.EVAL_INTERVAL,
        save_interval: int = config.SAVE_INTERVAL,
        max_buffer_size: int = config.MAX_BUFFER_SIZE,
        device: str = None,
    ):
        self.model = model
        self.mcts_sims = mcts_sims
        self.num_self_play_games = num_self_play_games
        self.num_eval_games = num_eval_games
        self.batch_size = batch_size
        self.lr = lr
        self.num_iterations = num_iterations
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.max_buffer_size = max_buffer_size
        self.device = device or config.INFERENCE_DEVICE

        self.model.to(self.device)

        # Optimizer & scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.LR_DECAY_STEP,
            gamma=config.LR_DECAY_GAMMA,
        )

        # Training example ring buffer
        self.training_buffer: deque = deque(maxlen=self.max_buffer_size)

        # Loss function
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

        # Ensures directories exist
        self._init_dirs()

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=config.TENSORBOARD_DIR)

        # Metrics tracking
        self.best_stockfish_score = -float("inf")
        self.iteration = 0

    def _init_dirs(self):
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.DATA_DIR, exist_ok=True)
        os.makedirs(config.TENSORBOARD_DIR, exist_ok=True)

    # ── Self-play ───────────────────────────────────────────────────────

    def generate_self_play_data(self, num_games: int) -> List[TrainingExample]:
        """Generate training examples via self-play."""
        worker = SelfPlayWorker(
            model=self.model,
            mcts_sims=self.mcts_sims,
            device=self.device,
        )
        examples = worker.play_batch(num_games, desc="Self-play")
        return examples

    # ── Training ────────────────────────────────────────────────────────

    def train_on_buffer(self, num_epochs: int = config.NUM_TRAIN_EPOCHS) -> dict:
        """
        Train the neural network on examples from the ring buffer.

        Args:
            num_epochs: number of passes over the buffer.

        Returns:
            dict with average losses.
        """
        if len(self.training_buffer) == 0:
            logger.warning("Training buffer is empty — skipping training step.")
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        self.model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_batches = 0

        # Convert buffer to tensors
        examples = list(self.training_buffer)
        states = np.stack([e[0] for e in examples])  # (N, 119, 8, 8)
        policies = np.stack([e[1] for e in examples])  # (N, 4672)
        values = np.array([e[2] for e in examples], dtype=np.float32)  # (N,)

        n_samples = len(examples)

        for epoch in range(num_epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start: start + self.batch_size]
                batch_states = torch.FloatTensor(states[batch_idx]).to(self.device)
                batch_policies = torch.FloatTensor(policies[batch_idx]).to(self.device)
                batch_values = torch.FloatTensor(values[batch_idx]).unsqueeze(1).to(self.device)

                # Forward
                policy_logits, value_pred = self.model(batch_states)

                # Policy loss: cross-entropy between policy targets and logits
                # The target is a probability distribution; use KL divergence / cross-entropy
                policy_log_softmax = torch.log_softmax(policy_logits, dim=1)
                policy_loss = -(batch_policies * policy_log_softmax).sum(dim=1).mean()

                # Value loss: MSE
                value_loss = self.value_loss_fn(value_pred, batch_values)

                # Combined loss
                loss = policy_loss + value_loss

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping (important for stability)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_batches += 1

        avg_policy = total_policy_loss / max(total_batches, 1)
        avg_value = total_value_loss / max(total_batches, 1)

        logger.info(
            "Training — policy_loss=%.4f, value_loss=%.4f, batches=%d",
            avg_policy, avg_value, total_batches,
        )

        return {
            "policy_loss": avg_policy,
            "value_loss": avg_value,
            "total_loss": avg_policy + avg_value,
        }

    # ── Evaluation vs Stockfish ─────────────────────────────────────────

    def evaluate_against_stockfish(self, num_games: int) -> float:
        """
        Play a match against Stockfish.

        Returns:
            Score from the neural network's perspective:
              +1.0 = won all, -1.0 = lost all, 0.0 = equal.
        """
        try:
            sf = StockfishEvaluator(elo=config.STOCKFISH_ELO)
            sf.start()
        except RuntimeError as e:
            logger.warning("Stockfish not available: %s. Skipping evaluation.", e)
            return 0.0

        total_score = 0.0
        games_played = 0

        for game_idx in range(num_games):
            board = chess.Board()
            # Alternate colors
            sf_plays_white = (game_idx % 2 == 1)

            try:
                result = sf.play_game(
                    model=self.model,
                    board=board,
                    time_per_move_ms=100,
                    mcts_sims=max(self.mcts_sims // 2, 100),  # fewer sims for speed
                    sf_plays_white=sf_plays_white,
                    temperature=0.0,
                )
                # result is from white's perspective; convert to NN's perspective
                if sf_plays_white:
                    # NN played black → negate
                    nn_score = -result
                else:
                    nn_score = result
                total_score += nn_score
                games_played += 1
                
                logger.info(
                    "Eval game %d/%d — NN score=%+.1f (cumulative=%.3f)",
                    game_idx + 1, num_games, nn_score, total_score / games_played,
                )
            except Exception as e:
                logger.error("Evaluation game %d failed: %s", game_idx, e)
                continue

        sf.stop()
        avg_score = total_score / max(games_played, 1)
        logger.info(
            "Evaluation vs Stockfish (ELO=%d): %.1f / %d = %.3f",
            config.STOCKFISH_ELO, total_score, games_played, avg_score,
        )
        return avg_score

    # ── Main loop ───────────────────────────────────────────────────────

    def train(self):
        """Run the full Fénix training loop."""
        logger.info("=" * 60)
        logger.info("Starting Fénix Chess training")
        logger.info("Iterations=%d, Self-play games=%d, MCTS sims=%d",
                     self.num_iterations, self.num_self_play_games, self.mcts_sims)
        logger.info("Neural network: %d res-blocks, %d filters",
                     config.NUM_RES_BLOCKS, config.NUM_FILTERS)
        logger.info("Device: %s", self.device)
        logger.info("=" * 60)

        start_time = time.time()

        for iteration in range(1, self.num_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()
            logger.info("─" * 50)
            logger.info("Iteration %d / %d", iteration, self.num_iterations)

            # ── Step 1: Self-play ──────────────────────────────────────
            logger.info("Generating self-play data...")
            new_examples = self.generate_self_play_data(
                num_games=self.num_self_play_games,
            )

            # Add to ring buffer
            self.training_buffer.extend(new_examples)
            logger.info(
                "Training buffer size: %d examples",
                len(self.training_buffer),
            )

            # ── Step 2: Train neural network ──────────────────────────
            logger.info("Training neural network on buffer...")
            train_metrics = self.train_on_buffer()

            # Step the LR scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Log to TensorBoard
            self.writer.add_scalar("Loss/policy", train_metrics["policy_loss"], iteration)
            self.writer.add_scalar("Loss/value", train_metrics["value_loss"], iteration)
            self.writer.add_scalar("Loss/total", train_metrics["total_loss"], iteration)
            self.writer.add_scalar("Training/buffer_size", len(self.training_buffer), iteration)
            self.writer.add_scalar("Training/lr", current_lr, iteration)
            self.writer.add_scalar("Training/examples_generated", len(new_examples), iteration)

            # ── Step 3: Evaluate against Stockfish ────────────────────
            if iteration % self.eval_interval == 0:
                logger.info("Evaluating against Stockfish...")
                sf_score = self.evaluate_against_stockfish(num_games=self.num_eval_games)
                self.writer.add_scalar("Eval/stockfish_score", sf_score, iteration)

                # Save best model
                if sf_score > self.best_stockfish_score:
                    self.best_stockfish_score = sf_score
                    best_path = os.path.join(config.MODEL_DIR, "best_model.pt")
                    torch.save(self.model.state_dict(), best_path)
                    logger.info(
                        "New best model! Score=%.3f against Stockfish. Saved to %s",
                        sf_score, best_path,
                    )

            # ── Step 4: Save checkpoint ───────────────────────────────
            if iteration % self.save_interval == 0:
                ckpt_path = os.path.join(config.MODEL_DIR, f"model_iter_{iteration}.pt")
                torch.save({
                    "iteration": iteration,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_score": self.best_stockfish_score,
                    "buffer_size": len(self.training_buffer),
                    "train_metrics": train_metrics,
                }, ckpt_path)
                logger.info("Checkpoint saved to %s", ckpt_path)

            elapsed = time.time() - iter_start
            total_elapsed = time.time() - start_time
            logger.info(
                "Iteration %d complete — time=%.1fs, total=%.1fs",
                iteration, elapsed, total_elapsed,
            )

        # ── Final save ────────────────────────────────────────────────
        final_path = os.path.join(config.MODEL_DIR, "final_model.pt")
        torch.save(self.model.state_dict(), final_path)
        logger.info("Training complete. Final model saved to %s", final_path)
        self.writer.close()
