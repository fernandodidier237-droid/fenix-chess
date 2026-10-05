"""
╔══════════════════════════════════════════════════════════════════════╗
║              AlphaZero Chess — Self-Play Training                   ║
║                                                                     ║
║  Train a deep reinforcement learning agent to play chess at a       ║
║  superhuman level using self-play + MCTS, inspired by DeepMind's   ║
║  AlphaZero.                                                         ║
╚══════════════════════════════════════════════════════════════════════╝

Usage:
    python main.py                          # full training
    python main.py --iterations 5           # short test run
    python main.py --resume path/to/ckpt    # resume from checkpoint
    python main.py --no-stockfish           # skip Stockfish evaluation
    python -m torch.utils.collect_env       # verify torch installation
"""

import argparse
import os
import sys
import logging

import config
from utils import setup_logging, get_device, print_model_summary, save_training_config
from stockfish_integration import find_stockfish

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AlphaZero Chess — self-play training with MCTS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training
    parser.add_argument("--iterations", type=int, default=config.NUM_ITERATIONS,
                        help="Number of training iterations")
    parser.add_argument("--self-play-games", type=int, default=config.NUM_SELF_PLAY_GAMES,
                        help="Self-play games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=config.NUM_MCTS_SIMS,
                        help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=config.INIT_LR,
                        help="Initial learning rate")
    parser.add_argument("--buffer-size", type=int, default=config.MAX_BUFFER_SIZE,
                        help="Maximum training examples in ring buffer")

    # Evaluation
    parser.add_argument("--eval-games", type=int, default=config.NUM_EVAL_GAMES,
                        help="Number of evaluation games vs Stockfish")
    parser.add_argument("--no-stockfish", action="store_true",
                        help="Skip Stockfish evaluation")

    # Model
    parser.add_argument("--res-blocks", type=int, default=config.NUM_RES_BLOCKS,
                        help="Number of residual blocks")
    parser.add_argument("--filters", type=int, default=config.NUM_FILTERS,
                        help="Number of convolutional filters")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu", "mps"],
                        help="Override device auto-detection")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Lazy torch import (torch may not be installed for light tasks) ──
    try:
        import torch
        from neural_network import NeuralNetwork
        from coach import Coach
    except ImportError as e:
        logger.error(
            "PyTorch is required for training. Install it with:\n"
            "    pip install torch\n"
            "Or if you have CUDA: pip install torch --index-url "
            "https://download.pytorch.org/whl/cu118\n"
        )
        logger.error("Failed to import: %s", e)
        sys.exit(1)

    # ── Setup ───────────────────────────────────────────────────────────
    setup_logging()
    save_training_config()

    logger.info("Starting AlphaZero Chess Training")
    logger.info("=" * 60)

    # ── Device detection ───────────────────────────────────────────────
    device = args.device or get_device()

    # ── Check Stockfish availability ───────────────────────────────────
    if not args.no_stockfish:
        sf_path = find_stockfish()
        if sf_path:
            logger.info("Stockfish found at: %s", sf_path)
            config.STOCKFISH_PATH = sf_path
        else:
            logger.warning(
                "Stockfish not found on the system. "
                "Evaluation against Stockfish will be DISABLED.\n"
                "  Install Stockfish: sudo apt install stockfish  (Linux)\n"
                "  or: brew install stockfish  (macOS)\n"
                "  or download from: https://stockfishchess.org/download/"
            )
            args.no_stockfish = True
    else:
        logger.info("Stockfish evaluation disabled by user.")

    # ── Build model ────────────────────────────────────────────────────
    model = NeuralNetwork(
        input_channels=config.INPUT_CHANNELS,
        num_res_blocks=args.res_blocks,
        num_filters=args.filters,
    )
    print_model_summary(model)

    # ── Resume from checkpoint ─────────────────────────────────────────
    start_iteration = 0
    if args.resume:
        logger.info("Loading checkpoint from: %s", args.resume)
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "iteration" in checkpoint:
            start_iteration = checkpoint["iteration"]
            logger.info("Resumed from iteration %d", start_iteration)

    # ── Create coach ───────────────────────────────────────────────────
    coach = Coach(
        model=model,
        mcts_sims=args.mcts_sims,
        num_self_play_games=args.self_play_games,
        num_eval_games=args.eval_games if not args.no_stockfish else 0,
        batch_size=args.batch_size,
        lr=args.lr,
        num_iterations=args.iterations + start_iteration,
        max_buffer_size=args.buffer_size,
        device=device,
    )

    # If resuming, also restore optimizer state
    if args.resume:
        if "optimizer_state_dict" in checkpoint:
            coach.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            coach.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "buffer_size" in checkpoint and checkpoint["buffer_size"] > 0:
            logger.warning(
                "Training buffer not persisted across runs. "
                "Buffer will be rebuilt from self-play."
            )
        if "best_score" in checkpoint:
            coach.best_stockfish_score = checkpoint["best_score"]
            logger.info("Best Stockfish score restored: %.3f", coach.best_stockfish_score)

    # ── Train ───────────────────────────────────────────────────────────
    try:
        coach.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving checkpoint...")
        save_path = os.path.join(
            config.MODEL_DIR, f"interrupted_iter_{coach.iteration}.pt"
        )
        torch.save({
            "iteration": coach.iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": coach.optimizer.state_dict(),
            "scheduler_state_dict": coach.scheduler.state_dict(),
            "best_score": coach.best_stockfish_score,
            "buffer_size": len(coach.training_buffer),
        }, save_path)
        logger.info("Interrupted checkpoint saved to: %s", save_path)
        sys.exit(0)


if __name__ == "__main__":
    main()
