"""
Configuration parameters for Fénix Chess AI.
All tunable hyperparameters are centralized here.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ── Board & Neural Network ──────────────────────────────────────────────
BOARD_SIZE = 8
INPUT_CHANNELS = 119
NUM_RES_BLOCKS = 10
NUM_FILTERS = 256
NUM_POLICY_CHANNELS = 32
POLICY_OUTPUT_SIZE = 4672  # 64 × 73 move types (AlphaZero standard)

# ── Optimizer / Training ────────────────────────────────────────────────
BATCH_SIZE = 256
INIT_LR = 0.0002
LR_DECAY_GAMMA = 0.9
LR_DECAY_STEP = 30          # decay every N iterations
WEIGHT_DECAY = 1.0e-4
MOMENTUM = 0.9
NUM_ITERATIONS = 100         # outer training iterations
NUM_SELF_PLAY_GAMES = 10_000  # games per iteration
SELF_PLAY_BATCH = 100        # how many games to run before training sub-iterations
NUM_TRAIN_EPOCHS = 4         # training sub-iterations per self-play batch
SAVE_INTERVAL = 5            # checkpoint every N iterations
MAX_BUFFER_SIZE = 300_000    # maximum training examples in ring buffer (~20 GB RAM)

# ── MCTS ────────────────────────────────────────────────────────────────
NUM_MCTS_SIMS = 800          # simulations per MCTS node (800 in paper)
C_PUCT = 2.5                 # exploration constant
DIRICHLET_ALPHA = 0.3        # Dirichlet noise concentration (chess)
DIRICHLET_EPSILON = 0.25     # noise mixing proportion
TEMPERATURE = 1.0            # initial temperature for move selection
TEMPERATURE_THRESHOLD = 30   # moves after which temperature → 0
TEMPERATURE_TAU = 1e-3       # near-zero temperature for deterministic selection

# ── Evaluation vs Stockfish ─────────────────────────────────────────────
STOCKFISH_PATH: str = "/usr/games/stockfish"  # auto-detected; override if needed
STOCKFISH_ELO = 2800
STOCKFISH_THREADS = 4
STOCKFISH_HASH_SIZE_MB = 512
NUM_EVAL_GAMES = 100          # games to play against Stockfish per evaluation
EVAL_INTERVAL = 1             # evaluate every N iterations

# ── Directories ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")
TENSORBOARD_DIR = os.path.join(BASE_DIR, "tensorboard")
INFERENCE_DEVICE = "cuda"     # "cuda" | "cpu" | "mps"
