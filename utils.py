"""
Utility functions for AlphaZero Chess.

Includes:
  - Logging configuration
  - Model parameter counting
  - GPU detection
  - Data persistence helpers
"""

import os
import sys
import logging
import json
from typing import Any, Dict
from datetime import datetime

import numpy as np

import config


# ── Logging (no torch dependency) ───────────────────────────────────────

def setup_logging(level: int = logging.INFO):
    """
    Configure a clean logger that writes both to stdout and a log file.
    """
    log_format = (
        "[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(console)

    os.makedirs(config.LOG_DIR, exist_ok=True)
    log_file = os.path.join(
        config.LOG_DIR,
        f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(file_handler)

    logging.getLogger("chess.engine").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def save_training_config():
    """
    Save the current configuration as JSON for reproducibility.
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)
    cfg_path = os.path.join(config.LOG_DIR, "config.json")

    cfg = {k: v for k, v in vars(config).items()
           if not k.startswith("_") and isinstance(v, (str, int, float, bool, list, dict))}

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    logging.getLogger(__name__).info("Configuration saved to %s", cfg_path)


# ── Torch-dependent helpers (lazy import) ───────────────────────────────

def count_parameters(model) -> Dict[str, Any]:
    """Count trainable and total parameters in a model (torch nn.Module)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
    }


def get_device() -> str:
    """
    Detect and return the best available device.
    Priority: CUDA > MPS (Apple Silicon) > CPU.
    """
    import torch
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logging.getLogger(__name__).info(
            "Using CUDA: %s (%.1f GB)", gpu_name, gpu_mem,
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logging.getLogger(__name__).info("Using MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        logging.getLogger(__name__).info("Using CPU")
    return device


def print_model_summary(model):
    """Print a human-readable model summary (model must be a torch nn.Module)."""
    params = count_parameters(model)
    summary = (
        f"Model summary:\n"
        f"  Architecture: AlphaZero ResNet\n"
        f"  Residual blocks: {config.NUM_RES_BLOCKS}\n"
        f"  Filters per block: {config.NUM_FILTERS}\n"
        f"  Input channels: {config.INPUT_CHANNELS}\n"
        f"  Policy output: {config.POLICY_OUTPUT_SIZE}\n"
        f"  Total parameters: {params['total']:,} ({params['total_millions']:.2f}M)\n"
        f"  Trainable parameters: {params['trainable']:,} ({params['trainable_millions']:.2f}M)"
    )
    print(summary)
    logging.getLogger(__name__).info(summary)
