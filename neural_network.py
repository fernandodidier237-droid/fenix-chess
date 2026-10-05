"""
Neural network architecture for AlphaZero Chess.

Architecture (following the AlphaZero paper):
  - Input: 8×8×119 (board encoding)
  - Initial convolution: 256 filters, 3×3, BN, ReLU
  - 10 residual blocks (256 filters, 3×3) with skip connections
  - Policy head: conv 1×1 (32 filters), BN, ReLU, flatten, FC → 4672
  - Value head: conv 1×1 (1 filter), BN, ReLU, flatten, FC 256, tanh → scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

import config


class ResidualBlock(nn.Module):
    """
    A single residual block:
      Conv2d(3×3) → BN → ReLU → Conv2d(3×3) → BN → +input → ReLU
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class NeuralNetwork(nn.Module):
    """
    AlphaZero neural network: maps board → (policy_logits, value).
    """

    def __init__(
        self,
        input_channels: int = config.INPUT_CHANNELS,
        num_res_blocks: int = config.NUM_RES_BLOCKS,
        num_filters: int = config.NUM_FILTERS,
        policy_channels: int = config.NUM_POLICY_CHANNELS,
        policy_output_size: int = config.POLICY_OUTPUT_SIZE,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters

        # ── Initial convolutional block ──────────────────────────────
        self.conv_init = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(num_filters)

        # ── Residual tower ──────────────────────────────────────────
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # ── Policy head ─────────────────────────────────────────────
        self.policy_conv = nn.Conv2d(num_filters, policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels * 8 * 8, policy_output_size)

        # ── Value head ──────────────────────────────────────────────
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 119, 8, 8) tensor
        Returns:
            policy: (batch, 4672) logits
            value: (batch, 1) scalar in (-1, 1)
        """
        # Initial conv
        out = self.conv_init(x)
        out = self.bn_init(out)
        out = F.relu(out)

        # Residual tower
        for block in self.res_blocks:
            out = block(out)

        # ── Policy head ──
        policy = self.policy_conv(out)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)  # flatten
        policy = self.policy_fc(policy)

        # ── Value head ──
        value = self.value_conv(out)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def get_policy_probs(self, board, legal_moves, device=None):
        """
        Convenience: given a python-chess board, return softmax
        probabilities over legal moves only.
        """
        import chess_env
        if device is None:
            device = next(self.parameters()).device
        tensor = torch.FloatTensor(chess_env.encode_board(board)).unsqueeze(0).to(device)
        self.eval()
        with torch.no_grad():
            logits, _ = self(tensor)
        logits = logits.squeeze(0)
        # Mask to legal moves only
        indices = [chess_env.move_to_index(m) for m in legal_moves]
        legal_logits = logits[indices]
        probs = F.softmax(legal_logits, dim=0).cpu().numpy()
        return probs

    def save_checkpoint(self, filepath: str):
        """Save model weights to filepath."""
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath: str, device: str = "cpu"):
        """Load model weights from filepath."""
        self.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
