#!/usr/bin/env python3
"""Quick benchmark: encode_board, model forward pass, MCTS sims/sec."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess
import numpy as np
import torch

import config, chess_env
from neural_network import NeuralNetwork

model = NeuralNetwork(config.INPUT_CHANNELS, config.NUM_RES_BLOCKS, config.NUM_FILTERS)
model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu", weights_only=True))
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params:,}")

board = chess.Board()
for _ in range(20):
    board.push(list(board.legal_moves)[np.random.randint(len(list(board.legal_moves)))])

# encode speed
t0 = time.time()
N = 50
for _ in range(N):
    enc = chess_env.encode_board(board)
print(f"encode_board: {(time.time()-t0)/N*1000:.2f} ms")

tensor = torch.FloatTensor(enc).unsqueeze(0)
with torch.no_grad():
    for _ in range(3):
        model(tensor)  # warmup
t0 = time.time()
with torch.no_grad():
    for _ in range(N):
        model(tensor)
print(f"model forward: {(time.time()-t0)/N*1000:.2f} ms")

# batch speed
batch = tensor.repeat(16, 1, 1, 1)
with torch.no_grad():
    for _ in range(3):
        model(batch)
t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        model(batch)
print(f"model forward batch16: {(time.time()-t0)/10*1000:.2f} ms  ({160/((time.time()-t0)):0.0f} evals/s)")

# MCTS speed
from mcts import MCTS
mcts = MCTS(model=model, num_simulations=100, device="cpu")
t0 = time.time()
root = mcts.search(board, add_root_noise=False)
dt = time.time() - t0
print(f"MCTS 100 sims: {dt:.2f} s -> {100/dt:.1f} sims/s")
top = sorted(root.children, key=lambda c: c.visit_count, reverse=True)[:5]
print("top moves:", [(c.action.uci(), c.visit_count, round(c.prior,3)) for c in top])
