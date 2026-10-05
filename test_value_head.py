#!/usr/bin/env python3
"""Sanity check of the NN value head on obvious positions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess, torch
import config, chess_env
from neural_network import NeuralNetwork

model = NeuralNetwork(config.INPUT_CHANNELS, config.NUM_RES_BLOCKS, config.NUM_FILTERS)
model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu", weights_only=True))
model.eval()

def val(fen):
    b = chess.Board(fen)
    enc = chess_env.encode_board(b)
    t = torch.FloatTensor(enc).unsqueeze(0)
    with torch.no_grad():
        logits, v = model(t)
    # policy: best legal moves
    legal = list(b.legal_moves)
    idxs = [chess_env.move_to_index(m) for m in legal]
    p = torch.softmax(logits[0][idxs], dim=0)
    top = sorted(zip(legal, p.tolist()), key=lambda x: -x[1])[:3]
    stm = "white" if b.turn else "black"
    return v.item(), [(m.uci(), round(pr, 3)) for m, pr in top]

cases = [
    (chess.STARTING_FEN, "startpos (side to move: white)"),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "after 1.e4 e5 (white)"),
    ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "scholar mate threat: black Qh4+?? no wait (white)"),
    ("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 3", "black to move, 1.f3 e5 2.g4 (black should love this: fools mate coming)"),
    ("7k/8/8/8/8/8/Q7/K7 w - - 0 1", "white up a queen (white)"),
    ("7k/8/8/8/8/8/8/K1Q5 b - - 0 1", "black to move, down a queen (black)"),
    ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "equal-ish rook endgame (white)"),
]
for fen, desc in cases:
    v, top = val(fen)
    print(f"{desc:60s} value={v:+.3f}  top: {top}")
