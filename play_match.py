#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║     Fénix Chess vs Stockfish — Match UCI (v2)                        ║
║                                                                          ║
║  Juega nuestro motor UCI (alpha_zero_engine.py, el mismo que usa         ║
║  PyChess) contra Stockfish, alternando colores.                           ║
║                                                                          ║
║  Ejemplos:                                                                ║
║    python play_match.py --games 6 --elo 1350                             ║
║    python play_match.py --games 4 --elo 1600 --az-time 1.5               ║
║    python play_match.py --games 2 --engine mcts                          ║
║    python play_match.py --games 1 --full-strength   # SF a fuerza total   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess
import chess.engine

import config
from stockfish_integration import find_stockfish

BASE = os.path.dirname(os.path.abspath(__file__))


def build_az_engine(args) -> chess.engine.SimpleEngine:
    cmd = [os.path.join(BASE, ".venv", "bin", "python"), "-u",
           os.path.join(BASE, "alpha_zero_engine.py"),
           "--search-type", args.engine]
    eng = chess.engine.SimpleEngine.popen_uci(cmd)
    return eng


def build_stockfish(args) -> chess.engine.SimpleEngine:
    path = find_stockfish()
    if not path:
        raise SystemExit("Stockfish no encontrado")
    eng = chess.engine.SimpleEngine.popen_uci(path)
    if not args.full_strength:
        eng.configure({
            "UCI_LimitStrength": True,
            "UCI_Elo": max(1320, min(3190, args.elo)),
        })
    return eng


def play_game(az, sf, args, az_white: bool, game_num: int, total_games):
    board = chess.Board()
    moves = []
    t_start = time.time()
    name_az = f"Fénix[{args.engine}] ({args.az_time:.1f}s/ jugada)"
    name_sf = ("Stockfish (fuerza total)" if args.full_strength
               else f"Stockfish (UCI_Elo {args.elo})")
    white_name = name_az if az_white else name_sf
    black_name = name_sf if az_white else name_az

    print(f"\n{'='*64}\n  Game {game_num}/{total_games}: "
          f"{white_name} (Blancas) vs {black_name} (Negras)\n{'='*64}",
          flush=True)

    while not board.is_game_over() and len(moves) < 300:
        is_az = (board.turn == chess.WHITE) == az_white
        t0 = time.time()
        if is_az:
            limit = chess.engine.Limit(time=args.az_time)
            engine, tag = az, "AZ"
        else:
            limit = chess.engine.Limit(time=args.sf_time_ms / 1000.0)
            engine, tag = sf, "SF"
        try:
            result = engine.play(board, limit)
        except chess.engine.EngineTerminatedError:
            print("  ¡Motor murió!", flush=True)
            break
        move = result.move
        if move is None:
            break
        board.push(move)
        moves.append(move.uci())
        dt = time.time() - t0
        print(f"  [{len(moves):3d}] {tag} {move.uci():7s} ({dt:4.1f}s)  "
              f"{board.fen().split(' ')[0][:30]}", flush=True)

    elapsed = time.time() - t_start
    outcome = board.outcome()
    if outcome is None:
        result_val, desc = 0.0, "sin resultado"
    elif outcome.winner is None:
        result_val, desc = 0.0, "Tablas"
    else:
        white_won = outcome.winner == chess.WHITE
        az_won = white_won == az_white
        result_val = 1.0 if az_won else -1.0
        desc = "¡Fénix gana!" if az_won else "Stockfish gana"
    term = outcome.termination.name if outcome else "-"
    print(f"  Resultado: {desc} ({term}) en {len(moves)} jugadas "
          f"({elapsed:.0f}s)", flush=True)
    print(f"  Moves: {' '.join(moves)}", flush=True)
    return result_val


def main():
    parser = argparse.ArgumentParser(
        description="Partido UCI: Fénix vs Stockfish",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=6,
                        help="Número de partidas (alterna colores)")
    parser.add_argument("--elo", type=int, default=1350,
                        help="UCI_Elo de Stockfish")
    parser.add_argument("--full-strength", action="store_true",
                        help="Stockfish a fuerza completa (sin limitar)")
    parser.add_argument("--engine", choices=["Hybrid", "MCTS"],
                        default="Hybrid",
                        help="Tipo de búsqueda de nuestro motor")
    parser.add_argument("--az-time", type=float, default=1.0,
                        help="Segundos por jugada para Fénix")
    parser.add_argument("--sf-time-ms", type=float, default=200,
                        help="Milisegundos por jugada para Stockfish")
    args = parser.parse_args()

    print(f"Motor: Fénix[{args.engine}]  {args.az_time}s/jugada | "
          f"Stockfish: "
          f"{'fuerza total' if args.full_strength else f'UCI_Elo {args.elo}'}"
          f"  {args.sf_time_ms:.0f}ms/jugada", flush=True)

    az = build_az_engine(args)
    sf = build_stockfish(args)

    scores = []
    try:
        for g in range(args.games):
            s = play_game(az, sf, args, az_white=(g % 2 == 0),
                          game_num=g + 1, total_games=args.games)
            scores.append(s)
            az_wins = sum(1 for x in scores if x > 0)
            sf_wins = sum(1 for x in scores if x < 0)
            draws = sum(1 for x in scores if x == 0)
            print(f"  [parcial] AZ {az_wins} - SF {sf_wins} "
                  f"(tablas {draws})", flush=True)
    finally:
        az.quit()
        sf.quit()

    az_wins = sum(1 for x in scores if x > 0)
    sf_wins = sum(1 for x in scores if x < 0)
    draws = sum(1 for x in scores if x == 0)
    print(f"\n{'='*64}\n  MARCADOR FINAL", flush=True)
    print(f"  Fénix: {az_wins} victorias, {sf_wins} derrotas, "
          f"{draws} tablas", flush=True)
    print(f"  Puntos: {sum(scores):+.1f} / {len(scores)}", flush=True)
    print(f"{'='*64}", flush=True)


if __name__ == "__main__":
    main()
