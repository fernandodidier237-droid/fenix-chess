#!/home/didier/alpha_zero_chess_selfplay/.venv/bin/python
"""
╔══════════════════════════════════════════════════════════════════════════╗
║         Fénix Chess — UCI Engine Interface (v2, híbrido)             ║
║                                                                          ║
║  Dos modos de búsqueda (opción UCI "SearchType"):                        ║
║    Hybrid (defecto): alfa-beta + quiescencia + TT, con la red            ║
║                      neuronal ordenando las jugadas (policy priors).     ║
║    MCTS:             Monte Carlo Tree Search + red (modo AlphaZero       ║
║                      clásico, más fiel pero más débil con esta red).     ║
║                                                                          ║
║  Compatible con cualquier GUI UCI (PyChess, Arena, CuteChess...).        ║
║                                                                          ║
║  Uso:                                                                    ║
║    python alpha_zero_engine.py                                           ║
║  En PyChess:                                                             ║
║    Preferencias → Motores → Añadir motor (o engines.json, ya listo)      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Dict, List, Optional

# Raíz del proyecto en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess

import config


def _log(msg: str) -> None:
    """Debug por stderr (nunca ensuciar stdout: es el canal UCI)."""
    print(f"info string {msg}", file=sys.stderr, flush=True)


def _load_model(checkpoint_path: str, device: str = "cpu"):
    """Carga la red neuronal desde un checkpoint."""
    import torch
    from neural_network import NeuralNetwork

    model = NeuralNetwork(
        input_channels=config.INPUT_CHANNELS,
        num_res_blocks=config.NUM_RES_BLOCKS,
        num_filters=config.NUM_FILTERS,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class FenixUCIEngine:
    """Motor UCI: híbrido alfa-beta + red neuronal, o MCTS + red."""

    def __init__(self, checkpoint_path: str, mcts_sims: int = 400,
                 device: str = "cpu", search_type: str = "Hybrid",
                 use_nn: bool = True, hash_mb: int = 64):
        self.checkpoint_path = checkpoint_path
        self.mcts_sims = mcts_sims          # nunca se muta por jugada
        self.device = device
        self.search_type = search_type
        self.use_nn = use_nn
        self.hash_mb = hash_mb

        self.board = chess.Board()
        self.model = None
        self._loaded = False
        self._load_failed = False

        self.name = "FenixChess-Hybrid"
        self.author = "Fénix Chess Project"

        # estado de búsqueda
        self._search_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── Inicialización ──────────────────────────────────────────────────

    def load_model(self):
        if self._loaded or self._load_failed or not self.use_nn:
            return
        try:
            _log(f"Loading model from {self.checkpoint_path}...")
            self.model = _load_model(self.checkpoint_path, self.device)
            self._loaded = True
            _log(f"Model loaded on {self.device}")
        except Exception as e:
            self._load_failed = True
            _log(f"Model not available ({e}); NN priors disabled")

    # ── Priors de la red para el root ───────────────────────────────────

    def _root_priors(self, board: chess.Board) -> Dict[chess.Move, float]:
        """Priors del policy head sobre las jugadas legales del root."""
        if not self.use_nn:
            return {}
        if not self._loaded and not self._load_failed:
            self.load_model()
        if self.model is None:
            return {}
        try:
            import numpy as np
            import torch
            import chess_env
            legal = list(board.legal_moves)
            if not legal:
                return {}
            enc = chess_env.encode_board(board)
            t = torch.FloatTensor(enc).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, _ = self.model(t)
            idxs = [chess_env.move_to_index(m) for m in legal]
            probs = torch.softmax(logits[0][idxs], dim=0).cpu().numpy()
            probs = probs / (probs.sum() + 1e-9)
            return {m: float(p) for m, p in zip(legal, probs)}
        except Exception as e:
            _log(f"priors error: {e}")
            return {}

    # ── Gestión de tiempo ───────────────────────────────────────────────

    def _time_budget(self, params: dict) -> tuple[Optional[float], int]:
        """Devuelve (presupuesto_segundos, profundidad_max)."""
        depth = params.get("depth", 64)

        if params.get("infinite"):
            return None, depth if params.get("depth") else 64

        if params.get("ponder"):
            # no soportamos ponder de verdad: acotado por seguridad
            # (PyChess solo envía 'go ponder' si mandamos bestmove+ponder,
            #  que no hacemos, pero así no puede colgarse)
            return 30.0, depth

        movetime = params.get("movetime")
        if movetime is not None:
            # deja un margen para el overhead de la GUI
            return max(0.05, movetime / 1000.0 * 0.92 - 0.03), depth

        wtime, btime = params.get("wtime"), params.get("btime")
        if wtime is not None and btime is not None:
            my_time = wtime if self.board.turn == chess.WHITE else btime
            my_inc = params.get("winc", 0) if self.board.turn == chess.WHITE \
                else params.get("binc", 0)
            movestogo = params.get("movestogo") or 30
            budget_ms = (my_time / movestogo) * 0.85 + my_inc * 0.75
            # nunca gastar más de la mitad del tiempo restante
            budget_ms = min(budget_ms, my_time * 0.5)
            budget_ms = min(budget_ms, my_time - 100)  # nunca a perder por tiempo
            return max(0.05, budget_ms / 1000.0), depth

        if params.get("depth"):
            return None, depth
        # sin control de tiempo: default razonable
        return 1.0, depth

    # ── Búsqueda ────────────────────────────────────────────────────────

    def _spawn_search(self, board: chess.Board, budget: Optional[float],
                      max_depth: int):
        """Lanza la búsqueda en un hilo (para poder atender 'stop')."""
        self._stop_event = threading.Event()
        self._search_thread = threading.Thread(
            target=self._search_worker,
            args=(board, budget, max_depth),
            daemon=True,
        )
        self._search_thread.start()

    def _search_worker(self, board: chess.Board, budget: Optional[float],
                       max_depth: int):
        try:
            if self.search_type == "MCTS":
                self._search_mcts(board, budget, max_depth)
            else:
                self._search_hybrid(board, budget, max_depth)
        except Exception as e:
            _log(f"search error: {e}")
            # fallback: cualquier jugada legal
            moves = list(board.legal_moves)
            print(f"bestmove {moves[0].uci() if moves else '0000'}", flush=True)

    def _search_hybrid(self, board: chess.Board, budget: Optional[float],
                       max_depth: int):
        from search import Searcher

        t_start = time.monotonic()
        searcher = Searcher(hash_mb=self.hash_mb)
        searcher.root_priors = self._root_priors(board)
        t_priors = time.monotonic()
        if budget is not None:
            # el tiempo de priors ya se gastó: ajustar el deadline
            budget = max(0.05, budget - (t_priors - t_start))

        def on_depth(res):
            score_str = _format_score(res.score)
            pv = " ".join(m.uci() for m in res.pv)
            time_ms = int(res.time_s * 1000)
            nps = res.nps
            line = (f"info depth {res.depth} seldepth {res.depth} "
                    f"multipv 1 {score_str} nodes {res.nodes} nps {nps} "
                    f"time {time_ms}")
            if pv:
                line += f" pv {pv}"
            print(line, flush=True)

        searcher.on_depth = on_depth
        deadline = (time.monotonic() + budget) if budget is not None else None

        res = searcher.search(
            board,
            time_budget_s=budget,
            max_depth=max_depth,
            stop_flag=self._stop_event.is_set,
            deadline=deadline,
        )
        move = res.move or (list(board.legal_moves) or [None])[0]
        print(f"bestmove {move.uci() if move else '0000'}", flush=True)

    def _search_mcts(self, board: chess.Board, budget: Optional[float],
                     max_depth: int):
        """Modo AlphaZero clásico: MCTS + red, con deadline real."""
        import numpy as np
        from mcts import MCTS

        if self.model is None:
            self.load_model()
        if self.model is None:
            # sin red no hay MCTS: cae al híbrido
            return self._search_hybrid(board, budget, max_depth)

        mcts = MCTS(
            model=self.model,
            num_simulations=self.mcts_sims,
            c_puct=config.C_PUCT,
            device=self.device,
        )
        deadline = (time.monotonic() + budget) if budget is not None else None
        t0 = time.monotonic()
        root = mcts.search(
            board,
            add_root_noise=False,
            deadline=deadline,
            stop_flag=self._stop_event.is_set,
        )
        if not root.children:
            print("bestmove 0000", flush=True)
            return
        # siempre determinista al jugar
        best_move, _ = root.get_visit_probs(temperature=config.TEMPERATURE_TAU)
        visits = max(c.visit_count for c in root.children)
        total = max(root.visit_count, 1)
        # valor del root como score (desde la perspectiva del que mueve)
        score_cp = int(root.value * 1000)
        elapsed = time.monotonic() - t0
        top = sorted(root.children, key=lambda c: c.visit_count, reverse=True)[:6]
        pv = " ".join(c.action.uci() for c in top)
        print(f"info depth 1 multipv 1 score cp {score_cp} "
              f"nodes {root.visit_count} "
              f"nps {int(root.visit_count / max(elapsed, 1e-6))} "
              f"time {int(elapsed * 1000)} pv {pv}", flush=True)
        print(f"bestmove {best_move.uci()}", flush=True)

    def _wait_search(self, timeout: Optional[float] = None):
        t = self._search_thread
        if t is not None and t.is_alive():
            t.join(timeout)

    # ── Handlers UCI ────────────────────────────────────────────────────

    def handle_uci(self):
        print(f"id name {self.name}", flush=True)
        print(f"id author {self.author}", flush=True)
        print("option name SearchType type combo default "
              f"{self.search_type} var Hybrid var MCTS", flush=True)
        print("option name UseNN type check default "
              f"{'true' if self.use_nn else 'false'}", flush=True)
        print("option name MCTS_Simulations type spin default "
              f"{self.mcts_sims} min 10 max 5000", flush=True)
        print("option name Hash type spin default "
              f"{self.hash_mb} min 1 max 1024", flush=True)
        print("option name UCI_AnalyseMode type check default false", flush=True)
        print("uciok", flush=True)

    def handle_isready(self):
        if not self._loaded and self.use_nn and not self._load_failed:
            self.load_model()
        print("readyok", flush=True)

    def handle_ucinewgame(self):
        self._stop_event.set()
        self._wait_search(5.0)
        self.board = chess.Board()

    def handle_position(self, fen: Optional[str] = None,
                        moves: Optional[List[str]] = None):
        self._stop_event.set()
        self._wait_search(5.0)
        if fen and fen != "startpos":
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        if moves:
            for move_uci in moves:
                move = chess.Move.from_uci(move_uci)
                if move in self.board.legal_moves:
                    self.board.push(move)
                else:
                    _log(f"Illegal move in position cmd: {move_uci}")

    def handle_go(self, params: dict):
        _log(f"go params: {params}")
        if self.board.is_game_over() or not self.board.legal_moves:
            print("bestmove 0000", flush=True)
            return
        budget, max_depth = self._time_budget(params)
        self._spawn_search(self.board.copy(), budget, max_depth)

    def handle_stop(self):
        self._stop_event.set()

    def handle_ponderhit(self):
        # sin ponder real: se mantiene la búsqueda actual
        pass

    # ── Loop UCI ────────────────────────────────────────────────────────

    def run(self):
        if not os.path.exists(self.checkpoint_path):
            _log(f"checkpoint not found: {self.checkpoint_path}")
        # precarga el modelo (queda dentro del handshake UCI)
        self.load_model()

        while True:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()
            try:
                if cmd == "uci":
                    self.handle_uci()
                elif cmd == "isready":
                    self.handle_isready()
                elif cmd == "ucinewgame":
                    self.handle_ucinewgame()
                elif cmd == "position":
                    self._handle_position(parts[1:])
                elif cmd == "go":
                    self._handle_go(parts[1:])
                elif cmd == "stop":
                    self.handle_stop()
                elif cmd == "ponderhit":
                    self.handle_ponderhit()
                elif cmd == "quit":
                    self._stop_event.set()
                    self._wait_search(5.0)
                    break
                elif cmd == "setoption":
                    self._handle_setoption(parts[1:])
                # comandos desconocidos: se ignoran (spec UCI)
            except Exception as e:
                _log(f"error handling '{line}': {e}")
                continue

    def _handle_position(self, args: List[str]):
        fen = None
        if "startpos" in args:
            fen = "startpos"
            idx = args.index("startpos") + 1
        elif "fen" in args:
            idx = args.index("fen") + 1
            fen_parts = []
            while idx < len(args) and args[idx] != "moves":
                fen_parts.append(args[idx])
                idx += 1
            fen = " ".join(fen_parts)
        else:
            idx = 0
            fen = "startpos"

        moves = None
        if idx < len(args) and args[idx] == "moves":
            moves = args[idx + 1:]
        self.handle_position(fen=fen, moves=moves)

    def _handle_go(self, args: List[str]):
        params = {}
        i = 0
        while i < len(args):
            arg = args[i]
            if arg in ("wtime", "btime", "winc", "binc", "movestogo",
                       "depth", "nodes", "movetime"):
                if i + 1 < len(args):
                    try:
                        params[arg] = int(args[i + 1])
                    except ValueError:
                        params[arg] = 0
                    i += 2
                    continue
            elif arg in ("infinite", "ponder"):
                params[arg] = True
            i += 1
        self.handle_go(params)

    def _handle_setoption(self, args: List[str]):
        if "name" not in args:
            return
        name_idx = args.index("name") + 1
        if "value" in args:
            val_idx = args.index("value")
            name = " ".join(args[name_idx:val_idx])
            value = " ".join(args[val_idx + 1:])
        else:
            name = " ".join(args[name_idx:])
            value = None

        key = name.lower().replace(" ", "_")
        if key == "mcts_simulations":
            try:
                self.mcts_sims = int(value)
                _log(f"MCTS sims = {self.mcts_sims}")
            except (ValueError, TypeError):
                pass
        elif key == "searchtype":
            if value in ("Hybrid", "MCTS"):
                self.search_type = value
                _log(f"SearchType = {self.search_type}")
        elif key == "usenn":
            self.use_nn = value == "true"
            if not self.use_nn:
                self.model = None
                self._loaded = False
        elif key == "hash":
            try:
                self.hash_mb = max(1, min(1024, int(value)))
            except (ValueError, TypeError):
                pass
        elif key in ("ponder", "uci_analysemode"):
            pass


def _format_score(score: int) -> str:
    """Convierte el score interno a 'score cp X' / 'score mate N' UCI."""
    from search import MATE_SCORE, MATE_BOUND
    if score >= MATE_BOUND:
        plies = MATE_SCORE - score
        return f"score mate {(plies + 1) // 2}"
    if score <= -MATE_BOUND:
        plies = MATE_SCORE + score
        return f"score mate -{(plies + 1) // 2}"
    return f"score cp {score}"


# ── Entrada ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fénix Chess — UCI Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", "-m",
                        default=os.path.join(config.MODEL_DIR, "best_model.pt"),
                        help="Checkpoint del modelo (.pt)")
    parser.add_argument("--sims", "-s", type=int, default=400,
                        help="Simulaciones MCTS por jugada (modo MCTS)")
    parser.add_argument("--device", "-d", choices=["cpu", "cuda", "mps"],
                        default="cpu", help="Dispositivo de inferencia")
    parser.add_argument("--search-type", choices=["Hybrid", "MCTS"],
                        default="Hybrid", help="Tipo de búsqueda por defecto")
    parser.add_argument("--no-nn", action="store_true",
                        help="No usar la red (búsqueda clásica pura)")
    parser.add_argument("--hash", type=int, default=64,
                        help="Tamaño de la tabla transpositoria (MB)")
    args = parser.parse_args()

    engine = FenixUCIEngine(
        checkpoint_path=args.checkpoint,
        mcts_sims=args.sims,
        device=args.device,
        search_type=args.search_type,
        use_nn=not args.no_nn,
        hash_mb=args.hash,
    )
    engine.run()


if __name__ == "__main__":
    main()
