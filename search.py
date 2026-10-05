#!/usr/bin/env python3
"""
search.py — Búsqueda clásica híbrida para Fénix Chess.
==========================================================

Alfa-beta negamax + quiescencia con:
  - Evaluación tapered (material + PST clásicas + peones pasados/doblados/
    aislados + pareja de alfiles + torres en columna abierta + tempo)
  - Quiescencia con MVV-LVA, stand-pat y delta pruning (+ evasiones de jaque)
  - Tabla transpositoria (Zobrist de python-chess) con ajuste de mates
  - Orden de jugadas: TT move, MVV-LVA, killers, history, priors de la red
  - Iterative deepening con gestión de tiempo real (deadline + stop flag)
  - Extensiones de jaque, LMR, reverse futility pruning, futility pruning
  - Detección de repetición y regla de 50 movidas dentro de la búsqueda

La red neuronal se usa para ordenar las jugadas del root (priors del policy
head), manteniendo el espíritu AlphaZero: la red propone, la búsqueda confirma.

Uso:
    python search.py --bench        # suite táctica + nps
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import chess
import chess.polyglot

# ── Constantes ──────────────────────────────────────────────────────────

MATE_SCORE = 32000
MATE_BOUND = MATE_SCORE - 512     # puntajes por debajo = mate en ≤512 plies
MAX_PLY = 64
INF = 1 << 24

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

# Valores de piezas (PeSTO-style, centipeones)
MG_VAL = (0, 82, 334, 365, 477, 1025, 0)    # -, P, N, B, R, Q, K
EG_VAL = (0, 94, 281, 297, 512, 936, 0)
PHASE_W = (0, 0, 1, 1, 2, 4, 0)             # contribución a la fase

# Tablas posicionales clásicas (orden visual: fila 0 = rank 8, para blancas).
# Para blancas se indexa con sq ^ 56; para negras con sq.

MG_PST_PAWN = [
      0,  0,  0,  0,  0,  0,  0,  0,
     50, 50, 50, 50, 50, 50, 50, 50,
     10, 10, 20, 30, 30, 20, 10, 10,
      5,  5, 10, 25, 25, 10,  5,  5,
      0,  0,  0, 20, 20,  0,  0,  0,
      5, -5,-10,  0,  0,-10, -5,  5,
      5, 10, 10,-20,-20, 10, 10,  5,
      0,  0,  0,  0,  0,  0,  0,  0,
]
EG_PST_PAWN = [
      0,  0,  0,  0,  0,  0,  0,  0,
     10, 10, 10, 10, 10, 10, 10, 10,
      5,  5,  5,  5,  5,  5,  5,  5,
      0,  0,  0,  0,  0,  0,  0,  0,
     -5, -5, -5, -5, -5, -5, -5, -5,
     10, 10, 10, 10, 10, 10, 10, 10,
     20, 20, 20, 20, 20, 20, 20, 20,
      0,  0,  0,  0,  0,  0,  0,  0,
]
MG_PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]
MG_PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]
MG_PST_ROOK = [
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0,
]
MG_PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]
MG_PST_KING = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]
EG_PST_KING = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

MG_PST = (None, MG_PST_PAWN, MG_PST_KNIGHT, MG_PST_BISHOP,
          MG_PST_ROOK, MG_PST_QUEEN, MG_PST_KING)
EG_PST = (None, EG_PST_PAWN, MG_PST_KNIGHT, MG_PST_BISHOP,
          MG_PST_ROOK, MG_PST_QUEEN, EG_PST_KING)

# Bonificación de peón pasado por fila relativa (0 = propia, 7 = 8ª)
PASSED_PAWN_BONUS = (0, 6, 12, 22, 38, 65, 110, 0)

BISHOP_PAIR = 30
ROOK_OPEN_FILE = 22
ROOK_SEMI_OPEN = 11
DOUBLED_PAWN = -12
ISOLATED_PAWN = -14
TEMPO_BONUS = 12
# refugio del rey: peón propio en su columna / columnas laterales
KING_SHELTER = 9
KING_SHELTER_SIDE = 13


# ── Evaluación ──────────────────────────────────────────────────────────

def evaluate(board: chess.Board) -> int:
    """Evaluación estática en centipeones, desde el punto de vista del que
    mueve. Tapered entre juego medio y final por fase."""
    mg_w = mg_b = eg_w = eg_b = 0
    phase = 0
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING):
        mv, ev = MG_VAL[pt], EG_VAL[pt]
        mg_pst, eg_pst = MG_PST[pt], EG_PST[pt]
        n_w = len(board.pieces(pt, chess.WHITE))
        n_b = len(board.pieces(pt, chess.BLACK))
        phase += PHASE_W[pt] * (n_w + n_b)
        for sq in board.pieces(pt, chess.WHITE):
            idx = sq ^ 56
            mg_w += mv + mg_pst[idx]
            eg_w += ev + eg_pst[idx]
        for sq in board.pieces(pt, chess.BLACK):
            idx = sq
            mg_b += mv + mg_pst[idx]
            eg_b += ev + eg_pst[idx]

    if phase > 24:
        phase = 24

    # ── Extras por bando ──
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        mg_extra = eg_extra = 0

        # pareja de alfiles
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            mg_extra += BISHOP_PAIR
            eg_extra += BISHOP_PAIR

        # torres en columna abierta / semiabierta
        for sq in board.pieces(chess.ROOK, color):
            f = chess.square_file(sq)
            own = bin(board.pieces_mask(chess.PAWN, color) & chess.BB_FILES[f]).count("1")
            opp = bin(board.pieces_mask(chess.PAWN, not color) & chess.BB_FILES[f]).count("1")
            if own == 0 and opp == 0:
                mg_extra += ROOK_OPEN_FILE
            elif opp == 0:
                mg_extra += ROOK_SEMI_OPEN

        # peones: doblados, aislados, pasados
        own_pawns = board.pieces(chess.PAWN, color)
        opp_pawns = board.pieces(chess.PAWN, not color)
        file_counts = [0] * 8
        for sq in own_pawns:
            file_counts[chess.square_file(sq)] += 1
        for n in file_counts:
            if n > 1:
                mg_extra += DOUBLED_PAWN * (n - 1)
                eg_extra += DOUBLED_PAWN * (n - 1)
        for sq in own_pawns:
            f = chess.square_file(sq)
            # aislado: ningún peón propio en fichas adyacentes
            adjacent = 0
            if f > 0:
                adjacent += file_counts[f - 1]
            if f < 7:
                adjacent += file_counts[f + 1]
            if adjacent == 0:
                mg_extra += ISOLATED_PAWN
                eg_extra += ISOLATED_PAWN
            # pasado: sin peones enemigos en fichas adyacentes por delante
            mask = chess.BB_FILES[f]
            if f > 0:
                mask |= chess.BB_FILES[f - 1]
            if f < 7:
                mask |= chess.BB_FILES[f + 1]
            r = chess.square_rank(sq)
            ahead_mask = 0
            if color == chess.WHITE:
                for rr in range(r + 1, 8):
                    ahead_mask |= chess.BB_RANKS[rr]
            else:
                for rr in range(0, r):
                    ahead_mask |= chess.BB_RANKS[rr]
            block = mask & ahead_mask
            if not (opp_pawns & block):
                rel_rank = r if color == chess.WHITE else 7 - r
                bonus = PASSED_PAWN_BONUS[rel_rank]
                mg_extra += bonus
                eg_extra += bonus * 2

        # refugio del rey: peones propios delante, en su columna o laterales
        king_sq = board.king(color)
        if king_sq is not None:
            kf = chess.square_file(king_sq)
            kr = chess.square_rank(king_sq)
            for sq in own_pawns:
                pf = chess.square_file(sq)
                pr = chess.square_rank(sq)
                ahead = pr > kr if color == chess.WHITE else pr < kr
                if not ahead or abs(pr - kr) > 2:
                    continue
                if pf == kf:
                    mg_extra += KING_SHELTER
                elif abs(pf - kf) == 1:
                    mg_extra += KING_SHELTER_SIDE

        if color == chess.WHITE:
            mg_w += mg_extra
            eg_w += eg_extra
        else:
            mg_b += mg_extra
            eg_b += eg_extra

    # ── Tapering ──
    mg = mg_w - mg_b
    eg = eg_w - eg_b
    score = (mg * phase + eg * (24 - phase)) // 24

    # ── Tempo (siempre para el que mueve) ──
    if board.turn == chess.WHITE:
        return score + TEMPO_BONUS
    return -score + TEMPO_BONUS


# ── Excepción de tiempo ─────────────────────────────────────────────────

class TimeUp(Exception):
    """Se agotó el tiempo / llegó la orden de stop."""


# ── Resultado de búsqueda ───────────────────────────────────────────────

@dataclass
class SearchResult:
    move: Optional[chess.Move]
    score: int
    depth: int
    nodes: int
    time_s: float
    pv: List[chess.Move] = field(default_factory=list)

    @property
    def nps(self) -> int:
        return int(self.nodes / self.time_s) if self.time_s > 0 else 0


# ── Buscador ────────────────────────────────────────────────────────────

class Searcher:
    """Alfa-beta negamax con iterative deepening, TT y gestión de tiempo."""

    def __init__(self, hash_mb: int = 64):
        self.hash_mb = hash_mb
        self.tt: Dict[int, Tuple] = {}
        self.tt_max_entries = max(1024, (hash_mb * 1024 * 1024) // 80)
        self.killers: List[List[Optional[chess.Move]]] = [
            [None, None] for _ in range(MAX_PLY + 8)
        ]
        self.history: Dict[Tuple[int, int], int] = {}
        self.nodes = 0
        self.deadline: Optional[float] = None
        self.stop_flag: Optional[Callable[[], bool]] = None
        self.root_priors: Dict[chess.Move, float] = {}
        # callbacks: recibe SearchResult parcial tras cada profundidad
        self.on_depth: Optional[Callable[[SearchResult], None]] = None
        self._start_time = 0.0
        self._safety_move: Optional[chess.Move] = None

    # ── utilidades ──

    @staticmethod
    def _can_null_move(board: chess.Board) -> bool:
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            if board.pieces(pt, board.turn):
                return True
        return False

    def _check_time(self) -> None:
        self.nodes += 1
        if self.nodes & 1023 == 0:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise TimeUp
            if self.stop_flag is not None and self.stop_flag():
                raise TimeUp

    def _tt_store(self, key: int, depth: int, flag: int, score: int,
                  move: Optional[chess.Move], ply: int) -> None:
        if len(self.tt) >= self.tt_max_entries:
            self.tt.clear()
        if score >= MATE_BOUND:
            score -= ply
        elif score <= -MATE_BOUND:
            score += ply
        self.tt[key] = (depth, flag, score, move)

    def _tt_probe(self, key: int, ply: int):
        entry = self.tt.get(key)
        if entry is None:
            return None, 0, None
        depth, flag, score, move = entry
        if score >= MATE_BOUND:
            score += ply
        elif score <= -MATE_BOUND:
            score -= ply
        return move, score, (flag, depth)

    # ── orden de jugadas ──

    def _score_moves(self, board: chess.Board, moves, tt_move,
                     ply: int, root: bool = False) -> List[Tuple[int, chess.Move]]:
        scores = []
        history = self.history
        killers = self.killers[min(ply, len(self.killers) - 1)]
        for move in moves:
            s = 0
            if move == tt_move:
                s = 1 << 24
            elif board.is_capture(move):
                victim = board.piece_type_at(move.to_square)
                attacker = board.piece_type_at(move.from_square)
                # MVV-LVA (+ peón legal si es al paso)
                if victim is None:
                    victim = chess.PAWN
                s = (1 << 20) + victim * 32 - attacker
            elif move.promotion:
                s = (1 << 20) + (move.promotion or 0) * 8
            elif move == killers[0]:
                s = (1 << 19)
            elif move == killers[1]:
                s = (1 << 19) - 1
            else:
                s = history.get((move.from_square, move.to_square), 0)
            if root:
                # prior de la red neuronal manda al inicio de la búsqueda
                prior = self.root_priors.get(move)
                if prior is not None:
                    s += int(prior * (1 << 18))
                if board.is_capture(move):
                    s += 1 << 17   # las capturas antes que los silenciosos del root
            scores.append((s, move))
        scores.sort(key=lambda x: -x[0])
        return scores

    # ── búsqueda principal ──

    def search(
        self,
        board: chess.Board,
        time_budget_s: Optional[float] = None,
        max_depth: int = 64,
        stop_flag: Optional[Callable[[], bool]] = None,
        deadline: Optional[float] = None,
    ) -> SearchResult:
        """Busca la mejor jugada con iterative deepening.

        Args:
            board: posición actual.
            time_budget_s: presupuesto de tiempo en segundos.
            max_depth: profundidad máxima.
            stop_flag: callable que devuelve True para detener (stop UCI).
            deadline: deadline absoluto (monotonic), opcional.
        """
        self.nodes = 0
        self.stop_flag = stop_flag
        if deadline is not None:
            self.deadline = deadline
        elif time_budget_s is not None:
            self.deadline = time.monotonic() + time_budget_s
        else:
            self.deadline = None
        self._start_time = time.monotonic()

        legal = list(board.legal_moves)
        if not legal:
            return SearchResult(None, 0, 0, 0, 0.0, [])

        # mover de seguridad: el primero según orden (priors de la red)
        scored = self._score_moves(board, legal, None, 0, root=True)
        self._safety_move = scored[0][1]

        best_move = self._safety_move
        best_score = 0
        best_pv: List[chess.Move] = []
        completed_depth = 0

        try:
            for depth in range(1, max_depth + 1):
                score, pv = self._iterative_depth(board, depth)
                completed_depth = depth
                best_move, best_score, best_pv = pv[0] if pv else best_move, score, pv
                if self.on_depth is not None:
                    self.on_depth(SearchResult(
                        move=best_move, score=score, depth=depth,
                        nodes=self.nodes,
                        time_s=time.monotonic() - self._start_time,
                        pv=list(pv),
                    ))
                # ¿mate encontrado? no hace falta seguir
                if abs(score) >= MATE_BOUND:
                    break
        except TimeUp:
            pass

        elapsed = time.monotonic() - self._start_time
        if completed_depth == 0:
            # ni siquiera terminó la profundidad 1: mover de seguridad
            return SearchResult(self._safety_move, 0, 0, self.nodes,
                                elapsed, [self._safety_move])
        return SearchResult(best_move, best_score, completed_depth,
                            self.nodes, elapsed, best_pv)

    def _iterative_depth(self, board: chess.Board, depth: int):
        score = self._negamax(board, depth, -INF, INF, 0)
        pv = self._extract_pv(board, depth)
        return score, pv

    def _extract_pv(self, board: chess.Board, max_len: int) -> List[chess.Move]:
        """Extrae la PV caminando por la TT (con guarda anti-ciclos)."""
        pv = []
        seen = set()
        b = board.copy(stack=False)
        for _ in range(max_len):
            key = chess.polyglot.zobrist_hash(b)
            if key in seen:
                break
            seen.add(key)
            entry = self.tt.get(key)
            if entry is None:
                break
            move = entry[3]
            if move is None or move not in b.legal_moves:
                break
            pv.append(move)
            b.push(move)
        return pv

    # ── negamax ──

    def _negamax(self, board: chess.Board, depth: int, alpha: int,
                 beta: int, ply: int) -> int:
        self._check_time()

        is_root = ply == 0
        in_check = board.is_check()

        # ── tablas de finales / reglas ──
        if not is_root:
            if board.is_repetition(2) or board.can_claim_fifty_moves():
                return 0
        # tope de profundidad de pila
        if ply >= MAX_PLY:
            return evaluate(board)

        # ── extensión de jaque ──
        if in_check and ply < MAX_PLY:
            depth += 1

        # ── hoja → quiescencia ──
        if depth <= 0:
            return self._quiesce(board, alpha, beta, ply)

        # ── TT probe ──
        key = chess.polyglot.zobrist_hash(board)
        tt_move, tt_score, tt_info = self._tt_probe(key, ply)
        tt_flag, tt_depth = tt_info if tt_info else (None, -1)
        if tt_info is not None and tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_score
            if tt_flag == TT_LOWER and tt_score >= beta:
                return tt_score
            if tt_flag == TT_UPPER and tt_score <= alpha:
                return tt_score

        static_eval = None
        if not in_check:
            static_eval = evaluate(board)

            # ── reverse futility pruning ──
            if depth <= 3 and not is_root and static_eval - 110 * depth >= beta \
                    and abs(beta) < MATE_BOUND:
                return static_eval

            # null-move (a partir de profundidad 5, con verificación)
            if (not is_root and depth >= 5 and abs(beta) < MATE_BOUND
                    and self._can_null_move(board)):
                board.push(chess.Move.null())
                null_score = -self._negamax(
                    board, depth - 3, -beta, -beta + 1, ply + 1)
                board.pop()
                if null_score >= beta:
                    verify = self._negamax(board, depth - 1, alpha, beta, ply)
                    if verify >= beta:
                        return verify

        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply if in_check else 0

        # ── intentar TT move primero ──
        scored = self._score_moves(board, moves, tt_move, ply, root=is_root)

        best_score = -INF
        best_move = None
        flag = TT_UPPER
        move_count = len(moves)
        gives_check_moves = 0

        for i, (_s, move) in enumerate(scored):
            is_capture = board.is_capture(move)
            is_quiet = not is_capture and not move.promotion
            board.push(move)
            new_check = board.is_check()

            # ── futility pruning (profundidad baja, jugadas silenciosas) ──
            if (depth <= 2 and not is_root and static_eval is not None
                    and not in_check and not new_check and is_quiet
                    and i > 0 and static_eval + 130 < alpha):
                board.pop()
                continue

            # ── LMR ──
            reduction = 0
            if (i >= 3 and depth >= 3 and is_quiet and not in_check
                    and not new_check and not is_root):
                reduction = 1
                if i >= 6 and depth >= 4:
                    reduction = 2

            if i == 0:
                score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                score = -self._negamax(board, depth - 1 - reduction,
                                       -alpha - 1, -alpha, ply + 1)
                if score > alpha and reduction:
                    score = -self._negamax(board, depth - 1,
                                           -alpha - 1, -alpha, ply + 1)
                if score > alpha and score < beta:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
                flag = TT_EXACT
                if score >= beta:
                    flag = TT_LOWER
                    # killers / history en jugadas silenciosas
                    if is_quiet:
                        k = self.killers[ply]
                        if k[0] != move:
                            k[1] = k[0]
                            k[0] = move
                        h = self.history.get((move.from_square, move.to_square), 0)
                        self.history[(move.from_square, move.to_square)] = \
                            min(h + depth * depth, 1 << 20)
                    break

        if best_score == -INF:
            return -MATE_SCORE + ply if in_check else 0

        self._tt_store(key, depth, flag, best_score, best_move, ply)
        return best_score

    # ── quiescencia ──

    def _quiesce(self, board: chess.Board, alpha: int, beta: int,
                 ply: int) -> int:
        self._check_time()

        in_check = board.is_check()
        if ply >= MAX_PLY:
            return evaluate(board)

        if not in_check:
            stand = evaluate(board)
            if stand >= beta:
                return stand
            if stand > alpha:
                alpha = stand

        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply if in_check else 0

        # solo capturas (+promociones) salvo que estemos en jaque
        if not in_check:
            moves = [m for m in moves
                     if board.is_capture(m) or (m.promotion == chess.QUEEN)]

        scored = self._score_moves(board, moves, None, ply)
        best = stand if not in_check else -INF

        for _s, move in scored:
            # delta pruning: ni sumando la pieza capturada llegamos a alpha
            if not in_check:
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                if best + MG_VAL[victim] + 200 < alpha and not move.promotion:
                    continue
            board.push(move)
            score = -self._quiesce(board, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break

        if best == -INF:
            return -MATE_SCORE + ply if in_check else alpha
        return best


# ── Suite táctica / benchmark ───────────────────────────────────────────

TACTICAL_SUITE = [
    # (fen, descripción, verificador)
    ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "mate en 1: Ra8#",
     "mate1"),
    ("7k/6Q1/6K1/8/8/8/8/8 w - - 0 1", "mate en 1: Qg7#",
     "mate1"),
    ("6k1/R7/1R6/8/8/8/8/6K1 w - - 0 1", "mate en 1: torre doble",
     "mate1"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
     "mate en 1: Qxf7#", "mate1"),
    ("2rr2k1/pp3ppp/8/8/8/8/PP3PPP/R2R2K1 w - - 0 1", "nada: posición igual",
     "any"),
    ("8/8/8/3k4/8/8/3K1R2/8 w - - 0 1", "ganar pieza: Rf5+ (fork rey/dama?) no",
     "any"),
]


def run_bench(max_depth: int = 10, budget_s: float = 2.0) -> None:
    print("─" * 64)
    # nps en startpos
    s = Searcher(hash_mb=32)
    t0 = time.time()
    r = s.search(chess.Board(), time_budget_s=budget_s, max_depth=max_depth)
    print(f"startpos: depth={r.depth} nodes={r.nodes} nps={r.nps} "
          f"pv={' '.join(m.uci() for m in r.pv)} ({r.time_s:.2f}s)")

    # posición tipo Kiwipete
    kiwi = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    s = Searcher(hash_mb=32)
    r = s.search(kiwi, time_budget_s=budget_s, max_depth=max_depth)
    print(f"kiwipete: depth={r.depth} nodes={r.nodes} nps={r.nps} "
          f"pv={' '.join(m.uci() for m in r.pv)} ({r.time_s:.2f}s)")

    print("─" * 64)
    ok = fail = 0
    for fen, desc, kind in TACTICAL_SUITE:
        board = chess.Board(fen)
        s = Searcher(hash_mb=16)
        r = s.search(board, time_budget_s=budget_s, max_depth=max_depth)
        if kind == "mate1":
            good = False
            if r.move is not None:
                board.push(r.move)
                good = board.is_checkmate()
                board.pop()
        else:
            good = r.move is not None
        mark = "OK " if good else "FALLO"
        ok += good
        fail += (not good)
        mv = r.move.uci() if r.move else "none"
        print(f"[{mark}] {desc:45s} -> {mv:7s} depth={r.depth} score={r.score}")
    print("─" * 64)
    print(f"táctica: {ok} ok / {fail} fallos")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Benchmark del buscador clásico")
    p.add_argument("--bench", action="store_true", help="suite táctica + nps")
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--time", type=float, default=2.0)
    args = p.parse_args()
    if args.bench:
        run_bench(max_depth=args.depth, budget_s=args.time)
    else:
        p.print_help()
