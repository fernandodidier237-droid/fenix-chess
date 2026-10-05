"""
Chess environment for AlphaZero.

Provides:
  - encode_board()        → 8×8×119 tensor of the current position
  - move_to_index()       → integer in [0, 4672) for a given chess.Move
  - index_to_move()       → reverse mapping (for inference)
  - flip_board()          → canonical orientation (current player = white)
"""

import numpy as np
import chess
from typing import List, Tuple, Optional

import config

# ── 8×8 square helpers ──────────────────────────────────────────────────

# 0-63, 0 = a1, 7 = h1, …, 56 = a8, 63 = h8
def _file(sq: int) -> int: return chess.square_file(sq)
def _rank(sq: int) -> int: return chess.square_rank(sq)

# All 8 queen-like direction vectors  (file_delta, rank_delta)
QUEEN_DIRS = [
    (0, 1),   # N
    (1, 1),   # NE
    (1, 0),   # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1), # SW
    (-1, 0),  # W
    (-1, 1),  # NW
]

# Knight offsets
KNIGHT_OFFSETS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

# Promotion piece types (the 9 promotion-underpromotion variants)
PROMO_OFFSETS = [
    (chess.QUEEN, 1, 0),   # forward
    (chess.QUEEN, 1, -1),  # left-capture
    (chess.QUEEN, 1, 1),   # right-capture
    (chess.KNIGHT, 1, 0),
    (chess.KNIGHT, 1, -1),
    (chess.KNIGHT, 1, 1),
    (chess.BISHOP, 1, 0),
    (chess.BISHOP, 1, -1),
    (chess.BISHOP, 1, 1),
]

# Number of move types per from-square
_MOVES_PER_SQUARE = 73  # 56 queen-type + 8 knight + 9 promotions


# ── Build forward and inverse look-up-tables ────────────────────────────

def _build_move_lut() -> Tuple[np.ndarray, dict]:
    """
    Returns:
        forward[from_sq, move_type] → (to_sq, promotion) or None
        reverse dict: (from_sq, to_sq, promo) → move_type
    """
    forward = np.full((64, 73), None, dtype=object)
    rev = {}

    for sq in range(64):
        f, r = _file(sq), _rank(sq)

        # --- Queen-type moves (types 0-55) ---
        for dir_idx, (df, dr) in enumerate(QUEEN_DIRS):
            for step in range(1, 8):  # distance 1..7
                nf, nr = f + df * step, r + dr * step
                if not (0 <= nf < 8 and 0 <= nr < 8):
                    break
                to_sq = chess.square(nf, nr)
                idx = dir_idx * 7 + (step - 1)
                forward[sq, idx] = (to_sq, None)
                rev[(sq, to_sq, None)] = idx

        # --- Knight moves (types 56-63) ---
        for knight_idx, (df, dr) in enumerate(KNIGHT_OFFSETS):
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                to_sq = chess.square(nf, nr)
                idx = 56 + knight_idx
                forward[sq, idx] = (to_sq, None)
                rev[(sq, to_sq, None)] = idx

        # --- Promotions (types 64-72) ---
        for promo_idx, (piece, dr, df_cap) in enumerate(PROMO_OFFSETS):
            nf, nr = f + df_cap, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                to_sq = chess.square(nf, nr)
                idx = 64 + promo_idx
                forward[sq, idx] = (to_sq, piece)
                rev[(sq, to_sq, piece)] = idx

    return forward, rev


MOVE_TO_INDEX_LUT, _ = _build_move_lut()

# The reverse LUT (rev dict from _build_move_lut) is intentionally unused here;
# move_to_index() re-computes indices dynamically from (from_sq, to_sq, promo).

def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess.Move to an integer in [0, 4672).
    The index = from_sq * 73 + move_type.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    promo = move.promotion

    f, r = _file(from_sq), _rank(from_sq)
    tf, tr = _file(to_sq), _rank(to_sq)
    df, dr = tf - f, tr - r

    if promo is not None:
        # It's a promotion — classify by piece type + direction
        for promo_idx, (piece, dr_p, df_cap) in enumerate(PROMO_OFFSETS):
            if piece == promo and dr == dr_p and df == df_cap:
                return from_sq * _MOVES_PER_SQUARE + (64 + promo_idx)
        # Fallback (shouldn't happen)
        return from_sq * _MOVES_PER_SQUARE + 64  # queen promo default

    # Try knight moves first
    for knight_idx, (kdf, kdr) in enumerate(KNIGHT_OFFSETS):
        if df == kdf and dr == kdr:
            return from_sq * _MOVES_PER_SQUARE + (56 + knight_idx)

    # Queen-type moves
    for dir_idx, (qdf, qdr) in enumerate(QUEEN_DIRS):
        if (qdf == 0 and df == 0) or (qdr == 0 and dr == 0) or (abs(df) == abs(dr)):
            # Check direction alignment
            if qdf != 0 and df != 0 and qdr != 0 and dr != 0:
                if df // qdf != dr // qdr:
                    continue
            elif qdf == 0 and df != 0:
                continue
            elif qdr == 0 and dr != 0:
                continue
            # Check it's aligned and step matches
            if qdf == 0 and df != 0:
                continue
            if qdr == 0 and dr != 0:
                continue
            if qdf != 0 and abs(df) % abs(qdf) != 0:
                continue
            if qdr != 0 and abs(dr) % abs(qdr) != 0:
                continue
            if qdf != 0 and qdf * df <= 0 and df != 0:
                continue
            if qdr != 0 and qdr * dr <= 0 and dr != 0:
                continue

            # Compute step distance
            if qdf != 0:
                step = df // qdf
            else:
                step = dr // qdr
            if step < 1 or step > 7:
                continue
            return from_sq * _MOVES_PER_SQUARE + (dir_idx * 7 + (step - 1))

    # Fallback — shouldn't be reached for legal moves
    return from_sq * _MOVES_PER_SQUARE


def index_to_move(board: chess.Board, index: int) -> Optional[chess.Move]:
    """
    Convert a policy index back to a chess.Move.
    Only returns the move if it is legal in the current position.
    """
    from_sq = index // _MOVES_PER_SQUARE
    move_type = index % _MOVES_PER_SQUARE

    entry = MOVE_TO_INDEX_LUT[from_sq, move_type]
    if entry is None:
        return None

    to_sq, promo = entry

    if promo is not None:
        move = chess.Move(from_sq, to_sq, promotion=promo)
    else:
        move = chess.Move(from_sq, to_sq)

    if move in board.legal_moves:
        return move
    return None


def legal_move_mask(board: chess.Board) -> np.ndarray:
    """
    Build a binary mask over the policy output indices for the current
    position: 1 = legal, 0 = illegal.
    """
    mask = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = 1.0
    return mask


# ── Board encoding (8×8×119) ───────────────────────────────────────────

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Encode a chess.Board as a (119, 8, 8) tensor (channel-first for PyTorch).
    Conceptual planes (same as AlphaZero 8×8×119):

      0-111  : 14 planes × 8 history steps (t, t-1, ..., t-7)
        0-5   : current player's pieces (P, N, B, R, Q, K)
        6-11  : opponent's pieces (P, N, B, R, Q, K)
        12    : en-passant target square (1-hot if any)
        13    : repetition count (binary: 0, 1, 2+ → value / 3)
      112    : side to move (1 = black, 0 = white)
      113-116: castling rights (K, Q, k, q)
      117    : no-progress count (fifty-move counter / 100)
      118    : all-ones constant plane

    The board is always oriented from the *current player's* perspective.
    """
    # ── helper: map piece type to plane index inside a 6-plane group ──
    PIECE_PLANE = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                   chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}

    def _put_plane(planes: np.ndarray, channel: int, sq_set):
        """Set plane[channel] to 1 for each square in sq_set."""
        for sq in sq_set:
            r, f = sq // 8, sq % 8   # row, col
            planes[channel, r, f] = 1.0

    # Shape: (channels, height, width) = (119, 8, 8) — channel-first for PyTorch
    planes = np.zeros((119, 8, 8), dtype=np.float32)

    # --- Build history from board's move stack ---
    history_boards = []
    temp_board = board.copy()
    for _ in range(8):
        history_boards.append(temp_board.copy())
        # Pop last move if possible
        if len(temp_board.move_stack) > 0:
            temp_board.pop()
        else:
            break

    # Fill back to 8 frames by repeating the earliest
    while len(history_boards) < 8:
        history_boards.append(history_boards[-1])

    # Reverse so index 0 = most recent, index 7 = oldest
    history_boards = history_boards[::-1]

    for step_idx, hist_board in enumerate(history_boards):
        ch_base = step_idx * 14

        opp_color = not hist_board.turn

        # Current player's pieces (planes 0-5)
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                           chess.ROOK, chess.QUEEN, chess.KING]:
            sqs = list(hist_board.pieces(piece_type, hist_board.turn))
            _put_plane(planes, ch_base + PIECE_PLANE[piece_type], sqs)

        # Opponent's pieces (planes 6-11)
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                           chess.ROOK, chess.QUEEN, chess.KING]:
            sqs = list(hist_board.pieces(piece_type, opp_color))
            _put_plane(planes, ch_base + 6 + PIECE_PLANE[piece_type], sqs)

        # En-passant square (plane 12)
        if hist_board.ep_square is not None:
            r, f = hist_board.ep_square // 8, hist_board.ep_square % 8
            planes[ch_base + 12, r, f] = 1.0

        # Repetition count (plane 13) — how many times this position
        # has occurred in the current game, capped at 3, normalized.
        rep = 0
        for _hist in history_boards[:step_idx + 1]:
            if _hist.fen().split(' ')[:2] == hist_board.fen().split(' ')[:2]:
                rep += 1
            if rep >= 3:
                break
        planes[ch_base + 13, :, :] = min(rep, 3) / 3.0

    # ── Constant planes (112-118) ────────────────────────────────────
    # 112: side to move
    planes[112, :, :] = 1.0 if board.turn == chess.BLACK else 0.0

    # 113-116: castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        planes[113, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[114, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[115, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[116, :, :] = 1.0

    # 117: no-progress count / 100
    planes[117, :, :] = board.halfmove_clock / 100.0

    # 118: all-ones
    planes[118, :, :] = 1.0

    return planes


# ── Canonical flip ──────────────────────────────────────────────────────

def flip_board(tensor: np.ndarray) -> np.ndarray:
    """
    Flip the board tensor so that it's always from white's perspective
    (used for training where we store examples in canonical form).
    The board is flipped vertically and the piece planes are swapped
    (current <-> opponent) for every history frame, and the side-to-move
    plane is set to 0.
    """
    flipped = tensor.copy()
    # Flip vertically (up-down) — axis 1 is height for (C, H, W) format
    flipped = flipped[:, ::-1, :]

    for step in range(8):
        base = step * 14
        # Swap current player (0-5) <-> opponent (6-11)
        flipped[base:base+6, :, :], flipped[base+6:base+12, :, :] = \
            flipped[base+6:base+12, :, :].copy(), flipped[base:base+6, :, :].copy()

    # Side to move → 0 (white)
    flipped[112, :, :] = 0.0
    return flipped


def encode_board_canonical(board: chess.Board) -> np.ndarray:
    """
    Encode and flip so the board is always from white's perspective.
    """
    tensor = encode_board(board)
    if board.turn == chess.BLACK:
        tensor = flip_board(tensor)
    return tensor


# ── API compatibility alias ─────────────────────────────────────────────
class ChessEnv:
    """Thin wrapper around the environment functions (compatibility with
    the example code from the AlphaZero spec)."""
    @staticmethod
    def encode(board: chess.Board) -> np.ndarray:
        return encode_board(board)

    @staticmethod
    def encode_canonical(board: chess.Board) -> np.ndarray:
        return encode_board_canonical(board)

    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        return move_to_index(move)

    @staticmethod
    def index_to_move(board: chess.Board, idx: int) -> Optional[chess.Move]:
        return index_to_move(board, idx)

    @staticmethod
    def legal_move_mask(board: chess.Board) -> np.ndarray:
        return legal_move_mask(board)
