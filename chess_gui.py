#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║       AlphaZero Chess — Web GUI                                         ║
║                                                                         ║
║  Interfaz web para jugar al ajedrez contra AlphaZero o Stockfish.       ║
║                                                                         ║
║  Usage:                                                                 ║
║    python chess_gui.py                                                  ║
║    # Abre http://localhost:8765 en tu navegador                         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import time
import json
import logging
from typing import Optional
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess
import numpy as np
import config
import chess_env

# ── Import chess engines ────────────────────────────────────────────────
from stockfish_integration import StockfishEvaluator, find_stockfish

# ── Estimated ELO for the untrained AlphaZero model ─────────────────────
# The model was trained for only 1 iteration with 5 self-play games.
# This is essentially a random network. Estimated ELO: ~500 (novice).
# As training progresses, this can reach 2000+.
AZ_ESTIMATED_ELO = 500

# ── FastAPI + uvicorn ───────────────────────────────────────────────────
try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:
    print("Error: FastAPI/uvicorn not installed. Install with:")
    print("  pip install fastapi uvicorn")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Global engine state ─────────────────────────────────────────────────
az_model = None
az_device = "cpu"
sf_engine = None


def load_alpha_zero(checkpoint_path: str, device: str = "cpu"):
    """Load the trained AlphaZero model."""
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


def alpha_zero_move(model, board_fen: str, mcts_sims: int = 200) -> dict:
    """Get the best move from AlphaZero using MCTS."""
    from mcts import MCTS
    board = chess.Board(board_fen)
    
    if board.is_game_over():
        return {"move": None, "info": "Game is over"}
    
    mcts = MCTS(
        model=model,
        num_simulations=mcts_sims,
        c_puct=config.C_PUCT,
        device=az_device,
    )
    t0 = time.time()
    root = mcts.search(board, add_root_noise=False)
    elapsed = time.time() - t0
    
    best_move, probs = root.get_visit_probs(temperature=config.TEMPERATURE_TAU)
    
    # Top moves
    top_moves = []
    for child in sorted(root.children, key=lambda c: c.visit_count, reverse=True)[:5]:
        pct = child.visit_count / max(root.visit_count, 1) * 100
        top_moves.append({
            "move": child.action.uci(),
            "visits": child.visit_count,
            "pct": round(pct, 1),
            "value": round(child.value, 3),
        })
    
    return {
        "move": best_move.uci() if best_move else None,
        "time": round(elapsed, 2),
        "sims": mcts_sims,
        "top_moves": top_moves,
    }


def stockfish_move(board_fen: str, time_ms: int = 200) -> dict:
    """Get the best move from Stockfish."""
    global sf_engine
    
    if sf_engine is None:
        sf_path = find_stockfish()
        if not sf_path:
            return {"move": None, "error": "Stockfish not found"}
        sf_engine = StockfishEvaluator(engine_path=sf_path, elo=config.STOCKFISH_ELO)
        sf_engine.start()
    
    board = chess.Board(board_fen)
    if board.is_game_over():
        return {"move": None, "info": "Game is over"}
    
    t0 = time.time()
    move = sf_engine.get_best_move(board, time_ms=time_ms)
    elapsed = time.time() - t0
    
    return {
        "move": move.uci() if move else None,
        "time": round(elapsed, 2),
        "elo": config.STOCKFISH_ELO,
    }


# ── FastAPI app ─────────────────────────────────────────────────────────
checkpoint_path = os.path.join(config.MODEL_DIR, "best_model.pt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global az_model, sf_engine
    
    # Load AlphaZero
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading AlphaZero model from {checkpoint_path}...")
        try:
            az_model = load_alpha_zero(checkpoint_path, az_device)
            params = sum(p.numel() for p in az_model.parameters())
            logger.info(f"  AlphaZero loaded: {params:,} params on {az_device}")
        except Exception as e:
            logger.error(f"  Failed to load AlphaZero model: {e}")
    else:
        logger.warning(f"  Checkpoint not found at {checkpoint_path}")
        logger.warning("  Train the model first or use --checkpoint")
    
    yield
    
    # Cleanup Stockfish
    if sf_engine is not None:
        try:
            sf_engine.stop()
        except Exception:
            pass


app = FastAPI(title="AlphaZero Chess GUI", lifespan=lifespan)


# ── API endpoints ───────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    """Get engine status."""
    return {
        "alpha_zero_loaded": az_model is not None,
        "stockfish_available": True,  # will check dynamically
        "model_checkpoint": os.path.basename(checkpoint_path) if os.path.exists(checkpoint_path) else None,
        "device": az_device,
    }


@app.post("/api/move")
async def api_move(request: Request):
    """Compute the best move for a given position."""
    data = await request.json()
    fen = data.get("fen", chess.STARTING_FEN)
    opponent = data.get("opponent", "alpha-zero")
    sims = data.get("sims", 200)
    
    board = chess.Board(fen)
    if board.is_game_over():
        outcome = board.outcome()
        result = None
        if outcome:
            if outcome.winner is None:
                result = "1/2-1/2"
            elif outcome.winner == chess.WHITE:
                result = "1-0"
            else:
                result = "0-1"
        return {
            "move": None,
            "game_over": True,
            "result": result,
            "termination": outcome.termination.name if outcome else None,
        }
    
    if opponent == "stockfish":
        result = stockfish_move(fen)
    else:
        if az_model is None:
            return {"move": None, "error": "AlphaZero model no cargado. Entrena el modelo primero con: python main.py --iterations 1 --self-play-games 5"}
        result = alpha_zero_move(az_model, fen, mcts_sims=sims)
    
    return result


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML_PAGE)


# ── HTML Frontend ───────────────────────────────────────────────────────

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AlphaZero Chess</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.css">
<style>
  :root {
    --bg: #1a1a2e;
    --surface: #16213e;
    --surface2: #0f3460;
    --accent: #e94560;
    --text: #eaeaea;
    --text-dim: #8892b0;
    --green: #64ffda;
    --gold: #ffd700;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
  }

  h1 {
    font-size: 1.8rem;
    margin-bottom: 4px;
    background: linear-gradient(135deg, var(--accent), var(--gold));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
  }

  .subtitle {
    color: var(--text-dim);
    font-size: 0.85rem;
    margin-bottom: 20px;
  }

  .container {
    display: flex;
    gap: 30px;
    align-items: flex-start;
    flex-wrap: wrap;
    justify-content: center;
    max-width: 1000px;
  }

  .board-section {
    flex-shrink: 0;
  }

  #board {
    width: 480px;
    height: 480px;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }

  .controls {
    background: var(--surface);
    border-radius: 12px;
    padding: 24px;
    min-width: 280px;
    max-width: 320px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }

  .controls h2 {
    font-size: 1.1rem;
    color: var(--text);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--surface2);
  }

  .control-group {
    margin-bottom: 16px;
  }

  .control-group label {
    display: block;
    font-size: 0.8rem;
    color: var(--text-dim);
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  select, input {
    width: 100%;
    padding: 10px 14px;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--surface2);
    border-radius: 8px;
    font-size: 0.9rem;
    outline: none;
    transition: border-color 0.2s;
    cursor: pointer;
  }

  select:hover, input:hover { border-color: var(--accent); }
  select:focus, input:focus { border-color: var(--green); }

  .btn {
    width: 100%;
    padding: 12px;
    border: none;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .btn-primary {
    background: linear-gradient(135deg, var(--accent), #c23152);
    color: white;
  }
  .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(233,69,96,0.4); }

  .btn-secondary {
    background: var(--surface2);
    color: var(--text);
    margin-top: 8px;
  }
  .btn-secondary:hover { background: #123a6e; }

  .status-box {
    background: var(--bg);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 16px 0;
    min-height: 48px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.9rem;
    border: 1px solid var(--surface2);
  }

  .status-box .indicator {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .indicator.thinking { background: var(--gold); animation: pulse 0.8s infinite; }
  .indicator.idle { background: var(--green); }
  .indicator.error { background: var(--accent); }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .move-history {
    background: var(--bg);
    border-radius: 8px;
    padding: 12px;
    max-height: 160px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 0.8rem;
    border: 1px solid var(--surface2);
    margin-top: 12px;
  }

  .move-history::-webkit-scrollbar { width: 4px; }
  .move-history::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 2px; }

  .move-entry {
    display: inline;
    padding: 2px 4px;
    margin: 1px;
    border-radius: 3px;
  }
  .move-entry:hover { background: var(--surface2); }
  .move-entry.white-move { color: var(--green); }
  .move-entry.black-move { color: var(--accent); }
  .move-entry.last-move { background: rgba(233,69,96,0.2); font-weight: bold; }

  .score-display {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--surface2);
    margin-bottom: 12px;
  }

  .score-item {
    text-align: center;
  }
  .score-item .label { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; }
  .score-item .value { font-size: 1.1rem; font-weight: 700; }
  .score-item .value.az-score { color: var(--accent); }
  .score-item .value.sf-score { color: var(--gold); }

  .top-moves {
    margin-top: 12px;
    padding: 8px;
    background: var(--bg);
    border-radius: 6px;
    border: 1px solid var(--surface2);
    display: none;
  }
  .top-moves.visible { display: block; }
  .top-moves .title { font-size: 0.7rem; color: var(--text-dim); margin-bottom: 4px; text-transform: uppercase; }
  .top-move-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    padding: 2px 4px;
    border-radius: 3px;
  }
  .top-move-row:hover { background: var(--surface2); }
  .top-move-row .move { font-weight: 600; color: var(--text); }
  .top-move-row .stat { color: var(--text-dim); }

  .button-row {
    display: flex;
    gap: 8px;
    margin-top: 8px;
  }
  .button-row .btn { flex: 1; }

  @media (max-width: 860px) {
    #board { width: 360px; height: 360px; }
    .controls { min-width: 240px; }
  }
  @media (max-width: 680px) {
    #board { width: 300px; height: 300px; }
    .controls { min-width: 100%; }
  }
</style>
</head>
<body>

<h1>♚ AlphaZero Chess</h1>
<p class="subtitle">Juega contra la IA o contra Stockfish</p>

<div class="container">
  <div class="board-section">
    <div id="board"></div>
  </div>

  <div class="controls">
    <h2>⚙️ Partida</h2>

    <div class="control-group">
      <label>Jugar contra</label>
      <select id="opponent-select">
        <option value="alpha-zero">🧠 AlphaZero (~500 ELO - novato)</option>
        <option value="stockfish">🤖 Stockfish (ELO 2800)</option>
      </select>
    </div>

    <div class="control-group">
      <label>Jugar como</label>
      <select id="color-select">
        <option value="white">♔ Blancas</option>
        <option value="black">♚ Negras</option>
      </select>
    </div>

    <div class="control-group">
      <label>Fuerza AlphaZero (simulaciones MCTS)</label>
      <select id="sims-select">
        <option value="25">⚡ Muy rápido (25 sims - CPU)</option>
        <option value="50">⚡ Rápido (50 sims)</option>
        <option value="200" selected>🎯 Normal (200 sims)</option>
        <option value="400">💪 Fuerte (400 sims)</option>
        <option value="800">🏆 Máximo (800 sims - GPU)</option>
      </select>
      <div style="margin-top:6px;font-size:0.75rem;color:var(--text-dim);">
        ⚠️ En CPU, &gt;200 sims puede ser muy lento por jugada
      </div>
    </div>

    <div class="status-box" id="status-box">
      <span class="indicator idle" id="status-indicator"></span>
      <span id="status-text">Listo para jugar</span>
    </div>

    <div id="top-moves-panel" class="top-moves">
      <div class="title">AlphaZero evalúa:</div>
      <div id="top-moves-list"></div>
    </div>

    <button class="btn btn-primary" onclick="newGame()">🔄 Nueva Partida</button>
    <div class="button-row">
      <button class="btn btn-secondary" onclick="flipBoard()">🔄 Girar tablero</button>
      <button class="btn btn-secondary" onclick="undoMove()">↩ Deshacer</button>
    </div>

    <h2 style="margin-top: 20px;">📋 Historial</h2>
    <div class="move-history" id="move-history">
      <span style="color: var(--text-dim);">Esperando jugadas...</span>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/chessboard-js/1.0.0/chessboard-1.0.0.min.js"></script>
<script>
// ── State ──────────────────────────────────────────────────────────────
const game = new Chess();
let board = null;
let playerColor = 'white';
let opponent = 'alpha-zero';
let sims = 200;
let isAIThinking = false;
let moveHistory = [];
let moveCount = 0;

// ── Board config ───────────────────────────────────────────────────────
const boardConfig = {
  draggable: true,
  position: 'start',
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
  onDrop: onPieceDrop,
  onDragStart: onDragStart,
  onMouseoutSquare: onMouseoutSquare,
  onSnapEnd: onSnapEnd,
  showNotation: true,
};

function onDragStart(source, piece) {
  if (isAIThinking) return false;
  if (game.game_over()) return false;
  if ((playerColor === 'white' && piece.search(/^b/) !== -1) ||
      (playerColor === 'black' && piece.search(/^w/) !== -1)) {
    return false;
  }
}

function onMouseoutSquare() {}

function onSnapEnd() {
  board.position(game.fen());
}

async function onPieceDrop(source, target) {
  if (isAIThinking) return 'snapback';
  
  const move = game.move({
    from: source,
    to: target,
    promotion: 'q',
  });

  if (move === null) return 'snapback';

  addMoveToHistory(move);
  board.position(game.fen());
  checkGameEnd();

  if (!game.game_over()) {
    await makeAIMove();
  }
}

// ── AI Move ────────────────────────────────────────────────────────────
async function makeAIMove() {
  if (game.game_over()) return;
  
  isAIThinking = true;
  setStatus('Pensando...', 'thinking');
  document.getElementById('top-moves-panel').classList.remove('visible');

  try {
    const response = await fetch('/api/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        fen: game.fen(),
        opponent: opponent,
        sims: sims,
      }),
    });

    const data = await response.json();
    
    if (data.error) {
      setStatus('Error: ' + data.error, 'error');
      isAIThinking = false;
      return;
    }

    if (data.game_over) {
      setStatus(`Partida terminada: ${data.result || data.termination}`, 'idle');
      isAIThinking = false;
      return;
    }

    if (data.move) {
      const aiMove = game.move(data.move, { sloppy: true });
      board.position(game.fen());
      addMoveToHistory(aiMove);
      
      let statusText = `IA jugó ${data.move}`;
      if (data.time) statusText += ` (${data.time}s)`;
      if (data.sims) statusText += ` · ${data.sims} sims`;
      if (data.elo) statusText += ` · ELO ${data.elo}`;
      setStatus(statusText, 'idle');

      // Show top moves from AlphaZero analysis
      if (data.top_moves && data.top_moves.length > 0) {
        const panel = document.getElementById('top-moves-panel');
        const list = document.getElementById('top-moves-list');
        list.innerHTML = data.top_moves.map(m => 
          `<div class="top-move-row">
            <span class="move">${m.move}</span>
            <span class="stat">${m.pct}% · v=${m.value}</span>
          </div>`
        ).join('');
        panel.classList.add('visible');
      }

      checkGameEnd();
    } else {
      setStatus('Error: La IA no devolvió una jugada', 'error');
    }
  } catch (err) {
    setStatus('Error de conexión: ' + err.message, 'error');
    // If the request timed out, suggest lower sims
    if (err.message && err.message.includes('timeout')) {
      setStatus('⏱ La jugada tardó demasiado. Prueba con menos simulaciones MCTS (50-100)', 'error');
    }
  }

  isAIThinking = false;
}

// ── UI helpers ─────────────────────────────────────────────────────────
function setStatus(text, type) {
  document.getElementById('status-text').textContent = text;
  const indicator = document.getElementById('status-indicator');
  indicator.className = 'indicator ' + (type || 'idle');
}

function addMoveToHistory(move) {
  moveCount++;
  moveHistory.push(move);

  const historyDiv = document.getElementById('move-history');
  if (moveCount === 1) historyDiv.innerHTML = '';

  const entry = document.createElement('span');
  entry.className = `move-entry ${move.color === 'w' ? 'white-move' : 'black-move'}`;

  if (move.color === 'w') {
    const moveNum = Math.ceil(moveCount / 2);
    entry.textContent = `${moveNum}. ${move.san} `;
  } else {
    entry.textContent = `${move.san} `;
  }

  // Remove last-move highlight from previous entries
  historyDiv.querySelectorAll('.last-move').forEach(e => e.classList.remove('last-move'));
  entry.classList.add('last-move');

  historyDiv.appendChild(entry);
  historyDiv.scrollTop = historyDiv.scrollHeight;
}

function checkGameEnd() {
  if (game.game_over()) {
    let msg = 'Partida terminada';
    if (game.in_checkmate()) msg += ' — ¡Jaque mate!';
    else if (game.in_stalemate()) msg += ' — Ahogado';
    else if (game.in_draw()) msg += ' — Tablas';
    else if (game.in_threefold_repetition()) msg += ' — Repetición';
    else if (game.insufficient_material()) msg += ' — Material insuficiente';
    setStatus(msg, 'idle');
  }
}

function newGame() {
  game.reset();
  board.start();
  moveHistory = [];
  moveCount = 0;
  document.getElementById('move-history').innerHTML = 
    '<span style="color: var(--text-dim);">Esperando jugadas...</span>';
  document.getElementById('top-moves-panel').classList.remove('visible');
  isAIThinking = false;
  setStatus('Nueva partida', 'idle');

  playerColor = document.getElementById('color-select').value;
  opponent = document.getElementById('opponent-select').value;
  sims = parseInt(document.getElementById('sims-select').value);

  if (playerColor === 'black') {
    board.orientation('black');
    setTimeout(() => makeAIMove(), 500);
  } else {
    board.orientation('white');
  }
}

function flipBoard() {
  board.flip();
}

async function undoMove() {
  if (isAIThinking) return;
  if (moveHistory.length < 2) return;

  // Undo both AI and player move
  game.undo();
  game.undo();
  moveHistory.pop();
  moveHistory.pop();
  moveCount -= 2;
  board.position(game.fen());
  
  // Rebuild history display
  const historyDiv = document.getElementById('move-history');
  historyDiv.innerHTML = '';
  if (moveHistory.length === 0) {
    historyDiv.innerHTML = '<span style="color: var(--text-dim);">Esperando jugadas...</span>';
  } else {
    moveHistory.forEach((m, i) => {
      const entry = document.createElement('span');
      entry.className = `move-entry ${m.color === 'w' ? 'white-move' : 'black-move'}`;
      if (m.color === 'w') {
        const moveNum = Math.floor(i / 2) + 1;
        entry.textContent = `${moveNum}. ${m.san} `;
      } else {
        entry.textContent = `${m.san} `;
      }
      historyDiv.appendChild(entry);
    });
  }
  
  document.getElementById('top-moves-panel').classList.remove('visible');
  checkGameEnd();
  setStatus('Jugada deshecha', 'idle');
}

// ── Start ──────────────────────────────────────────────────────────────
window.onload = function() {
  board = Chessboard('board', boardConfig);

  document.getElementById('opponent-select').addEventListener('change', function() {
    opponent = this.value;
  });
  document.getElementById('color-select').addEventListener('change', function() {
    playerColor = this.value;
  });
  document.getElementById('sims-select').addEventListener('change', function() {
    sims = parseInt(this.value);
  });
};
</script>
</body>
</html>
"""


# ── Run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="AlphaZero Chess Web GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=checkpoint_path,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--sims", type=int, default=200,
                        help="Default MCTS simulations per move")
    parser.add_argument("--port", type=int, default=8765,
                        help="Server port")
    parser.add_argument("--device", default="cpu",
                        help="Device for AlphaZero inference")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    az_device = args.device

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║              AlphaZero Chess — Web GUI                          ║
║                                                                 ║
║  Abre en tu navegador:                                          ║
║    →  http://localhost:{args.port}                               ║
║                                                                 ║
║  Juega contra AlphaZero o Stockfish desde el navegador.         ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
