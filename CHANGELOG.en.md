# Changelog — Fénix Chess

Iteration history of the engine, in real development order.
English version · [Spanish version](CHANGELOG.md)

> **Naming note**: as of 1.7 the engine is called **Fénix Chess**
> (`FenixChess-Hybrid` over UCI, entries `FenixChess` and `FenixChess-MCTS`
> in PyChess). The project is renamed because “AlphaZero” is already a DeepMind
> trademark. This engine is not an AlphaZero implementation: it is a classical
> alpha-beta searcher with priors from its own network, inspired by the
> self-play + network + search idea.

## 1.7 — verified null-move + more aggressive LMR

Small change in `search.py` (+21 / −2 lines), no new UCI option.

- **Null-move pruning** from `depth >= 5`, reduction 3, with verification
  (`_can_null_move`, static): never applied in check, in positions prone to
  zugzwang, or in mate nodes.
- **More aggressive LMR**: threshold `i >= 3` and reduction 2 at `i >= 6`
  (previously `i >= 6` and reduction 1).

### Measured ladder vs Stockfish 16 (4 games per level, 1 s per move for us)

| Level | v1.5 | v1.6 | **v1.7** |
|---|---|---|---|
| `UCI_Elo 1350` | 6-0-0 | 4-0-0 | **3-0-1 (+3.0/4)** |
| `UCI_Elo 1600` | 3-1-0 (+2.0/4) | 3-1-0 (+2.0/4) | **3-0-1 (+2.0/4)** |
| `UCI_Elo 1800` | 0-1-3 (−3/4) | 2-0-2 (0,0/4) | **2-0-2 (0.0/4)** |
| Full strength (0.5 s) | 0-0-1 | 0-0-2 | **0-0-2** |

### Head-to-head A/B v1.6 vs v1.7 (8 games, 1 s per move, alternating colours)

**4.0/8 – 4.0/8: a draw.** The outcome flips exactly with the colour (whoever
plays Black wins), so the strength difference between 1.6 and 1.7 is
indistinguishable from noise. Hence this ships as **1.7**, not 2.0: same
criterion as 1.6, no demonstrable jump in strength.

### Efficiency (nodes at fixed depth)

| Measurement | v1.6 | v1.7 | Δ |
|---|---|---|---|
| Nodes, depth 7 (4 positions) | 239,527 | 238,566 | **−0.4 %** |
| Nodes, depth 7 (startpos) | 212,706 | 220,882 | +3.8 % |
| Nodes, depth 8 (startpos) | 366,285 | 450,152 | +22.9 % |
| Time to depth 8 (startpos) | 4.16 s | 4.34 s | +4.4 % |
| NPS | ~27k | ~27k | 0 % |

Honest conclusion: **v1.7 is not more efficient than v1.6**; the only thing it
adds over it is the 1800 ladder (0.0/4 vs −3/4 for v1.5). Ablations
implemented, measured and discarded: extra mobility (−12 % nodes but loses a
ply: `d6` instead of `d7`), PV-first ordering after the transposition table
(+12 % nodes), null-move only (+5 % at depth 7, −29 % time to depth 8) and
LMR only (−6 % nodes).

## 1.0 — Original engine (starting point)

- `alpha_zero_engine.py` UCI + pure MCTS on top of the trained network
  (1 iteration × 10 self-play games, 100 sims, 52 min on CPU).
- Diagnostics when measured:
  - **0 wins, 0 draws, 4 losses vs Stockfish (UCI_Elo 1350)**.
  - Value head unusable: returns ≈0 with an extra queen (`+0.0` instead of
    ≈+0.9); weak policy (proposes `g2g4` in the top-3 of the opening).
  - Slow search: 114 sims/s (encode 1.65 ms + forward 6.4 ms per sim, no
    batching); no real time management (the `sims` value was lowered
    **permanently** after a fast move); `stop` was not handled (blocking
    search inside the UCI thread).

## 1.1 — `search.py`: classic hybrid searcher (alpha-beta)

New module: the network proposes (policy-head priors order the root moves),
classic search confirms.

- Alpha-beta negamax + quiescence with MVV-LVA, stand-pat and delta pruning.
- Tapered evaluation: material (PeSTO values) + classic PSTs + passed /
  doubled / isolated pawns + bishop pair + open files + tempo.
- Transposition table (python-chess Zobrist) with mate-score adjustment.
- Iterative deepening with deadline + stop flag; killers, history, LMR,
  reverse/futility pruning, check extensions, repetitions and the 50-move rule.
- Built-in tactics suite: `python search.py --bench`.

Fixed during that iteration:
- `Move.is_capture()` does not exist in python-chess → `board.is_capture(move)`.
- Futility: `pop()` without `continue` → empty stack (`IndexError`).
- Delta pruning had no effect (`alpha + v + 200 < alpha` is never true) → it
  now uses stand-pat.
- Tempo was applied to Black incorrectly; dead code in `evaluate`; stack limit
  (`ply >= MAX_PLY`) so check extensions cannot overflow the recursion.
- API bug: `board.pawns(...)` does not exist → `pieces_mask`.

Result: **depth 7 in 2 s (29k nps), 6/6 tactics** (mate-in-1s, tactics).

## 1.2 — UCI engine v2 (`alpha_zero_engine.py` rewritten)

- Search on a **separate thread**: the UCI loop really handles `stop`/`quit`
  (it used to block; PyChess/CuteChess stop analysis with `stop`).
- Real time management: `movetime`, `wtime/btime + inc + movestogo` with a
  budget (85 % of the time per move, capped at half of the remaining time and
  a safety margin so we never lose on time); `mcts_sims` is never mutated.
- `info depth/score cp|mate/nodes/nps/time/pv` lines for the GUIs.
- UCI option `SearchType = Hybrid | MCTS` (default Hybrid), `UseNN`,
  `Hash`, `MCTS_Simulations`.
- Network priors at the root (one evaluation per move); if the model fails to
  load, it keeps working as a pure classical engine.
- Clean output: all debug goes to stderr (it used to pollute stdout with
  lines before the handshake).

Fixed during that iteration:
- The budget did not subtract the time spent computing the priors.
- Model not preloaded during the handshake (first move paid ~0.9 s extra) →
  preload in `run()`.

Verified: full smoke test with python-chess (handshake, movetime, clocks,
10 incremental moves, `go infinite` + `stop`/analysis).

## 1.2.1 — `mcts.py`: deadline + stop (MCTS mode)

- Optional and backward compatible `MCTS.search(..., deadline=, stop_flag=)`
  (checked every 16 sims), so pure AlphaZero mode also respects time and
  `stop`.
- Verified: MCTS mode answers `bestmove` within the budget.

## 1.3 — `play_match.py` v2: real UCI match

- Our engine runs **as a UCI process** (exactly the path PyChess uses), not
  in-process MCTS.
- Flags: `--engine Hybrid|MCTS`, `--az-time`, `--sf-time-ms`, `--elo`,
  `--full-strength`, `--games`; running score per game.
- Fixed: the Stockfish label showed a fixed `ELO 2800` (config value) instead
  of the real Elo of the match.

Measured results (1 s/move for us, 200 ms for SF):

| Version | Opponent | Score |
|---|---|---|
| 1.0 (before) | Stockfish UCI_Elo 1350 | **0W – 0D – 4L** |
| 1.4 (then) | Stockfish UCI_Elo 1350 | **6W – 0D – 0L** |
| 1.4 (then) | Stockfish UCI_Elo 1600 | **3W – 1D – 0L** (+2/4) |
| 1.4 (then) | Stockfish UCI_Elo 1800 | **0W – 1D – 3L** (−3/4) |
| 1.4 (then) | Stockfish full strength (0.5 s) | 0W – 0D – 1L (−1/1, expected) |

Estimated strength of the hybrid engine: **≈1650–1750** (it beats Stockfish
limited to 1600 clearly and loses to 1800). At full strength Stockfish 16 is
still far stronger: that is what you should expect from a Python engine with a
lightly trained network, and there is no trick to hide it.

## 1.4 — PyChess integration

- `alpha_zero_engine.py` executable with the venv shebang
  (`#!/home/didier/.../.venv/bin/python`) → PyChess launches it directly.
- `FenixChess` entry in `~/.config/pychess/engines.json` (backup in
  `engines.json.bak-alphazero`): UCI protocol, `normal` variant, level 20,
  engine options.
- `verify_pychess_integration.py`: verification with the **real PyChess code**
  (EngineDiscoverer):
  - discovery: PyChess launches the engine and parses the `uci` handshake
    (`FenixChess` discovered, 0 failures, md5 filled in);
  - real game: `initEngine` + `UCIEngine.makeMove` returns moves
    (`g1f3`, `g8f6`) with `go wtime ...`.
- GUI: PyChess starts on DISPLAY=:0 with the registered entry.

Fixed in that iteration:
- The `Ponder` option was declared but not supported → removed; `go ponder`
  capped at 30 s for safety (although PyChess only ponders if the engine
  sends `bestmove ... ponder ...`, which we do not do).
- GObject signal signature in the verification script
  `(emitter, name, engine)`; `Board(True)` to initialise the board.

## 1.5 — Selecting each iteration/mode from PyChess

**What can be chosen, and from where:**

1. **New Game dialog** → engine dropdown: **two** registered entries show up:
   - `FenixChess` → **Hybrid** mode (alpha-beta + network priors): the
     strongest one, it is the default;
   - `FenixChess-MCTS` → pure **MCTS** mode (classic AlphaZero).
2. **Tools → Engines** → select the entry → **editable** options table
   (`SearchType`, `UseNN`, `MCTS_Simulations`, `Hash`). When the dialog is
   closed, PyChess stores it in `engines.json` and applies it when the engine
   starts (`engineNest.initEngine → optionsCallback`).
3. The **1.0 … 1.4 versions in this changelog are file history**, not runtime
   options: PyChess always runs the project's current `.py` files. To "use an
   older iteration" you must restore its files (e.g. from the backup) and
   re-run the verification.

Changes:
- `FenixChess-MCTS` entry cloned into `~/.config/pychess/engines.json`
  (previous backup: `engines.json.bak-iteraciones`) with
  `options[SearchType].value = "MCTS"`; the original entry keeps
  `value = "Hybrid"`. That `value` is exactly what PyChess writes when you
  edit an option in its Engines dialog.
- `verify_pychess_integration.py` extended: recheck + discovery of **both**
  entries, `value` injection, and `initEngine + makeMove` per entry checking
  the `setoption` PyChess really sends (`SearchType=MCTS` on the second one;
  the first sends nothing because it is the default, which is correct
  `optionsCallback` behaviour).
  Output: `INTEGRACIÓN PYCHESS (Hybrid + MCTS): OK`, exit 0.
- Direct UCI test of the binary: with `setoption name SearchType value MCTS`
  the engine logs `SearchType = MCTS` and answers `info depth 1 ...` (no
  `seldepth`, MCTS visits); without the setoption it does iterative deepening
  (`info depth 1..N seldepth ...`) → both paths work.
- Fixed in the verification: legality was checked with `board.legal_moves`,
  which **does not exist** in PyChess' `Board` → now validated with
  `python-chess` (`chess.Move.from_uci(str(move))`).
- The GUI was restarted to load the new entry (pid in `/tmp/pychess.pid`);
  `engines.json` keeps `recheck=false` on both entries so PyChess does not
  regenerate the options at startup.

Known limit: if PyChess rechecks the engine (the binary changed, e.g. editing
`alpha_zero_engine.py` → different md5), it regenerates the option list and
**loses the `value` fields**; you have to set them again (edit in Tools →
Engines, or re-run `verify_pychess_integration.py`). Both entries also share
the same md5 (same file), so "remember last engine" may preselect
`FenixChess` instead of the MCTS variant.

## 1.6 — Evaluation: king shelter (and two classic tricks discarded)

- **Added** in `search.py::evaluate`: a **king shelter** term — +9 cp for
  every own pawn on the king's file, +13 cp on the adjacent files, counting
  only pawns up to 2 ranks ahead (`KING_SHELTER`, `KING_SHELTER_SIDE`).
  Minimal diff: +19 lines, the search itself is untouched.
- **Tested and DISCARDED** (with measurements, not opinions): *null-move
  pruning* with verification and *ordering the PV move ahead of the TT move*.
  Both **hurt** at this scale (narrow windows, depth ~7). They are not in the
  code.

Ablation (nodes at fixed depth 7, 4 positions: startpos, kiwipete, pos3,
pos4; same hardware, no clock):

| variant | nodes | vs base |
|---|---:|---:|
| base (v1.5) | 263,479 | 0.0 % |
| **+ king shelter (what shipped)** | **239,527** | **−9.1 %** |
| + null-move (with verification) | 360,068 | +36.7 % |
| + PV move above TT move | 458,900 | +74.2 % |
| shelter + null-move | 328,903 | +24.8 % |
| all three | 533,297 | +102.4 % |

Measurements for this iteration:

- Bench: **depth 7 (startpos) / 5 (kiwipete) in 2 s**, ~27 kNPS, **6/6
  tactics** (unchanged).
- Matches (AZ 1 s/move, Stockfish 200 ms):
  - `UCI_Elo 1350` → **4W – 0D – 0L**
  - `UCI_Elo 1600` → **3W – 1D – 0L** (+2.0/4)
- PyChess integration: `verify_pychess_integration.py` → **exit 0**
  (Hybrid + MCTS).

Comparison with the previous versions (same script and hardware):

| Metric | v1.0 | v1.5 | v1.6 |
|---|---|---|---|
| vs SF `UCI_Elo 1350` | 0W – 0D – 4L | 6W – 0D – 0L | **4W – 0D – 0L** |
| vs SF `UCI_Elo 1600` | — | 3W – 1D – 0L | **3W – 1D – 0L** |
| vs SF `UCI_Elo 1800` | — | 0W – 1D – 3L | not measured |
| vs SF full strength | — | 0W – 0D – 1L | not measured |
| Search | pure MCTS (114 sims/s) | hybrid alpha-beta | hybrid alpha-beta |
| Nodes at depth 7 (4 pos) | — | 263,479 | **239,527 (−9.1 %)** |
| Estimated strength | ≤1300 | ≈1650–1750 | ≈1650–1750 |

**Honest reading:** the strength gain is inside the noise of 4 games; the
engine stays in the same band (≈1650–1750). That is why the version is **1.6**
and not 2.0. What *is* a reproducible improvement is the node cost (−9 %) and,
above all, the finding: at this scale the big classic pruning tricks do **not**
pay off.

**Why this is not 2.0 (explicit criteria):**

1. The only measurable improvement is node cost, not strength → there is no
   demonstrable jump that would justify a major version bump.
2. It is backward compatible: it only adds one evaluation term (19 lines); no
   UCI options, engine format or PyChess configuration change → minor
   according to semver.
3. The big boosters (null-move, PV ordering) were measured and they hurt.
4. A real 2.0 would require: beating Stockfish 1800/full strength with a
   margin, or a long training run making MCTS competitive, or an architecture
   change (GPU, bigger net, new API).

## Known / pending

- The network is still barely trained: the published checkpoint comes from
  **a single self-play iteration with 10 games** (log 30/06/2026 17:59,
  3160 s). The later runs of 5×20 and 7×40 games die instantly (1 s logs) and
  never produced weights. Pure `MCTS` mode plays worse than the hybrid; more
  iterations would improve the priors.
- At full strength, Stockfish 16 is still better (see the tables).