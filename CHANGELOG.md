# Changelog — AlphaZero Chess

Historial de iteraciones del motor, en orden real de desarrollo.

## 1.0 — Motor original (estado inicial)

- `alpha_zero_engine.py` UCI + MCTS puro sobre la red neuronal entrenada
  (7 iteraciones × 40 partidas de autojuego/self-play, 100 sims).
- Diagnóstico al medirlo:
  - **0 victorias, 0 empates, 4 derrotas vs Stockfish (UCI_Elo 1350)**.
  - Value head inservible: devuelve ≈0 con una dama de más (`+0.0` en vez de
    ≈+0.9); policy débil (propone `g2g4` en el top-3 de apertura).
  - Búsqueda lenta: 114 sims/s (encode 1.65 ms + forward 6.4 ms por sim,
    sin batching); sin gestión de tiempo real (los `sims` se rebajaban
    **permanentemente** tras una jugada rápida); `stop` no se atendía
    (búsqueda bloqueante en el hilo UCI).

## 1.1 — `search.py`: buscador clásico híbrido (alfa-beta)

Nuevo módulo: la red propone (priors del policy head ordenan las jugadas del
root), la búsqueda clásica confirma.

- Negamax alfa-beta + quiescencia con MVV-LVA, stand-pat y delta pruning.
- Evaluación tapered: material (valores PeSTO) + PSTs clásicas + peones
  pasados/doblados/aislados + pareja de alfiles + columnas abiertas + tempo.
- Tabla transpositoria (Zobrist de python-chess) con ajuste de mates.
- Iterative deepening con deadline + flag de stop; killers, history, LMR,
  reverse/futility pruning, extensiones de jaque, repes y regla de 50.
- Suite táctica integrada: `python search.py --bench`.

Corregido durante la iteración:
- `Move.is_capture()` no existe en python-chess → `board.is_capture(move)`.
- Futility: `pop()` sin `continue` → stack vacío (`IndexError`).
- Delta pruning sin efecto (`alpha + v + 200 < alpha` nunca cierto) → usa
  stand-pat.
- Tempo mal aplicado a negras; código muerto en `evaluate`; tope de pila
  (`ply >= MAX_PLY`) para no desbordar recursión con extensiones de jaque.
- Bug de API: `board.pawns(...)` no existe → `pieces_mask`.

Resultado: **depth 7 en 2 s (29k nps), 6/6 táctica** (mates en 1, tácticas).

## 1.2 — Motor UCI v2 (`alpha_zero_engine.py` reescrito)

- Búsqueda en **hilo aparte**: el loop UCI atiende `stop`/`quit` de verdad
  (antes bloqueaba; PyChess/CuteChess cortan análisis con `stop`).
- Gestión de tiempo real: `movetime`, `wtime/btime + inc + movestogo` con
  presupuesto (85% del tiempo por jugada, tope a la mitad del remanente y
  margen para no perder por tiempo); nunca muta `mcts_sims`.
- Líneas `info depth/score cp|mate/nodes/nps/time/pv` para las GUIs.
- Opción UCI `SearchType = Hybrid | MCTS` (defecto Hybrid), `UseNN`,
  `Hash`, `MCTS_Simulations`.
- Priors de la red en el root (1 evaluación por jugada); si el modelo no
  carga, sigue funcionando como clásico puro.
- Salida limpia: todo lo de debug va a stderr (antes ensuciaba stdout con
  mensajes previos al handshake).

Corregido durante la iteración:
- Presupuesto no descontaba el tiempo de calcular los priors.
- Modelo no precargado en el handshake (1ª jugada con ~0.9 s de más) →
  precarga en `run()`.

Verificado: smoke test completo con python-chess (handshake, movetime,
relojes, 10 jugadas incrementales, `go infinite`+`stop`/análisis).

## 1.2.1 — `mcts.py`: deadline + stop (modo MCTS)

- `MCTS.search(..., deadline=, stop_flag=)` opcional y retrocompatible
  (comprobado cada 16 sims), para que el modo AlphaZero puro también
  respete tiempo y `stop`.
- Verificado: modo MCTS responde `bestmove` dentro del presupuesto.

## 1.3 — `play_match.py` v2: partido UCI real

- Nuestro motor corre **como proceso UCI** (exactamente el camino que usa
  PyChess), no el MCTS en proceso.
- Flags: `--engine Hybrid|MCTS`, `--az-time`, `--sf-time-ms`, `--elo`,
  `--full-strength`, `--games`; marcador parcial por partida.
- Corregido: el rótulo de Stockfish mostraba `ELO 2800` fijo (config) en
  vez del Elo real del partido.

Resultados medidos (1 s/jugada para AZ, 200 ms para SF):

| Versión | Rival | Marcador |
|---|---|---|
| 1.0 (antes) | Stockfish UCI_Elo 1350 | **0 V – 0 E – 4 D** |
| 1.4 (ahora) | Stockfish UCI_Elo 1350 | **6 V – 0 E – 0 D** |
| 1.4 (ahora) | Stockfish UCI_Elo 1600 | **3 V – 0 E – 1 D** (+2/4) |
| 1.4 (ahora) | Stockfish UCI_Elo 1800 | **0 V – 1 E – 3 D** (−3/4) |
| 1.4 (ahora) | Stockfish fuerza completa (0.5 s) | 0 V – 0 E – 1 D (−1/1, esperado) |

Fuerza estimada del motor híbrido: **≈1650–1750** (gana con claridad a
Stockfish limitado a 1600, pierde con 1800). A fuerza completa Stockfish 16
sigue siendo muy superior: es lo esperable para un motor en Python con una
red entrenada poco; aquí no hay truco posible.

## 1.4 — Integración con PyChess

- `alpha_zero_engine.py` ejecutable con shebang del venv
  (`#!/home/didier/.../.venv/bin/python`) → PyChess lo lanza directamente.
- Entrada `AlphaZeroChess` en `~/.config/pychess/engines.json`
  (backup en `engines.json.bak-alphazero`): protocolo UCI, variant
  `normal`, nivel 20, opciones del motor.
- `verify_pychess_integration.py`: verificación con el **código real de
  PyChess** (EngineDiscoverer):
  - descubrimiento: PyChess lanza el motor, parsea el handshake `uci`
    (`AlphaZeroChess` descubierto, 0 fallos, md5 rellenado);
  - partida real: `initEngine` + `UCIEngine.makeMove` devuelve jugadas
    (`g1f3`, `g8f6`) con `go wtime ...`.
- GUI: PyChess arranca en DISPLAY=:0 con la entrada registrada.

Corregido en la iteración:
- Se declaraba la opción `Ponder` sin soportarla → retirada; `go ponder`
  acotado a 30 s por seguridad (aunque PyChess solo pondera si el motor
  envía `bestmove ... ponder ...`, cosa que no hacemos).
- Firma de los signals de GObject en el script de verificación
  `(emitter, name, engine)`; `Board(True)` para inicializar el tablero.

## 1.5 — Selección de cada iteración/modo desde PyChess

**Qué se puede elegir y desde dónde:**

1. **Diálogo «Nueva partida»** → desplegable de motor: aparecen **dos**
   entradas registradas:
   - `AlphaZeroChess` → modo **Hybrid** (alfa-beta + priors de la red): el
     más fuerte, es el defecto;
   - `AlphaZeroChess-MCTS` → modo **MCTS** puro (el AlphaZero clásico).
2. **Herramientas → Motores** → seleccionar la entrada → tabla de opciones
   **editable** (`SearchType`, `UseNN`, `MCTS_Simulations`, `Hash`). Al
   cerrar el diálogo, PyChess guarda en `engines.json` y lo aplica al
   arrancar el motor (`engineNest.initEngine → optionsCallback`).
3. Las versiones **1.0 … 1.4 de este CHANGELOG son historial de archivos**,
   no opciones de runtime: PyChess siempre ejecuta los `.py` actuales del
   proyecto. Para "usar una iteración anterior" hay que restaurar sus
   ficheros (p. ej. desde el backup) y luego re-ejecutar la verificación.

Cambios:
- Entrada `AlphaZeroChess-MCTS` clonada en `~/.config/pychess/engines.json`
  (backup previo: `engines.json.bak-iteraciones`) con
  `options[SearchType].value = "MCTS"`; la entrada original queda con
  `value = "Hybrid"`. El `value` es exactamente lo que PyChess escribe al
  editar una opción en su diálogo de Motores.
- `verify_pychess_integration.py` ampliado: recheck + descubrimiento de
  **ambas** entradas, inyección del `value`, y `initEngine + makeMove` por
  entrada comprobando el `setoption` que PyChess manda de verdad
  (`SearchType=MCTS` en la segunda; en la primera no se envía porque es el
  default, comportamiento correcto de `optionsCallback`).
  Salida: `INTEGRACIÓN PYCHESS (Hybrid + MCTS): OK`, exit 0.
- Test UCI directo del binario: con `setoption name SearchType value MCTS`
  el motor loguea `SearchType = MCTS` y responde `info depth 1 ...` (sin
  `seldepth`, visits del MCTS); sin setoption hace iterative deepening
  (`info depth 1..N seldepth ...`) → los dos caminos funcionan.
- Corregido en la verificación: se validaba la legalidad con
  `board.legal_moves`, que **no existe** en el `Board` de PyChess → se
  valida con `python-chess` (`chess.Move.from_uci(str(move))`).
- La GUI se reinició para cargar la entrada nueva (pid en
  `/tmp/pychess.pid`); `engines.json` queda con `recheck=false` en ambas
  para que PyChess no regenere las opciones al arrancar.

Límite conocido: si PyChess hace un recheck del motor (cambia el binario,
p. ej. editar `alpha_zero_engine.py` → md5 distinto), regenera la lista de
opciones y **pierde los `value`**; hay que volver a ponerlos (editar en
Herramientas → Motores, o re-ejecutar `verify_pychess_integration.py`).
Además ambas entradas comparten md5 (mismo fichero), así que «recordar el
último motor» puede precargar `AlphaZeroChess` en lugar de la variante MCTS.

## Pendiente / conocido

- La red sigue siendo débil (solo 7 iteraciones de self-play): el modo
  `MCTS` puro juega
  peor que el híbrido. Entrenar más iteraciones mejoraría los priors.
- A fuerza completa, Stockfish 16 sigue siendo superior (ver tabla).
