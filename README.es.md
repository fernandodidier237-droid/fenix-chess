# AlphaZero Chess — autojuego desde cero

[![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![UCI](https://img.shields.io/badge/motor-UCI-green)

> Entrena un motor de ajedrez estilo AlphaZero **desde cero** (Monte-Carlo
> Tree Search + red neuronal, solo partidas contra sí mismo) y juega con él
> desde cualquier GUI UCI — este repo incluye la integración con **PyChess**
> y resultados medidos y honestos.

[English](README.md) | **Español**

## Resumen

```bash
pip install -r requirements.txt
python main.py --iterations 10 --self-play-games 40   # entrenar (vale CPU)
python search.py --bench                             # bench alfa-beta: profundidad 7 / 2 s
python play_match.py --games 4 --elo 1350            # partido vs Stockfish
```

Después apunta cualquier GUI UCI a `alpha_zero_engine.py` y juega.

## Qué hay dentro

- **`self_play.py` / `mcts.py` / `neural_network.py`** — el ciclo AlphaZero:
  MCTS guiado por una red ResNet policy+value, partidas de autojuego, buffer
  de replay, evaluación en arena y checkpoints. Sin código de AlphaZero ajeno.
- **`search.py`** — la apuesta: un buscador **alfa-beta clásico** (negamax,
  quiescencia, tabla de transposición, evaluación tapered PeSTO, iterative
  deepening, killers/history/LMR/futility) que recibe los priors de la red
  como probabilidades en la raíz. Este modo *híbrido* es mucho más fuerte
  que el MCTS puro con el presupuesto de entrenamiento actual.
- **`alpha_zero_engine.py`** — un motor **UCI** de verdad, con dos modos de
  búsqueda seleccionables, gestión de tiempo, `stop` real y salida
  `info`/`pv`. Opción `SearchType = Hybrid | MCTS`.
- **`play_match.py`** — runner de partidas UCI (puertas de Elo o fuerza
  completa).
- **`verify_pychess_integration.py`** — test headless con el **código real
  de PyChess**: descubrimiento, handshake de opciones y una jugada desde
  cada entrada.

## Resultados medidos

Todo salió de `play_match.py` (1 s/jugada para nosotros, 200 ms/jugada para
Stockfish salvo indicación). Sin trucos: las derrotas están en la tabla.

| Rival (Stockfish 16) | Antes (v1.0) | Ahora (v1.5) |
|---|---|---|
| `UCI_Elo 1350` | 0V – 0E – 4D | **6V – 0E – 0D** |
| `UCI_Elo 1600` | — | **3V – 1E – 0D** (+2/4) |
| `UCI_Elo 1800` | — | 0V – 1E – 3D (−3/4) |
| Fuerza completa (0.5 s) | — | 0V – 0E – 1D (esperado) |

Bench alfa-beta: **profundidad 7 en 2 s (~29 kNPS)**, **6/6** tácticas.

Fuerza estimada: **≈1650–1750 Elo**. A fuerza completa Stockfish 16 sigue
siendo muy superior: es una limitación honesta y medida (motor en Python con
una red entrenada en portátil).

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # torch, python-chess, numpy, tensorboard
```

Opcional, para la GUI y los tests: PyChess del sistema (`apt install pychess`)
y Stockfish (`apt install stockfish`). Ninguno va empaquetado: son programas
GPL que instalas tú.

## Entrenar

```bash
python main.py --iterations 20 --self-play-games 40 --mcts-sims 100
tensorboard --logdir logs
```

`config.py` trae los valores por defecto tipo artículo (100 iteraciones ×
10 000 partidas); empieza pequeño en CPU. Los checkpoints van a `models/`
(los pesos **no** se suben — ver `.gitignore`).

## Jugar

**Cualquier GUI UCI** (Arena, CuteChess, PyChess…) → añadir motor → comando:

```bash
python /ruta/a/alpha_zero_engine.py
```

> El shebang del repo apunta a la venv del autor original; edita la primera
> línea de `alpha_zero_engine.py` o lánzalo con tu propio `python`.

Opciones UCI útiles: `SearchType` (Hybrid|MCTS), `UseNN`,
`MCTS_Simulations`, `Hash`.

**PyChess** lo registra en `~/.config/pychess/engines.json` (entrada
`AlphaZeroChess`, protocolo `uci`, `level` 20, `recheck` true). Este repo
incluye **dos** entradas para elegir el modo en el diálogo de nueva partida:
`AlphaZeroChess` (Hybrid) y `AlphaZeroChess-MCTS` (MCTS puro). Las opciones
se editan en *Herramientas → Motores* y el `value` guardado se aplica al
arrancar el motor. Si tocas el binario del motor, vuelve a ejecutar
`python verify_pychess_integration.py`: el cambio de checksum hace que
PyChess redescubra el motor y pierda los `value` guardados.

**Por línea de comandos:**

```bash
python play_match.py --games 4 --elo 1600 --engine Hybrid
python play_match.py --games 1 --full-strength --sf-time-ms 500
python search.py --bench
```

## Estructura del repo

| Ruta | Qué es |
|---|---|
| `main.py` / `coach.py` | bucle de entrenamiento (autojuego → buffer → train → arena) |
| `chess_env.py` / `neural_network.py` | entorno + ResNet (policy y value) |
| `mcts.py` / `self_play.py` | MCTS con priors de la red + generación de partidas |
| `search.py` | buscador híbrido alfa-beta (el fuerte) |
| `alpha_zero_engine.py` | front-end UCI (Hybrid / MCTS) |
| `play_match.py` | runner de partidas UCI con puertas de Elo |
| `verify_pychess_integration.py` | test de integración con PyChess |
| `CHANGELOG.md` | historial de versiones 1.0 → 1.5 con cada corrección |

## Limitaciones / hoja de ruta

- La cabeza `value` sigue siendo débil; el checkpoint incluido tiene pocas
  iteraciones de autojuego. Entrenar más mejora directamente el MCTS.
- Stockfish a fuerza completa gana con claridad: el hueco está medido y es
  honesto.
- Rendimiento en Python: ~29 kNPS de alfa-beta, ~115 sims/s de MCTS. Portar
  los caminos calientes a una extensión compilada es el paso obvio.

## Licencia

[MIT](LICENSE). Stockfish y PyChess son programas GPL independientes y aquí
no se redistribuyen.
