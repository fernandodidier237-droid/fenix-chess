# AlphaZero Chess — autojuego desde cero

[![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![UCI](https://img.shields.io/badge/motor-UCI-green)

> Entrena un motor de ajedrez estilo AlphaZero **desde cero** (Monte-Carlo
> Tree Search + red neuronal, solo partidas contra sí mismo) y juega con él
> desde cualquier GUI UCI — este repo incluye la integración con **PyChess**
> y resultados medidos y honestos.

**Español** | [English](README.en.md)

![AlphaZero (Hybrid) da mate a Stockfish 16 limitado a 1350 Elo](docs/demo.gif)

*Partida real registrada por `play_match.py`: AlphaZero (Hybrid, 1 s/jugada)
contra Stockfish 16 con `UCI_Elo 1350` — mate en 65 jugadas, sin editar.*

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
  quiescencia, tabla de transposición, evaluación tapered PeSTO con refugio del rey, iterative
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

Bench alfa-beta: **profundidad 7 en 2 s (~27 kNPS)**, **6/6** tácticas.

La v1.6 vuelve a medirse igual: `UCI_Elo 1350` → 4V–0E–0D y
`UCI_Elo 1600` → 3V–1E–0D — mismo tramo de fuerza (el término de refugio del
rey baja los nodos un 9 % sin cambiar el resultado). En `CHANGELOG.md` está
la ablación que además demuestra que el *null-move pruning* y la ordenación
por jugada PV **empeoran** a esta escala.

## Comparativa entre versiones

Todo medido con el mismo script y hardware; los partidos son cortos (4–6 por
nivel), así que léelos como tendencia, no como Elo exacto.

| Métrica | v1.0 (original) | v1.5 | v1.6 (actual) |
|---|---|---|---|
| vs Stockfish `UCI_Elo 1350` | 0V – 0E – 4D | **6V – 0E – 0D** | **4V – 0E – 0D** |
| vs Stockfish `UCI_Elo 1600` | — | 3V – 1E – 0D (+2/4) | 3V – 1E – 0D (+2/4) |
| vs Stockfish `UCI_Elo 1800` | — | 0V – 1E – 3D (−3/4) | sin medir |
| vs Stockfish fuerza completa | — | 0V – 0E – 1D | sin medir |
| Búsqueda | MCTS puro, 114 sims/s | alfa-beta híbrido | alfa-beta híbrido |
| Bench 2 s | sin bench | depth 7 / 5, ~29 kNPS | depth 7 / 5, ~27 kNPS |
| Nodos a depth 7 (4 posiciones) | — | 263.479 | **239.527 (−9,1 %)** |
| Motor UCI con `stop` real | no | sí | sí |
| Integración con GUI | ninguna | PyChess (2 entradas) | PyChess (2 entradas) |
| Fuerza estimada | ≤1300 (perdía 0–4 a 1350) | ≈1650–1750 | ≈1650–1750 |

## Por qué es 1.6 y no 2.0

1. **Lo único medible que mejoró fue el coste, no la fuerza**: −9,1 % de
   nodos; en fuerza el resultado cae dentro del ruido de 4 partidas
   (mismo tramo, ≈1650–1750 Elo). Un 2.0 debe traer un salto demostrable.
2. **Compatibilidad**: la v1.6 solo *añade* un término a la evaluación
   (19 líneas). No cambia opciones UCI, ni el formato del motor, ni la
   configuración de PyChess → por semver es una minor.
3. **Los refuerzos "grandes" no se sostienen**: null-move pruning y
   ordenación PV se implementaron, se midieron y **empeoraron** (+36,7 % y
   +74,2 % de nodos). No hay salto cualitativo que justifique un 2.0.
4. **Lo que sí convertiría esto en un 2.0**: ganarle a Stockfish 1800 o a
   fuerza completa con margen, un entrenamiento largo que dé un MCTS
   competitivo, o un cambio de arquitectura (GPU / red mayor / API nueva).

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
| `CHANGELOG.md` / `CHANGELOG.en.md` | historial de versiones 1.0 → 1.6 con cada corrección (español / inglés) |

## Limitaciones / hoja de ruta

- La cabeza `value` sigue siendo débil; el checkpoint publicado viene de **una
  sola iteración** de autojuego (10 partidas, 100 sims MCTS, ~53 min en CPU).
  Entrenar más mejora directamente el MCTS.
- Stockfish a fuerza completa gana con claridad: el hueco está medido y es
  honesto.
- Rendimiento en Python: ~29 kNPS de alfa-beta, ~115 sims/s de MCTS. Portar
  los caminos calientes a una extensión compilada es el paso obvio.

## Licencia

[MIT](LICENSE). Stockfish y PyChess son programas GPL independientes y aquí
no se redistribuyen.
