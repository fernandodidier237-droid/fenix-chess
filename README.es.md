# Fénix Chess — autojuego desde cero

[![Licencia: MIT](https://img.shields.io/badge/licencia-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![UCI](https://img.shields.io/badge/motor-UCI-green)
![Versión](https://img.shields.io/badge/versión-1.7-informational)

> Entrena un motor de ajedrez estilo AlphaZero **desde cero** (Monte-Carlo
> Tree Search + red neuronal, solo partidas contra sí mismo) y juega con él
> desde cualquier GUI UCI — este repo incluye la integración con **PyChess** y
> resultados medidos y honestos.

> **Nombre**: el motor se llama **Fénix Chess** (`FenixChess-Hybrid` en UCI,
> `FenixChess` / `FenixChess-MCTS` en PyChess). Se renombró en la 1.7 porque
> «AlphaZero» ya es una marca de DeepMind: esto es un motor alfa-beta clásico
> con los priors de su propia red, no una implementación de AlphaZero.

🌐 [English](README.md) · **[Español](README.es.md)** ·
[Changelog (es)](CHANGELOG.md) · [Changelog (en)](CHANGELOG.en.md) ·
[Releases](https://github.com/fernandodidier237-droid/fenix-chess/releases)

![Fénix (Hybrid) da mate a Stockfish 16 limitado a 1350 Elo](docs/demo.gif)

*Partida real registrada por `play_match.py`: Fénix (Hybrid, 1 s/jugada)
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
  MCTS guiado por una red ResNet policy+value, partidas de autojuego, buffer de
  replay, evaluación en arena y checkpoints. Sin código de AlphaZero ajeno.
- **`search.py`** — la apuesta: un buscador **alfa-beta clásico** (negamax,
  quiescencia, tabla de transposición, evaluación tapered PeSTO con refugio del
  rey, iterative deepening, killers/history/LMR/futility) que recibe los priors
  de la red como probabilidades en la raíz. Este modo *híbrido* es mucho más
  fuerte que el MCTS puro con el presupuesto de entrenamiento actual.
- **`alpha_zero_engine.py`** — un motor **UCI** de verdad, con dos modos de
  búsqueda seleccionables, gestión de tiempo, `stop` que funciona, salida
  `info`/`pv` y la búsqueda en un hilo aparte. Opción
  `SearchType = Hybrid | MCTS`.
- **`play_match.py`** — runner de partidas UCI (puertas de Elo o fuerza
  completa).
- **`verify_pychess_integration.py`** — test headless con el **código real de
  PyChess**: descubrimiento, handshake de opciones y una jugada desde cada
  entrada registrada.

## Resultados y comparativa entre versiones

Todo se produjo con `play_match.py` (1 s/jugada para nosotros, 200 ms/jugada
para Stockfish). Sin trucos: las derrotas se quedan en la tabla. Los partidos
son cortos (4–6 por nivel), así que léelos como tendencia, no como Elo exacto.

| Métrica | v1.0 (original) | v1.5 | v1.6 | **v1.7 (actual)** |
|---|---|---|---|---|
| vs Stockfish `UCI_Elo 1350` | 0V – 0E – 4D | **6V – 0E – 0D** | **4V – 0E – 0D** | 3V – 0E – 1D (+3,0/4) |
| vs Stockfish `UCI_Elo 1600` | — | 3V – 1E – 0D (+2/4) | 3V – 1E – 0D (+2/4) | 3V – 0E – 1D (+2,0/4) |
| vs Stockfish `UCI_Elo 1800` | — | 0V – 1E – 3D (−3/4) | 2V – 0E – 2D (0,0/4) | 2V – 0E – 2D (0,0/4) |
| vs Stockfish fuerza completa (0.5 s) | — | 0V – 0E – 1D | 0V – 0E – 2D | 0V – 0E – 2D |
| A/B directo contra v1.6 (8 partidas) | — | — | — | 4,0/8 – 4,0/8 (empate) |
| Búsqueda | MCTS puro, 114 sims/s | alfa-beta híbrido | alfa-beta híbrido | alfa-beta híbrido + null-move + LMR |
| Bench, 2 s de presupuesto | sin bench | prof. 7 / 5, ~29 kNPS | prof. 7 / 5, ~27 kNPS | prof. 7 / 5, ~27 kNPS |
| Nodos a profundidad fija 7 (4 posiciones) | — | 263.479 | 239.527 (−9,1 %) | 238.566 (−0,4 %) |
| Suite táctica | ninguna | 6/6 | 6/6 | 6/6 |
| Motor UCI con `stop` real | no | sí | sí | sí |
| Integración con GUI | ninguna | PyChess, 2 entradas | PyChess, 2 entradas | PyChess, 2 entradas |
| Fuerza estimada | ≤1300 (perdía 0–4 a 1350) | ≈1650–1750 | ≈1650–1750 | ≈1650–1750 |

A fuerza completa Stockfish 16 sigue siendo muy superior: es una limitación
honesta, no oculta — es un motor en Python cuya red se entrenó en la CPU de un
portátil.

## Por qué 1.7 y no 2.0

1. **El A/B contra la 1.6 fue un empate**: 4,0/8 – 4,0/8 en 8 partidas a 1 s
   por jugada. El resultado se alterna exactamente con el color, así que la
   diferencia de fuerza entre 1.6 y 1.7 no se distingue del ruido. Un 2.0
   tiene que traer un salto demostrable.
2. **La ganancia real es la escalera a 1800** (0,0/4 frente a −3/4 de la 1.5) y
   el null-move verificado; no es una mejora de fuerza medible frente a la 1.6.
3. **Compatible hacia atrás**: la 1.7 solo cambia dos reglas de poda dentro de
   `search.py` (+21 / −2 líneas). Sin opciones UCI nuevas ni cambios en el
   formato del motor o en la configuración de PyChess → versión menor.
4. **Eficiencia**: 238.566 nodos a prof. 7 frente a 239.527 (−0,4 %, ruido), y
   +22,9 % a prof. 8. No se puede honestamente vender la 1.7 como más
   eficiente que la 1.6; el `CHANGELOG` trae la tabla completa.
5. **Qué haría falta para un 2.0**: ganar a Stockfish 1800 o a fuerza completa
   con margen, un entrenamiento largo que haga competitivo el MCTS, o un
   cambio de arquitectura (GPU / red más grande / API nueva).

<details><summary>Por qué la 1.6 fue menor y no 2.0</summary>

1. **Lo medible es el coste, no la fuerza**: −9,1 % de nodos; en fuerza el
   resultado cae dentro del ruido de 4 partidas (mismo tramo, ≈1650–1750 Elo).
   Un 2.0 debe traer un salto demostrable.
2. **Compatible hacia atrás**: la v1.6 solo *añade* un término de evaluación
   (19 líneas). No cambia ninguna opción UCI, ni el formato del motor, ni la
   configuración de PyChess → minor según semver.
3. **Los refuerzos "grandes" no se sostienen**: null-move pruning y ordenación
   por jugada PV se implementaron, se midieron y **empeoran** (+36,7 % y
   +74,2 % de nodos). La ablación completa está en el CHANGELOG.
4. **Lo que sí sería un 2.0**: ganarle a Stockfish 1800 o a fuerza completa con
   margen, un entrenamiento largo que haga competitivo el MCTS, o un cambio de
   arquitectura (GPU / red mayor / API nueva).

</details>

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
10 000 partidas); empieza pequeño en CPU. Los checkpoints van a `models/` (los
pesos **no** se suben — ver `.gitignore`).

## Jugar

**Cualquier GUI UCI** (Arena, CuteChess, PyChess…) → añadir motor →

```bash
python /ruta/a/alpha_zero_engine.py
```

> El shebang del repo apunta a la venv del autor original; edita la primera
> línea de `alpha_zero_engine.py` o lánzalo con tu propio `python`.

Opciones UCI útiles: `SearchType` (Hybrid|MCTS), `UseNN`,
`MCTS_Simulations`, `Hash`.

**PyChess** lo registra en `~/.config/pychess/engines.json`:

```json
{
  "name": "FenixChess",
  "command": "/ruta/a/alpha_zero_engine.py",
  "protocol": "uci",
  "level": 20,
  "analyze": true,
  "recheck": true
}
```

Este repo incluye **dos** entradas para elegir el modo en el diálogo de nueva
partida: `FenixChess` (Hybrid) y `FenixChess-MCTS` (MCTS puro). Las
opciones se editan en *Herramientas → Motores* y el `value` guardado se aplica
al arrancar el motor. Si tocas el binario del motor, vuelve a ejecutar
`python verify_pychess_integration.py`: el cambio de checksum hace que
PyChess lo redescubra y pierda los valores guardados.

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
| `CHANGELOG.md` / `CHANGELOG.en.md` | historial 1.0 → 1.7 con cada corrección (es / en) |

## Limitaciones / hoja de ruta

- La cabeza `value` sigue siendo débil; el checkpoint publicado viene de **una
  sola iteración** de autojuego (10 partidas, 100 sims MCTS, ~53 min en CPU).
  Entrenar más mejora directamente el MCTS.
- Stockfish a fuerza completa gana, y con holgura: el hueco está medido y es
  honesto.
- Rendimiento en Python: ~27 kNPS de alfa-beta, ~115 sims/s de MCTS. Portar los
  caminos calientes a una extensión compilada es el paso obvio.

## Licencia

[MIT](LICENSE) — © 2026 Fernando Didier Leal Mendoza. Stockfish y PyChess son
programas GPL independientes y aquí no se redistribuyen.