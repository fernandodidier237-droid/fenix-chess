#!/usr/bin/env python3
"""
Verificación headless de la integración con PyChess (con el código REAL de
PyChess, sin GUI), para las DOS entradas registradas:

    FenixChess       -> SearchType = Hybrid
    FenixChess-MCTS  -> SearchType = MCTS

Pasos:
  1. Fuerza un re-check de las entradas Fénix (como en su 1ª instalación).
  2. Descubrimiento: PyChess lanza el motor y parsea su respuesta 'uci'.
  3. Inyecta el valor de SearchType en engines.json (exactamente lo que hace
     el diálogo "Motores" de PyChess al editar una opción) y lo refleja en la
     entrada en memoria.
  4. initEngine + makeMove para CADA entrada, comprobando que PyChess envía
     de verdad el `setoption name SearchType value ...` esperado y que el
     motor devuelve una jugada legal.

Ejecutar con el python del sistema (donde está pychess):
    python3 verify_pychess_integration.py
"""

import asyncio
import json
import os
import sys

import gi
gi.require_version("Gtk", "3.0")

from pychess.Players.engineNest import EngineDiscoverer

JSONPATH = os.path.expanduser("~/.config/pychess/engines.json")
TIMEOUT = 60

# Entrada esperada -> valor de la opción SearchType
EXPECTED = {
    "FenixChess": "Hybrid",
    "FenixChess-MCTS": "MCTS",
}
NEW_ENTRY_NAME = "FenixChess-MCTS"


def load_engines():
    with open(JSONPATH) as fh:
        return json.load(fh)


def save_engines(engines):
    with open(JSONPATH, "w") as fh:
        json.dump(engines, fh, indent=1)


def ensure_entries(engines):
    """Crea la entrada MCTS (clon de la principal) si no existe todavía."""
    base = next((e for e in engines if e.get("name") == "FenixChess"), None)
    if base is None:
        raise SystemExit("FALLO: no existe la entrada FenixChess")
    if not any(e.get("name") == NEW_ENTRY_NAME for e in engines):
        clone = json.loads(json.dumps(base))  # copia profunda
        clone["name"] = NEW_ENTRY_NAME
        clone["comment"] = (
            "Fénix: búsqueda MCTS puro (misma red neuronal)"
        )
        clone["recheck"] = True
        engines.append(clone)
    return engines


def force_recheck(engines):
    """Re-check de las entradas Fénix, como en su primera instalación."""
    for e in engines:
        if e.get("name") in EXPECTED:
            e["recheck"] = True
    save_engines(engines)


def inject_values(engines):
    """Pone option['value'] en SearchType de cada entrada y guarda el JSON.

    Es lo mismo que hace el usuario desde Herramientas → Motores de PyChess;
    PyChess lo aplica en engineNest.initEngine (optionsCallback) al arrancar
    el motor.
    """
    for e in engines:
        name = e.get("name")
        if name not in EXPECTED:
            continue
        opts = e.get("options") or []
        st = next((o for o in opts if o.get("name") == "SearchType"), None)
        if st is None:
            print(f"FALLO: {name} no declara la opción SearchType")
            return False
        st["value"] = EXPECTED[name]
    save_engines(engines)
    return True


async def wait_discovery(nest, timeout=45):
    found, failed = {}, {}

    def on_found(*args):
        found[args[1]] = args[2]   # (emitter, name, engine)

    def on_failed(*args):
        failed[args[1]] = args[2]

    nest.connect("engine_discovered", on_found)
    nest.connect("engine_failed", on_failed)

    nest.pre_discover()
    nest.discover()

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.5)
        if nest.toBeRechecked and all(
            done for _, done in nest.toBeRechecked.values()
        ):
            break
        if not nest.toBeRechecked:
            break
    await asyncio.sleep(1)
    return found, failed


async def play_with(nest, name, expected_search_type):
    """initEngine + makeMove para una entrada, verificando el setoption."""
    entry = next(
        (e for e in nest.engines if e.get("name") == name), None
    )
    if entry is None:
        print(f"FALLO: la entrada {name} no está en engines.json")
        return False

    from pychess.Utils.const import WHITE
    player = await nest.initEngine(entry, WHITE, False)

    # Grabar lo que PyChess manda al motor (optionsCallback de engineNest)
    sent = []
    original_set_option = player.setOption

    def recording_set_option(key, value):
        sent.append((key, value))
        original_set_option(key, value)

    player.setOption = recording_set_option

    player.prestart()
    event = asyncio.Event()
    is_dead = set()
    player.start(event, is_dead)
    await asyncio.wait_for(event.wait(), TIMEOUT)
    if is_dead:
        print(f"FALLO: el proceso de {name} murió durante el arranque")
        return False

    player.setOptionStrength(20, True)
    player.setOptionTime(2, 0, 0)   # 2 s por jugada

    sent_opts = [(k, str(v)) for k, v in sent]
    print(f"[{name}] setoptions enviados por PyChess: {sent_opts}")

    # PyChess solo envía una opción si "value" difiere de "default"
    # (engineNest.optionsCallback), así que el valor guardado en engines.json
    # es lo que manda: si es el default, no se envía nada y el motor usa el suyo.
    st = next(
        (o for o in (entry.get("options") or [])
         if o.get("name") == "SearchType"),
        None,
    )
    stored = st.get("value", st.get("default")) if st else None
    default = st.get("default") if st else None
    if stored != expected_search_type:
        print(f"FALLO: {name} guardado SearchType={stored!r}, "
              f"esperado {expected_search_type!r}")
        return False
    expected_sent = None if stored == default else str(stored)
    got = dict(sent_opts).get("SearchType")
    if got != expected_sent:
        print(f"FALLO: {name} -> PyChess envió SearchType={got!r}, "
              f"esperado {expected_sent!r}")
        return False
    print(f"[{name}] SearchType OK (guardado={stored!r}, default={default!r}, "
          f"enviado={got!r})")

    from pychess.Utils.Board import Board
    from pychess.Utils.Move import toAN
    board = Board(True)   # posición inicial

    player.set_board(board)
    move = await asyncio.wait_for(player.makeMove(board, None, None), TIMEOUT)
    san = toAN(board, move) if move else None
    print(f"[{name}] JUGADA a través de PyChess: {move} (AN: {san})")
    if move is None:
        print(f"FALLO: {name} no devolvió jugada")
        return False
    import chess
    try:
        pcm = chess.Move.from_uci(str(move))
        legal = pcm in chess.Board().legal_moves
    except Exception as exc:
        print(f"FALLO: {name} jugada no interpretable ({exc})")
        return False
    if not legal:
        print(f"FALLO: {name} devolvió una jugada ilegal: {move}")
        return False

    try:
        print("quit", file=player.engine)
    except Exception:
        pass
    return True


async def main():
    engines = ensure_entries(load_engines())
    force_recheck(engines)

    nest = EngineDiscoverer()
    found, failed = await wait_discovery(nest)

    print("=" * 64)
    print(f"Descubiertos: {sorted(found)}")
    print(f"Fallidos    : {sorted(failed)}")

    for name in EXPECTED:
        if name in failed:
            print(f"FALLO: {name} fue descubierto con errores")
            return 1

    # La descubrición regenera 'options' sin 'value': lo inyectamos ahora
    # (en disco y en memoria) igual que haría el diálogo de Motores.
    engines = ensure_entries(load_engines())
    for e in engines:
        name = e.get("name")
        if name in EXPECTED:
            fresh = next(
                (f for f in nest.engines if f.get("name") == name), None
            )
            if fresh is not None:
                e["options"] = fresh.get("options")
                e["recheck"] = False
    if not inject_values(engines):
        return 1
    for e in engines:
        name = e.get("name")
        if name in EXPECTED:
            mem = next(
                (f for f in nest.engines if f.get("name") == name), None
            )
            if mem is not None:
                mem["options"] = e["options"]
    print("SearchType inyectado en engines.json para ambas entradas.")

    ok = True
    for name, expected in EXPECTED.items():
        print("=" * 64)
        ok = await play_with(nest, name, expected) and ok

    print("=" * 64)
    if not ok:
        print("INTEGRACIÓN PYCHESS: FALLOS")
        return 1
    print("INTEGRACIÓN PYCHESS (Hybrid + MCTS): OK")
    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
