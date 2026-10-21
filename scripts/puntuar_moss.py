"""Puntúa las transcripciones de acordes de MOSS-Music con la métrica del proyecto.

Para cada acorde escrito (instante t en negras) se calcula su segundo en el audio sintetizado
(``(t − q0) · 60 / bpm``, ver ``exportar_para_audio.py``) y se toma el acorde de MOSS vigente en
ese segundo (el último que empezó antes). Su símbolo se traduce a las 60 clases con music21, igual
que el cifrado de PDMX. Versión estricta (segundo exacto) y con tolerancia (vale si MOSS dijo el
acorde correcto en algún momento de ±0,5 s: no le penaliza por desfases pequeños al marcar tiempos).

    PYTHONDONTWRITEBYTECODE=1 uv run python scripts/puntuar_moss.py data/audio/moss_test test modelos/combo_s0.pt …
"""

import bisect
import json
import re
import statistics as st
import sys
import warnings
from pathlib import Path

import torch

from acordes_gnn.dataset import cargar, nombres_de
from acordes_gnn.datos import DATOS
from acordes_gnn.etiquetas import CALIDAD_M21, calidad_por_intervalos, clase
from acordes_gnn.evaluar import cargar_modelo, comparar, evaluar_piezas

warnings.filterwarnings("ignore")
HARTE = {"maj": "", "min": "m", "dim": "dim", "aug": "+", "sus2": "sus2", "sus4": "sus4", "maj7": "maj7",
         "min7": "m7", "7": "7", "hdim7": "m7b5", "dim7": "dim7", "maj6": "6", "min6": "m6", "9": "9"}  # fmt: skip


def clase_de(simbolo: str) -> int | None:
    """'Am7', 'F#', 'Bb/D', 'C:maj', 'A:min7'… → clase de 60 (None si es "sin acorde" o no se entiende)."""
    from music21 import harmony

    s = simbolo.strip()
    if not s or s.upper() in ("N", "NC", "N.C.", "X"):
        return None
    if ":" in s:  # sintaxis de Harte (MIREX)
        raiz, resto = s.split(":", 1)
        tipo, _, bajo = resto.partition("/")
        s = raiz + HARTE.get(tipo, tipo) + ("/" + bajo if bajo and not bajo.isdigit() else "")
    try:
        cs = harmony.ChordSymbol(s.replace("b", "-") if re.match(r"^[A-G]b", s) else s)
        raiz = cs.root().pitchClass
        return clase(raiz, calidad_por_intervalos({(p.pitchClass - raiz) % 12 for p in cs.pitches}))
    except Exception:  # noqa: BLE001
        return None


def leer_respuesta(texto: str) -> list[tuple[float, int | None]]:
    texto = re.sub(r"^```(json)?|```$", "", texto.strip(), flags=re.M)
    try:
        eventos = json.loads(texto)
    except json.JSONDecodeError:
        eventos = [{"time": float(t), "chord": c} for t, c in re.findall(r'"time"\s*:\s*([\d.]+)\s*,\s*"chord"\s*:\s*"([^"]*)"', texto)]
    if isinstance(eventos, dict):
        eventos = next((v for v in eventos.values() if isinstance(v, list)), [])
    salida = []
    for e in eventos:
        t = e.get("time", e.get("start", e.get("onset")))
        if t is not None:
            salida.append((float(t), clase_de(str(e.get("chord", e.get("label", ""))))))
    return sorted(salida, key=lambda x: x[0])


carpeta, split, modelos = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
piezas = [cargar(n) for n in nombres_de(split)]
estricto = {"raiz": 0, "calidad": 0, "equivalente": 0}
tolerante = 0
total = sin_respuesta = 0
for p in piezas:
    base = p["nombre"].rsplit(".", 1)[0]
    meta = json.loads((DATOS / "audio" / split / f"{base}.json").read_text())
    ruta = carpeta / f"{base}.json"
    eventos = []
    if ruta.exists():
        datos = json.loads(ruta.read_text())
        for t in datos.get("trozos", [{"inicio": 0.0, "respuesta": datos.get("respuesta", "")}]):
            eventos += [(t["inicio"] + s, c) for s, c in leer_respuesta(t["respuesta"])]  # tiempos del trozo → de la canción
        eventos.sort(key=lambda x: x[0])
    if not eventos:
        sin_respuesta += 1
    tiempos = [t for t, _ in eventos]
    for a in p["acordes"]:
        seg = (a["tiempo"] - meta["q0"]) * 60 / meta["bpm"]
        k = bisect.bisect_right(tiempos, seg) - 1
        c = eventos[k][1] if k >= 0 else None
        total += 1
        if c is not None:
            r, q, e = comparar(c, a)
            estricto["raiz"] += r; estricto["calidad"] += q; estricto["equivalente"] += e
        cerca = {eventos[j][1] for j in range(len(eventos)) if abs(eventos[j][0] - seg) <= 0.5} | ({c} if c is not None else set())
        tolerante += any(x is not None and comparar(x, a)[1] for x in cerca)
print(f"{split}: {len(piezas)} partituras, {total} acordes · sin respuesta de MOSS: {sin_respuesta}")
print(f"MOSS-Music (estricto)       raíz {estricto['raiz'] / total:.1%} · raíz+calidad {estricto['calidad'] / total:.1%} · equivalente {estricto['equivalente'] / total:.1%}")
print(f"MOSS-Music (±0,5 s)         raíz+calidad {tolerante / total:.1%}")
if modelos:
    torch.set_num_threads(4)  # en esta máquina, más hilos es mucho más lento
    rs = [evaluar_piezas(cargar_modelo(m)[0], piezas) for m in modelos]
    print(f"Nuestra GNN (media de {len(rs)})    raíz {st.mean(r['raiz'] for r in rs):.1%} · raíz+calidad {st.mean(r['calidad'] for r in rs):.1%} · equivalente {st.mean(r['equivalente'] for r in rs):.1%}")
