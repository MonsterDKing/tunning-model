"""Exporta las notas de un split en SEGUNDOS (tempo fijo) para sintetizar audio y comparar con
modelos que escuchan (p. ej. MOSS-Music). El acorde escrito i suena en (tiempo_i − q0) · 60 / BPM.

    PYTHONDONTWRITEBYTECODE=1 uv run python scripts/exportar_para_audio.py test   # → data/audio/test/*.json
"""

import json
import sys

from acordes_gnn.dataset import cargar, nombres_de
from acordes_gnn.datos import DATOS

BPM = 90  # tempo fijo: las partituras no siempre lo indican, y así la conversión es exacta
split = sys.argv[1]
salida = DATOS / "audio" / split
salida.mkdir(parents=True, exist_ok=True)
for n in nombres_de(split):
    p = cargar(n)
    nota = p["grafo"]["note"]
    q0 = float(nota.onset_q.min())  # la anacrusa puede empezar en negativo
    seg = 60 / BPM
    notas = [[(float(o) - q0) * seg, float(d) * seg, int(a)] for o, d, a in zip(nota.onset_q, nota.dur_q, nota.pitch)]
    (salida / f"{n.rsplit('.', 1)[0]}.json").write_text(json.dumps({"nombre": n, "bpm": BPM, "q0": q0, "notas": notas}))
print(f"{len(nombres_de(split))} partituras → {salida}")
