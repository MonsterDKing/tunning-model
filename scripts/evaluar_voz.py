"""Melodía frente a acompañamiento en las canciones de examen de POP909 (``pop909_voz_test``).

Compara los modelos con salida de voz con el método clásico *skyline*: en cada instante en que
empiezan notas, la más aguda es la melodía. Mide, sobre las notas con etiqueta: acierto, y para la
clase "melodía" precisión (de lo que dijo melodía, cuánto lo era), cobertura (de la melodía real,
cuánto encontró) y F1 (la media armónica de las dos).

    PYTHONDONTWRITEBYTECODE=1 uv run python scripts/evaluar_voz.py modelos/voz_s0.pt ...
"""

import statistics as st
import sys

import torch

from acordes_gnn.dataset import cargar
from acordes_gnn.datos import cargar_splits
from acordes_gnn.evaluar import cargar_modelo

torch.set_num_threads(8)
piezas = []
for n in cargar_splits()["pop909_voz_test"]:
    try:
        p = cargar(n)
    except FileNotFoundError:
        continue
    if (p["grafo"]["note"].voz >= 0).any():
        piezas.append(p)


def medir(predecir) -> dict:
    vp = fp = fn = ok = tot = 0
    for p in piezas:
        g = p["grafo"]
        voz = g["note"].voz
        pred = predecir(g)
        con = voz >= 0
        ok += int((pred[con] == voz[con]).sum()); tot += int(con.sum())
        vp += int(((pred == 1) & (voz == 1)).sum()); fp += int(((pred == 1) & (voz == 0)).sum()); fn += int(((pred == 0) & (voz == 1)).sum())
    prec, cob = vp / max(vp + fp, 1), vp / max(vp + fn, 1)
    return {"acierto": ok / tot, "precision": prec, "cobertura": cob, "f1": 2 * prec * cob / max(prec + cob, 1e-9)}


def skyline(g):
    on, alt = g["note"].onset_q, g["note"].pitch
    pred = torch.zeros(len(on), dtype=torch.long)
    for t in on.unique():
        idx = (on == t).nonzero().flatten()
        pred[idx[alt[idx].argmax()]] = 1
    return pred


def con_modelo(m):
    def predecir(g):
        with torch.no_grad():
            return m.salida_voz(m.codificar(g.x_dict, g.edge_index_dict)).argmax(-1)
    return predecir


print(f"{len(piezas)} canciones de examen con etiqueta de voz · {sum(int((p['grafo']['note'].voz >= 0).sum()) for p in piezas)} notas")
fila = lambda nom, r: print(f"{nom:28} acierto {r['acierto']:.1%} · melodía: precisión {r['precision']:.1%}, cobertura {r['cobertura']:.1%}, F1 {r['f1']:.1%}")
fila("skyline (la más aguda)", medir(skyline))
res = [medir(con_modelo(cargar_modelo(r)[0].eval())) for r in sys.argv[1:]]
if res:
    fila(f"GNN (media de {len(res)})", {k: st.mean(r[k] for r in res) for k in res[0]})
