"""Ajusta el prior armónico (s, w) en dev_todo y lo mide en test, con los modelos del Experimento 13.

    PYTHONDONTWRITEBYTECODE=1 uv run python scripts/probar_prior.py modelos/combo_s0.pt modelos/combo_s1.pt modelos/combo_s2.pt
"""

import statistics as st
import sys
from collections import Counter

import numpy as np
import torch

from acordes_gnn.dataset import cargar, nombres_de
from acordes_gnn.etiquetas import nombre
from acordes_gnn.evaluar import cargar_modelo, comparar, predecir_en_puntos
from acordes_gnn.prior import PRIOR, matriz_cambio, predecir_con_prior, predecir_por_segmentos

torch.set_num_threads(8)
log_cambio = matriz_cambio(np.load(PRIOR))
modelos = [cargar_modelo(r)[0].eval() for r in sys.argv[1:]]


def probabilidades(split):
    """[(prob, onset, dur, acordes)] por modelo y partitura: la GNN se ejecuta una sola vez."""
    piezas = [cargar(n) for n in nombres_de(split)]
    salida = []
    for m in modelos:
        filas = []
        for p in piezas:
            g = p["grafo"]
            with torch.no_grad():
                prob = m(g.x_dict, g.edge_index_dict, onset=g["note"].onset_q).softmax(-1)
            filas.append((prob, g["note"].onset_q, g["note"].dur_q, p["acordes"]))
        salida.append(filas)
    return salida


def acierto(datos, s=None, w=None, confusiones=None):
    por_modelo = []
    for filas in datos:
        ok = n = 0
        for prob, on, du, A in filas:
            tiempos = [a["tiempo"] for a in A]
            if w is None:
                preds = predecir_en_puntos(prob, on, du, tiempos)
            elif s == "segmentos":
                preds = predecir_por_segmentos(prob, on, du, tiempos, log_cambio, w)
            else:
                preds = predecir_con_prior(prob, on, du, tiempos, log_cambio, s, w)
            for c, a in zip(preds, A):
                bien = comparar(c, a)[1]
                ok += bien
                n += 1
                if confusiones is not None and not bien:
                    confusiones[(a["figura"].split("/")[0], nombre(c))] += 1
        por_modelo.append(ok / n)
    return st.mean(por_modelo), por_modelo


dev = probabilidades("dev_todo")
print(f"sin prior (como siempre):        {acierto(dev)[0]:.2%}")
print(f"Viterbi con w = 0 (comprobación): {acierto(dev, 0.9, 0.0)[0]:.2%}")
resultados = {}
for s in ("segmentos", 0.5, 0.7, 0.8, 0.9, 0.95):
    for w in (0.05, 0.1, 0.25, 0.5, 1.0):
        resultados[(s, w)] = acierto(dev, s, w)[0]
        print(f"   s {s!s:9}  w {w:4}  dev {resultados[(s, w)]:.2%}", flush=True)
s, w = max(resultados, key=resultados.get)
print(f"\nMEJOR EN DEV: s = {s}, w = {w} → {resultados[(s, w)]:.2%}\n")
for split in ("test", "test_ampliado"):
    datos = probabilidades(split)
    base, base_m = acierto(datos)
    con, con_m = acierto(datos, s, w)
    print(f"{split:14} sin prior {base:.2%} {[round(x * 100, 1) for x in base_m]} · con prior {con:.2%} {[round(x * 100, 1) for x in con_m]}")
