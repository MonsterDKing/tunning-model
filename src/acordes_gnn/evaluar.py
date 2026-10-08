"""Paso 5 — Evaluar con LA MISMA métrica que Analisis-AI.

El modelo predice un acorde POR NOTA, pero la referencia es un acorde por PUNTO del cifrado
(donde el arreglista escribió un <harmony>). ``validar()`` de Analisis-AI mira, en cada uno de esos
puntos, qué acorde dice el analizador en ese instante. Aquí hacemos lo mismo:

¿Qué acorde "dice" el modelo en el instante t de un acorde escrito?
-------------------------------------------------------------------
Se promedian las probabilidades (softmax) de las notas que:

1. ATACAN en t (empiezan justo ahí); si no hay ninguna,
2. SUENAN en t (empezaron antes y siguen sonando); si tampoco (un silencio),
3. las del PRIMER ataque después de t (y, al final de la pieza, las del último antes de t).

y gana la clase con más probabilidad media. ¿Por qué así? Es lo que hace un oyente: el acorde de
un instante es lo que suena en ese instante. Preferir los ataques evita que una nota larga del
acorde anterior (cuya etiqueta es la del acorde ANTERIOR, ver etiquetas.py) vote en el nuevo.
Promediar probabilidades (en vez de votar cada nota su favorita) deja que una nota segura pese
más que una dudosa. Importante: solo usamos el instante t, no dónde termina el acorde escrito; el
heurístico y Claude tampoco conocen la segmentación de la referencia.

Métricas (las tres de la tabla de línea base)
---------------------------------------------
- **raíz**: la raíz predicha = ``cs.root()`` del acorde escrito.
- **raíz + calidad** (estricta): además, la calidad predicha (mayor/menor/dim/aum/sus→"other")
  = ``chord.Chord(cs.pitches).quality`` de music21, exactamente como ``validar()``. Ojo con una
  rareza heredada: music21 llama "minor" a C6 (lo lee como Am7) y "diminished" a Cm6; esos
  acordes penalizan igual al heurístico, a Claude y a este modelo, así que la comparación es justa.
- **equivalente**: acierto si las clases de altura de la predicción ⊆ las del acorde escrito o
  al revés (C ⊆ C7, Am ⊆ Fmaj7, Csus = {C, G} ⊆ C) o si ya acertó la estricta.

Uso::

    uv run evaluar                                  # modelos/mejor.pt en test
    uv run evaluar --split dev --modelo modelos/mejor.pt
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch

from acordes_gnn.datos import RAIZ
from acordes_gnn.etiquetas import CALIDAD_M21, NUM_CLASES, clases_de_altura, nombre, raiz_y_calidad

# Set de prueba (30 partituras, 1532 acordes), medido en Analisis-AI
LINEA_BASE = {
    "Heurístico (Analisis-AI)": {"raiz": 0.807, "calidad": 0.762, "equivalente": 0.831},
    "Claude Opus (prompt v3)": {"raiz": 0.908, "calidad": 0.884, "equivalente": 0.926},
}


def notas_del_instante(onset: torch.Tensor, dur: torch.Tensor, t: float, eps: float = 1e-4) -> torch.Tensor:
    """Índices de las notas que deciden el acorde del instante t (ver docstring del módulo)."""
    ataques = (onset - t).abs() < eps
    if ataques.any():
        return ataques.nonzero().flatten()
    sonando = (onset < t) & (onset + dur > t + eps)
    if sonando.any():
        return sonando.nonzero().flatten()
    despues = onset > t
    if despues.any():
        primero = onset[despues].min()
        return ((onset - primero).abs() < eps).nonzero().flatten()
    ultimo = onset.max()
    return ((onset - ultimo).abs() < eps).nonzero().flatten()


def predecir_en_puntos(prob: torch.Tensor, onset: torch.Tensor, dur: torch.Tensor, tiempos: list[float]) -> list[int]:
    """Clase predicha en cada instante de ``tiempos`` a partir de las probabilidades por nota."""
    return [int(prob[notas_del_instante(onset, dur, t)].mean(0).argmax()) for t in tiempos]


def comparar(pred: int, ref: dict) -> tuple[bool, bool, bool]:
    """(raíz, raíz+calidad, equivalente) de una predicción frente a un acorde escrito."""
    raiz, calidad = raiz_y_calidad(pred)
    ok_raiz = raiz == ref["raiz"]
    ok_calidad = ok_raiz and CALIDAD_M21[calidad] == ref["calidad_m21"]
    alturas, escrito = clases_de_altura(pred), frozenset(ref["alturas"])
    ok_equiv = ok_calidad or alturas <= escrito or escrito <= alturas
    return ok_raiz, ok_calidad, ok_equiv


@torch.no_grad()
def evaluar_piezas(modelo: torch.nn.Module, piezas: list[dict], detalle: bool = False) -> dict:
    """Métricas por punto de cifrado (y por nota) sobre una lista de piezas de la caché."""
    modelo.eval()
    total = Counter()
    filas = []
    for p in piezas:
        g = p["grafo"]
        logits = modelo(g.x_dict, g.edge_index_dict, onset=g["note"].onset_q)
        prob = logits.softmax(-1)
        y = g["note"].y
        con = y >= 0
        total["notas"] += int(con.sum())
        total["notas_ok"] += int((logits.argmax(-1)[con] == y[con]).sum())
        # predecir_en_puntos recorre los acordes uno a uno: en GPU cada uno esperaría a la tarjeta,
        # así que se traen las probabilidades a la CPU UNA vez por partitura
        preds = predecir_en_puntos(prob.cpu(), g["note"].onset_q.cpu(), g["note"].dur_q.cpu(), [a["tiempo"] for a in p["acordes"]])
        c = Counter()
        fallos = []
        for pred, ref in zip(preds, p["acordes"]):
            r, q, e = comparar(pred, ref)
            c.update(acordes=1, raiz=r, calidad=q, equivalente=e)
            if detalle and not q:
                fallos.append({"compas": ref["indice"] + 1, "offset": ref["offset"], "referencia": ref["figura"],
                               "predicho": nombre(pred), "raiz_ok": r, "equivalente": e})  # fmt: skip
        total.update(c)
        filas.append({"partitura": p["nombre"], **{k: c[k] for k in ("acordes", "raiz", "calidad", "equivalente")},
                      **({"fallos": fallos} if detalle else {})})  # fmt: skip
    n = max(total["acordes"], 1)
    return {
        "acordes": total["acordes"],
        "raiz": total["raiz"] / n,
        "calidad": total["calidad"] / n,
        "equivalente": total["equivalente"] / n,
        "precision_por_nota": total["notas_ok"] / max(total["notas"], 1),
        "partituras": filas,
    }


def cargar_modelo(ruta: Path):
    from acordes_gnn.modelo import crear_modelo

    ck = torch.load(ruta, weights_only=False, map_location="cpu")  # aunque se entrenara en GPU
    cfg = ck["config"]
    modelo = crear_modelo(cfg["arquitectura"], ck["metadata"], voz=cfg.get("peso_voz", 0) > 0, oculto=cfg["oculto"],
                          capas=cfg["capas"], dropout=cfg["dropout"], clases=NUM_CLASES)  # fmt: skip
    modelo.load_state_dict(ck["estado"])
    return modelo, ck


def imprimir_tabla(res: dict, etiqueta: str = "Este modelo (GNN)") -> None:
    print(f"\n{'':28} {'raíz':>7} {'raíz+calidad':>13} {'equivalente':>12}")
    for k, v in LINEA_BASE.items():
        print(f"{k:28} {v['raiz']:7.1%} {v['calidad']:13.1%} {v['equivalente']:12.1%}")
    print(f"{etiqueta:28} {res['raiz']:7.1%} {res['calidad']:13.1%} {res['equivalente']:12.1%}"
          f"   ({res['acordes']} acordes; por nota {res['precision_por_nota']:.1%})")  # fmt: skip


def main() -> None:
    from acordes_gnn.dataset import cargar, nombres_de

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modelo", type=Path, default=RAIZ / "modelos" / "mejor.pt")
    p.add_argument("--split", default="test", choices=["test", "test_ampliado", "dev", "dev_ampliado", "dev_todo", "train"])
    args = p.parse_args()
    torch.set_num_threads(4)  # ver --hilos en entrenar.py
    modelo, ck = cargar_modelo(args.modelo)
    piezas = [cargar(n) for n in nombres_de(args.split)]
    res = evaluar_piezas(modelo, piezas, detalle=True)
    print(f"Modelo {args.modelo.name} (época {ck.get('epoca')}, {ck['config']['arquitectura']}) en {args.split}:")
    for f in res["partituras"]:
        n = max(f["acordes"], 1)
        print(f"  {f['partitura'][:46]:46} {f['acordes']:4d}  raíz {f['raiz'] / n:5.0%}  "
              f"+calidad {f['calidad'] / n:5.0%}  equiv. {f['equivalente'] / n:5.0%}")  # fmt: skip
    if args.split == "test":
        imprimir_tabla(res)
    else:
        print(f"\nraíz {res['raiz']:.1%} · raíz+calidad {res['calidad']:.1%} · equivalente {res['equivalente']:.1%}")
    salida = RAIZ / "salidas" / f"evaluacion_{args.split}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\nDetalle de cada fallo: {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
