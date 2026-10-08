"""Paso 7 — Prior armónico: qué acorde suele venir después de cuál (Chordonomicon) + Viterbi.

La GNN mira las NOTAS y dice "aquí suena 70 % Am, 25 % F". No sabe que en pop, después de C, F es
mucho más habitual que Am. Chordonomicon (Kantarelis et al., 2024, CC-BY-NC-4.0, uso personal) trae
las progresiones de ~680 000 canciones escritas por usuarios (sin notas). De ahí se cuenta:

    P(siguiente acorde | acorde actual)   en forma RELATIVA: (calidad, calidad, intervalo de raíz)

Relativa = "un menor seguido del mayor 8 semitonos arriba" (vi → IV), igual en cualquier tonalidad.
Es estadística general del lenguaje armónico (de UN acorde al siguiente): no recuerda canciones.

Combinar las dos fuentes es un modelo oculto de Markov (como en reconocimiento de voz):

- los **estados** son los 60 acordes; los **pasos** son los ataques de la partitura (instantes en
  que empieza alguna nota), en orden;
- **emisión** (lo que "oye" la GNN): la media de las probabilidades de las notas de ese ataque;
- **transición**: quedarse en el mismo acorde con probabilidad ``s``; cambiar con ``1 − s``,
  repartido según el prior ``P(b | a)``. El peso ``w`` dice cuánto se cree a la transición frente
  a lo que oye la GNN (``w = 0`` → cada ataque por separado, como sin prior);
- **Viterbi** encuentra la secuencia de acordes más probable sumando las dos cosas (programación
  dinámica: en cada paso, para cada acorde, el mejor camino que llega a él).

No usa dónde están escritos los acordes (eso sería ventaja sobre el heurístico): decodifica todos
los ataques y luego se lee el acorde en cada instante del cifrado, como ``predecir_en_puntos``.

Uso::

    uv run python -m acordes_gnn.prior     # cuenta las transiciones → data/chordonomicon/prior.npy
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import torch

from acordes_gnn.datos import DATOS
from acordes_gnn.etiquetas import CALIDADES, NUM_CLASES, clase

CSV = DATOS / "chordonomicon" / "chordonomicon_v2.csv"
PRIOR = DATOS / "chordonomicon" / "prior.npy"
NOTAS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SIMBOLO = re.compile(r"^([A-G])(s(?!us)|b)?([^/]*)")  # raíz, alteración ("s" = sostenido, salvo en "sus"), resto


def clase_de_simbolo(simbolo: str) -> int | None:
    """'Fsmin7/A' → clase de F#m. Mismo pliegue que PDMX: séptimas y extensiones → su tríada."""
    m = SIMBOLO.match(simbolo)
    if not m:
        return None
    raiz = (NOTAS[m.group(1)] + {"s": 1, "b": -1, None: 0}[m.group(2)]) % 12
    resto = m.group(3)
    if resto.startswith("min"):
        calidad = "dim" if "b5" in resto else "menor"  # min7b5 = semidisminuido
    elif resto.startswith(("dim", "o")):
        calidad = "dim"
    elif resto.startswith(("aug", "+")):
        calidad = "aum"
    elif resto.startswith(("sus", "5")):  # sus2, sus4 y power chords (C5): sin tercera
        calidad = "sus"
    else:
        calidad = "mayor"  # "", 7, maj7, add9, 6, 9…
    return clase(raiz, calidad)


def contar_transiciones(ruta: Path = CSV) -> np.ndarray:
    """Cuentas relativas [calidad_a, calidad_b, intervalo] de cambios de acorde (sin repeticiones)."""
    csv.field_size_limit(10**9)
    cuenta = np.zeros((len(CALIDADES), len(CALIDADES), 12))
    with open(ruta, newline="", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            previo = None
            for t in fila["chords"].split():
                if t.startswith("<"):
                    continue  # etiquetas de sección: la progresión sigue
                c = clase_de_simbolo(t)
                if c is None or c == previo:
                    continue  # solo CAMBIOS de acorde: la permanencia la modela s
                if previo is not None:
                    ra, qa = divmod(previo, len(CALIDADES))
                    rb, qb = divmod(c, len(CALIDADES))
                    cuenta[qa, qb, (rb - ra) % 12] += 1
                previo = c
    return cuenta


def matriz_cambio(cuenta: np.ndarray, suavizado: float = 1.0) -> np.ndarray:
    """log P(b | a, cambio) para las 60 × 60 clases (la diagonal no se usa: "quedarse" es s)."""
    log_t = np.full((NUM_CLASES, NUM_CLASES), -np.inf)
    for a in range(NUM_CLASES):
        ra, qa = divmod(a, len(CALIDADES))
        fila = np.array([cuenta[qa, b % len(CALIDADES), (b // len(CALIDADES) - ra) % 12] + suavizado
                         if b != a else 0.0 for b in range(NUM_CLASES)])  # fmt: skip
        log_t[a, np.arange(NUM_CLASES) != a] = np.log(fila[np.arange(NUM_CLASES) != a] / fila.sum())
    return log_t


def viterbi(emision: np.ndarray, log_cambio: np.ndarray, s: float, w: float) -> np.ndarray:
    """Secuencia de clases más probable. ``emision``: (pasos, 60) probabilidades de la GNN."""
    log_e = np.log(emision + 1e-9)
    trans = w * (np.log(1 - s) + log_cambio)
    np.fill_diagonal(trans, w * np.log(s))
    puntos = log_e[0].copy()
    de_donde = np.zeros(emision.shape, dtype=np.int64)
    for k in range(1, len(emision)):
        candidatos = puntos[:, None] + trans  # [desde, hacia]
        de_donde[k] = candidatos.argmax(0)
        puntos = candidatos.max(0) + log_e[k]
    camino = np.zeros(len(emision), dtype=np.int64)
    camino[-1] = puntos.argmax()
    for k in range(len(emision) - 1, 0, -1):
        camino[k - 1] = de_donde[k, camino[k]]
    return camino


def predecir_con_prior(prob: torch.Tensor, onset: torch.Tensor, dur: torch.Tensor, tiempos: list[float],
                       log_cambio: np.ndarray, s: float, w: float) -> list[int]:  # fmt: skip
    """Como ``predecir_en_puntos``, pero decodificando todos los ataques con Viterbi."""
    from acordes_gnn.evaluar import notas_del_instante

    onset_np = onset.numpy()
    ataques, inversa = np.unique(np.round(onset_np, 4), return_inverse=True)
    emision = np.zeros((len(ataques), prob.shape[1]))
    np.add.at(emision, inversa, prob.numpy())
    emision /= np.bincount(inversa, minlength=len(ataques))[:, None]
    camino = viterbi(emision, log_cambio, s, w)
    preds = []
    for t in tiempos:
        # el paso del instante t: el ataque de las notas que usaría predecir_en_puntos
        k = np.searchsorted(ataques, round(float(onset_np[notas_del_instante(onset, dur, t)[0]]), 4))
        preds.append(int(camino[min(k, len(camino) - 1)]))
    return preds


def predecir_por_segmentos(prob: torch.Tensor, onset: torch.Tensor, dur: torch.Tensor, tiempos: list[float],
                           log_cambio: np.ndarray, w: float) -> list[int]:  # fmt: skip
    """Variante sin "peaje por quedarse": la GNN decide DÓNDE cambia el acorde y Viterbi QUÉ es.

    1. Cada ataque toma su acorde más probable; los ataques seguidos con el mismo acorde forman un
       segmento (la segmentación es la de la GNN, no la del cifrado escrito).
    2. Viterbi recorre los SEGMENTOS: emisión = media de las probabilidades de sus ataques;
       transición = ``w · log P(b | a)`` del prior (y 0 si el acorde se repite: dos segmentos
       vecinos pueden acabar siendo el mismo acorde).
    """
    from acordes_gnn.evaluar import notas_del_instante

    onset_np = onset.numpy()
    ataques, inversa = np.unique(np.round(onset_np, 4), return_inverse=True)
    emision = np.zeros((len(ataques), prob.shape[1]))
    np.add.at(emision, inversa, prob.numpy())
    emision /= np.bincount(inversa, minlength=len(ataques))[:, None]
    mejor = emision.argmax(1)
    segmento = np.concatenate([[0], np.cumsum(mejor[1:] != mejor[:-1])])
    em_seg = np.zeros((segmento[-1] + 1, emision.shape[1]))
    np.add.at(em_seg, segmento, emision)
    em_seg /= np.bincount(segmento)[:, None]
    trans = w * log_cambio.copy()
    np.fill_diagonal(trans, 0.0)
    log_e = np.log(em_seg + 1e-9)
    puntos, de_donde = log_e[0].copy(), np.zeros(em_seg.shape, dtype=np.int64)
    for k in range(1, len(em_seg)):
        candidatos = puntos[:, None] + trans
        de_donde[k] = candidatos.argmax(0)
        puntos = candidatos.max(0) + log_e[k]
    camino = np.zeros(len(em_seg), dtype=np.int64)
    camino[-1] = puntos.argmax()
    for k in range(len(em_seg) - 1, 0, -1):
        camino[k - 1] = de_donde[k, camino[k]]
    preds = []
    for t in tiempos:
        k = np.searchsorted(ataques, round(float(onset_np[notas_del_instante(onset, dur, t)[0]]), 4))
        preds.append(int(camino[segmento[min(k, len(segmento) - 1)]]))
    return preds


def main() -> None:
    cuenta = contar_transiciones()
    np.save(PRIOR, cuenta)
    total = cuenta.sum()
    print(f"{int(total):,} cambios de acorde contados → {PRIOR}")
    nombres_q = ["", "m", "dim", "aug", "sus"]
    grados = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]
    for qa, desde in ((0, "un mayor"), (1, "un menor")):
        fila = cuenta[qa] / cuenta[qa].sum()
        top = np.argsort(fila, axis=None)[::-1][:6]
        print(f"después de {desde}:", ", ".join(f"{grados[i % 12]}{nombres_q[i // 12]} {fila.flat[i]:.0%}" for i in top))


if __name__ == "__main__":
    main()
