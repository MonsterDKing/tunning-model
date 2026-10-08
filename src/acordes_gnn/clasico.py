"""Paso 0c — Corpus clásico de BACHI: obras de piano con análisis armónico de musicólogos.

El repositorio de BACHI en Hugging Face (``Itsuki-music/BACHI_Chord_Recognition``, MIT; los corpus
de origen son *When in Rome*, CC-BY-SA, y DCML, CC-BY-NC-SA: uso personal) trae, por obra, la
partitura (MusicXML o MIDI) y ``chord_symbol.csv``: cada acorde con su posición en negras desde el
inicio (``offset_qb``), tonalidad local, numeral romano, raíz, calidad y bajo. Son análisis
HUMANOS (``analysis_raw.txt`` lleva el nombre del analista): sirven como etiquetas.

Este módulo los convierte en lo mismo que sale de un MusicXML de PDMX, como ``pop909.py``:

- **Notas**: con partitura. En MusicXML los pentagramas dicen qué mano toca; en MIDI no, y la mano
  izquierda se aproxima por altura (por debajo del Do central), como en POP909.
- **Acordes**: la calidad de BACHI (M, m, o, D7, M7, /o7…) se pliega a nuestras 5 calidades.
  ``offset_qb`` está en el eje de music21 (0 = inicio del primer compás); se pasa a
  (compás, offset dentro del compás) con los inicios de compás de music21 y después al eje de
  partitura con ``ubicar``, igual que el cifrado de PDMX. En MIDI el eje es el mismo que el de
  partitura (ticks / ppq).
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np

from acordes_gnn.etiquetas import CALIDAD_M21, AcordeEscrito, clase, clases_de_altura, inicios_de_compas
from acordes_gnn.grafo import Notas, leer_notas

RAICES = {"C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4, "E#": 5, "F": 5,
          "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11}  # fmt: skip
CALIDADES = {"M": "mayor", "D7": "mayor", "M7": "mayor", "m": "menor", "m7": "menor", "mM7": "menor",
             "o": "dim", "o7": "dim", "/o7": "dim", "+": "aum", "+7": "aum", "sus2": "sus", "sus4": "sus"}  # fmt: skip


def partitura_de(carpeta: Path) -> Path | None:
    """El archivo de la partitura de una obra (MusicXML preferido; si no, MIDI)."""
    for patron in ("*.musicxml", "*.mxl", "*.xml", "*.mid", "*.midi"):
        encontrados = sorted(carpeta.glob(patron))
        if encontrados:
            return encontrados[0]
    return None


def _notas_midi(ruta: Path) -> Notas:
    import partitura as pt

    score = pt.load_score_midi(str(ruta))
    na = score.note_array(include_time_signature=True, include_metrical_position=True,
                          include_pitch_spelling=True, include_staff=True, include_key_signature=True)  # fmt: skip
    na = na[np.lexsort((na["pitch"], na["onset_div"]))]
    na["staff"] = np.where(na["pitch"] < 60, 2, 1)  # sin pentagramas: mano izquierda ≈ por debajo del Do central
    parte = score.parts[0]
    medidas = np.array([[m.start.t, m.end.t] for m in parte.measures], dtype=np.int64)
    compas = np.searchsorted(medidas[:, 0], na["onset_div"], side="right") - 1
    return Notas(array=na, compas=np.clip(compas, 0, None), inicios_compas=inicios_de_compas(parte), medidas_div=medidas)


def _inicios_music21(ruta: Path) -> list[float]:
    from music21 import converter, stream

    sc = converter.parse(str(ruta))
    parte = sc.parts[0] if sc.hasPartLikeStreams() else sc
    return [float(m.offset) for m in parte.getElementsByClass(stream.Measure)]


def leer_clasico(carpeta: Path) -> tuple[Notas, list[AcordeEscrito], list[float]]:
    """Obra del corpus clásico → (notas, acordes escritos, tiempos "sin acorde")."""
    warnings.filterwarnings("ignore")
    ruta = partitura_de(carpeta)
    if ruta is None:
        raise ValueError("sin partitura")
    es_midi = ruta.suffix.lower() in (".mid", ".midi")
    notas = _notas_midi(ruta) if es_midi else leer_notas(ruta)
    # inicios de compás en el eje de offset_qb: el de partitura (MIDI) o el de music21 (MusicXML)
    inicios = list(notas.inicios_compas) if es_midi else _inicios_music21(ruta)
    if not es_midi and len(inicios) != len(notas.inicios_compas):
        raise ValueError(f"music21 ve {len(inicios)} compases y partitura {len(notas.inicios_compas)}")
    acordes, sin_acorde = [], []
    with open(carpeta / "chord_symbol.csv", newline="") as f:
        for fila in csv.DictReader(f):
            t = float(fila["offset_qb"])
            i = max(int(np.searchsorted(inicios, t + 1e-6, side="right")) - 1, 0)
            raiz, calidad = RAICES.get(fila["root"]), CALIDADES.get(fila["quality"])
            if raiz is None or calidad is None:  # N, other: sin etiqueta a partir de aquí
                sin_acorde.append(float(notas.inicios_compas[min(i, len(notas.inicios_compas) - 1)]) + t - inicios[i])
                continue
            c = clase(raiz, calidad)
            acordes.append(AcordeEscrito(indice=i, offset=t - float(inicios[i]), figura=f"{fila['root']}{fila['quality']}",
                                         raiz=raiz, calidad=calidad, calidad_m21=CALIDAD_M21[calidad],
                                         alturas=clases_de_altura(c)))  # fmt: skip
    return notas, acordes, sin_acorde
