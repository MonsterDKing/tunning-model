"""Paso 0b — POP909-CL: 909 canciones pop para piano (MIDI) con acordes corregidos por humanos.

POP909-CL (Yao, Chen, Dubnov & Berg-Kirkpatrick, BACHI, ICASSP 2026; licencia MIT) amplía POP909
(Wang et al., ISMIR 2020) con acordes revisados por expertos. Cada MIDI de ``POP909_processed/``
trae tres pistas: metadatos (tempo, compás, armadura), ``piano`` (melodía + acompañamiento juntos,
UN solo canal: no dice qué mano toca cada nota) y la de acordes (``chords`` en 453 archivos,
``MIDI 01`` en 454: se toma la última pista, como hace su propio script): cada acorde escrito como
un bloque de notas que empieza donde empieza el acorde. Dos archivos vienen incompletos (658 sin
piano, uno sin acordes): se descartan al leerlos.

Este módulo convierte cada MIDI en lo mismo que sale de un MusicXML de PDMX (``Notas`` + acordes
escritos + "sin acorde"), así el grafo, las etiquetas y la métrica no cambian:

- **Notas**: solo la pista ``piano``, leída con partitura (compases desde el compás del MIDI). Un
  MIDI no dice qué mano toca cada nota: se aproxima con la altura (por debajo del Do central =
  mano izquierda). Es una aproximación: el rasgo ``mano_izquierda`` es más ruidoso que en PDMX.
- **Acordes**: las notas de cada bloque se comparan con las plantillas que usa POP909-CL (7 séptimas
  y 6 tríadas, en ese orden) y se pliegan a nuestras 60 clases como el cifrado de PDMX (D7, maj7 →
  mayor; m7 → menor; ø7, °7 → dim…). Un bloque que no encaja en ninguna plantilla, o un hueco entre
  bloques, cuenta como "sin acorde" (sus notas quedan sin etiqueta).

Las canciones 518 y 620 conservan etiquetas del algoritmo (no humanas, lo dice el README): fuera.
Ninguna de las 909 repite canción de test/dev: se comprobó el 2026-10-07 contra el índice de POP909
(``POP909/index.xlsx`` del repositorio original: título y artista, casi todo pop chino).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from acordes_gnn.etiquetas import CALIDAD_M21, RAICES, AcordeEscrito, inicios_de_compas
from acordes_gnn.grafo import Notas


# Plantillas de POP909-CL (process_pop909.py): intervalos desde la raíz → nuestra calidad
SEPTIMAS = {
    frozenset({0, 4, 7, 10}): ("7", "mayor"),
    frozenset({0, 4, 7, 11}): ("maj7", "mayor"),
    frozenset({0, 3, 7, 10}): ("m7", "menor"),
    frozenset({0, 3, 6, 10}): ("m7b5", "dim"),
    frozenset({0, 3, 6, 9}): ("dim7", "dim"),
    frozenset({0, 3, 7, 11}): ("mMaj7", "menor"),
    frozenset({0, 4, 8, 10}): ("aug7", "aum"),
}
TRIADAS = {
    frozenset({0, 4, 7}): ("", "mayor"),
    frozenset({0, 3, 7}): ("m", "menor"),
    frozenset({0, 3, 6}): ("dim", "dim"),
    frozenset({0, 4, 8}): ("aug", "aum"),
    frozenset({0, 2, 7}): ("sus2", "sus"),
    frozenset({0, 5, 7}): ("sus4", "sus"),
}


def nombrar(alturas: frozenset[int]) -> tuple[int, str, str] | None:
    """(raíz, sufijo, calidad) del bloque de notas, o None si no encaja en ninguna plantilla."""
    for plantillas in (SEPTIMAS, TRIADAS):
        for raiz in sorted(alturas):
            encaje = plantillas.get(frozenset((a - raiz) % 12 for a in alturas))
            if encaje:
                return raiz, *encaje
    return None


def leer_pop909(ruta: str | Path) -> tuple[Notas, list[AcordeEscrito], list[float]]:
    """MIDI de POP909-CL → (notas de la pista piano, acordes escritos, tiempos "sin acorde")."""
    import mido
    import partitura as pt

    warnings.filterwarnings("ignore")
    midi = mido.MidiFile(str(ruta))
    if len(midi.tracks) != 3 or midi.tracks[1].name != "piano":
        raise ValueError("falta la pista de piano o la de acordes")
    pista_piano, pista_acordes = midi.tracks[1], midi.tracks[2]
    solo_piano = mido.MidiFile(ticks_per_beat=midi.ticks_per_beat)
    solo_piano.tracks = [midi.tracks[0], pista_piano]
    score = pt.load_score_midi(solo_piano)
    na = score.note_array(include_time_signature=True, include_metrical_position=True,
                          include_pitch_spelling=True, include_staff=True, include_key_signature=True)  # fmt: skip
    na = na[np.lexsort((na["pitch"], na["onset_div"]))]
    na["staff"] = np.where(na["pitch"] < 60, 2, 1)  # sin pentagramas: mano izquierda ≈ por debajo del Do central
    parte = score.parts[0]
    medidas = np.array([[m.start.t, m.end.t] for m in parte.measures], dtype=np.int64)
    compas = np.searchsorted(medidas[:, 0], na["onset_div"], side="right") - 1
    inicios = inicios_de_compas(parte)
    notas = Notas(array=na, compas=np.clip(compas, 0, None), inicios_compas=inicios, medidas_div=medidas)

    # Bloques de la pista de acordes: notas que empiezan a la vez (en negras desde el inicio del MIDI)
    bloques: dict[int, list[tuple[int, int]]] = {}
    t, abiertas = 0, {}
    for msg in pista_acordes:
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            abiertas[msg.note] = t
        elif msg.type in ("note_off", "note_on") and msg.note in abiertas:
            inicio = abiertas.pop(msg.note)
            bloques.setdefault(inicio, []).append((msg.note, t))
    acordes, sin_acorde = [], []
    inicios_bloque = sorted(bloques)
    for k, inicio in enumerate(inicios_bloque):
        tiempo = inicio / midi.ticks_per_beat
        fin = max(f for _, f in bloques[inicio]) / midi.ticks_per_beat
        nombre = nombrar(frozenset(n % 12 for n, _ in bloques[inicio]))
        if nombre is None:
            sin_acorde.append(tiempo)  # no encaja en ninguna plantilla: sin etiqueta
        else:
            raiz, sufijo, calidad = nombre
            i = max(int(np.searchsorted(inicios, tiempo + 1e-6, side="right")) - 1, 0)
            acordes.append(AcordeEscrito(
                indice=i, offset=tiempo - float(inicios[i]), figura=RAICES[raiz] + sufijo, raiz=raiz,
                calidad=calidad, calidad_m21=CALIDAD_M21[calidad],
                alturas=frozenset(n % 12 for n, _ in bloques[inicio])))  # fmt: skip
        siguiente = inicios_bloque[k + 1] / midi.ticks_per_beat if k + 1 < len(inicios_bloque) else None
        if siguiente is None or siguiente > fin + 1e-6:
            sin_acorde.append(fin)  # hueco hasta el siguiente acorde (o final): "N.C."
    return notas, acordes, sin_acorde
