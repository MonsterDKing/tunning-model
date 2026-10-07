"""Paso 2 — Las etiquetas: qué acorde "suena" debajo de cada nota.

Un modelo supervisado aprende de pares (entrada, respuesta correcta). Aquí la entrada es cada
nota del grafo y la respuesta es el acorde que el ARREGLISTA escribió en el cifrado (<harmony>)
y que está vigente cuando esa nota empieza a sonar. Nada de etiquetas generadas por Claude ni por
el heurístico: solo cifrado humano (más las 20 correcciones revisadas a mano de
``correcciones.json`` para las partituras originales).

Vocabulario (60 clases = 12 raíces × 5 calidades)
--------------------------------------------------
La calidad se deduce de los intervalos que el acorde escrito forma sobre su raíz:

==========  =====================================  ============================
calidad     regla (intervalos desde la raíz)        ejemplos que caen aquí
==========  =====================================  ============================
``dim``     3ª menor + 5ª disminuida, sin 5ª justa  Bdim, Bm7b5, Bdim7
``aum``     3ª mayor + 5ª aumentada, sin 5ª justa   C+, C7#5
``mayor``   3ª mayor                                C, C7, Cmaj7, C6, Cadd9, C9, C/E
``menor``   3ª menor                                Cm, Cm7, Cm6, Cm9
``sus``     sin tercera                             Csus4, Csus2, C5, C7sus4
==========  =====================================  ============================

Las séptimas, novenas, inversiones (C/E) se pliegan a la tríada: con ~190 partituras no hay
ejemplos suficientes para aprender 12 × 20 tipos de acorde, y la métrica del proyecto (raíz +
mayor/menor/dim/aum) tampoco las distingue. El número de clase es ``raiz * 5 + calidad``:
0 = C, 1 = Cm, 2 = Cdim, 3 = Caum, 4 = Csus, 5 = C#, …, 59 = Bsus.

Notas sin etiqueta (``-100``, el valor que ``cross_entropy`` ignora): las que empiezan antes del
primer acorde escrito, después de un "N.C." (sin acorde), o más de ``VIGENCIA_COMPASES`` compases
después del último acorde escrito. Esto último importa: muchos arreglistas solo cifran la primera
sección y, sin límite, el último acorde "sonaría" hasta el final y enseñaría basura al modelo. En
el corpus, el 95 % de los acordes van seguidos de otro a 1–2 compases y saltos de 3–4 compases son
acordes largos legítimos; a partir de 5 compases casi siempre es que se dejó de cifrar.
Una nota que se sostiene a través de un cambio de acorde lleva el acorde de su ATAQUE.

Alineación de tiempos
---------------------
La referencia se lee con music21 (igual que ``validar()`` en Analisis-AI, para que las métricas
sean comparables) y las notas con partitura (lo que usa GraphMuse). Los dos miden en negras pero
con distinto cero: partitura pone la anacrusa en tiempo negativo y music21 en 0. Por eso cada
acorde se ubica por (índice de compás, offset dentro del compás) y se traduce al eje de partitura
con el inicio de ese mismo compás.

Uso::

    uv run etiquetas data/xmls_originales/2_When_I_Was_Your_Man.musicxml
    uv run etiquetas --split train      # resumen de notas sin etiqueta en todo un split
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RAICES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
CALIDADES = ("mayor", "menor", "dim", "aum", "sus")
SUFIJOS = ("", "m", "dim", "aug", "sus")
NUM_CLASES = len(RAICES) * len(CALIDADES)  # 60
SIN_ETIQUETA = -100
VIGENCIA_COMPASES = 4  # un acorde escrito en el compás i etiqueta notas hasta el final del compás i+3

# Conjunto de clases de altura de cada calidad (desde la raíz). "sus" = la quinta sin tercera:
# así "Csus" ⊆ C, Csus4, Csus2, C5 → cuenta como equivalente a cualquiera de ellos.
INTERVALOS = {"mayor": (0, 4, 7), "menor": (0, 3, 7), "dim": (0, 3, 6), "aum": (0, 4, 8), "sus": (0, 7)}

# Para la métrica estricta: lo que music21 devuelve en chord.Chord(...).quality
CALIDAD_M21 = {"mayor": "major", "menor": "minor", "dim": "diminished", "aum": "augmented", "sus": "other"}

CORRECCIONES = Path(__file__).resolve().parents[2] / "data" / "xmls_originales" / "correcciones.json"


def clase(raiz: int, calidad: str) -> int:
    return (raiz % 12) * len(CALIDADES) + CALIDADES.index(calidad)


def raiz_y_calidad(c: int) -> tuple[int, str]:
    return c // len(CALIDADES), CALIDADES[c % len(CALIDADES)]


def nombre(c: int) -> str:
    r, q = raiz_y_calidad(c)
    return RAICES[r] + SUFIJOS[CALIDADES.index(q)]


def clases_de_altura(c: int) -> frozenset[int]:
    r, q = raiz_y_calidad(c)
    return frozenset((r + i) % 12 for i in INTERVALOS[q])


def transponer_clase(c: int, semitonos: int) -> int:
    """Subir k semitonos la partitura sube k semitonos la raíz; la calidad no cambia."""
    if c < 0:
        return c
    r, q = raiz_y_calidad(c)
    return clase(r + semitonos, q)


def calidad_por_intervalos(intervalos: set[int]) -> str:
    """Calidad de un acorde dado el conjunto de intervalos (en semitonos) desde su raíz."""
    if 3 in intervalos and 6 in intervalos and 7 not in intervalos:
        return "dim"
    if 4 in intervalos and 8 in intervalos and 7 not in intervalos:
        return "aum"
    if 4 in intervalos:
        return "mayor"
    if 3 in intervalos:
        return "menor"
    return "sus"


@dataclass
class AcordeEscrito:
    """Un acorde del cifrado humano, tal como lo mide la métrica de Analisis-AI."""

    indice: int  # índice del compás (0 = primero, como en validar())
    offset: float  # negras desde el inicio del compás
    figura: str  # "Am7", "G/B"…
    raiz: int  # clase de altura de la raíz (music21 cs.root())
    calidad: str  # una de CALIDADES (por intervalos)
    calidad_m21: str  # chord.Chord(cs.pitches).quality → lo que compara la métrica estricta
    alturas: frozenset[int]  # clases de altura del acorde escrito (para la métrica "equivalente")
    original: str | None = None  # si se corrigió con correcciones.json, la figura original
    tiempo: float = 0.0  # negras en el eje de partitura (se rellena al alinear)
    hasta: float = float("inf")  # fin de su vigencia en ese eje (ver VIGENCIA_COMPASES)
    clase: int = field(init=False)

    def __post_init__(self) -> None:
        self.clase = clase(self.raiz, self.calidad)


def correcciones(ruta: str | Path) -> dict[tuple[int, float], dict]:
    """Acordes corregidos para esta partitura (clave: índice de compás, offset). Igual que Analisis-AI."""
    if not CORRECCIONES.exists():
        return {}
    filas = json.loads(CORRECCIONES.read_text()).get(Path(ruta).name, [])
    return {(f["compas"] - 1, float(f["offset"])): f for f in filas}


def _desde_simbolo(cs, i: int, off: float, original: str | None = None) -> AcordeEscrito:
    from music21 import chord

    raiz = cs.root().pitchClass
    alturas = frozenset(p.pitchClass for p in cs.pitches)
    return AcordeEscrito(
        indice=i,
        offset=off,
        figura=cs.figure.replace("-", "b"),
        raiz=raiz,
        calidad=calidad_por_intervalos({(a - raiz) % 12 for a in alturas}),
        calidad_m21=chord.Chord(cs.pitches).quality,
        alturas=alturas,
        original=original,
    )


def leer_referencia(ruta: str | Path, score=None) -> tuple[list[AcordeEscrito], list[tuple[int, float]], int]:
    """Acordes escritos (ya corregidos) + posiciones de los "N.C." + nº de compases.

    Recorre las partes y se queda con la primera que tenga cifrado, como ``validar()``.
    """
    from music21 import converter, harmony, stream

    warnings.filterwarnings("ignore")
    score = score or converter.parse(str(ruta))
    arreglos = correcciones(ruta)
    for parte in score.parts:
        acordes, silencios = [], []
        compases = list(parte.getElementsByClass(stream.Measure))
        for i, m in enumerate(compases):
            for cs in m.recurse().getElementsByClass(harmony.ChordSymbol):
                off = float(cs.offset)
                if isinstance(cs, harmony.NoChord):
                    silencios.append((i, off))
                    continue
                if not cs.pitches:
                    continue
                arreglo = arreglos.get((i, off))
                if arreglo:
                    corregido = harmony.ChordSymbol(arreglo["corregido"].replace("b", "-"))
                    acordes.append(_desde_simbolo(corregido, i, off, original=cs.figure.replace("-", "b")))
                else:
                    acordes.append(_desde_simbolo(cs, i, off))
        if acordes:
            return acordes, silencios, len(compases)
    return [], [], 0


def inicios_de_compas(part) -> np.ndarray:
    """Inicio (en negras, eje de partitura) de cada compás de una parte de partitura."""
    return np.array([float(part.quarter_map(m.start.t)) for m in part.measures])


def ubicar(acordes: list[AcordeEscrito], inicios: np.ndarray) -> None:
    """Rellena ``tiempo`` de cada acorde en el eje de partitura (ver "Alineación de tiempos")."""
    for a in acordes:
        base = inicios[a.indice] if a.indice < len(inicios) else inicios[-1]
        a.tiempo = float(base) + a.offset
        fin = a.indice + VIGENCIA_COMPASES
        a.hasta = float(inicios[fin]) if fin < len(inicios) else float("inf")


def etiquetar_notas(
    inicio_notas: np.ndarray, acordes: list[AcordeEscrito], silencios_en_tiempo: list[float] | None = None
) -> np.ndarray:
    """Clase del acorde vigente en el ataque de cada nota (``SIN_ETIQUETA`` si no hay ninguno).

    ``inicio_notas``: tiempo de ataque de cada nota en negras (mismo eje que ``AcordeEscrito.tiempo``).
    ``silencios_en_tiempo``: tiempos de los "N.C." (a partir de ahí no hay acorde hasta el siguiente).
    """
    inf = float("inf")
    eventos = sorted([(a.tiempo, 1, a.clase, a.hasta) for a in acordes]
                     + [(t, 0, SIN_ETIQUETA, inf) for t in silencios_en_tiempo or []])  # fmt: skip
    if not eventos:
        return np.full(len(inicio_notas), SIN_ETIQUETA, dtype=np.int64)
    tiempos = np.array([e[0] for e in eventos])
    clases = np.array([e[2] for e in eventos])
    hastas = np.array([e[3] for e in eventos])
    # índice del último evento con tiempo <= ataque (tolerancia por redondeos de tresillos)
    pos = np.searchsorted(tiempos, inicio_notas + 1e-6, side="right") - 1
    p = np.clip(pos, 0, None)
    y = np.where((pos >= 0) & (inicio_notas < hastas[p] - 1e-6), clases[p], SIN_ETIQUETA)
    return y.astype(np.int64)


def main() -> None:
    from acordes_gnn.datos import cargar_splits, ruta_de
    from acordes_gnn.grafo import leer_notas

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rutas", nargs="*")
    p.add_argument("--split", help="train, dev, test u originales")
    args = p.parse_args()
    rutas = [Path(r) for r in args.rutas]
    if args.split:
        rutas += [ruta_de(n) for n in cargar_splits()[args.split]]
    total, sin, por_clase = 0, 0, Counter()
    for r in rutas:
        notas = leer_notas(r)
        acordes, silencios, _ = leer_referencia(r)
        ubicar(acordes, notas.inicios_compas)
        sil_t = [float(notas.inicios_compas[min(i, len(notas.inicios_compas) - 1)]) + o for i, o in silencios]
        y = etiquetar_notas(notas.array["onset_quarter"], acordes, sil_t)
        n_sin = int((y == SIN_ETIQUETA).sum())
        total += len(y)
        sin += n_sin
        por_clase.update(nombre(int(c)) for c in y if c >= 0)
        corregidos = sum(a.original is not None for a in acordes)
        print(f"{r.name[:50]:50} notas {len(y):5d}  sin etiqueta {n_sin:4d} ({n_sin / max(len(y), 1):5.1%})  "
              f"acordes escritos {len(acordes):4d}" + (f"  corregidos {corregidos}" if corregidos else ""))  # fmt: skip
        if len(rutas) == 1:
            print("\nPrimeras 12 notas → acorde:")
            for k in range(min(12, len(y))):
                fila = notas.array[k]
                print(f"  compás {notas.compas[k] + 1:3d}  t={fila['onset_quarter']:6.2f}  {nombre_nota(fila):5}"
                      f" → {nombre(int(y[k])) if y[k] >= 0 else '(sin etiqueta)'}")  # fmt: skip
    print(f"\nTOTAL {len(rutas)} partituras · {total} notas · sin etiqueta {sin} ({sin / max(total, 1):.1%})")
    print("Clases más frecuentes:", ", ".join(f"{k} {v}" for k, v in por_clase.most_common(12)))
    print(f"Clases usadas: {len(por_clase)} de {NUM_CLASES}")


def nombre_nota(fila) -> str:
    """'Mi4', 'Fa#3'… a partir de una fila del note_array de partitura."""
    solfeo = {"C": "Do", "D": "Re", "E": "Mi", "F": "Fa", "G": "Sol", "A": "La", "B": "Si"}
    alter = {-2: "bb", -1: "b", 0: "", 1: "#", 2: "x"}.get(int(fila["alter"]), "?")
    return f"{solfeo[str(fila['step'])]}{alter}{int(fila['octave'])}"


if __name__ == "__main__":
    main()
