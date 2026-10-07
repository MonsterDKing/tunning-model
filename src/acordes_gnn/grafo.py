"""Paso 1 — De partitura a grafo.

Una red neuronal no "lee" una partitura: necesita números. La idea de GraphMuse es convertir la
partitura en un GRAFO donde cada nota es un nodo y las aristas dicen cómo se relacionan dos notas
en el tiempo. Así el modelo puede mirar, para cada nota, qué otras suenan con ella, cuáles vienen
antes y después, etc.

Nodos ``note`` (uno por nota; las ligaduras ya vienen fusionadas por partitura). Cada uno lleva un
vector de rasgos (features) — ver ``NOMBRES_RASGOS``:

- clase de altura one-hot (12): Do, Do#, … Si. Es lo más importante para reconocer acordes.
- altura normalizada (1): (midi − 60) / 24. Permite saber qué nota es más grave/aguda.
- duración (1): 1 − tanh(duración / compás), como en GraphMuse: notas largas → valor bajo.
- posición en el compás (1): 0 = primer tiempo, 0.5 = mitad del compás…
- tiempo fuerte (1): 1 si cae en el primer tiempo del compás.
- en el pulso (1): 1 si empieza justo en un pulso (no a contratiempo).
- mano izquierda (1): 1 si está en el pentagrama de abajo (suele llevar la armonía).
- grado en la armadura (12): qué grado de la escala mayor de la armadura es la nota (1, ♭2, 2…
  7). En Re mayor (2♯) un Fa♯ es la 3ª. Es el "director" que sabe la tonalidad: la armadura la
  escribió el arreglista. Ojo: no distingue mayor de su relativa menor (Do mayor y La menor no
  llevan alteraciones), así que el grado se cuenta siempre desde la tónica MAYOR.

Aristas entre notas (las calcula GraphMuse en C, ``create_score_graph``):

- ``onset``: las dos notas empiezan a la vez (un acorde plaqué).
- ``consecutive``: la segunda empieza justo cuando termina la primera (melodía, arpegio).
- ``during``: la segunda empieza mientras la primera todavía suena (bajo largo + melodía encima).
- ``rest``: entre las dos hay un silencio; conecta con lo siguiente que suena.
- ``*_rev``: las mismas al revés, para que la información fluya en los dos sentidos.

Además (grafo "métrico") se añaden nodos virtuales ``measure`` (uno por compás) y ``beat`` (uno
por pulso) conectados a sus notas. Un nodo-pulso junta todo lo que suena en ese pulso: es la
"ventana" del heurístico, pero aprendida.

¿Por qué un grafo? El heurístico de Analisis-AI decide el acorde sumando duraciones en ventanas
y premiando a mano lo que suena al principio (PESO_ATAQUE), castigando la nota más aguda porque
suele ser melodía con notas de paso (PESO_MELODIA) o premiando la raíz en el bajo (PESO_BAJO).
Una GNN aprende esos pesos (y otros que no se nos ocurrieron) a partir de los ejemplos: cada capa
mezcla la información de una nota con la de sus vecinas por cada tipo de arista.

Uso::

    uv run explorar-grafo data/xmls_originales/2_When_I_Was_Your_Man.musicxml
    uv run explorar-grafo <ruta> --compas 5
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

NOMBRES_RASGOS = [f"pc_{n}" for n in ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")] + [
    "altura",
    "duracion",
    "posicion_compas",
    "tiempo_fuerte",
    "en_pulso",
    "mano_izquierda",
] + [f"grado_{g}" for g in ("1", "b2", "2", "b3", "3", "4", "#4", "5", "b6", "6", "b7", "7")]
NUM_RASGOS = len(NOMBRES_RASGOS)  # 30

EXPLICACION_ARISTAS = {
    "onset": "empieza a la vez que",
    "consecutive": "termina justo cuando empieza",
    "during": "sigue sonando cuando empieza",
    "rest": "después de un silencio, va seguida de",
    "consecutive_rev": "empieza justo cuando termina",
    "during_rev": "empieza mientras suena",
    "rest_rev": "viene, tras un silencio, después de",
}


@dataclass
class Notas:
    """Las notas de una partitura, ya ordenadas, más lo necesario para alinear compases."""

    array: np.ndarray  # note_array estructurado de partitura (ordenado por ataque y altura)
    compas: np.ndarray  # índice de compás de cada nota
    inicios_compas: np.ndarray  # inicio de cada compás en negras (eje de partitura)
    medidas_div: np.ndarray  # [[inicio, fin], …] de cada compás en "divisions" (para GraphMuse)


def leer_notas(ruta: str | Path) -> Notas:
    """Lee un MusicXML con partitura y devuelve las notas ordenadas."""
    import partitura as pt

    from acordes_gnn.etiquetas import inicios_de_compas

    warnings.filterwarnings("ignore")
    score = pt.load_musicxml(str(ruta))
    na = score.note_array(include_time_signature=True, include_metrical_position=True,
                          include_pitch_spelling=True, include_staff=True, include_key_signature=True)  # fmt: skip
    if len(na) == 0:
        raise ValueError("la partitura no tiene notas")
    # GraphMuse asigna notas a compases recorriendo en orden: hay que ordenar por ataque (y altura)
    na = na[np.lexsort((na["pitch"], na["onset_div"]))]
    parte = score.parts[0]
    medidas = np.array([[m.start.t, m.end.t] for m in parte.measures], dtype=np.int64)
    compas = np.searchsorted(medidas[:, 0], na["onset_div"], side="right") - 1
    return Notas(array=na, compas=np.clip(compas, 0, None), inicios_compas=inicios_de_compas(parte), medidas_div=medidas)


def rasgos_fijos(na: np.ndarray) -> np.ndarray:
    """Los rasgos que NO cambian al transponer (todo menos la altura)."""
    duracion = 1 - np.tanh(na["duration_beat"] / na["ts_beats"])
    posicion = na["rel_onset_div"] / np.maximum(na["tot_measure_div"], 1)
    fuerte = na["is_downbeat"].astype(float)
    en_pulso = (np.abs(na["onset_beat"] - np.round(na["onset_beat"])) < 1e-6).astype(float)
    izquierda = (na["staff"] >= 2).astype(float)
    return np.stack([duracion, posicion, fuerte, en_pulso, izquierda], axis=1).astype(np.float32)


def tonica_de_armadura(quintas: np.ndarray) -> np.ndarray:
    """Clase de altura de la tónica mayor: cada sostenido sube una quinta (7 semitonos) desde Do.

    0 → Do, 1♯ → Sol, 2♯ → Re, 1♭ (−1) → Fa, 2♭ → Si♭… (el círculo de quintas).
    """
    return (7 * quintas) % 12


def calcular_rasgos(altura: torch.Tensor, fijos: torch.Tensor, tonica: torch.Tensor) -> torch.Tensor:
    """Vector de rasgos de cada nota: [clase de altura (12), altura, fijos (5), grado en la armadura (12)]."""
    pc = torch.nn.functional.one_hot(altura % 12, 12).float()
    grado = torch.nn.functional.one_hot((altura - tonica) % 12, 12).float()
    return torch.cat([pc, ((altura.float() - 60) / 24).unsqueeze(1), fijos, grado], dim=1)


def actualizar_rasgos(grafo) -> None:
    """Recalcula ``x`` de notas, compases y pulsos (después de transponer, por ejemplo)."""
    nota = grafo["note"]
    nota.x = calcular_rasgos(nota.pitch, nota.fijos, nota.tonica)
    for tipo, grupo in (("measure", "measure_cluster"), ("beat", "beat_cluster")):
        if tipo in grafo.node_types:
            x = torch.zeros(grafo[tipo].num_nodes, nota.x.shape[1])
            x.index_add_(0, nota[grupo], nota.x)
            grafo[tipo].x = x


def construir_grafo(notas: Notas, etiquetas: np.ndarray | None = None):
    """Notas → ``HeteroData`` de PyTorch Geometric con GraphMuse."""
    import graphmuse as gm

    na = notas.array
    fijos = torch.from_numpy(rasgos_fijos(na))
    altura = torch.from_numpy(na["pitch"].astype(np.int64))
    tonica = torch.from_numpy(tonica_de_armadura(na["ks_fifths"].astype(np.int64)))
    x = calcular_rasgos(altura, fijos, tonica)
    grafo = gm.create_score_graph(
        x.numpy(), na, labels=etiquetas, sort=False, add_reverse=True, measures=notas.medidas_div, add_beats=True
    )
    grafo["note"].fijos = fijos
    grafo["note"].tonica = tonica  # por nota: la armadura puede cambiar a mitad de pieza
    grafo["note"].onset_q = torch.from_numpy(na["onset_quarter"].astype(np.float32))
    grafo["note"].dur_q = torch.from_numpy(na["duration_quarter"].astype(np.float32))
    grafo["note"].compas = torch.from_numpy(notas.compas.astype(np.int64))
    actualizar_rasgos(grafo)  # x de compases/pulsos con el mismo cálculo que usaremos al transponer
    return grafo


# ---------------------------------------------------------------- explorador


def _describir(i: int, notas: Notas) -> str:
    from acordes_gnn.etiquetas import nombre_nota

    f = notas.array[i]
    tiempo = (float(f["onset_quarter"]) - notas.inicios_compas[notas.compas[i]]) + 1
    mano = "izq." if f["staff"] >= 2 else "der."
    return f"{nombre_nota(f)} (compás {notas.compas[i] + 1}, negra {tiempo:g}, mano {mano})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ruta", type=Path)
    p.add_argument("--compas", type=int, default=3, help="compás del ejemplo (1 = primero)")
    args = p.parse_args()

    notas = leer_notas(args.ruta)
    g = construir_grafo(notas)
    print(f"Partitura: {args.ruta.name}\n")
    print("NODOS")
    for t in g.node_types:
        print(f"  {t:8} {g[t].num_nodes:6d}" + (f"   rasgos por nodo: {g[t].x.shape[1]}" if "x" in g[t] else ""))
    print("\nARISTAS (origen —tipo→ destino: cuántas)")
    for (a, rel, b) in g.edge_types:
        print(f"  {a:7} —{rel:16}→ {b:7} {g[a, rel, b].edge_index.shape[1]:7d}")
    print("\nRasgos de cada nota:", ", ".join(NOMBRES_RASGOS))

    # ejemplo legible: la nota más aguda del primer ataque del compás pedido (suele ser la melodía)
    en_compas = np.where(notas.compas == args.compas - 1)[0]
    if len(en_compas) == 0:
        print(f"\nEl compás {args.compas} no tiene notas.")
        return
    primer = notas.array["onset_div"][en_compas].min()
    candidatas = en_compas[notas.array["onset_div"][en_compas] == primer]
    i = int(candidatas[np.argmax(notas.array["pitch"][candidatas])])
    print(f"\nEJEMPLO: la nota {_describir(i, notas)} — nodo {i}")
    print("  rasgos:", ", ".join(f"{n}={v:.2f}" for n, v in zip(NOMBRES_RASGOS, g["note"].x[i].tolist()) if v))
    for (a, rel, b) in g.edge_types:
        if a != "note" or b != "note":
            continue
        ei = g[a, rel, b].edge_index
        vecinos = ei[1, ei[0] == i].tolist()
        if vecinos:
            muestra = ", ".join(_describir(j, notas) for j in vecinos[:4]) + (f" … (+{len(vecinos) - 4})" if len(vecinos) > 4 else "")
            print(f"  [{rel}] {EXPLICACION_ARISTAS.get(rel, rel)}: {muestra}")
    print(f"  [connects] pertenece al nodo-compás {int(g['note'].measure_cluster[i]) + 1} "
          f"y al nodo-pulso {int(g['note'].beat_cluster[i])}")  # fmt: skip
    print("\nEn cada capa de la GNN esta nota recibe un resumen (la media) de los rasgos de esos vecinos,")
    print("con un peso aprendido distinto por tipo de arista. Con 2–3 capas 've' unas pocas negras alrededor.")


if __name__ == "__main__":
    main()
