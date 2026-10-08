"""Paso 0 — Los datos: copiar el corpus y repartirlo en train / dev / test.

Las partituras vienen del proyecto hermano Analisis-AI (que NO se modifica). Este módulo:

1. ``--copiar-desde RUTA``: copia el corpus a ``data/`` (solo hace falta una vez).
2. Construye ``data/splits.json`` con los criterios de abajo y comprueba que no se solapan.

Criterios de los splits (todo con partituras de piano de PDMX con cifrado humano <harmony>):

- **test** = las 30 de ``muestra30.json``: el set de prueba CONGELADO con el que se midieron el
  heurístico y Claude. Nunca se entrena ni se ajusta nada mirándolo.
- **dev** = las 20 de ``desarrollo20.json``: para elegir hiperparámetros y la mejor época.
- **train** = el resto de arreglos a dos manos (``_dos_manos.json``) que cumplen:
    * legibles: el heurístico pudo analizarlas (sin ``error`` en ``heuristico_dos_manos.json``);
    * referencia confiable: ``raiz_ausente <= 15 %`` de los acordes y ``contradichos <= 5 %``
      (si la raíz escrita no suena casi nunca, o las notas contradicen el acorde, la "verdad"
      que aprendería el modelo sería ruido);
    * no es la misma canción que una de test/dev (mismo ``song_name`` normalizado): en PDMX hay
      varios arreglos de la misma obra y entrenar con uno inflaría la nota en el otro.
- **originales** = las 9 partituras de ``xmls/`` de Analisis-AI que traen cifrado y notas (con las
  correcciones de ``correcciones.json``). Se usan como train extra salvo las que repiten una
  canción de test/dev (``Shallow`` está en test).
- **train_nuevas** = arreglos a dos manos con cifrado que NO estaban en el corpus de Analisis-AI:
  salen de recorrer el PDMX completo (``scripts/buscar_cifrado_pdmx.py`` → ``extraer_pdmx.py`` →
  ``auditar_pdmx.py``). Mismo criterio de referencia confiable y sin repetir canción de test/dev
  (``nuevas_auditoria.json``). Van a train: ``nombres_de("train")`` las incluye.
- **test_ampliado** (100) y **dev_ampliado** (50) = exámenes extra sacados de las nuevas, para medir
  con menos ruido (test 30 y dev 20 son pequeños: una partitura rara mueve el resultado). Se reparte
  POR CANCIÓN (todos los arreglos de una canción van al mismo lado) y solo canciones que no están
  en el resto de train; al azar con semilla fija. test y dev de Analisis-AI no cambian.
- **pop909** = las canciones de POP909-CL (MIDI de piano pop con acordes corregidos por humanos, ver
  ``pop909.py``) salvo 518 y 620 (etiquetas del algoritmo). Solo para ``--datos dos_manos_pop909``.
- **clasico** = obras de piano del corpus clásico de BACHI (When in Rome + DCML, análisis de
  musicólogos; ver ``clasico.py``). Solo con ``--con-clasico``.
- **lead_sheets_nuevas** = las ~19 800 lead sheets del PDMX completo que no estaban en el corpus de
  Analisis-AI, sin las que repiten canción de test/dev (``lead_nuevas.json``). Solo para
  ``--datos todo`` / ``lead_sheets_todas``.
- **lead_sheets** = partituras de un solo pentagrama (solo melodía + cifrado). No entran en el
  entrenamiento por defecto; quedan listadas para un posible preentrenamiento (paso opcional).
  Las 62 de dos pentagramas que no están en ``_dos_manos.json`` (mano izquierda casi vacía) no
  van a ningún split.

Uso::

    uv run python -m acordes_gnn.datos --copiar-desde ../Analisis-AI   # copia + splits
    uv run python -m acordes_gnn.datos                                 # solo regenerar splits
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data"
PDMX = DATOS / "pdmx"
CORPUS = PDMX / "con_cifrado"
NUEVAS = PDMX / "nuevas"  # extraídas del PDMX completo (scripts/extraer_pdmx.py)
NUEVAS_LEAD = PDMX / "nuevas_lead"  # ídem, lead sheets
ORIGINALES = DATOS / "xmls_originales"
POP909 = DATOS / "pop909cl" / "POP909_processed"  # git clone de POP909-CL-Dataset
PREFIJO_POP909 = "pop909_"  # en splits.json: "pop909_001.mid" → POP909 / "001.mid"
POP909_EXCLUIDAS = {"518.mid", "620.mid"}  # conservan etiquetas del algoritmo (README de POP909-CL)
CLASICO = DATOS / "bachi" / "hf" / "unique_data_collection"  # hf download Itsuki-music/BACHI_Chord_Recognition
PREFIJO_CLASICO = "clasico_"  # en splits.json: "clasico_<carpeta de la obra>"
CLASICO_PIANO = ("Piano_Sonatas", "Keyboard_Other", "scarlatti", "beethoven_piano", "jc_bach", "wf_bach",
                 "rachmaninoff", "medtner", "liszt", "Variations_and", "debussy", "ravel", "schulhoff")  # fmt: skip
CLASICO_EXCLUIDAS = {  # posibles repeticiones de los exámenes, revisadas a mano (2026-10-07)
    "Keyboard_Other_Bach_The_Well-Tempered_Clavier_I_12": "dev_ampliado tiene «Prelude in F Minor (MIDIFlip)»",
}
SPLITS = DATOS / "splits.json"

TAMANO_TEST_AMPLIADO, TAMANO_DEV_AMPLIADO, SEMILLA_EXAMENES = 100, 50, 0

MAX_RAIZ_AUSENTE = 0.15
MAX_CONTRADICHOS = 0.05

# Repeticiones de canciones de test/dev que el filtro por ``song_name`` no ve: erratas o nombres
# distintos para la misma obra. Se encontraron comparando también el campo ``title`` de PDMX.csv
# (2026-10-07). En la duda, fuera: una fuga infla test sin avisar, y quitar una partitura no cuesta nada.
MISMA_CANCION_REVISADA = {
    "Qmf6PErv95W6XRHWfnwo5x8m3XtFfkqZ2Jt5zT69Ss4wPR.musicxml": "«Scool of Velocity op.299 no.1» (errata): test tiene dos del Op. 299",
    "QmYbCX4wBsX88CHeHAs6Sv3qWWceCX2Ewuj18FTVkiUg2v.musicxml": "«How Great Thou Art b2»: test/dev tiene «How Great Thou Art F»",
    "QmYe3WJngeNomcdxjjEtHGgVXnjAuySJwPV6LMLUHd6P3S.musicxml": "«Santo»: test/dev tiene «Santo santo grande eterno Dios»",
    "Qmf62FQ2J9VoFvVvN2GZBf1jd1JadjZfE654LnpRRi3KAR.musicxml": "«Santo»: test/dev tiene «Santo santo grande eterno Dios»",
    "QmUYarR3EgawzLHDw9UtPGihR2aSL71ybBoGQfXCRuEp9R.musicxml": "«SALMO»: test/dev tiene «Salmo 50B Parte 1»",
}


def ruta_de(nombre: str) -> Path:
    """Ruta de una partitura a partir del nombre que aparece en splits.json."""
    if nombre.startswith(PREFIJO_CLASICO):
        return CLASICO / nombre.removeprefix(PREFIJO_CLASICO)  # una CARPETA: partitura + chord_symbol.csv
    if nombre.startswith(PREFIJO_POP909):
        return POP909 / nombre.removeprefix(PREFIJO_POP909)
    for carpeta in (CORPUS, NUEVAS, NUEVAS_LEAD):
        if (carpeta / nombre).exists():
            return carpeta / nombre
    return ORIGINALES / nombre


def cargar_splits() -> dict:
    return json.loads(SPLITS.read_text(encoding="utf-8"))


def _normalizar(titulo: str) -> str:
    t = re.sub(r"[^a-z0-9]", "", (titulo or "").lower())
    return "" if t in ("", "na") else t


def _con_cifrado(carpeta: Path) -> list[Path]:
    """Igual que con_cifrado() de Analisis-AI: con <harmony> y con notas (no un chart de acordes)."""
    out = []
    for r in sorted(carpeta.iterdir()):
        if r.suffix in (".musicxml", ".xml"):
            texto = r.read_text(errors="ignore")
            if "<harmony" in texto and "<pitch>" in texto:
                out.append(r)
    return out


def copiar(origen: Path) -> None:
    """Copia el corpus desde Analisis-AI. Solo lee del origen."""
    pdmx = origen / "datasets" / "pdmx"
    PDMX.mkdir(parents=True, exist_ok=True)
    ORIGINALES.mkdir(parents=True, exist_ok=True)
    if not CORPUS.exists():
        shutil.copytree(pdmx / "con_cifrado", CORPUS, symlinks=False)
    for nombre in ("heuristico_dos_manos.json", "PDMX.csv"):
        if not (PDMX / nombre).exists():
            shutil.copy2(pdmx / nombre, PDMX / nombre)
    # muestra30 / desarrollo20 traen rutas de Analisis-AI → aquí solo el nombre de archivo
    for nombre in ("muestra30.json", "desarrollo20.json"):
        rutas = json.loads((pdmx / nombre).read_text(encoding="utf-8"))
        (PDMX / nombre).write_text(json.dumps([Path(r).name for r in rutas], indent=1) + "\n")
    shutil.copy2(origen / "xmls" / "correcciones.json", ORIGINALES / "correcciones.json")
    for r in _con_cifrado(origen / "xmls"):
        shutil.copy2(r, ORIGINALES / r.name)


def construir_splits() -> dict:
    indice = {f["archivo"]: f for f in json.loads((CORPUS / "_indice.json").read_text(encoding="utf-8"))}
    dos_manos = set(json.loads((CORPUS / "_dos_manos.json").read_text(encoding="utf-8")))
    auditoria = {f["archivo"]: f for f in json.loads((PDMX / "heuristico_dos_manos.json").read_text(encoding="utf-8"))}
    test = json.loads((PDMX / "muestra30.json").read_text(encoding="utf-8"))
    dev = json.loads((PDMX / "desarrollo20.json").read_text(encoding="utf-8"))

    titulos_congelados = {_normalizar(indice[a]["song_name"]) for a in test + dev} - {""}
    descartes: dict[str, list[str]] = {"ilegible": [], "referencia_poco_confiable": [], "misma_cancion_que_test_dev": []}
    train = []
    for a in sorted(dos_manos - set(test) - set(dev)):
        f = auditoria.get(a)
        if f is None or "error" in f:
            descartes["ilegible"].append(a)
        elif f["raiz_ausente"] > MAX_RAIZ_AUSENTE * f["acordes"] or f["contradichos"] > MAX_CONTRADICHOS * f["acordes"]:
            descartes["referencia_poco_confiable"].append(a)
        elif _normalizar(indice[a]["song_name"]) in titulos_congelados or a in MISMA_CANCION_REVISADA:
            descartes["misma_cancion_que_test_dev"].append(a)
        else:
            train.append(a)

    # Las 9 originales: el título sale del nombre de archivo ("3215_Shallow_Lady_Gaga" → "shallowladygaga")
    originales, originales_fuera = [], []
    for r in _con_cifrado(ORIGINALES):
        titulo = _normalizar(r.stem.split("_", 1)[-1])
        if any(t and t in titulo for t in titulos_congelados if len(t) >= 5):
            originales_fuera.append(r.name)
        else:
            originales.append(r.name)

    lead_sheets = []
    for a in sorted(indice):
        if a in dos_manos or a in test or a in dev:
            continue
        if "<staves>2</staves>" in (CORPUS / a).read_text(errors="ignore"):
            continue  # dos pentagramas con mano izquierda casi vacía: ni lead sheet ni arreglo
        if _normalizar(indice[a]["song_name"]) not in titulos_congelados and a not in MISMA_CANCION_REVISADA:
            lead_sheets.append(a)

    # Partituras nuevas del PDMX completo: mismo criterio que train, auditadas por scripts/auditar_pdmx.py
    train_nuevas, nuevas_descartes = [], {"ilegible": 0, "referencia_poco_confiable": 0, "misma_cancion_que_test_dev": 0}
    auditoria_nuevas = PDMX / "nuevas_auditoria.json"
    if auditoria_nuevas.exists():
        for a, f in sorted(json.loads(auditoria_nuevas.read_text(encoding="utf-8")).items()):
            if "error" in f:
                nuevas_descartes["ilegible"] += 1
            elif f["raiz_ausente"] > MAX_RAIZ_AUSENTE * f["acordes"] or f["contradichos"] > MAX_CONTRADICHOS * f["acordes"]:
                nuevas_descartes["referencia_poco_confiable"] += 1
            elif f["misma_cancion_que_test_dev"] or a in MISMA_CANCION_REVISADA:
                nuevas_descartes["misma_cancion_que_test_dev"] += 1
            else:
                train_nuevas.append(a)

    # Exámenes ampliados: se sacan de train_nuevas, por canción, sin canciones que sigan en train
    test_ampliado, dev_ampliado = [], []
    if train_nuevas:
        titulos_nuevas = {a: f["titulo"] for a, f in json.loads(auditoria_nuevas.read_text(encoding="utf-8")).items()}
        titulos_train = {_normalizar(indice[a]["song_name"]) for a in train} | {
            _normalizar(r.split("_", 1)[-1].rsplit(".", 1)[0]) for r in originales}
        titulos_train.discard("")
        grupos: dict[str, list[str]] = {}
        for a in train_nuevas:
            grupos.setdefault(_normalizar(titulos_nuevas[a]) or a, []).append(a)  # sin título: grupo propio

        def en_train(t: str) -> bool:
            return t in titulos_train or any(c in t or t in c for c in titulos_train if len(c) >= 6 and len(t) >= 6)

        claves = sorted(k for k in grupos if not en_train(k))
        random.Random(SEMILLA_EXAMENES).shuffle(claves)
        for k in claves:
            if len(test_ampliado) < TAMANO_TEST_AMPLIADO:
                test_ampliado += grupos[k]
            elif len(dev_ampliado) < TAMANO_DEV_AMPLIADO:
                dev_ampliado += grupos[k]
        examen = set(test_ampliado) | set(dev_ampliado)
        titulos_examen = {_normalizar(titulos_nuevas[a]) for a in examen} - {""}

        def como_examen(t: str) -> bool:  # misma canción que un examen ampliado (exacta o contenida)
            return bool(t) and (t in titulos_examen or any(c in t or t in c for c in titulos_examen if len(c) >= 6 and len(t) >= 6))

        # fuera de train lo que es examen y lo que se le parece (títulos casi iguales en grupos distintos)
        train_nuevas = [a for a in train_nuevas if a not in examen and not como_examen(_normalizar(titulos_nuevas[a]))]

    lead_nuevas_json = PDMX / "lead_nuevas.json"
    lead_sheets_nuevas = []
    if lead_nuevas_json.exists():
        lead_sheets_nuevas = [a for a, f in sorted(json.loads(lead_nuevas_json.read_text(encoding="utf-8")).items())
                              if not f["misma_cancion_que_test_dev"] and a not in MISMA_CANCION_REVISADA]  # fmt: skip
        if test_ampliado:  # tampoco canciones de los exámenes ampliados
            lead_titulos = json.loads(lead_nuevas_json.read_text(encoding="utf-8"))
            lead_sheets_nuevas = [a for a in lead_sheets_nuevas if not como_examen(_normalizar(lead_titulos[a]["titulo"]))]
    if test_ampliado:
        lead_sheets = [a for a in lead_sheets if not como_examen(_normalizar(indice[a]["song_name"]))]

    pop909 = sorted(PREFIJO_POP909 + r.name for r in POP909.glob("*.mid") if r.name not in POP909_EXCLUIDAS)
    # encaje < 45 % (notas que pertenecen a su acorde): probablemente desalineadas → fuera
    encaje_bajo = DATOS / "bachi" / "clasico_encaje_bajo.json"
    bajas = set(json.loads(encaje_bajo.read_text(encoding="utf-8"))) if encaje_bajo.exists() else set()
    clasico = sorted(PREFIJO_CLASICO + d.name for d in CLASICO.glob("*")
                     if d.name.startswith(CLASICO_PIANO) and d.name not in CLASICO_EXCLUIDAS
                     and d.name not in bajas and (d / "chord_symbol.csv").exists())  # fmt: skip
    # examen de la tarea de melodía: su melodía nunca se usa al entrenar (sus acordes sí: no la revelan)
    pop909_voz_test = sorted(random.Random(SEMILLA_EXAMENES).sample(pop909, 60))

    # Comprobaciones: nada de test/dev puede filtrarse a otro split
    conjuntos = {"test": set(test), "dev": set(dev), "train": set(train), "originales": set(originales),
                 "lead_sheets": set(lead_sheets), "train_nuevas": set(train_nuevas),
                 "lead_sheets_nuevas": set(lead_sheets_nuevas), "test_ampliado": set(test_ampliado),
                 "dev_ampliado": set(dev_ampliado), "pop909": set(pop909)}  # fmt: skip
    nombres = list(conjuntos)
    for i, a in enumerate(nombres):
        for b in nombres[i + 1 :]:
            comun = conjuntos[a] & conjuntos[b]
            assert not comun, f"{a} y {b} se solapan: {sorted(comun)[:3]}"
    assert len(test) == 30 and len(dev) == 20

    return {
        "_leeme": (
            "Splits del proyecto. test (30) y dev (20) están CONGELADOS: vienen de Analisis-AI y nunca se "
            "entrena con ellos. train = arreglos a dos manos legibles con referencia confiable "
            f"(raiz_ausente <= {MAX_RAIZ_AUSENTE:.0%} y contradichos <= {MAX_CONTRADICHOS:.0%} de los acordes) "
            "que no repiten canción de test/dev. originales = partituras de xmls/ de Analisis-AI (train extra). "
            "lead_sheets = solo melodía, para un preentrenamiento opcional. Ver src/acordes_gnn/datos.py."
        ),
        "criterios": {
            "max_raiz_ausente": MAX_RAIZ_AUSENTE,
            "max_contradichos": MAX_CONTRADICHOS,
            "descartados_de_train": {k: len(v) for k, v in descartes.items()},
            "descartados_detalle": descartes,
            "originales_fuera_por_repetir_cancion": originales_fuera,
            "descartados_de_train_nuevas": nuevas_descartes,
        },
        "test": test,
        "dev": dev,
        "train": train,
        "originales": originales,
        "train_nuevas": train_nuevas,
        "test_ampliado": sorted(test_ampliado),
        "dev_ampliado": sorted(dev_ampliado),
        "lead_sheets": lead_sheets,
        "lead_sheets_nuevas": lead_sheets_nuevas,
        "pop909": pop909,
        "pop909_voz_test": pop909_voz_test,
        "clasico": clasico,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--copiar-desde", type=Path, help="carpeta del proyecto Analisis-AI")
    args = p.parse_args()
    if args.copiar_desde:
        copiar(args.copiar_desde.resolve())
    splits = construir_splits()
    SPLITS.write_text(json.dumps(splits, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for k in ("test", "dev", "test_ampliado", "dev_ampliado", "train", "originales", "train_nuevas",
              "lead_sheets", "lead_sheets_nuevas", "pop909", "clasico"):
        print(f"{k:12} {len(splits[k]):5d} partituras")
    print("descartados de train:", splits["criterios"]["descartados_de_train"])
    print("originales fuera:", splits["criterios"]["originales_fuera_por_repetir_cancion"])
    print(f"→ {SPLITS.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
