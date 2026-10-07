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
import re
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data"
PDMX = DATOS / "pdmx"
CORPUS = PDMX / "con_cifrado"
ORIGINALES = DATOS / "xmls_originales"
SPLITS = DATOS / "splits.json"

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
    r = CORPUS / nombre
    return r if r.exists() else ORIGINALES / nombre


def cargar_splits() -> dict:
    return json.loads(SPLITS.read_text())


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
        rutas = json.loads((pdmx / nombre).read_text())
        (PDMX / nombre).write_text(json.dumps([Path(r).name for r in rutas], indent=1) + "\n")
    shutil.copy2(origen / "xmls" / "correcciones.json", ORIGINALES / "correcciones.json")
    for r in _con_cifrado(origen / "xmls"):
        shutil.copy2(r, ORIGINALES / r.name)


def construir_splits() -> dict:
    indice = {f["archivo"]: f for f in json.loads((CORPUS / "_indice.json").read_text())}
    dos_manos = set(json.loads((CORPUS / "_dos_manos.json").read_text()))
    auditoria = {f["archivo"]: f for f in json.loads((PDMX / "heuristico_dos_manos.json").read_text())}
    test = json.loads((PDMX / "muestra30.json").read_text())
    dev = json.loads((PDMX / "desarrollo20.json").read_text())

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

    # Comprobaciones: nada de test/dev puede filtrarse a otro split
    conjuntos = {"test": set(test), "dev": set(dev), "train": set(train), "originales": set(originales),
                 "lead_sheets": set(lead_sheets)}  # fmt: skip
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
        },
        "test": test,
        "dev": dev,
        "train": train,
        "originales": originales,
        "lead_sheets": lead_sheets,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--copiar-desde", type=Path, help="carpeta del proyecto Analisis-AI")
    args = p.parse_args()
    if args.copiar_desde:
        copiar(args.copiar_desde.resolve())
    splits = construir_splits()
    SPLITS.write_text(json.dumps(splits, ensure_ascii=False, indent=1) + "\n")
    for k in ("test", "dev", "train", "originales", "lead_sheets"):
        print(f"{k:12} {len(splits[k]):5d} partituras")
    print("descartados de train:", splits["criterios"]["descartados_de_train"])
    print("originales fuera:", splits["criterios"]["originales_fuera_por_repetir_cancion"])
    print(f"→ {SPLITS.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
