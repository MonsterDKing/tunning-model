"""Audita el cifrado de las partituras nuevas de PDMX y busca repeticiones de test/dev.

Para cada partitura de ``data/pdmx/nuevas`` (ver ``extraer_pdmx.py``) cuenta, como la auditoría
de ``heuristico_dos_manos.json`` con la que se filtraron las antiguas:

- ``raiz_ausente``: acordes escritos cuya raíz no suena en ningún momento mientras dura el acorde;
- ``contradichos``: acordes escritos mayores en los que suena la 3ª menor y no la mayor (o al revés).

El código que generó la auditoría antigua no se conserva. Esta reconstrucción da los mismos
``acordes``, ``raiz_ausente`` con correlación 0.93 y la misma decisión "referencia confiable" en
el 92 % de las 311 partituras antiguas (``contradichos`` sale algo más bajo: la definición original
debía de ser más estricta). Las partituras antiguas no se re-filtran: esto solo decide las nuevas.

Además marca las que repiten canción de test/dev comparando ``title`` y ``song_name`` de PDMX.csv:
igual exacto, o un título contenido en el otro (de 6 letras o más). En la duda, fuera.

Con ``lead_sheets`` solo se buscan repeticiones de test/dev en ``data/pdmx/nuevas_lead``: a una
melodía sola no se le puede auditar el cifrado así (la raíz del acorde a menudo no suena).

Uso::

    PYTHONDONTWRITEBYTECODE=1 uv run python scripts/auditar_pdmx.py               # → data/pdmx/nuevas_auditoria.json
    PYTHONDONTWRITEBYTECODE=1 uv run python scripts/auditar_pdmx.py lead_sheets   # → data/pdmx/lead_nuevas.json
"""

import csv
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from acordes_gnn.datos import NUEVAS, NUEVAS_LEAD, PDMX, _normalizar, cargar_splits
from acordes_gnn.etiquetas import leer_referencia, ubicar
from acordes_gnn.grafo import leer_notas

SALIDA = PDMX / "nuevas_auditoria.json"
SALIDA_LEAD = PDMX / "lead_nuevas.json"


def auditar(ruta: Path) -> dict:
    warnings.filterwarnings("ignore")
    try:
        notas = leer_notas(ruta)
        acordes, _, n_compases = leer_referencia(ruta)
        if not acordes:
            return {"error": "sin cifrado"}
        if n_compases != len(notas.inicios_compas):
            return {"error": "music21 y partitura no ven los mismos compases"}
        ubicar(acordes, notas.inicios_compas)
    except Exception as e:  # noqa: BLE001 - se informa y se sigue
        return {"error": type(e).__name__}
    na = notas.array
    inicio, fin, pc = na["onset_quarter"], na["onset_quarter"] + na["duration_quarter"], na["pitch"] % 12
    ausente = contra = 0
    for i, a in enumerate(acordes):
        t0 = a.tiempo
        t1 = min(acordes[i + 1].tiempo if i + 1 < len(acordes) else np.inf, a.hasta)
        suenan = set(pc[(inicio < t1 - 1e-6) & (fin > t0 + 1e-6)].tolist())
        ausente += a.raiz not in suenan
        menor, mayor = (a.raiz + 3) % 12 in suenan, (a.raiz + 4) % 12 in suenan
        contra += (a.calidad == "mayor" and menor and not mayor) or (a.calidad == "menor" and mayor and not menor)
    return {"acordes": len(acordes), "raiz_ausente": ausente, "contradichos": contra}


def titulos(m: dict) -> list[str]:
    return [t for t in (_normalizar(m["title"]), _normalizar(m["song_name"])) if len(t) >= 4]


def main() -> None:
    lead = len(sys.argv) > 1 and sys.argv[1] == "lead_sheets"
    rutas = sorted((NUEVAS_LEAD if lead else NUEVAS).glob("*.musicxml"))
    if lead:
        auditoria = {r.name: {} for r in rutas}
    else:
        with ProcessPoolExecutor(16) as pool:
            auditoria = dict(zip((r.name for r in rutas), pool.map(auditar, rutas)))
    csv.field_size_limit(10**9)
    with open(PDMX / "PDMX.csv", newline="") as f:
        meta = {r["path"].rsplit("/", 1)[-1].split(".")[0]: r for r in csv.DictReader(f)}
    s = cargar_splits()
    congelados = {t for a in s["test"] + s["dev"] for t in titulos(meta[Path(a).stem])}
    for nombre, f in auditoria.items():
        m = meta[Path(nombre).stem]
        f["titulo"] = m["title"]
        ts = titulos(m)
        f["misma_cancion_que_test_dev"] = any(t in congelados for t in ts) or any(
            c in t or t in c for t in ts for c in congelados if len(c) >= 6 and len(t) >= 6
        )
    salida = SALIDA_LEAD if lead else SALIDA
    salida.write_text(json.dumps(auditoria, ensure_ascii=False, indent=1) + "\n")
    ok = [f for f in auditoria.values() if "error" not in f]
    print(f"{len(auditoria)} nuevas · legibles {len(ok)} · repiten canción de test/dev "
          f"{sum(f['misma_cancion_que_test_dev'] for f in auditoria.values())} → {salida}")  # fmt: skip


if __name__ == "__main__":
    main()
