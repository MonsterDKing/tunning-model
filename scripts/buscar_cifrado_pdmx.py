"""Busca en el PDMX completo las partituras con cifrado humano <harmony>.

PDMX (Zenodo, record 15571083) publica todas sus partituras en ``mxl.tar.gz`` (1,9 GB, ~254 000
archivos .mxl, que son MusicXML comprimido en zip). Este script recorre el tar SIN descomprimirlo
a disco y, para cada partitura, mira:

- si tiene cifrado escrito (``<harmony``) y notas (``<pitch>``): si no, no sirve para entrenar;
- cuántas partes (instrumentos) y si alguna tiene dos pentagramas (``<staves>2``): piano a dos manos;
- si ya la teníamos (está en ``data/pdmx/con_cifrado``).

Solo lee: no ejecuta nada de lo descargado. El resultado va a un JSON para decidir después qué
partituras nuevas pasan por los filtros de siempre (``datos.py``).

Uso::

    uv run python -I scripts/buscar_cifrado_pdmx.py data/pdmx/completo/mxl.tar.gz data/pdmx/con_cifrado \\
        data/pdmx/completo/_hallazgos.json
"""

import json
import re
import sys
import tarfile
import zipfile
from collections import Counter
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path


def principal_xml(datos: bytes) -> str | None:
    """Texto del MusicXML principal dentro de un .mxl (zip)."""
    with zipfile.ZipFile(BytesIO(datos)) as z:
        nombres = z.namelist()
        raiz = None
        if "META-INF/container.xml" in nombres:
            m = re.search(rb'full-path="([^"]+)"', z.read("META-INF/container.xml"))
            raiz = m.group(1).decode() if m else None
        if raiz not in nombres:
            raiz = next((n for n in nombres if n.endswith((".xml", ".musicxml")) and not n.startswith("META-INF")), None)
        return z.read(raiz).decode("utf-8", errors="ignore") if raiz else None


def examinar(item: tuple[str, bytes]) -> dict | None:
    ruta, datos = item
    try:
        texto = principal_xml(datos)
    except Exception as e:  # noqa: BLE001 - un .mxl roto no debe parar la búsqueda
        return {"ruta": ruta, "error": type(e).__name__}
    if not texto or "<harmony" not in texto or "<pitch>" not in texto:
        return None
    return {
        "ruta": ruta,
        "id": Path(ruta).stem,
        "acordes": texto.count("<harmony"),
        "partes": texto.count("<score-part "),
        "dos_pentagramas": "<staves>2</staves>" in texto,
    }


def miembros(tar_ruta: str):
    with tarfile.open(tar_ruta, "r:gz") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".mxl"):
                f = tar.extractfile(m)
                if f is not None:
                    yield m.name, f.read()


def main() -> None:
    tar_ruta, carpeta_actual, salida = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    ya = {p.stem for p in carpeta_actual.glob("*.musicxml")}
    total, hallazgos, errores = 0, [], Counter()
    with Pool(16) as pool:
        for r in pool.imap_unordered(examinar, miembros(tar_ruta), chunksize=64):
            total += 1
            if total % 20000 == 0:
                print(f"  {total} partituras revisadas, {len(hallazgos)} con cifrado", flush=True)
            if r is None:
                continue
            if "error" in r:
                errores[r["error"]] += 1
                continue
            r["ya_la_teniamos"] = r["id"] in ya
            hallazgos.append(r)
    salida.write_text(json.dumps(hallazgos, ensure_ascii=False, indent=1))
    nuevas = [h for h in hallazgos if not h["ya_la_teniamos"]]
    piano2 = [h for h in nuevas if h["partes"] == 1 and h["dos_pentagramas"]]
    print(f"\nRevisadas: {total} · ilegibles: {sum(errores.values())} {dict(errores)}")
    print(f"Con cifrado y notas: {len(hallazgos)} (ya teníamos {len(hallazgos) - len(nuevas)})")
    print(f"NUEVAS: {len(nuevas)} · de ellas, una sola parte con dos pentagramas (piano a dos manos): {len(piano2)}")
    print(f"→ {salida}")


if __name__ == "__main__":
    main()
