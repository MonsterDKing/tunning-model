"""Extrae del PDMX completo las partituras con cifrado que aún no tenemos.

Lee ``_hallazgos.json`` (de ``buscar_cifrado_pdmx.py``), se queda con las nuevas de un tipo
(``dos_manos``: una parte con dos pentagramas; ``lead_sheets``: una parte con un pentagrama) y
escribe su MusicXML principal (descomprimido) en la carpeta de salida, con el mismo nombre que
usa ``data/pdmx/con_cifrado`` (``<id>.musicxml``). Solo lee el tar: no ejecuta nada.

Uso::

    uv run python -I scripts/extraer_pdmx.py data/pdmx/completo/mxl.tar.gz \\
        data/pdmx/completo/_hallazgos.json data/pdmx/nuevas
    uv run python -I scripts/extraer_pdmx.py data/pdmx/completo/mxl.tar.gz \\
        data/pdmx/completo/_hallazgos.json data/pdmx/nuevas_lead lead_sheets
"""

import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # -I no añade la carpeta del script
from buscar_cifrado_pdmx import principal_xml  # noqa: E402


def main() -> None:
    tar_ruta, hallazgos, salida = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    dos_pentagramas = (sys.argv[4] if len(sys.argv) > 4 else "dos_manos") == "dos_manos"
    elegidas = {h["ruta"]: h["id"] for h in json.loads(hallazgos.read_text())
                if not h["ya_la_teniamos"] and h["partes"] == 1 and h["dos_pentagramas"] == dos_pentagramas}  # fmt: skip
    salida.mkdir(parents=True, exist_ok=True)
    escritas = 0
    with tarfile.open(tar_ruta, "r:gz") as tar:
        for m in tar:
            if m.name in elegidas:
                texto = principal_xml(tar.extractfile(m).read())
                (salida / f"{elegidas[m.name]}.musicxml").write_text(texto, encoding="utf-8")
                escritas += 1
    print(f"{escritas} de {len(elegidas)} partituras → {salida}")


if __name__ == "__main__":
    main()
