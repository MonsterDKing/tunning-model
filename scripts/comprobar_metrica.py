"""Comprueba que nuestra métrica da LO MISMO que validar() de Analisis-AI.

Corre el analizador heurístico de Analisis-AI (solo lectura: se importa desde su carpeta) sobre
las 30 partituras de test y lo puntúa dos veces: con su validar() y con evaluar.comparar() de
este proyecto. Si los números coinciden, las tablas de los dos proyectos son comparables.

    PYTHONDONTWRITEBYTECODE=1 uv run python scripts/comprobar_metrica.py ../Analisis-AI

Resultado (2026-10-07): 1532 acordes; validar() raíz 80.7 % y raíz+calidad 76.2 %; aquí
idénticos, y equivalente 83.2 % (Analisis-AI reportó 83.1 %: 1–2 acordes de diferencia en la
definición exacta de "equivalente").
"""

import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(sys.argv[1] if len(sys.argv) > 1 else "../Analisis-AI").resolve() / "src"))

from music21 import converter, pitch  # noqa: E402
from musica import analisis as H  # noqa: E402

from acordes_gnn.datos import cargar_splits, ruta_de  # noqa: E402
from acordes_gnn.etiquetas import leer_referencia  # noqa: E402


def medir(nombre: str):
    ruta = ruta_de(nombre)
    sc = converter.parse(str(ruta))
    segs = H.detectar_acordes(sc)[0]
    v = H.validar(sc, segs, ruta)
    refs, _, _ = leer_referencia(ruta, score=sc)
    ok = [0, 0, 0]
    for a in refs:
        s = H._activo_en(segs, a.indice, a.offset)
        if s is None:
            continue
        raiz = pitch.Pitch(s.raiz.replace("b", "-")).pitchClass
        ok_r = raiz == a.raiz
        ok_q = ok_r and H._CALIDAD_SUFIJO.get(s.sufijo, "other") == a.calidad_m21
        alturas = frozenset((raiz + i) % 12 for i in H._SEMITONOS[s.sufijo])
        ok[0] += ok_r
        ok[1] += ok_q
        ok[2] += ok_q or alturas <= a.alturas or a.alturas <= alturas
    return v["total"], round(v["precision_raiz"] * v["total"]), round(v["precision_raiz_y_calidad"] * v["total"]), len(refs), ok


if __name__ == "__main__":
    with ProcessPoolExecutor(15) as pool:
        res = list(pool.map(medir, cargar_splits()["test"]))
    n, n2 = sum(r[0] for r in res), sum(r[3] for r in res)
    print(f"validar() de Analisis-AI: {n} acordes · raíz {sum(r[1] for r in res) / n:.1%} · "
          f"raíz+calidad {sum(r[2] for r in res) / n:.1%}")  # fmt: skip
    print(f"nuestra métrica:          {n2} acordes · raíz {sum(r[4][0] for r in res) / n2:.1%} · "
          f"raíz+calidad {sum(r[4][1] for r in res) / n2:.1%} · equivalente {sum(r[4][2] for r in res) / n2:.1%}")  # fmt: skip
