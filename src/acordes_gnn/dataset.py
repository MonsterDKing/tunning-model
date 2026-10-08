"""Paso 3 — El dataset: grafos etiquetados en caché + aumento por transposición.

Leer un MusicXML con partitura y music21 tarda ~0,5 s; en un entrenamiento de 100 épocas eso se
repetiría miles de veces. Por eso ``preparar-datos`` convierte cada partitura UNA vez en
``data/cache/<nombre>.pt`` (el grafo + los acordes escritos para evaluar) usando los 24 núcleos.

Aumento por transposición (solo en train)
-----------------------------------------
Un Do mayor y un Re mayor son "el mismo" acorde desplazado 2 semitonos. Si el modelo solo ve las
tonalidades de las ~190 partituras de train, puede aprender atajos como "en este corpus casi
siempre hay Sol, así que digo Sol". Transponer cada partitura a las 12 tonalidades multiplica por
12 los ejemplos y obliga a aprender la RELACIÓN entre notas, no las alturas absolutas.

Transponer en el grafo es barato: las aristas (quién suena con quién) no cambian; solo se suma k
a la altura de cada nota, se recalculan sus rasgos, y la raíz de la etiqueta sube k semitonos.
En cada época cada partitura se transpone a una tonalidad al azar (k ∈ −5…+6): a lo largo de las
épocas el modelo ve todas, sin multiplicar por 12 el coste de cada época. Dev y test NUNCA se
transponen: se miden tal cual las escribió el arreglista.

Uso::

    uv run preparar-datos            # train + originales + dev + test → data/cache/
    uv run preparar-datos --rehacer  # ignorar la caché
"""

from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from acordes_gnn.datos import DATOS, PREFIJO_POP909, cargar_splits, ruta_de
from acordes_gnn.etiquetas import SIN_ETIQUETA, etiquetar_notas, leer_referencia, ubicar

CACHE = DATOS / "cache"
VERSION_CACHE = 3  # súbelo si cambias grafo.py o etiquetas.py: invalida la caché


def nombres_de(split: str, splits: dict | None = None) -> list[str]:
    """Partituras de un split.

    ``train`` = arreglos de PDMX + las originales de Analisis-AI + las nuevas del PDMX completo.
    ``train_antiguo`` = lo mismo sin las nuevas (el train de los experimentos 1–8).
    """
    s = splits or cargar_splits()
    if split == "train_antiguo":
        return s["train"] + s["originales"]
    if split == "train":
        return s["train"] + s["originales"] + s.get("train_nuevas", [])
    if split == "train_pop909":  # train + las canciones de POP909-CL
        return s["train"] + s["originales"] + s.get("train_nuevas", []) + s.get("pop909", [])
    if split == "dev_todo":  # dev de Analisis-AI (20) + dev ampliado (50): decide con menos ruido
        return s["dev"] + s.get("dev_ampliado", [])
    if split == "lead_sheets_todas":
        return s["lead_sheets"] + s.get("lead_sheets_nuevas", [])
    return s[split]


def _ruta_cache(nombre: str) -> Path:
    return CACHE / (Path(nombre).stem + ".pt")


def procesar(nombre: str) -> dict:
    """MusicXML (o MIDI de POP909-CL) → {grafo, acordes, estadísticas}. Corre en un proceso aparte."""
    from acordes_gnn.grafo import construir_grafo, leer_notas

    ruta = ruta_de(nombre)
    if nombre.startswith(PREFIJO_POP909):
        from acordes_gnn.pop909 import leer_pop909

        notas, acordes, sil_t = leer_pop909(ruta)  # los "sin acorde" ya vienen en tiempo
        if not acordes:
            raise ValueError("sin acordes")
        ubicar(acordes, notas.inicios_compas)
    else:
        notas = leer_notas(ruta)
        acordes, silencios, n_compases = leer_referencia(ruta)
        if not acordes:
            raise ValueError("sin cifrado")
        if n_compases != len(notas.inicios_compas):
            raise ValueError(f"music21 ve {n_compases} compases y partitura {len(notas.inicios_compas)}")
        ubicar(acordes, notas.inicios_compas)
        inicios = notas.inicios_compas
        sil_t = [float(inicios[min(i, len(inicios) - 1)]) + o for i, o in silencios]
    y = etiquetar_notas(notas.array["onset_quarter"], acordes, sil_t)
    grafo = construir_grafo(notas, y)
    return {
        "version": VERSION_CACHE,
        "nombre": nombre,
        "grafo": grafo,
        "acordes": [asdict(a) for a in acordes],
        "notas": len(y),
        "sin_etiqueta": int((y == SIN_ETIQUETA).sum()),
    }


def _procesar_y_guardar(nombre: str, rehacer: bool) -> dict:
    destino = _ruta_cache(nombre)
    if destino.exists() and not rehacer:
        d = torch.load(destino, weights_only=False)
        if d.get("version") == VERSION_CACHE:
            return {"nombre": nombre, "notas": d["notas"], "sin_etiqueta": d["sin_etiqueta"], "acordes": len(d["acordes"])}
    try:
        d = procesar(nombre)
    except Exception as e:  # noqa: BLE001 - se informa y se sigue con las demás
        return {"nombre": nombre, "error": f"{type(e).__name__}: {e}", "traza": traceback.format_exc(limit=3)}
    torch.save(d, destino)
    return {"nombre": nombre, "notas": d["notas"], "sin_etiqueta": d["sin_etiqueta"], "acordes": len(d["acordes"])}


def _un_hilo() -> None:
    torch.set_num_threads(1)  # 20 procesos × 24 hilos de torch cada uno se pisarían entre sí


def preparar(nombres: list[str], rehacer: bool = False, procesos: int = 20) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    filas = []
    with ProcessPoolExecutor(max_workers=procesos, initializer=_un_hilo) as pool:
        futuros = [pool.submit(_procesar_y_guardar, n, rehacer) for n in nombres]
        for f in as_completed(futuros):
            filas.append(f.result())
    return filas


# ---------------------------------------------------------------- carga y transposición


def cargar(nombre: str) -> dict:
    d = torch.load(_ruta_cache(nombre), weights_only=False)
    if d.get("version") != VERSION_CACHE:
        raise RuntimeError(f"caché vieja para {nombre}: corre `uv run preparar-datos --rehacer`")
    return d


def transponer(grafo, k: int):
    """Copia del grafo transportada k semitonos (rasgos y etiquetas)."""
    from acordes_gnn.grafo import actualizar_rasgos

    g = grafo.clone()
    if k == 0:
        return g
    g["note"].pitch = g["note"].pitch + k
    g["note"].tonica = (g["note"].tonica + k) % 12  # la armadura se transpone con las notas: el grado no cambia
    y = g["note"].y
    raiz, calidad = y // 5, y % 5
    g["note"].y = torch.where(y >= 0, ((raiz + k) % 12) * 5 + calidad, y)
    actualizar_rasgos(g)
    return g


class CorpusAcordes(torch.utils.data.Dataset):
    """Lista de grafos de un split. Con ``aumentar=True`` cada acceso transpone al azar."""

    def __init__(self, nombres: list[str], aumentar: bool = False, semilla: int = 0):
        self.piezas, self.fallidas = [], []
        for n in nombres:
            try:
                self.piezas.append(cargar(n))
            except FileNotFoundError:
                self.fallidas.append(n)
        self.aumentar = aumentar
        self.rng = random.Random(semilla)

    def __len__(self) -> int:
        return len(self.piezas)

    def __getitem__(self, i: int):
        g = self.piezas[i]["grafo"]
        return transponer(g, self.rng.randint(-5, 6)) if self.aumentar else g


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    p.add_argument("--rehacer", action="store_true")
    p.add_argument("-j", "--procesos", type=int, default=20)
    args = p.parse_args()
    splits = cargar_splits()
    resumen = {}
    for split in args.splits:
        t0 = time.time()
        filas = preparar(nombres_de(split, splits), args.rehacer, args.procesos)
        ok = [f for f in filas if "error" not in f]
        malas = [f for f in filas if "error" in f]
        notas = sum(f["notas"] for f in ok)
        sin = sum(f["sin_etiqueta"] for f in ok)
        acordes = sum(f["acordes"] for f in ok)
        print(f"{split:6} {len(ok):4d} partituras OK, {len(malas)} fallidas · {notas} notas "
              f"({sin / max(notas, 1):.1%} sin etiqueta) · {acordes} acordes escritos · {time.time() - t0:.0f}s")  # fmt: skip
        for f in malas:
            print(f"   ✗ {f['nombre']}: {f['error']}")
        resumen[split] = {"ok": len(ok), "fallidas": {f["nombre"]: f["error"] for f in malas}, "notas": notas,
                          "sin_etiqueta": sin, "acordes": acordes}  # fmt: skip
    (CACHE / "_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
