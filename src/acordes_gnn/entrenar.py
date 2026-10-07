"""Paso 4 — Entrenar: el bucle de descenso de gradiente.

Una época = pasar una vez por todas las partituras de train. Para cada lote:

1. **forward**: el modelo calcula 60 logits por nota.
2. **pérdida**: entropía cruzada entre esos logits y el acorde escrito de cada nota (las notas
   sin etiqueta, -100, se ignoran). Es pequeña si el modelo da mucha probabilidad al acorde bueno.
3. **backward**: PyTorch calcula cuánto cambiaría la pérdida al mover cada peso (el gradiente).
4. **paso**: AdamW mueve cada peso un poquito (``--lr``) en la dirección que baja la pérdida.
   ``--peso-decay`` empuja los pesos hacia 0 (regularización: modelos más simples generalizan mejor).

Al final de cada época se mide en DEV (nunca en test) con la métrica de los puntos de cifrado.
Se guarda el modelo de la época con mejor raíz+calidad en dev (early stopping "a posteriori"):
si train sigue mejorando y dev empeora, el modelo está memorizando (sobreajuste).

Uso::

    uv run entrenar --humo          # 1 época con 5 partituras: comprueba que todo encaja (~segundos)
    uv run entrenar                 # entrenamiento normal (CPU, 4 hilos: ver --hilos)
    uv run entrenar --sin-transposicion --epocas 40   # experimento: ¿cuánto aporta el aumento?
    uv run entrenar --arquitectura metrical           # la MetricalGNN de GraphMuse
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from acordes_gnn.datos import RAIZ
from acordes_gnn.dataset import CorpusAcordes, nombres_de
from acordes_gnn.etiquetas import NUM_CLASES, SIN_ETIQUETA
from acordes_gnn.evaluar import evaluar_piezas
from acordes_gnn.modelo import contar_parametros, crear_modelo


def argumentos(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--humo", action="store_true", help="1 época sobre 5 partituras de train y 3 de dev")
    p.add_argument("--epocas", type=int, default=60)
    p.add_argument("--lote", type=int, default=8, help="partituras por lote")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--peso-decay", type=float, default=1e-4)
    p.add_argument("--oculto", type=int, default=64)
    p.add_argument("--capas", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--arquitectura", default="propio", choices=["propio", "metrical"])
    p.add_argument("--sin-transposicion", action="store_true", help="no aumentar train con transposiciones")
    p.add_argument("--semilla", type=int, default=0)
    # con grafos pequeños, muchos hilos se estorban: 24 hilos fue ~400× más lento que 4 en esta máquina
    p.add_argument("--hilos", type=int, default=4)
    p.add_argument("--salida", type=Path, default=RAIZ / "modelos" / "mejor.pt")
    return p.parse_args(argv)


def fijar_semilla(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def main(argv=None) -> None:
    args = argumentos(argv)
    args.salida = args.salida.resolve()
    fijar_semilla(args.semilla)
    torch.set_num_threads(args.hilos)

    nombres_train, nombres_dev = nombres_de("train"), nombres_de("dev")
    if args.humo:
        nombres_train, nombres_dev, args.epocas = nombres_train[:5], nombres_dev[:3], 1
        args.salida = args.salida.with_name("humo.pt")
    train = CorpusAcordes(nombres_train, aumentar=not args.sin_transposicion, semilla=args.semilla)
    dev = CorpusAcordes(nombres_dev)
    if train.fallidas or dev.fallidas:
        raise SystemExit(f"Faltan grafos en caché ({len(train.fallidas) + len(dev.fallidas)}): corre `uv run preparar-datos`")
    cargador = DataLoader(train, batch_size=args.lote, shuffle=True)

    metadata = train.piezas[0]["grafo"].metadata()
    modelo = crear_modelo(args.arquitectura, metadata, oculto=args.oculto, capas=args.capas, dropout=args.dropout,
                          clases=NUM_CLASES)  # fmt: skip
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=args.peso_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epocas, 1))
    print(f"train: {len(train)} partituras ({'con' if train.aumentar else 'sin'} transposición) · dev: {len(dev)} · "
          f"modelo {args.arquitectura}: {contar_parametros(modelo):,} parámetros")  # fmt: skip

    historial, mejor = [], -1.0
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    for epoca in range(1, args.epocas + 1):
        t0 = time.time()
        modelo.train()
        suma, n_ok, n_tot, n_lotes = 0.0, 0, 0, 0
        for lote in cargador:
            logits = modelo(lote.x_dict, lote.edge_index_dict)
            y = lote["note"].y
            perdida = torch.nn.functional.cross_entropy(logits, y, ignore_index=SIN_ETIQUETA)
            opt.zero_grad()
            perdida.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            con = y != SIN_ETIQUETA
            suma += perdida.item()
            n_lotes += 1
            n_ok += int((logits.argmax(-1)[con] == y[con]).sum())
            n_tot += int(con.sum())
        sched.step()
        res = evaluar_piezas(modelo, dev.piezas)
        fila = {"epoca": epoca, "perdida": suma / max(n_lotes, 1), "train_por_nota": n_ok / max(n_tot, 1),
                "dev_por_nota": res["precision_por_nota"], "dev_raiz": res["raiz"], "dev_calidad": res["calidad"],
                "dev_equivalente": res["equivalente"], "segundos": time.time() - t0}  # fmt: skip
        historial.append(fila)
        marca = ""
        if res["calidad"] > mejor:
            mejor = res["calidad"]
            torch.save({"estado": modelo.state_dict(), "config": vars(args) | {"salida": str(args.salida)},
                        "metadata": metadata, "epoca": epoca, "dev": fila}, args.salida)  # fmt: skip
            marca = "  ← mejor en dev, guardado"
        print(f"época {epoca:3d}  pérdida {fila['perdida']:.3f}  train/nota {fila['train_por_nota']:5.1%}  "
              f"dev/nota {fila['dev_por_nota']:5.1%}  dev raíz {res['raiz']:5.1%}  raíz+cal {res['calidad']:5.1%}  "
              f"equiv {res['equivalente']:5.1%}  ({fila['segundos']:.1f}s){marca}", flush=True)  # fmt: skip

    destino = RAIZ / "salidas" / f"historial_{'humo_' if args.humo else ''}{time.strftime('%Y%m%d-%H%M%S')}.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps({"config": vars(args) | {"salida": str(args.salida)}, "historial": historial},
                                  ensure_ascii=False, indent=1))  # fmt: skip
    print(f"\nMejor raíz+calidad en dev: {mejor:.1%} → {args.salida.relative_to(RAIZ)} · historial: {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
