"""Paso 4 — Entrenar: el bucle de descenso de gradiente.

Una época = pasar una vez por todas las partituras de train. Para cada lote:

1. **forward**: el modelo calcula 60 logits por nota.
2. **pérdida**: entropía cruzada entre esos logits y el acorde escrito de cada nota (las notas
   sin etiqueta, -100, se ignoran). Es pequeña si el modelo da mucha probabilidad al acorde bueno.
   Por defecto cada ACORDE escrito cuenta igual (ver ``pesos_por_acorde``), que es como cuenta la
   métrica; con ``--no-peso-por-acorde`` cada NOTA cuenta igual (la versión original).
3. **backward**: PyTorch calcula cuánto cambiaría la pérdida al mover cada peso (el gradiente).
4. **paso**: AdamW mueve cada peso un poquito (``--lr``) en la dirección que baja la pérdida.
   ``--peso-decay`` empuja los pesos hacia 0 (regularización: modelos más simples generalizan mejor).

Al final de cada época se mide en DEV (nunca en test) con la métrica de los puntos de cifrado.
Se guarda el modelo de la época con mejor raíz+calidad en dev (early stopping "a posteriori"):
si train sigue mejorando y dev empeora, el modelo está memorizando (sobreajuste).

Preentrenar y afinar (fine-tuning)
---------------------------------
Con ``--datos lead_sheets`` se entrena con las ~3700 lead sheets de PDMX (solo melodía + cifrado
humano: 13 veces más acordes que train). Ese modelo ya "sabe" qué melodías van con qué acordes.
Con ``--partir-de`` otro entrenamiento EMPIEZA con esos pesos en lugar de al azar y los ajusta con
los arreglos a dos manos, normalmente con un ``--lr`` más bajo para no borrar lo aprendido.

Uso::

    uv run entrenar --humo          # 1 época con 5 partituras: comprueba que todo encaja (~segundos)
    uv run entrenar                 # entrenamiento normal (CPU, 4 hilos: ver --hilos)
    uv run entrenar --sin-transposicion --epocas 40   # experimento: ¿cuánto aporta el aumento?
    uv run entrenar --arquitectura metrical           # la MetricalGNN de GraphMuse
    uv run entrenar --arquitectura temporal           # GNN + GRU sobre los ataques (paso 6)
    uv run entrenar --datos lead_sheets --epocas 20 --salida modelos/pre.pt     # preentrenar
    uv run entrenar --partir-de modelos/pre.pt --lr 1e-3                          # afinar
    uv run entrenar --no-peso-por-acorde              # pérdida por nota (la original, ~2 puntos peor en dev)
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

from acordes_gnn.datos import RAIZ, cargar_splits
from acordes_gnn.dataset import CorpusAcordes, nombres_de
from acordes_gnn.etiquetas import CALIDADES, NUM_CLASES, SIN_ETIQUETA
from acordes_gnn.evaluar import evaluar_piezas
from acordes_gnn.modelo import ARQUITECTURAS, CABEZAS, agrupar_por_ataque, contar_parametros, crear_modelo


def argumentos(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--humo", action="store_true", help="1 época sobre 5 partituras de train y 3 de dev")
    p.add_argument("--epocas", type=int, default=60)
    p.add_argument("--lote", type=int, default=8, help="partituras por lote")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--peso-decay", type=float, default=1e-4)
    p.add_argument("--oculto", type=int, default=128)  # 64 → 86.2 %, 128 → 88.0 %, 256 → 87.9 % en dev (Exp. 7–8)
    p.add_argument("--capas", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--arquitectura", default="propio", choices=list(ARQUITECTURAS))
    p.add_argument("--cabeza", default="conjunta", choices=list(CABEZAS),
                   help="salida: 60 clases (conjunta) o raíz × calidad (factorizada, Exp. 25; ver modelo.py)")
    p.add_argument("--sin-transposicion", action="store_true", help="no aumentar train con transposiciones")
    p.add_argument("--mutar-calidad", type=float, default=0.0,
                   help="fracción de acordes escritos de train que se mutan de calidad en cada época (Exp. 26, ver dataset.py)")
    p.add_argument("--mutaciones", default="raras", choices=["raras", "todas"],
                   help="qué mutaciones: solo hacia dim/aum/sus (raras) o también mayor ↔ menor (todas)")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--peso-por-acorde", action=argparse.BooleanOptionalAction, default=True,
                   help="cada acorde escrito pesa igual en la pérdida (--no-peso-por-acorde: cada nota)")
    p.add_argument("--perdida-por-ataque", action="store_true",
                   help="entrenar lo que se mide: la pérdida se calcula sobre la MEDIA de probabilidades de las "
                        "notas de cada ataque (como predecir_en_puntos), no nota a nota")
    p.add_argument("--datos", default="dos_manos", choices=["dos_manos", "dos_manos_antiguo", "dos_manos_pop909", "lead_sheets", "lead_sheets_todas", "todo",
                            "todo_pop909"],
                   help="con qué partituras entrenar (lead_sheets: para preentrenar; todo: dos manos + todas las lead sheets)")
    p.add_argument("--dev", default="dev_todo", choices=["dev", "dev_todo"],
                   help="con qué partituras elegir la mejor época (dev_todo = dev 20 + dev ampliado 50)")
    p.add_argument("--peso-dos-manos", type=int, default=1,
                   help="con --datos todo: cuántas veces aparece cada partitura a dos manos por época")
    p.add_argument("--partir-de", type=Path, help="empezar con los pesos de este modelo (fine-tuning)")
    p.add_argument("--peso-calidad", type=float, default=0.0,
                   help="alfa de pesos_por_calidad: >0 da más peso a los acordes raros (dim, aum, sus)")
    p.add_argument("--peso-voz", type=float, default=0.0,
                   help=">0 añade la tarea melodía/acompañamiento (POP909) con este peso en la pérdida")
    p.add_argument("--con-clasico", action="store_true",
                   help="añadir a train las obras de piano del corpus clásico de BACHI (análisis de musicólogos)")
    p.add_argument("--fraccion", type=float, default=1.0, help="fracción de las partituras de train (curva de datos)")
    p.add_argument("--trabajadores", type=int, default=0,
                   help="procesos que preparan lotes en paralelo (transponer + juntar). Útil en Linux con lotes "
                        "grandes (--lote 32: ~2,4× más rápido en la RTX 4090); en Windows cada proceso copiaría todo")
    p.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"],
                   help="dónde calcular: auto = la GPU NVIDIA (CUDA) si hay, si no la CPU")
    # con grafos pequeños, muchos hilos se estorban: 24 hilos fue ~400× más lento que 4 en esta máquina
    p.add_argument("--hilos", type=int, default=4)
    p.add_argument("--salida", type=Path, default=RAIZ / "modelos" / "mejor.pt")
    return p.parse_args(argv)


def pesos_por_acorde(y: torch.Tensor, pieza: torch.Tensor) -> torch.Tensor:
    """Peso de cada nota para que cada acorde escrito sume 1 en la pérdida.

    Por qué: la pérdida normal hace la media por NOTA, pero la métrica cuenta cada ACORDE escrito
    por igual. Un acorde que dura dos compases tiene ~40 notas y uno de paso de una negra ~4: fallar
    el de paso le cuesta al modelo 10 veces menos, así que aprende a "seguir con el acorde de la
    zona" y se traga los acordes cortos (lo vimos en el diagnóstico: 69 % de acierto en acordes de
    una negra o menos frente a 85 % en los de más de un compás).

    Un "tramo" es una racha de notas seguidas (en orden de ataque) con la misma etiqueta y de la
    misma pieza: las notas que empiezan a la vez llevan la misma etiqueta, así que cada tramo es
    un acorde escrito (o dos seguidos iguales, que da lo mismo). Cada nota pesa 1 / (notas de su
    tramo); las notas sin etiqueta pesan 0.
    """
    cambio = torch.ones_like(y, dtype=torch.bool)
    cambio[1:] = (y[1:] != y[:-1]) | (pieza[1:] != pieza[:-1])
    tramo = cambio.cumsum(0) - 1  # 0, 0, 0, 1, 1, 2…: a qué tramo pertenece cada nota
    notas_del_tramo = torch.bincount(tramo)[tramo].float()
    return torch.where(y == SIN_ETIQUETA, 0.0, 1.0 / notas_del_tramo)


def perdida_por_ataque(logits: torch.Tensor, y: torch.Tensor, pieza: torch.Tensor, onset: torch.Tensor,
                       peso_calidad: torch.Tensor | None = None) -> torch.Tensor:  # fmt: skip
    """Entropía cruzada sobre la predicción de cada ATAQUE, no de cada nota (experimento 23).

    Por qué: la métrica no mira notas sueltas. En cada punto del cifrado promedia las probabilidades
    de las notas que atacan ahí y se queda con la mayor (``predecir_en_puntos``). La pérdida por nota
    empuja a TODAS las notas del ataque hacia el acorde, también a la de paso que lo contradice; aquí
    basta con que la MEDIA del ataque acierte, igual que en el examen: el modelo puede dejar dudosa una
    nota de paso si las demás notas del acorde la compensan. Es la idea "onset-wise" de ChordGNN.

    Cada ataque pesa como en ``pesos_por_acorde`` (cada acorde escrito suma 1): las notas del mismo
    ataque comparten etiqueta (se etiquetan por su tiempo de inicio), así que la etiqueta del ataque
    es la de cualquiera de sus notas.
    """
    ataque, conteos = agrupar_por_ataque(onset, pieza)  # ordenados por pieza y tiempo
    n = sum(conteos)
    prob = logits.softmax(-1)
    suma = torch.zeros(n, prob.size(1), device=prob.device).index_add_(0, ataque, prob)
    cuantas = torch.zeros(n, device=prob.device).index_add_(0, ataque, torch.ones(len(prob), device=prob.device))
    prob_ataque = suma / cuantas[:, None]
    y_ataque = torch.full((n,), SIN_ETIQUETA, dtype=y.dtype, device=y.device).scatter_(0, ataque, y)
    pieza_ataque = torch.zeros(n, dtype=pieza.dtype, device=pieza.device).scatter_(0, ataque, pieza)
    nll = -torch.log(prob_ataque.gather(1, y_ataque.clamp(min=0)[:, None]).squeeze(1) + 1e-9)
    w = pesos_por_acorde(y_ataque, pieza_ataque)
    if peso_calidad is not None:
        w = w * peso_calidad[y_ataque.clamp(min=0) % len(CALIDADES)]
    return (nll * w).sum() / w.sum()


def pesos_por_calidad(piezas: list[dict], alfa: float) -> torch.Tensor:
    """Peso extra para cada calidad (mayor, menor, dim, aum, sus) según lo rara que sea en train.

    Por qué: el 66 % de los acordes escritos son mayores y solo el 2 % disminuidos. Con la pérdida
    normal al modelo le compensa no decir nunca "dim" (34 % de acierto en dim, 7 % en sus). El peso
    de una calidad es ``(acordes de la calidad más frecuente / acordes de esta calidad) ** alfa``:
    ``alfa = 0`` → todas igual (lo de siempre); ``alfa = 0.5`` → un dim pesa ~5 veces un mayor.
    Se cuentan ACORDES escritos (tramos de ``pesos_por_acorde``), no notas.
    """
    cuenta = torch.zeros(len(CALIDADES))
    for p in piezas:
        y = p["grafo"]["note"].y
        w = pesos_por_acorde(y, torch.zeros_like(y))
        con = y != SIN_ETIQUETA
        cuenta.index_add_(0, y[con] % len(CALIDADES), w[con])
    return (cuenta.max() / cuenta.clamp(min=1)) ** alfa


def fijar_semilla(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def main(argv=None) -> None:
    args = argumentos(argv)
    args.salida = args.salida.resolve()
    fijar_semilla(args.semilla)
    torch.set_num_threads(args.hilos)
    if args.dispositivo == "auto":
        args.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    dispositivo = torch.device(args.dispositivo)

    if args.datos == "todo_pop909":
        # PDMX a dos manos ×K (las que más se parecen a test) + POP909 ×1 (piano pop, ya son muchas) + lead sheets
        nombres_train = (nombres_de("train") * args.peso_dos_manos + nombres_de("pop909")
                         + nombres_de("lead_sheets_todas"))  # fmt: skip
    elif args.datos == "todo":
        # Las ~23 500 lead sheets ahogarían a las 461 partituras a dos manos (las que se parecen a test):
        # --peso-dos-manos K hace que cada una aparezca K veces por época
        nombres_train = nombres_de("train") * args.peso_dos_manos + nombres_de("lead_sheets_todas")
    else:
        nombres_train = nombres_de({"dos_manos": "train", "dos_manos_antiguo": "train_antiguo", "dos_manos_pop909": "train_pop909",
                                    "lead_sheets": "lead_sheets", "lead_sheets_todas": "lead_sheets_todas"}[args.datos])  # fmt: skip
    nombres_dev = nombres_de(args.dev)
    if args.humo:
        nombres_train, nombres_dev, args.epocas = nombres_train[:5], nombres_dev[:3], 1
        args.salida = args.salida.with_name("humo.pt")
    if args.con_clasico:
        nombres_train = nombres_train + nombres_de("clasico")
    if args.fraccion < 1.0:
        # Curva de datos: ¿mejora el modelo si le damos más partituras? Con la misma semilla los
        # subconjuntos están anidados (el 25 % está dentro del 50 %…), así solo cambia la cantidad.
        orden = random.Random(args.semilla).sample(nombres_train, len(nombres_train))
        nombres_train = orden[: max(1, round(len(orden) * args.fraccion))]
    train = CorpusAcordes(nombres_train, aumentar=not args.sin_transposicion, semilla=args.semilla, mutar=args.mutar_calidad,
                          mutaciones=args.mutaciones)  # fmt: skip
    dev = CorpusAcordes(nombres_dev)
    if train.fallidas and args.datos != "dos_manos":
        # unas pocas lead sheets no se pueden leer (preparar-datos lo informa): se saltan
        print(f"lead sheets sin grafo (se saltan): {len(train.fallidas)}")
    elif train.fallidas or dev.fallidas:
        raise SystemExit(f"Faltan grafos en caché ({len(train.fallidas) + len(dev.fallidas)}): corre `uv run preparar-datos`")
    if args.trabajadores > 0:  # con muchos lotes en vuelo, Linux se queda sin descriptores de archivo
        torch.multiprocessing.set_sharing_strategy("file_system")
    cargador = DataLoader(train, batch_size=args.lote, shuffle=True, num_workers=args.trabajadores,
                          persistent_workers=args.trabajadores > 0)  # fmt: skip

    metadata = train.piezas[0]["grafo"].metadata()
    modelo = crear_modelo(args.arquitectura, metadata, voz=args.peso_voz > 0, oculto=args.oculto, capas=args.capas,
                          dropout=args.dropout, clases=NUM_CLASES, cabeza=args.cabeza)  # fmt: skip
    if args.peso_voz > 0:  # la melodía del examen de voz no se ve nunca al entrenar
        examen_voz = set(cargar_splits()["pop909_voz_test"])
        for p in train.piezas:
            if p["nombre"] in examen_voz:
                p["grafo"]["note"].voz = torch.full_like(p["grafo"]["note"].voz, SIN_ETIQUETA)
    if args.partir_de:
        previo = torch.load(args.partir_de, weights_only=False, map_location="cpu")
        if previo["config"]["arquitectura"] != args.arquitectura:
            raise SystemExit(f"{args.partir_de} es {previo['config']['arquitectura']}, no {args.arquitectura}")
        modelo.load_state_dict(previo["estado"])  # mismos tamaños: falla si cambian --oculto/--capas
        print(f"partiendo de {args.partir_de.name} (época {previo['epoca']}, dev raíz+cal {previo['dev']['dev_calidad']:.1%})")
    peso_calidad = None
    if args.peso_calidad > 0:
        peso_calidad = pesos_por_calidad(train.piezas, args.peso_calidad).to(dispositivo)
        print("peso por calidad:", " · ".join(f"{q} {w:.1f}" for q, w in zip(CALIDADES, peso_calidad.tolist())))
    # La GPU necesita tener a mano el modelo y los datos: se mueven una vez (modelo, dev) o por lote
    modelo.to(dispositivo)
    for p in dev.piezas:
        p["grafo"] = p["grafo"].to(dispositivo)
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=args.peso_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epocas, 1))
    print(f"train: {len(train)} partituras ({'con' if train.aumentar else 'sin'} transposición) · dev: {len(dev)} · "
          f"modelo {args.arquitectura}: {contar_parametros(modelo):,} parámetros · {dispositivo}")  # fmt: skip

    historial, mejor = [], -1.0
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    for epoca in range(1, args.epocas + 1):
        t0 = time.time()
        modelo.train()
        suma, n_ok, n_tot, n_lotes = 0.0, 0, 0, 0
        for lote in cargador:
            lote = lote.to(dispositivo)
            # onset y pieza (de qué partitura del lote es cada nota) los usa el modelo temporal
            if args.peso_voz > 0:  # multitarea: el mismo vector de cada nota alimenta las dos salidas
                h = modelo.codificar(lote.x_dict, lote.edge_index_dict)
                logits, logits_voz = modelo.salida(h), modelo.salida_voz(h)
            else:
                logits = modelo(lote.x_dict, lote.edge_index_dict, onset=lote["note"].onset_q, pieza=lote["note"].batch)
            y = lote["note"].y
            if args.perdida_por_ataque:
                perdida = perdida_por_ataque(logits, y, lote["note"].batch, lote["note"].onset_q, peso_calidad)
            elif args.peso_por_acorde:
                # pérdida de cada nota (0 en las sin etiqueta) × su peso; dividir entre la suma de
                # pesos (= nº de acordes) da la media POR ACORDE en vez de por nota
                por_nota = torch.nn.functional.cross_entropy(logits, y, ignore_index=SIN_ETIQUETA, reduction="none")
                w = pesos_por_acorde(y, lote["note"].batch)
                if peso_calidad is not None:
                    w = w * peso_calidad[y.clamp(min=0) % len(CALIDADES)]  # las notas sin etiqueta ya pesan 0
                perdida = (por_nota * w).sum() / w.sum()
            else:
                perdida = torch.nn.functional.cross_entropy(logits, y, ignore_index=SIN_ETIQUETA)
            voz = lote["note"].voz
            if args.peso_voz > 0 and (voz != SIN_ETIQUETA).any():
                perdida = perdida + args.peso_voz * torch.nn.functional.cross_entropy(logits_voz, voz, ignore_index=SIN_ETIQUETA)
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

    destino = RAIZ / "salidas" / f"historial_{args.salida.stem}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps({"config": vars(args) | {"salida": str(args.salida)}, "historial": historial},
                                  ensure_ascii=False, indent=1, default=str))  # fmt: skip
    print(f"\nMejor raíz+calidad en dev: {mejor:.1%} → {args.salida.relative_to(RAIZ)} · historial: {destino.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
