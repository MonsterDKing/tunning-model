"""Exp. 27: acordes dudosos de dev_todo (margen top1-top2 < UMBRAL con el ensemble del Exp. 21) → JSON con el
pasaje en texto (dos manos, compás anterior y siguiente), los candidatos de la GNN y la referencia humana.
Lo lee scripts/preguntar_llm.py.

    PYTHONDONTWRITEBYTECODE=1 uv run python scripts/casos_dudosos.py 0.5 salidas/casos_dudosos.json
"""
import json, sys, torch
from collections import Counter
from acordes_gnn.dataset import cargar, nombres_de
from acordes_gnn.etiquetas import nombre, RAICES
from acordes_gnn.evaluar import cargar_modelo, comparar, notas_del_instante, VENTANA_NEGRAS, PESO_VENTANA
torch.set_num_threads(8)
UMBRAL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
M = "modelos/"  # bajo_s0..2.pt: los tres modelos del Exp. 21
modelos = [cargar_modelo(M + f"bajo_s{s}.pt")[0].eval() for s in range(3)]
NOMBRE = {0: "Do", 1: "Do#", 2: "Re", 3: "Mib", 4: "Mi", 5: "Fa", 6: "Fa#", 7: "Sol", 8: "Lab", 9: "La", 10: "Sib", 11: "Si"}
ARMADURA = {0: "Do mayor / La menor", 1: "Sol mayor / Mi menor", 2: "Re mayor / Si menor", 3: "La mayor / Fa# menor", 4: "Mi mayor / Do# menor",
            5: "Si mayor / Sol# menor", 6: "Fa# mayor / Re# menor", 7: "Reb mayor / Sib menor", 8: "Lab mayor / Fa menor", 9: "Mib mayor / Do menor",
            10: "Sib mayor / Sol menor", 11: "Fa mayor / Re menor"}

def nota(p): return f"{NOMBRE[int(p) % 12]}{int(p) // 12 - 1}"

def dur(d):
    for v, n in ((4, "redonda"), (3, "blanca con puntillo"), (2, "blanca"), (1.5, "negra con puntillo"), (1, "negra"), (0.75, "corchea con puntillo"), (0.5, "corchea"), (0.25, "semicorchea")):
        if abs(d - v) < 0.01: return n
    return f"{d:g} negras"

def pasaje(g, compas, inicios):
    """Texto de los compases compas-1 .. compas+1: por ataque, mano derecha y mano izquierda."""
    n = g["note"]; lineas = []
    izq = n.fijos[:, 4] > 0.5
    for c in range(max(0, compas - 1), min(int(n.compas.max()) + 1, compas + 2)):
        en = (n.compas == c).nonzero().flatten()
        if len(en) == 0: continue
        lineas.append(f"Compás {c + 1}:")
        for t in sorted(set(n.onset_q[en].tolist())):
            idx = en[(n.onset_q[en] - t).abs() < 1e-4]
            pulso = t - inicios[c] + 1
            md = [i for i in idx.tolist() if not izq[i]]; mi = [i for i in idx.tolist() if izq[i]]
            f = lambda ids: ", ".join(f"{nota(n.pitch[i])} ({dur(float(n.dur_q[i]))})" for i in sorted(ids, key=lambda i: -int(n.pitch[i])))
            lineas.append(f"  pulso {pulso:g}: MD: {f(md) or '—'} | MI: {f(mi) or '—'}")
    return "\n".join(lineas)

casos, resumen = [], Counter()
for nombre_pieza in nombres_de("dev_todo"):
    p = cargar(nombre_pieza); g = p["grafo"]; n = g["note"]
    with torch.no_grad():
        prob = sum(m(g.x_dict, g.edge_index_dict, onset=n.onset_q).softmax(-1) for m in modelos) / 3
    # inicio de cada compás (en negras) a partir de las notas
    inicios = {}
    for c, t in zip(n.compas.tolist(), n.onset_q.tolist()):
        inicios[c] = min(inicios.get(c, 1e9), t)
    preds_gnn = []
    for k, ref in enumerate(p["acordes"]):
        t = ref["tiempo"]
        w = torch.zeros(len(n.onset_q)); w[notas_del_instante(n.onset_q, n.dur_q, t)] = 1.0
        desp = (n.onset_q > t + 1e-4) & (n.onset_q < t + VENTANA_NEGRAS - 1e-4) & (w == 0); w[desp] = PESO_VENTANA
        v = (prob * w[:, None]).sum(0) / w.sum()
        top = v.topk(3); margen = float(top.values[0] - top.values[1])
        pred = int(top.indices[0]); ok = comparar(pred, ref)[1]; preds_gnn.append(nombre(pred))
        resumen["acordes"] += 1; resumen["ok"] += ok
        if margen < UMBRAL:
            resumen["dudosos"] += 1; resumen["dudosos_ok"] += ok
            resumen["dudosos_top3"] += any(comparar(int(c), ref)[1] for c in top.indices.tolist())
            i0 = int(notas_del_instante(n.onset_q, n.dur_q, t)[0]); compas = int(n.compas[i0])
            casos.append({"pieza": nombre_pieza, "k": k, "compas": compas + 1, "offset": ref["offset"], "referencia": ref["figura"],
                          "ref_raiz": ref["raiz"], "ref_calidad_m21": ref["calidad_m21"], "ref_alturas": sorted(ref["alturas"]),
                          "candidatos": [(nombre(int(c)), round(float(pv), 3)) for c, pv in zip(top.indices.tolist(), top.values.tolist())],
                          "gnn": nombre(pred), "gnn_ok": bool(ok), "armadura": ARMADURA[int(n.tonica[i0])],
                          "anteriores_gnn": preds_gnn[max(0, k - 3):k],
                          "pasaje": pasaje(g, compas, inicios)})
print(f"acordes {resumen['acordes']} · acierto GNN {resumen['ok']/resumen['acordes']:.1%} · dudosos (margen<{UMBRAL}) {resumen['dudosos']} "
      f"({resumen['dudosos']/resumen['acordes']:.1%}) · acierto GNN en dudosos {resumen['dudosos_ok']/resumen['dudosos']:.1%} · "
      f"la verdad está en el top-3 {resumen['dudosos_top3']/resumen['dudosos']:.1%} · fallos cubiertos "
      f"{(resumen['dudosos']-resumen['dudosos_ok'])/(resumen['acordes']-resumen['ok']):.0%}")
json.dump(casos, open(sys.argv[2] if len(sys.argv) > 2 else "casos_dudosos.json", "w"), ensure_ascii=False, indent=1)
print(casos[0]["pasaje"][:600]); print(casos[0]["candidatos"], casos[0]["referencia"])
