"""Exp. 27: ¿cuánto acierta un LLM en los acordes dudosos de dev? Lee el JSON de casos_dudosos.py, pregunta a
un modelo de OpenRouter N veces por acorde (voto por mayoría) y puntúa con la métrica del proyecto.

Es una MEDIDA, no etiquetado: dev tiene la respuesta humana, así que sale "en los acordes dudosos el LLM
acierta X %". Nada de lo que responde el LLM entra en ningún split ni entrenamiento (regla 3).

    PYTHONDONTWRITEBYTECODE=1 uv run python scripts/casos_dudosos.py 0.5 salidas/casos_dudosos.json
    OPENROUTER_API_KEY=… uv run python scripts/preguntar_llm.py salidas/casos_dudosos.json google/gemini-3.8-flash 1 [limite]
"""
import json, os, re, sys, time, random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import requests
from acordes_gnn.etiquetas import clase, CALIDADES
from acordes_gnn.evaluar import comparar

casos = json.load(open(sys.argv[1])); modelo = sys.argv[2]; n_votos = int(sys.argv[3])
limite = int(sys.argv[4]) if len(sys.argv) > 4 else len(casos)
casos = casos[:limite]
clave = os.environ.get("OPENROUTER_API_KEY") or [l.split("=", 1)[1].strip() for l in open("../Analisis-AI/.env") if l.startswith("OPENROUTER_API_KEY=")][0]
NOTAS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

SISTEMA = """Eres un arreglista de piano experto en cifrado americano (chord symbols) de música pop.
Te doy un pasaje de piano en texto (mano derecha MD y mano izquierda MI, nota más aguda primero), la armadura,
los acordes que un sistema automático puso en los compases anteriores y sus candidatos para el acorde que hay que decidir.
Tu tarea: decir qué acorde escribiría un arreglista humano en ese punto exacto. Ten en cuenta el bajo, la melodía,
las notas de paso y la progresión. Puedes elegir uno de los candidatos o proponer otro.
Responde SOLO con una línea con el formato exacto:  ACORDE: <símbolo>
(ejemplos de símbolo: C, Am, F#m, Bb, Ddim, Eaug, Gsus, A7, Dm7). Nada más."""

def mensaje(c):
    cand = ", ".join(f"{s} ({p:.0%})" for s, p in c["candidatos"])
    ant = ", ".join(c["anteriores_gnn"]) or "ninguno (es el primero)"
    return (f"Armadura: {c['armadura']}.\nAcordes del sistema automático en los puntos anteriores: {ant}.\n"
            f"Hay que decidir el acorde vigente en el compás {c['compas']}, pulso {c['offset'] + 1:g}.\n"
            f"Candidatos del sistema automático: {cand}.\n\nPasaje (compases {max(1, c['compas'] - 1)} a {c['compas'] + 1}):\n{c['pasaje']}\n\nACORDE:")

def interpretar(texto):
    hallazgos = re.findall(r"ACORDE:\s*\**\s*([A-G])([#b]?)([^\s/*]*)", texto)
    m = re.match(r"([A-G])([#b]?)(.*)", "".join(hallazgos[-1])) if hallazgos else None  # la última línea ACORDE: es la decisión
    if not m:
        m = re.search(r"\b([A-G])([#b]?)(m7b5|maj7|min7|dim7|dim|aug|sus[24]?|m7|m|7|\+|°|ø)?\b", texto.strip().split("\n")[-1])
        if not m: return None
    raiz = (NOTAS[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)]) % 12
    suf = (m.group(3) or "").lower()
    if suf.startswith(("dim", "°", "o", "m7b5", "ø")): cal = "dim"
    elif suf.startswith(("aug", "+")): cal = "aum"
    elif suf.startswith(("sus", "5")): cal = "sus"
    elif suf.startswith(("m", "min", "-")) and not suf.startswith("maj"): cal = "menor"
    else: cal = "mayor"
    return clase(raiz, cal)

def preguntar(c, intento=0):
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {clave}", "Content-Type": "application/json"},
                          json={"model": modelo, "temperature": 0.7 if n_votos > 1 else 0.0, "max_tokens": 4000,
                                "messages": [{"role": "system", "content": SISTEMA}, {"role": "user", "content": mensaje(c)}]})
        d = r.json()
        if "choices" not in d:
            raise RuntimeError(str(d)[:200])
        m = d["choices"][0]["message"]
        texto = m.get("content") or ""
        if "ACORDE" not in texto:  # modelos razonadores: el texto puede venir en reasoning
            texto = texto + "\n" + (m.get("reasoning") or "")
        return texto or "VACIO", d.get("usage", {})
    except Exception as e:  # noqa
        if intento < 3:
            time.sleep(2 + 3 * intento); return preguntar(c, intento + 1)
        return f"ERROR {e}", {}

def votar(c):
    votos, usos, crudos = [], [], []
    for _ in range(n_votos):
        txt, uso = preguntar(c); crudos.append(txt); usos.append(uso)
        k = interpretar(txt)
        if k is not None: votos.append(k)
    eleccion = Counter(votos).most_common(1)[0][0] if votos else None
    return c, eleccion, votos, crudos, usos

t0 = time.time(); res = Counter(); tokens = Counter(); salida = []
with ThreadPoolExecutor(8) as pool:
    for c, elec, votos, crudos, usos in pool.map(votar, casos):
        ref = {"raiz": c["ref_raiz"], "calidad_m21": c["ref_calidad_m21"], "alturas": set(c["ref_alturas"])}
        ok = comparar(elec, ref)[1] if elec is not None else False
        res["n"] += 1; res["ok"] += ok; res["gnn_ok"] += c["gnn_ok"]; res["sin_respuesta"] += elec is None
        res["unanime"] += len(set(votos)) == 1 and len(votos) == n_votos
        for u in usos: tokens["in"] += u.get("prompt_tokens", 0); tokens["out"] += u.get("completion_tokens", 0)
        salida.append({"pieza": c["pieza"], "k": c["k"], "referencia": c["referencia"], "gnn": c["gnn"], "gnn_ok": c["gnn_ok"],
                       "llm": elec, "llm_ok": bool(ok), "votos": votos, "crudo": crudos[0][:200]})
        if limite <= 10: print(f"ref {c['referencia']:8} gnn {c['gnn']:6} {'✓' if c['gnn_ok'] else '✗'}   llm → {crudos[0].strip()[:60]!r} {'✓' if ok else '✗'}")
n = res["n"]
print(f"\n{modelo} × {n_votos} votos · {n} acordes dudosos · LLM {res['ok']/n:.1%} · GNN en los mismos {res['gnn_ok']/n:.1%} · "
      f"sin respuesta {res['sin_respuesta']} · unánimes {res['unanime']/n:.0%} · tokens in {tokens['in']} out {tokens['out']} · {time.time()-t0:.0f}s")
json.dump(salida, open(sys.argv[1].replace(".json", f"_{modelo.replace('/', '_')}_x{n_votos}.json"), "w"), ensure_ascii=False, indent=1)
