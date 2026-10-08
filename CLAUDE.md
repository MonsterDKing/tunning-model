# tunning-model — reglas del proyecto

Proyecto de APRENDIZAJE: Pedro quiere entender cómo se entrena un modelo especializado. Todo en
español (código, comentarios, mensajes, docs) y cada paso explicado. Prioriza la claridad sobre
el truco: si un cambio mejora la métrica pero no se entiende, explícalo o no lo hagas.

## Reglas que no se rompen

1. **test (30) y dev (20) están congelados** (`data/splits.json`, vienen de Analisis-AI).
   - Nunca entran en el entrenamiento, ni transpuestos, ni "solo un poco".
   - Hiperparámetros, arquitectura, época a guardar… se eligen mirando **dev**, nunca test.
   - Test se mide al final de un experimento, para informar; no para decidir el siguiente.
   - Ninguna partitura de train puede ser la misma canción que una de test/dev (ver `datos.py`).
   - Lo mismo para `test_ampliado` (100) y `dev_ampliado` (50), sacados de las nuevas de PDMX: nunca
     entrenan, ni ellos ni sus canciones (tampoco como lead sheets). Se decide con `dev_todo` (70).
2. **Métricas comparables con Analisis-AI.** `evaluar.comparar()` reproduce `validar()` de
   Analisis-AI (raíz, raíz+calidad con `chord.Chord(cs.pitches).quality`, y "equivalente").
   Si tocas la métrica o la extracción de la referencia, vuelve a correr
   `PYTHONDONTWRITEBYTECODE=1 uv run python scripts/comprobar_metrica.py ../Analisis-AI`:
   debe dar 1532 acordes, raíz 80.7 % y raíz+calidad 76.2 % para el heurístico.
3. **Nada de salidas de Claude como etiquetas.** Las etiquetas son el cifrado humano `<harmony>`
   de PDMX (+ las 20 correcciones revisadas a mano en `correcciones.json`) y los acordes corregidos
   por humanos de POP909-CL (sin 518 y 620, que conservan los del algoritmo). No uses informes ni
   trazas de `Analisis-AI/salida/` para entrenar (eso sería destilación, no es el objetivo).
4. **No modifiques Analisis-AI.** Solo se lee de ahí.
5. **Datos = datos.** No ejecutes nada desde dentro de `data/`. No se versionan partituras ni
   PDMX.csv (licencia CC-BY-4.0, atribución en el README); solo los JSON pequeños de splits.

## Comandos

```bash
uv sync                               # entorno (torch CPU 2.8.0 + PyG + graphmuse)
uv run python -m acordes_gnn.datos    # regenerar data/splits.json
uv run preparar-datos                 # grafos en data/cache/ (si cambias grafo/etiquetas: sube VERSION_CACHE)
uv run entrenar --humo                # comprobación rápida de punta a punta
uv run entrenar                       # entrenamiento (elige la mejor época en dev)
uv run evaluar                        # modelos/mejor.pt en test (tabla con la línea base)
uv run pytest -q
```

- En esta máquina usa `--hilos 4` (por defecto): con 24 hilos torch es ~400× más lento.
- Si cambias `grafo.py` o `etiquetas.py`, sube `VERSION_CACHE` en `dataset.py` y corre
  `uv run preparar-datos --rehacer`.
