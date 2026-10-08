# tunning-model — un modelo pequeño de acordes con GraphMuse

Proyecto de aprendizaje: entrenar desde cero una **red neuronal de grafos (GNN)** que reconozca
los acordes de una partitura de piano, para entender **cómo trabajan los modelos especializados**:
cómo se representan los datos, de dónde salen las etiquetas, cómo aprende, cómo se sobreajusta y
cómo se mide de forma honesta.

- **Etiquetas = cifrado humano.** El modelo aprende del `<harmony>` que escribió cada arreglista
  en las partituras de [PDMX](#datos-y-licencia). **No** se entrena con salidas de Claude (eso sería
  destilación: copiar a otro modelo, no aprender música de los datos).
- **Misma prueba que Analisis-AI.** Se mide en el mismo set de 30 partituras (1532 acordes) y con
  la misma métrica que el heurístico y Claude, así que los números se pueden comparar directamente
  (comprobado: ver [Métrica](#5-evaluar-en-test)).

## La línea base a batir

Set de prueba congelado (30 partituras de PDMX, 1532 acordes escritos):

| método                         | raíz      | raíz + calidad | equivalente |
|--------------------------------|-----------|----------------|-------------|
| Heurístico (Analisis-AI)       | 80.7 %    | 76.2 %         | 83.1 %      |
| Claude Opus (prompt v3)        | 90.8 %    | 88.4 %         | 92.6 %      |
| **GNN (este proyecto)**        | **87.2 %**| **84.7 %**     | **85.8 %**  |

GNN = media de 3 semillas del mejor modelo (Experimento 13: `--datos todo_pop909 --peso-dos-manos 10
--oculto 256 --epocas 10`, entrenado en la RTX 4090), medida al cerrar el experimento (ver
[Experimentos](#experimentos-realizados)). En el test ampliado (100 partituras más, 4067 acordes):
raíz 88.9 %, raíz+calidad 85.8 %, equivalente 89.0 %, frente a 87.2 % / 83.6 % / 87.9 % del heurístico.

*raíz + calidad* = raíz y mayor/menor/dim/aum correctos. *Equivalente* = las notas del acorde
predicho están contenidas en las del escrito o al revés (C ⊆ C7, Am ⊆ Fmaj7).

**Expectativa honesta.** Con ~190 partituras de entrenamiento lo realista es **batir al
heurístico**. Igualar a Claude es difícil: Claude trae millones de partituras y teoría musical
"de serie", reconoce canciones y razona sobre la tonalidad global; nuestro modelo solo sabe lo
que vea en 196 arreglos. Que un modelo de 300 mil parámetros entrenado en un minuto de CPU se
acerque ya es una lección sobre por qué existen los modelos especializados.

---

## Las ideas, en cristiano

### ¿Qué es una GNN sobre partituras?

Una red neuronal solo entiende números. Una imagen es una cuadrícula de píxeles; un texto, una
secuencia de palabras. ¿Y una partitura? Es más rara: varias voces a la vez, notas largas que
siguen sonando mientras empiezan otras, silencios… [GraphMuse](https://github.com/manoskary/graphmuse)
(Karystinaios & Widmer) la convierte en un **grafo**:

- cada **nota** es un nodo, con unos pocos números que la describen (qué nota es, su altura,
  cuánto dura, dónde cae en el compás, en qué mano está);
- las **aristas** unen notas según su relación temporal: *empiezan a la vez* (`onset`), *una
  empieza justo cuando acaba la otra* (`consecutive`), *una empieza mientras la otra suena*
  (`during`), *hay un silencio entre medias* (`rest`);
- nodos virtuales de **pulso** y **compás** que agrupan lo que suena en cada pulso o compás.

Una **GNN** procesa ese grafo por rondas (*message passing*): en cada capa, cada nota recibe un
resumen de sus vecinas por cada tipo de arista, lo mezcla con lo suyo usando matrices que se
aprenden, y actualiza su vector. Tras 3 capas, el vector de una nota "sabe" qué suena con ella y
alrededor. Al final, una capa lineal lo convierte en 60 puntuaciones: una por acorde posible.

### ¿Por qué un grafo y no reglas? El problema de las notas de paso

El heurístico de Analisis-AI decide el acorde sumando cuánto suena cada nota en una ventana y
comparándolo con plantillas. Funciona… hasta que la melodía mete **notas de paso**: un Re de paso
sobre un acorde de Do "ensucia" la ventana. La solución fueron pesos ajustados a mano:

```python
PESO_ATAQUE = 1.5     # lo que suena al inicio de la ventana pesa más
PESO_MELODIA = 0.6    # la nota más aguda suele ser melodía (con notas de paso)
PESO_BAJO = 0.15      # premio si la raíz es la nota más grave
CASTIGO_CROMATICO = 0.06
```

Cada peso es una hipótesis de un humano sobre cómo escuchar, y se ajustó mirando 9 partituras
(y se sobreajustó a ellas). En la GNN esas decisiones son **parámetros aprendidos**: la matriz
de la arista `during` aprende cuánto debe influir un bajo largo sobre la melodía que suena
encima; el rasgo `mano_izquierda` y la `altura` le permiten descubrir solo que la nota más grave
importa más; la arista `consecutive` le deja ver que una nota corta entre dos notas del acorde es
"de paso". Nadie escribe esas reglas: salen de 100 000 notas etiquetadas por arreglistas.

---

## El plan de aprendizaje, paso a paso

Cada paso tiene su módulo en `src/acordes_gnn/`, con la explicación en el docstring de arriba.
Léelos en orden.

### 1. De partitura a grafo — `grafo.py`

```bash
uv run explorar-grafo data/xmls_originales/2_When_I_Was_Your_Man.musicxml --compas 3
```

```
NODOS
  note        699   rasgos por nodo: 30
  measure      44   rasgos por nodo: 30
  beat        169   rasgos por nodo: 30

ARISTAS (origen —tipo→ destino: cuántas)
  note    —onset           → note       2645
  note    —consecutive     → note       2340
  note    —during          → note        641
  note    —rest            → note         75
  note    —consecutive_rev → note       2340
  …
EJEMPLO: la nota Do4 (compás 3, negra 1, mano der.) — nodo 22
  rasgos: pc_C=1.00, duracion=0.54, tiempo_fuerte=1.00, en_pulso=1.00
  [onset] empieza a la vez que: Re2 (compás 3, negra 1, mano izq.), Re3 (…), Fa#3 (…), La3 (…) … (+1)
  [consecutive] termina justo cuando empieza: Re2 (compás 3, negra 3, mano izq.), …
  [consecutive_rev] empieza justo cuando termina: Mi4 (compás 2, negra 4.33333, mano der.), …
  [connects] pertenece al nodo-compás 3 y al nodo-pulso 8
```

**Para pensar:** prueba con una partitura con arpegios en la mano izquierda: ¿qué aristas dominan?
¿Qué vecinos tendría que mirar el modelo para saber que un arpegio Do-Mi-Sol es un acorde de Do?

### 2. Las etiquetas — `etiquetas.py`

Cada nota se etiqueta con el acorde escrito **vigente en su ataque**. Vocabulario: **60 clases =
12 raíces × 5 calidades** (mayor, menor, dim, aum, sus = "sin tercera"). Las séptimas, novenas e
inversiones se pliegan a su tríada (C7 → C, Bm7b5 → Bdim, C/E → C): con ~190 partituras no hay
ejemplos para 12 × 20 tipos de acorde, y la métrica tampoco los distingue. La tabla completa de
reglas está en el docstring.

```bash
uv run etiquetas data/xmls_originales/21_PERFECT_-_Lead_Sheet.musicxml   # nota → acorde, con las 16 correcciones aplicadas
uv run etiquetas --split train                                           # cuántas notas quedan sin etiqueta
```

Notas **sin etiqueta** (se ignoran en la pérdida): las anteriores al primer acorde, las que van
tras un "N.C." y las que caen **más de 4 compases** después del último acorde escrito (muchos
arreglistas solo cifran la primera sección; sin ese límite el último acorde "sonaría" hasta el
final y enseñaría basura). Resultado: 10.8 % de notas sin etiqueta en train, 7.7 % en dev,
12.5 % en test.

Para las 9 partituras originales de Analisis-AI se aplican las 20 correcciones revisadas a mano de
`data/xmls_originales/correcciones.json` (acordes escritos que contradicen las notas).

**Detalle técnico (alineación):** la referencia se lee con music21 (como `validar()`), las notas
con partitura (como GraphMuse). Partitura pone la anacrusa en tiempo negativo y music21 en 0, así
que cada acorde se ubica por *(nº de compás, offset en el compás)*, no por tiempo absoluto.

### 3. El modelo mínimo — `dataset.py`, `modelo.py`, `entrenar.py`

```bash
uv run preparar-datos      # 1 vez: cada partitura → data/cache/*.pt (≈1 min con 20 procesos)
uv run entrenar --humo     # 1 época, 5 partituras: comprueba que todo encaja
uv run entrenar            # 60 épocas (~1,2 s/época en CPU)
```

- **Modelo** (`ModeloAcordes`): proyección lineal de los 30 rasgos (18 + 12 de grado en la
  armadura) → 3 capas `HeteroConv` con un `SAGEConv` por tipo de arista (13 tipos), conexión
  residual + `LayerNorm` + dropout → lineal a 60 clases. Con `--oculto 128` (por defecto),
  ~1,3 millones de parámetros; con `--oculto 64`, ~333 000. Alternativas:
  `--arquitectura metrical` (la `MetricalGNN` de GraphMuse tal cual) y `--arquitectura temporal`
  (GNN + GRU bidireccional sobre los ataques).
- **Entrenamiento**: entropía cruzada **pesada por acorde** (cada acorde escrito suma lo mismo,
  como en la métrica; `--no-peso-por-acorde` vuelve a la media por nota), AdamW, *cosine
  annealing*, recorte de gradiente. Al final de cada época mide en **dev** y guarda la época con
  mejor raíz+calidad.

Salida del modo humo (5 partituras, 1 época; cifras de un modelo que aún no aprendió nada —
la pérdida inicial ≈ ln 60 = 4.1 es "adivinar al azar entre 60"):

```
$ uv run entrenar --humo
train: 5 partituras (con transposición) · dev: 3 · modelo propio: 330,684 parámetros
época   1  pérdida 4.405  train/nota  1.3%  dev/nota 10.6%  dev raíz 11.5%  raíz+cal 10.9%  equiv 11.5%  (0.0s)  ← mejor en dev, guardado

Mejor raíz+calidad en dev: 10.9% → modelos/humo.pt · historial: salidas/historial_humo_20261007-021847.json
```

Primera prueba real (solo para comprobar que aprende; **medida en dev, no en test**):
`uv run entrenar --epocas 40` → mejor época 32: **dev raíz 86.1 %, raíz+calidad 84.3 %,
equivalente 87.4 %** en ~50 s de CPU. Como referencia, en ese mismo dev el heurístico sacó
89.1 % / 84.7 % y Claude (prompt v2) 94.2 % / 93.1 %. Fíjate en el historial: `train/nota`
(≈75 %) va por debajo de `dev/nota` (≈85 %) porque train se mide con dropout y transpuesto a
tonalidades raras; todavía no hay sobreajuste → hay margen para un modelo más grande.

### 4. Sobreajuste, transposición, regularización

Experimentos sugeridos (todos midiendo en dev):

- **Sin aumento**: `uv run entrenar --sin-transposicion`. ¿Train sube y dev baja? Eso es
  sobreajuste: el modelo memoriza las tonalidades de las 196 partituras.
- **Capacidad**: `--oculto 128 --capas 4` vs `--oculto 32 --capas 2`. Dibuja (o mira en
  `salidas/historial_*.json`) la distancia entre train y dev.
- **Regularización**: `--dropout 0.0` vs `0.5`, `--peso-decay 1e-3`.
- **Arquitectura**: `--arquitectura metrical` (la de GraphMuse) vs la propia.
- **Rasgos**: añade a `grafo.py` "es la nota más grave de su ataque" (lo que hacía PESO_BAJO).
  ¿Mejora, o la GNN ya lo había aprendido con la altura + aristas `onset`?
- **Semillas**: repite con `--semilla 1, 2, 3`. Si dos configuraciones difieren menos que la
  variación entre semillas, la diferencia no es real (dev tiene solo 1099 acordes).

**Aumento por transposición:** transponer una partitura k semitonos no cambia las aristas, solo
las alturas y la raíz de las etiquetas. En cada época cada partitura de train se transpone a una
tonalidad al azar, así el modelo ve las 12 y aprende *relaciones* entre notas, no alturas fijas.
Dev y test nunca se transponen.

#### Experimentos realizados

Todos con 40 épocas y 3 semillas (0, 1, 2), medidos en **dev** (raíz + calidad, media). El
detalle, con gráficas, está en el documento de aprendizaje paso a paso.

| # | Experimento | Resultado en dev | Conclusión |
|---|-------------|------------------|------------|
| 1 | Curva de datos (`--fraccion 0.25/0.5/0.75/1`) | 76.4 → 80.8 → 82.8 → 84.2 % | Más datos ayudan, cada vez menos (~+4 puntos por cada vez que se duplican). La semilla mueve ±1 punto |
| 2 | Análisis de errores | — | 80 % de los fallos son acordes "vecinos" (quinta, tercera, relativo); 40 % son de frontera |
| 3 | GNN + GRU (`--arquitectura temporal`) | 83.7 % (base 84.2 %) | Sin mejora: la GRU suaviza en el tiempo y empeora las fronteras |
| 4 | Grado en la armadura (12 rasgos) | 84.2 % (base 84.2 %) | Sin mejora: la tonalidad no dice cuándo cambia el acorde, ni separa mayor/relativa menor |
| — | Diagnóstico de fronteras (en train) | — | Se traga los acordes cortos: 69 % de acierto en los de ≤ 1 negra frente a 85 % en los de > 1 compás |
| 5 | Pérdida pesada por acorde (`pesos_por_acorde`) | **86.6 %** (base 84.2 %) | Mejora real: acordes cortos +6.5 puntos. En test, 79.7 % frente a 78.6 % |
| 6 | Preentrenar con 3692 lead sheets y afinar (`--datos lead_sheets`, `--partir-de`) | 86.4 % (desde cero 86.2 %) | Sin ganancia: acelera el arranque, pero la melodía sola no enseña la mano izquierda. Con `--lr 1e-3`, peor (83.3 %). Se corrigió una fuga (Op. 299): train 196 → 195 |
| 7 | Modelo 4× más grande (`--oculto 128`) | **88.0 %** (64: 86.2 %) | El modelo pequeño se quedaba corto. Preentrenar sigue sin ayudar (88.0 %) |
| 8 | Más ancho (`--oculto 256`) o más profundo (`--capas 4`) | 87.9 % / 87.3 % | Techo: ningún tamaño memoriza (train limpio 82 % < dev 88 %); el límite son las etiquetas de train. En test, 128 saca 81.1 % frente a 80.8 % de 64 |
| 9 | +266 partituras a dos manos del PDMX completo (train 195 → 461; `scripts/buscar_cifrado_pdmx.py`, `extraer_pdmx.py`, `auditar_pdmx.py`) | **89.2 %** (128) / **89.5 %** (256) | Más datos del mismo tipo sí ayudan, y con ellos el modelo grande empieza a rendir. En test: 82.5 % / 82.6 % (antes 81.1 %); supera al heurístico en las tres métricas |
| 10 | Todo PDMX con cifrado: 461 a dos manos + 23 400 lead sheets (`--datos todo`), 10 épocas en la RTX 4090 | 86.0 % (sin peso) / **90.3 %** (`--peso-dos-manos 10`) | Mezclar sin más empeora; con más peso para las de dos manos mejora un poco. En test: 83.4 % frente a 82.6 % |
| 11 | Exámenes ampliados: test +100 y dev +50 (de las nuevas, por canción); train 461 → 310 | — | El heurístico saca 76.2 % en el test original y 83.6 % en el ampliado: con 30 partituras la cifra depende mucho de cuáles tocaron. Desde aquí se decide con dev 70 |
| 12 | +905 canciones de POP909-CL (`--datos dos_manos_pop909`) | dev 70: 85.7 % (sin: 84.9 %) | Mejora en los tres conjuntos: test original 83.3 % (81.6 %), test ampliado 85.3 % (84.7 %; heurístico 83.6 %) |
| 13 | Todo junto: 310 a dos manos ×10 + 905 POP909 + 23 400 lead sheets (`--datos todo_pop909`), 10 épocas en la RTX 4090 | dev 70: **86.4 %** | Mejor modelo: test original 84.7 %, test ampliado 85.8 %, los 130 juntos 85.5 % (raíz 88.4 %) |
| 14 | Prior armónico: 50 M de cambios de acorde de Chordonomicon + Viterbi (`prior.py`, `scripts/probar_prior.py`) | dev 70: 86.47 % (sin: 86.44 %) | No mejora: en el 63 % de los fallos el prior prefiere la respuesta equivocada (los errores son progresiones habituales) |
| 15 | kNN contra las 60 plantillas de acordes | dev 70: 43 % (instante) · 60 % (ventana perfecta) | El tiempo importa, pero lo difícil es qué notas son del acorde; quitar la melodía empeora (la tercera suele estar en la mano derecha) |
| 16 | Más peso a los acordes raros (`--peso-calidad 0.5`) | dev 70: 83.4 % | aum 33 % → 56 %, dim 27 % → 33 %, pero mayores −4 y global −2 a −3: desequilibrio de clases |
| 17 | Multitarea melodía/acompañamiento (`--peso-voz 1`, etiquetas del POP909 original alineadas con POP909-CL; `scripts/evaluar_voz.py`) | dev 70: 86.4 % (acordes) | Melodía: F1 63.7 % frente a 52.1 % del *skyline* (acierto 85.8 % frente a 65.0 %). A los acordes no les ayuda |

### 5. Evaluar en test

```bash
uv run evaluar                       # modelos/mejor.pt en test; imprime la tabla con la línea base
uv run evaluar --split dev
```

Para cada acorde escrito en el instante t, el acorde del modelo es la **media de las
probabilidades** de las notas que **atacan** en t (si no hay, las que **suenan** en t; si es un
silencio, las del siguiente ataque). Se usa solo t, nunca dónde acaba el acorde escrito: el
heurístico y Claude tampoco conocen la segmentación de la referencia. Detalles y razones en el
docstring de `evaluar.py`.

**La métrica es la de Analisis-AI, comprobado.** `scripts/comprobar_metrica.py` corre el
heurístico de Analisis-AI en las 30 de test y lo puntúa con su `validar()` y con nuestro
`comparar()`: 1532 acordes, raíz 80.7 % y raíz+calidad 76.2 % en ambos; equivalente 83.2 %
(Analisis-AI reportó 83.1 %, 1–2 acordes de diferencia en la definición exacta).
Rareza heredada (a propósito): music21 llama "minor" a C6 y "diminished" a Cm6; esos acordes
penalizan igual a los tres métodos.

**Regla:** test se mide al final de un experimento para informar. Si miras test, cambias algo y
vuelves a mirar, test se convierte en dev y el número deja de ser honesto (es exactamente cómo
el heurístico se sobreajustó a las 9 partituras originales).

### 6. (Opcional) Numerales romanos estilo ChordGNN

[ChordGNN](https://arxiv.org/abs/2307.03544) (Karystinaios & Widmer, ISMIR 2023) predice, además
del acorde, la tonalidad local, el grado, la inversión… con varias cabezas de salida sobre el
mismo grafo (*multitarea*), y resume las predicciones por ataque (*onset-wise*). Siguiente paso
natural: añadir a `ModeloAcordes` una cabeza de tonalidad (24 clases) y derivar el numeral
romano. Las etiquetas de tonalidad se podrían sacar de la armadura + Krumhansl como en
Analisis-AI (¡ojo: serían etiquetas heurísticas, no humanas!), o de corpus anotados como
DCML/ABC (Beethoven, Mozart) que usa ChordGNN.

Otro opcional: **preentrenar** con las 3714 *lead sheets* de `splits.json["lead_sheets"]`
(solo melodía + cifrado: el modelo aprende qué melodías van con qué acordes) y luego afinar con
los arreglos a dos manos.

---

## Estructura

```
tunning-model/
├── src/acordes_gnn/
│   ├── datos.py        # paso 0: copiar corpus + construir data/splits.json
│   ├── grafo.py        # paso 1: MusicXML → grafo (GraphMuse) + explorar-grafo
│   ├── etiquetas.py    # paso 2: vocabulario de 60 acordes, nota → acorde escrito
│   ├── dataset.py      # paso 3: caché de grafos + aumento por transposición
│   ├── modelo.py       # paso 3: la GNN
│   ├── entrenar.py     # paso 4: bucle de entrenamiento (+ --humo)
│   └── evaluar.py      # paso 5: métrica de Analisis-AI en test/dev
├── tests/              # pytest: etiquetas y métrica
├── scripts/comprobar_metrica.py   # paridad con validar() de Analisis-AI
├── data/               # (ignorado por git salvo los JSON pequeños)
│   ├── splits.json
│   ├── pdmx/con_cifrado/          # 4132 MusicXML + _indice.json + _dos_manos.json
│   ├── pdmx/muestra30.json        # test congelado (nombres de archivo)
│   ├── pdmx/desarrollo20.json     # dev
│   ├── pdmx/heuristico_dos_manos.json   # auditoría de la referencia por partitura
│   ├── pdmx/PDMX.csv              # metadatos completos de PDMX (215 MB)
│   ├── xmls_originales/           # 9 partituras originales con cifrado + correcciones.json
│   └── cache/                     # grafos preprocesados (uv run preparar-datos)
├── modelos/            # checkpoints (ignorado)
└── salidas/            # historiales y evaluaciones (ignorado)
```

## Datos y licencia

Los datos se copiaron del proyecto Analisis-AI (que no se modifica). Para rehacerlo:
`uv run python -m acordes_gnn.datos --copiar-desde ../Analisis-AI`.

**PDMX** — *Public Domain MusicXML*, Long, P., Novack, Z., Berg-Kirkpatrick, T. & McAuley, J.
(2024), [arXiv:2409.10831](https://arxiv.org/abs/2409.10831), datos en
[Zenodo (record 15571083)](https://zenodo.org/records/15571083). Licencia **CC-BY-4.0**. Las
partituras de PDMX son de dominio público o CC; este proyecto usa un subconjunto (piano con
cifrado `<harmony>`) y no redistribuye los archivos (están fuera de git). Analisis-AI solo tenía
4132 de las 26 354 partituras de PDMX con cifrado; el resto sale de recorrer el PDMX completo
(`mxl.tar.gz`) con `scripts/buscar_cifrado_pdmx.py` → `extraer_pdmx.py` → `auditar_pdmx.py`.

**POP909-CL** — Yao, M., Chen, K., Dubnov, S. & Berg-Kirkpatrick, T. (2026), *BACHI: Boundary-Aware
Symbolic Chord Recognition Through Masked Iterative Decoding on Pop and Classical Music*, ICASSP,
[arXiv:2510.06528](https://arxiv.org/abs/2510.06528), datos en
[POP909-CL-Dataset](https://github.com/AndyWeasley2004/POP909-CL-Dataset) (licencia **MIT**), sobre
POP909 (Wang, Z., Chen, K. et al., ISMIR 2020). 909 canciones pop para piano en MIDI con acordes
corregidos por humanos; se leen con `pop909.py` (`git clone` en `data/pop909cl/`, fuera de git).
Ninguna repite canción de los exámenes (comprobado con su `index.xlsx`). Para la tarea de melodía
se usa además el POP909 original (`data/pop909_original/`, [music-x-lab/POP909-Dataset](https://github.com/music-x-lab/POP909-Dataset)),
que separa MELODY, BRIDGE y PIANO.

**Chordonomicon** — Kantarelis, S. et al. (2024), *CHORDONOMICON: A Dataset of 666,000 Songs and their
Chord Progressions*, [arXiv:2410.22046](https://arxiv.org/abs/2410.22046), datos en
[Hugging Face](https://huggingface.co/datasets/ailsntua/Chordonomicon) (licencia **CC-BY-NC-4.0**:
uso personal y de aprendizaje). Solo progresiones de acordes, sin notas: se usa para el prior
armónico (`prior.py`), no para entrenar la GNN.

| conjunto   | partituras | criterio |
|------------|-----------:|----------|
| test       | 30   | `muestra30.json`: el set de prueba con el que se midieron heurístico y Claude. **Congelado.** |
| test_ampliado | 100 | sacadas de las nuevas de PDMX, por canción, al azar (semilla 0): examen más grande. |
| dev        | 20   | `desarrollo20.json`. |
| dev_ampliado | 50 | ídem. `--dev dev_todo` (por defecto) elige la época con dev + dev_ampliado (70). |
| train      | 187 + 8 + 115 | arreglos a dos manos legibles, con referencia confiable (`raiz_ausente` ≤ 15 % y `contradichos` ≤ 5 % de los acordes) y que no repiten canción de los exámenes: 187 de PDMX (Analisis-AI), 8 originales de Analisis-AI (`Shallow` queda fuera: está en test) y 115 nuevas del PDMX completo. |
| pop909     | 905  | POP909-CL (sin 518 y 620, con etiquetas del algoritmo, ni 2 archivos incompletos). `--datos dos_manos_pop909`. |
| lead_sheets(_nuevas) | 3690 + 19 706 | un solo pentagrama, sin canciones de los exámenes. Solo con `--datos todo` / `todo_pop909`. |

Descartados de train: 42 ilegibles, 69 con referencia poco confiable, 4 por repetir canción de
test/dev (3 de *School of Velocity Op. 299* y un *Shallow*). Las 62 partituras de dos pentagramas
con la mano izquierda casi vacía no van a ningún split. `datos.py` comprueba que los splits no se
solapan. En total: train 97 682 notas / 8245 acordes escritos; dev 12 064 / 1099; test 20 333 / 1532.

## Cuaderno paso a paso

`notebooks/paso_a_paso.ipynb` recorre todo el proceso con una partitura de dev («Believer»): notas y
acordes escritos, el grafo y los rasgos de una nota, las etiquetas, el modelo sin entrenar, un
entrenamiento en directo de segundos en la CPU, el mejor modelo compás a compás y sus errores
típicos. Se abre con `uv run jupyter lab` (JupyterLab y matplotlib están en el grupo `dev`).

## Instalación

```bash
uv sync          # Python 3.12, torch 2.8.0+cpu, PyG 2.8, graphmuse 0.0.5, partitura 1.9
uv run pytest -q
```

Versiones instaladas: `torch 2.8.0+cpu`, `torch-geometric 2.8.0.post1`, `pyg-lib 0.5.0+pt28cpu`,
`torch-scatter 2.1.2+pt28cpu`, `torch-sparse 0.6.18+pt28cpu`, `torch-cluster 1.6.3+pt28cpu`,
`graphmuse 0.0.5`, `partitura 1.9.0`, `numpy 2.3.5`, `music21`.

Por qué así (ver `pyproject.toml`):

- **torch para CPU** desde `https://download.pytorch.org/whl/cpu` (esta máquina no tiene GPU
  NVIDIA; la rueda de PyPI trae CUDA y pesa ~2 GB).
- **torch 2.8.0** porque es la última versión para la que `data.pyg.org` publica las 4 ruedas
  nativas que pide graphmuse (pyg-lib, torch-scatter, torch-sparse, torch-cluster) para cp312
  (para torch 2.9 ya no hay pyg-lib). Se instalan con `find-links`, sin compilar nada.
- **Workaround numpy < 2.4**: partitura 1.9 usa `np.row_stack`, que numpy eliminó en 2.4
  (`AttributeError: module 'numpy' has no attribute 'row_stack'`). Con numpy 2.3.5 funciona.
- graphmuse 0.0.5 trae su extensión en C precompilada para cp312: `import graphmuse` funciona.
- **Hilos**: `entrenar` usa 4 hilos por defecto. Con los 24 núcleos, cada paso fue ~400× más
  lento (los grafos son pequeños y la sincronización entre hilos domina).

**RTX 4090 (Windows, 192.168.1.114)**: `uv run entrenar` usa la GPU sola si la hay
(`--dispositivo auto`). Allí no se instalan graphmuse ni pyg-lib (no hay ruedas para Windows):
basta con copiar `data/cache/` y entrenar con torch (cu128) + torch_geometric + numpy en un
`python -m venv` normal (el venv de `uv` falla por SSH: el Python que gestiona uv está detrás de
un *junction* que Windows no deja recorrer). Una época del modelo de 256 tarda ~4,7 s frente a
~45 s en la CPU de esta máquina; con la GPU, 128 y 256 tardan casi lo mismo. Los modelos se
cargan siempre en CPU (`map_location`), así que se evalúan aquí.

**Mac Studio M4 Max (48 GB)**, disponible por SSH: opción futura para modelos más grandes. Torch
tiene backend MPS en Apple Silicon, pero no todas las operaciones de PyG lo soportan y las
extensiones nativas (torch-scatter/sparse, pyg-lib) puede que haya que compilarlas allí (no se
ha probado). Con modelos de este tamaño la CPU de esta máquina basta (una época ≈ 1,2 s).

## Decisiones que conviene revisar

- **Vigencia de 4 compases** de un acorde escrito (si no hay otro antes). En test deja sin
  etiqueta un 4 % más de notas (de 8.5 % a 12.5 %); está en `etiquetas.VIGENCIA_COMPASES`.
- **Una nota larga lleva el acorde de su ataque**, aunque cambie el acorde mientras suena.
- **Las 8 originales entran en train.** Son pocas notas, pero tienen las correcciones a mano.
- **Duplicados por título**: se compara `song_name` normalizado; títulos genéricos o mal
  codificados podrían colarse. PDMX.csv trae `subset:deduplicated` si se quiere afinar.
- **Selección de época por raíz+calidad en dev**: dev tiene 20 partituras; la "mejor época" tiene
  ruido. Con varias semillas se ve cuánto.

## Referencias

- Karystinaios, E. & Widmer, G. (2024). *GraphMuse: A Library for Symbolic Music Graph
  Processing*. ISMIR. <https://github.com/manoskary/graphmuse>
- Karystinaios, E. & Widmer, G. (2023). *Roman Numeral Analysis with Graph Neural Networks:
  Onset-wise Predictions from Note-wise Features* (ChordGNN). ISMIR. arXiv:2307.03544
- Long, P. et al. (2024). *PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music
  Processing*. arXiv:2409.10831
- Yao, M., Chen, K., Dubnov, S. & Berg-Kirkpatrick, T. (2026). *BACHI* (POP909-CL). ICASSP.
  arXiv:2510.06528
- Wang, Z., Chen, K. et al. (2020). *POP909: A Pop-song Dataset for Music Arrangement Generation*.
  ISMIR.
- Pardo, B. & Birmingham, W. (2002). *Algorithms for Chordal Analysis* — el método de plantillas
  del heurístico.
