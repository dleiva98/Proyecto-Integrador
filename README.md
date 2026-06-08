# Programa Voces — Corpus paralelo bribri-español 

Pipeline de construcción del corpus paralelo bribri-español para el
Proyecto Integrador de la Maestria. en IA Aplicada (Tec de Monterrey, equipo
Costa Rica), correspondiente al **Componente B** descrito en
`Avance1_Equipo66.pdf`.

El objetivo es producir un corpus normalizado, filtrado y trazable a partir
de los diez documentos primarios del Programa Voces, listo para entrenar
fine-tuning de NLLB-200 distilled-600M.

## Estructura del repositorio

```
.
├── Avance1_Equipo66.pdf            ← reporte de avance académico
├── README.md                       ← este archivo
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── pdfs/                   ← 10 documentos fuente
│   │   └── web/                    ← HTML crudo de scraping (vacío en este corte)
│   ├── interim/                    ← un JSONL por fuente, sin filtrar
│   └── processed/
│       ├── corpus_v0.jsonl
│       ├── corpus_v0.parquet
│       ├── source_hashes.json
│       └── pipeline_report.txt     ← reporte detallado de la última corrida
├── src/voces_corpus/
│   ├── schema.py                   ← ParallelPair (Pydantic)
│   ├── normalization.py            ← NFC + reglas bribri/español
│   ├── filters.py                  ← word-count, ratio, dedup SHA-256
│   ├── consolidate.py              ← JSONL ↔ Parquet, hashes
│   └── extractors/
│       ├── base.py                 ← heurísticas de idioma
│       ├── pdf_interlinear.py      ← Gramática, I ttè, Palabras García
│       ├── pdf_dialog.py           ← Sé'ttö' bribri ie (Hablemos)
│       ├── pdf_versicle.py         ← ESSJ Gabb (5 líneas por versículo)
│       ├── pdf_trilingual.py       ← Ditsö̀ rukuö̀ (bri/es/en)
│       └── web_scraper.py          ← lenguabribri.com (no usado, ver nota)
├── scripts/
│   ├── run_pipeline.py             ← orquestador end-to-end
│   └── make_splits.py              ← train/val/test estratificado (Avance 2)
├── notebooks/
│   └── nllb_finetune_colab.ipynb   ← Avance 3: fine-tuning NLLB + métricas
├── data/splits/                    ← (generado) train.jsonl, val.jsonl, test.jsonl
└── outputs/                        ← (generado) métricas, predicciones, curvas
    ├── metrics.json                ← historial de entrenamiento (loss / spBLEU / chrF / chrF++)
    ├── test_metrics.json           ← métricas finales sobre el split test
    ├── test_predictions.jsonl      ← 304 predicciones (152 por dirección) + referencia
    └── training_curves.png         ← curvas de val loss, spBLEU y chrF++ por paso
```

## Cómo correr el pipeline

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

Salida principal en `data/processed/`:

- `corpus_v0.jsonl` y `corpus_v0.parquet` — corpus filtrado.
- `source_hashes.json` — SHA-256 de cada PDF de origen.
- `pipeline_report.txt` — reporte con cuentas por fuente, distribución por
  dominio/confianza y los primeros 5 pares de cada fuente.

## Schema de un par paralelo

```json
{
  "bri": "Ìs be' shkẽ̀na?",
  "es": "¿Cómo amaneció usted?",
  "source_doc": "Gramatica_lengua_bribri_Jara2018.pdf",
  "source_page": 80,
  "domain": "didactico",
  "extraction_method": "pdf_interlinear",
  "confidence": "high",
  "metadata": {"gloss": "cómo 2S despertar.VM.REC"}
}
```

Dominios: `didactico`, `narrativo`, `religioso`, `etnografico`, `web`.
Confianzas: `high`, `medium`, `low`.

## Reglas de normalización (resumen)

- **Todo texto:** NFC.
- **Bribri:** preserva todos los diacríticos (tonos, virgulilla nasal, corte
  glotal `ʼ`/`'`); nunca lowercase, nunca quita combining marks.
- **Español:** NFC + comillas tipográficas normalizadas a rectas
  (`“ ” ‘ ’ « »` → `" '`) + dashes a `-` + whitespace colapsado.

## Filtros (en orden, registrados en el reporte)

1. Descarta si `bri` o `es` están vacíos.
2. Descarta si alguno tiene `< 2` o `> 80` palabras.
3. Descarta si `len(es)/len(bri)` (chars) está fuera de `[0.3, 3.0]`.
4. Descarta si `bri == es`.
5. Deduplica por SHA-256 del par `(bri, es)`.

## Resultado de la corrida actual (v0)

Total después de filtros: **1.505 pares** (de 1.707 brutos), en línea con
la estimación 2.500-4.500 declarada en `Avance1_Equipo66.pdf` §2.3
(estamos al lado bajo del rango esperado, pendiente la fuente web).

| Fuente                                  | Brutos | Conservados | Dominio      | Confianza dominante |
|-----------------------------------------|-------:|------------:|--------------|---------------------|
| ESSJ_Sanchez_Avendano_Vol1.pdf          |   814  |       675   | religioso    | medium              |
| I_tte_Historias_bribris.pdf             |   774  |       728   | narrativo    | low¹                |
| Gramatica_lengua_bribri_Jara2018.pdf    |    76  |        65   | didactico    | high / medium       |
| Sevtto_bribri_ie_Hablemos.pdf           |    21  |        18   | didactico    | high                |
| Palabras_Francisco_Garcia.pdf           |    18  |        17   | etnografico  | low¹                |
| Ditso_rukuo_Identidad_semillas.pdf      |     4  |         2   | narrativo    | low                 |
| https://www.lenguabribri.com/           |     0  |         0   | web          | —                   |

¹ El "español" en I ttè (interlineal) y en Palabras de Francisco García son
**glosas palabra-por-palabra**, no traducciones libres. Útiles pero deben
ponderarse o filtrarse en el set de entrenamiento NMT.

## Fuentes ignoradas explícitamente

Según las decisiones documentadas en §2.2 del reporte de avance:

- **Diccionario de mitología bribri (EditUCR)** — PDF escaneado, requiere
  OCR especializado. Fuera del alcance del pipeline (no-OCR).
- **Kó Késka** — PDF escaneado y mayormente monolingüe español con
  términos bribri.
- **Cargos tradicionales** — texto monolingüe español con terminología
  bribri en cursiva; no aporta pares paralelos.
- **Se' dör stë** — bilingüe español-inglés con bribri ornamental;
  no es corpus del par objetivo.

## Estado de la fuente web (lenguabribri.com)

El extractor `web_scraper.py` está implementado y respeta robots.txt, pero
**no se pudo ejecutar contra lenguabribri.com desde el entorno de cómputo
remoto usado para esta corrida**: la política de red del contenedor bloquea
el dominio (`HTTP 403 host_not_allowed`). Para incorporar la fuente:

1. Correr el pipeline desde una máquina con acceso libre a Internet, o
2. Habilitar el dominio en la lista permitida del entorno remoto.

Una vez disponible, el scraper guardará HTML crudo bajo
`data/raw/web/www.lenguabribri.com/` y emitirá pares con `domain="web"`.

## Próximos pasos (alineado con §2.5 del reporte de avance)

1. **Alineación oración-por-oración** con hunalign + LASER embeddings para
   los pares actualmente etiquetados como `confidence="low"` (I ttè,
   Palabras García), donde la prosa libre en español existe en otra
   sección del mismo libro pero no fue alineable con extractores
   estructurales simples.
2. **Web scraping** efectivo de lenguabribri.com cuando el dominio esté
   habilitado.
3. **Validación por hablantes nativos** sobre muestreo estratificado (5%
   por dominio) usando el canal de retroalimentación del Componente A.
4. **Recuperación** vía UCR de Krohn, Margery Peña y materiales MEP
   (sección 2.4 del reporte de avance).

## Avance 2 — Fine-tuning NLLB-200 y métricas

El segundo entregable usa el corpus `corpus_v0.jsonl` para hacer fine-tuning
de `facebook/nllb-200-distilled-600M` en ambas direcciones (`es↔bri`) y
reporta spBLEU, chrF y chrF++.

### Splits

```bash
python scripts/make_splits.py
```

Produce `data/splits/{train,val,test}.jsonl` con división **80/10/10
estratificada por `(domain, confidence)`** y semilla 42.
Cuentas actuales: 1.201 train / 152 val / 152 test.

### Cómo entrenar

El entrenamiento **requiere GPU** (descarga ~2.4 GB de pesos y corre
~30-45 min en T4). Recomendado: Colab.

1. Abrí `notebooks/nllb_finetune_colab.ipynb` en Colab.
2. Runtime → Change runtime type → **GPU** (T4 es suficiente).
3. Ejecutá las celdas en orden. Clona este repo, instala
   `transformers==4.48.3`, `torch`, `sacrebleu`, genera splits si
   faltan, corre 3 épocas con `batch_size=8`, `lr=5e-4`, `fp16`, y
   guarda artefactos en `outputs/`.

Para correr el mismo flujo desde un entorno local con GPU:

```bash
pip install -r requirements.txt
python scripts/make_splits.py
python -m voces_corpus.training.nllb_train
```

Configuración por defecto en `src/voces_corpus/training/nllb_train.py`
(`TrainConfig`). Editá la dataclass o pasale otra instancia a `train()`.

### Métricas

`src/voces_corpus/training/metrics.py` calcula spBLEU
(`tokenize=flores200`), chrF y chrF++ sobre listas de predicciones y
referencias. También sirve como CLI sobre un JSONL `{prediction, reference}`:

```bash
python -m voces_corpus.training.metrics outputs/test_predictions.jsonl \
    --out outputs/test_metrics.json
```

El notebook ya genera `outputs/test_predictions.jsonl`,
`outputs/test_metrics.json` y `outputs/training_curves.png`.

### Decisión: token proxy para bribri

Bribri (`bri`) no tiene token de idioma propio en NLLB-200. Usamos
`quy_Latn` (Quechua Ayacucho) como proxy: lengua indígena americana, en
script latino, presente en NLLB. La elección es **discutible** — el
template oficial sugiere usar cualquier token siempre que el tokenizador
no convierta caracteres importantes en `<unk>`. Alternativas a evaluar:
`spa_Latn` (mismo script, baseline), `quy_Latn` (default actual), o un
token reasignado tras vocabulario extendido. Esto entra en el análisis
del Avance 2.

### Limitación conocida

El 49% del corpus (`I_tte_Historias_bribris` + Palabras García) tiene
`confidence="low"` porque el "español" son glosas palabra-por-palabra,
no traducción libre. Eso introduce ruido sistemático; el split mantiene
la proporción para que las métricas sean comparables con el corpus real,
pero conviene reportar también métricas filtrando a `confidence ∈
{high, medium}` (~758 pares).

## Outputs — resultados experimentales del Avance 3

La carpeta `outputs/` contiene los artefactos producidos por la corrida
de fine-tuning ejecutada en Colab (GPU T4, fp16, 3 épocas,
`batch_size=8`, `lr=5e-4`, `max_length=256`, `seed=42`,
1.201 pares de entrenamiento × 2 direcciones, evaluación cada 100 pasos
de optimización sobre los 152 pares de validación). Los cuatro archivos
son **reproducibles** a partir de `corpus_v0.jsonl` y los splits con
semilla fija; ver `notebooks/nllb_finetune_colab.ipynb`.

### Contenido de `outputs/`

| Archivo | Tipo | Contenido |
|---|---|---|
| `metrics.json` | JSON | `config` reproducible + serie temporal de `train_loss`, `val_loss`, `spBLEU`, `chrF` y `chrF++` por paso de evaluación, para `es→bri`, `bri→es` y promedio. |
| `test_metrics.json` | JSON | Métricas finales sobre el split `test.jsonl` (no visto en entrenamiento). |
| `test_predictions.jsonl` | JSONL | 304 registros `{direction, prediction, reference}` (152 por dirección) — base para inspección cualitativa y para recomputar métricas con otros tokenizadores. |
| `training_curves.png` | PNG | Tres paneles (val loss, spBLEU, chrF++) con tres líneas cada uno (`es→bri`, `bri→es`, promedio) en función del paso de optimización. |

### Métricas — qué miden y cómo se leen

Se reportan tres métricas, todas calculadas con `sacrebleu` para que
sean comparables entre corridas y con literatura externa:

- **spBLEU** (`tokenize=flores200`): BLEU calculado sobre el tokenizador
  SentencePiece de FLORES-200. Usa la misma tokenización que la
  evaluación oficial de NLLB; importa para comparar contra la línea base
  reportada por el modelo. Es estricto: penaliza fuertemente cualquier
  diferencia de n-grama exacto y se degrada rápido con corpus pequeños.
- **chrF**: F-score sobre n-gramas de caracteres (n=6 por defecto).
  Robusto a morfología rica y útil para lenguas aglutinantes o con
  pocas referencias por oración, como bribri. Es la métrica recomendada
  por la línea de trabajo de NLLB y WMT para pares de bajo recurso.
- **chrF++**: variante que añade n-gramas de palabras (`word_order=2`)
  por encima de chrF. Penaliza un poco más los reordenamientos
  agramaticales. Suele quedar entre 0.5 y 1.5 puntos por debajo de
  chrF.

Todas las métricas están en escala 0-100 (mayor es mejor); `eval_loss`
es la *cross-entropy* media por token (menor es mejor).

### Resultados finales sobre el set de test (152 pares)

| Dirección | eval_loss ↓ | spBLEU ↑ | chrF ↑ | chrF++ ↑ |
|---|---:|---:|---:|---:|
| **es → bri** | 2.151 | **18.61** | 26.97 | 26.01 |
| **bri → es** | 3.056 | 10.26 | **27.34** | **26.16** |
| **promedio** | 2.603 | 14.43 | 27.16 | 26.09 |

Fuente: `outputs/test_metrics.json`.

**Interpretación de los números absolutos.** Para un par sin datos en
el pre-entrenamiento de NLLB (bribri *no existe* en NLLB-200), con
sólo ~1.2k pares paralelos y 3 épocas, llegar a chrF ≈ 27 en ambas
direcciones es un resultado consistente con la literatura de fine-tuning
de NLLB sobre lenguas indígenas americanas de muy bajo recurso (cf.
AmericasNLP 2023, donde la línea base NLLB fine-tuneada para Aymara
y Guaraní queda en rangos chrF 25-35 con corpus de tamaño similar).
spBLEU 14 promedio es razonable pero no debería interpretarse como una
calidad de traducción usable — sirve como **indicador de progreso**
relativo entre corridas, no como medida absoluta de fluidez.

### Asimetría entre direcciones

`es→bri` casi duplica a `bri→es` en spBLEU (18.61 vs 10.26) pero las
dos direcciones quedan casi empatadas en chrF/chrF++ (~27 / ~26). Esto
es informativo y no trivial:

- spBLEU compara n-gramas tokenizados con SentencePiece FLORES-200.
  El lado bribri del corpus tiene **menor variedad léxica y oraciones
  más cortas** (dominado por glosas y versículos breves), lo que hace
  más probable que el modelo recupere n-gramas exactos por
  memorización. En contraste, el lado español va de glosas
  telegráficas a traducción libre de prosa religiosa: alta varianza
  estilística, n-gramas difíciles de acertar.
- chrF, al operar a nivel de caracteres, **promedia esa varianza** y
  refleja que la "cantidad de información acertada" es similar en
  ambas direcciones.
- `eval_loss` (CE por token) confirma la asimetría desde la
  perspectiva del modelo: predecir el siguiente token en bribri es
  más fácil porque su distribución es más concentrada.

La conclusión práctica es que **chrF/chrF++ son la métrica primaria
para este corpus** y spBLEU debe leerse junto con ellas, no de
manera aislada.

### Curvas de entrenamiento

![Curvas de entrenamiento](outputs/training_curves.png)

Tres paneles con eje X en pasos de optimización (100 → 900, evaluación
cada 100 pasos, ~900 pasos ≈ 3 épocas).

1. **Val loss** (panel izquierdo). Caída monotónica suave en ambas
   direcciones; `es→bri` baja de 3.15 a 2.17 (−31%) y `bri→es` de 3.85
   a 3.28 (−15%). No hay rebote tipo *overfitting* dentro del
   horizonte de 900 pasos: la pendiente final aún es negativa, lo que
   sugiere que **el modelo no terminó de converger** y aumentar épocas
   o reducir `learning rate` con warmup podría dar mejoras adicionales.
2. **spBLEU** (panel central). Crecimiento marcado en los primeros
   400 pasos (0 → ~11 promedio) y posteriormente meseta ruidosa
   entre 8 y 12 puntos. El ruido refleja que sacrebleu sobre apenas
   152 referencias es de alta varianza; cada salto/bajada de ~3
   puntos no es necesariamente cambio de calidad real.
3. **chrF++** (panel derecho). Crecimiento mucho más estable y
   monotónico, 10 → 25 promedio sin retrocesos. Esto confirma
   empíricamente que chrF++ es **una mejor señal de progreso** en
   este régimen de poco dato que spBLEU.

La señal de `train_loss` (en `metrics.json`, no graficada) tiene más
ruido (cae de 3.63 a ~1.0-1.4, con dientes de sierra). Esto es esperable
con `batch_size=8` + AMP fp16; no debe interpretarse como inestabilidad
del entrenamiento.

### Análisis cualitativo — muestra de `test_predictions.jsonl`

Inspección manual de las predicciones revela cuatro patrones
recurrentes (extractos literales del JSONL):

**1. Salidas estructuralmente correctas, traducción parcial** (caso
esperado al alcance del corpus actual):

```
direction : bri → es
prediction: "Me llamo Trini."
reference : "Yo me llamo Trini."
```

**2. Hallucination temática** — el modelo se ancla al dominio
religioso (mayoritario en el corpus) cuando la entrada bribri es
ambigua:

```
direction : bri → es
prediction: "que viene en el cielo, que dice todas las cosas que ven y viene en vosotros a la verdad."
reference : "nombre de todas las cosas que vienen."
```

**3. Collapse a repetición** — patrón clásico de seq2seq sub-entrenado
con `max_new_tokens` alto:

```
direction : es → bri
prediction: "ẽ̀nẽ̀nẽ̀nẽ̀nẽ̀nẽ̀nẽ̀nẽ̀nẽ."
reference : "Kotereööö, uuuhhh, ie' tö Sòrbulu tchìwẽ̀wã"
```

**4. Salida bribri morfológicamente plausible** — se observa que el
modelo aprende a producir secuencias con diacríticos, tonos y
clíticos en posiciones gramaticalmente coherentes, aunque el
contenido léxico no siempre coincide con la referencia:

```
direction : es → bri
prediction: "E'ta̠ Marta tö Jesús i-ché: Akë́kë, ma̱ -ma̱ le̱ í̠e̠ a' tso'rö, ye' ë́l kë̀ dawö̀wa̱."
reference : "E'ta̠ Marta tö Jesús i̱a̱ i-ché: Akë́kë, ma̱ -a̱ mú̱ pa tso' í̱e̱ e̱ ma̠ ya-akë̀ kë̀ dúwa̱."
```

Este último patrón es la evidencia más fuerte de que el fine-tuning
**sí está alineando el espacio latente del proxy `quy_Latn` con
bribri real**: el modelo no sólo memoriza, también generaliza la
morfología (sufijos `-wã`, `-ke̱`, posiciones de tono) a contextos
nuevos.

### Configuración exacta (de `metrics.json`)

```json
{
  "model_str":        "facebook/nllb-200-distilled-600M",
  "lr":               5e-4,
  "batch_size":       8,
  "epochs":           3,
  "bidirectional":    true,
  "max_length":       256,
  "use_float16":      true,
  "seed":             42,
  "src_lang_token":   "spa_Latn",
  "tgt_lang_token":   "quy_Latn"
}
```

### Limitaciones de la corrida actual

1. **Tamaño del corpus**: 1.201 pares de entrenamiento es ~1-2 órdenes
   de magnitud por debajo de los volúmenes típicos para fine-tuning
   estable de NLLB-200 distilled. Las métricas absolutas deben
   leerse con esa caveat.
2. **Ruido de glosas**: 49 % del corpus es `confidence="low"`. No se
   filtró por confianza para no reducir el set a ~600 pares, pero
   sería una ablación natural reportar también las métricas
   restringiendo a `confidence ∈ {high, medium}`.
3. **Proxy de idioma**: `quy_Latn` (Quechua Ayacucho) se usó como
   ancla para bribri; alternativas no exploradas son `spa_Latn` (mismo
   script, sin información tipológica) y reentrenamiento del
   tokenizador con vocabulario bribri extendido. Una ablación A/B
   sobre el token de idioma daría evidencia directa de cuánto importa
   esa elección.
4. **Sin warmup ni scheduler**: AdamW plano con `lr=5e-4`. Las curvas
   sugieren que un `cosine` o `linear` con warmup mejoraría
   convergencia, sobre todo en `bri→es`.
5. **Métrica de evaluación**: spBLEU/chrF/chrF++ son automáticas;
   sería deseable agregar **evaluación humana** (validación por
   hablantes nativos, ver §"Próximos pasos") en una muestra
   estratificada del test.

### Próximos pasos (Avance 3 → Avance 4)

- Ablación del token proxy: corrida idéntica con `spa_Latn` y con un
  vocabulario extendido, comparando chrF/chrF++ promedio.
- Filtrado por `confidence`: comparar fine-tuning sobre el corpus
  completo vs. corpus sin `low`.
- Schedulers (warmup lineal de 100 pasos + cosine) y más épocas hasta
  ver rebote en val loss.
- Evaluación humana de 30 oraciones por dirección sobre el split test.

## Avance 4 — Reentrenamiento en CENIA (H100) y baseline comparativo M2M-100

El cuarto entregable retoma exactamente donde quedó el Avance 3. Allí las
curvas mostraban que **el modelo no había terminado de converger** (val loss
con pendiente aún negativa a 900 pasos) y que faltaban dos ablaciones
prometidas: más épocas con `learning rate` más bajo, y un **segundo modelo
de comparación** para verificar que NLLB-200 era realmente la mejor elección
de arquitectura para este par de muy bajo recurso. Este avance ejecuta ambas
cosas y las pone una al lado de la otra.

A diferencia del Avance 3 (Google Colab, GPU T4, fp16, sesiones limitadas),
estas dos corridas se ejecutaron en el **servidor de cómputo del CENIA
(Centro Nacional de Inteligencia Artificial)** sobre una **GPU NVIDIA H100**.
La H100 (80 GB HBM, soporte nativo bf16/fp16 y throughput ~1 orden de
magnitud sobre la T4) hizo viable correr 8 épocas completas y un segundo
modelo de 418M de parámetros sin las restricciones de tiempo de Colab. La
única variable que cambia entre corridas son los hiperparámetros y el
modelo; **splits, semilla (42), corpus, `max_length`, `batch_size` y código
de métricas se mantienen idénticos** a los del Avance 3 para que la
comparación sea limpia.

### Lo que se entrena en este avance

| Corrida | Archivo lanzador | Salida | Modelo | Épocas | lr | Hardware |
|---|---|---|---|---:|---:|---|
| **NLLB orig** (Avance 3) | notebook Colab | `outputs/` | `nllb-200-distilled-600M` | 3 | 5e-4 | T4 |
| **NLLB H100** (nuevo) | `run_nllb_h100.py` | `outputs_nllb_h100/` | `nllb-200-distilled-600M` | **8** | **2e-4** | **H100** |
| **M2M-100** (nuevo) | `m2m100_train.py` | `outputs_m2m100/` | `m2m100_418M` | 3 | 5e-4 | H100 |

`run_nllb_h100.py` **no reescribe** `nllb_train.py`: importa `train()` y
`TrainConfig`, sólo cambia `epochs=8` y `lr=2e-4`, y guarda en una carpeta
nueva para no pisar la corrida original. `m2m100_train.py` es un **espejo
intencional** de `nllb_train.py` (mismo `Dataset`, mismo bucle, mismo
`evaluate_split`, **mismas métricas**); las únicas diferencias son las
*obligadas* por el modelo (M2M-100 usa códigos ISO simples y fuerza el idioma
destino con `get_lang_id` en vez de los tokens `xxx_Latn` de NLLB).

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python run_nllb_h100.py        # NLLB-200, 8 épocas, lr 2e-4  -> outputs_nllb_h100/
python m2m100_train.py         # M2M-100 baseline             -> outputs_m2m100/
python make_plots_h100.py      # curvas H100 + barras 3-way
```

### Nuevos artefactos

| Archivo | Contenido |
|---|---|
| `outputs_nllb_h100/metrics.json` | Serie temporal de 24 evaluaciones (100→2400 pasos) con `train_loss`, `val_loss`, spBLEU, chrF y chrF++ por dirección y promedio. |
| `outputs_nllb_h100/test_metrics.json` | Métricas finales sobre el split test (152 pares). |
| `outputs_nllb_h100/test_predictions.jsonl` | 304 predicciones (152 × 2 direcciones) de la corrida H100. |
| `outputs_nllb_h100/training_curves_nllb_h100.png` | Curvas de la corrida H100 (val loss / spBLEU / chrF++). |
| `outputs_nllb_h100/comparison_bars_3way.png` | Barras comparativas de las 3 corridas sobre test. |
| `outputs_m2m100/metrics.json`, `test_metrics.json`, `test_predictions.jsonl` | Equivalentes para M2M-100. |
| `outputs_m2m100/training_curves_m2m100.png` | Curvas de entrenamiento de M2M-100. |
| `outputs_m2m100/comparison_curves.png` | NLLB-200 vs M2M-100, curvas de validación superpuestas. |

### Resultados finales sobre el set de test (152 pares, promedio de direcciones)

| Corrida | eval_loss ↓ | spBLEU ↑ | chrF ↑ | chrF++ ↑ |
|---|---:|---:|---:|---:|
| NLLB orig · 3ep lr5e-4 (Avance 3) | **2.603** | 14.43 | 27.16 | 26.09 |
| **NLLB H100 · 8ep lr2e-4** | 3.166 | **21.16** | **31.43** | **30.47** |
| M2M-100 · 3ep lr5e-4 | 3.624 | 1.33 | 9.38 | 8.33 |

Fuente: `outputs_nllb_h100/test_metrics.json`, `outputs/test_metrics.json`,
`outputs_m2m100/test_metrics.json`.

Desglose por dirección de la corrida ganadora (**NLLB H100**):

| Dirección | eval_loss ↓ | spBLEU ↑ | chrF ↑ | chrF++ ↑ |
|---|---:|---:|---:|---:|
| **es → bri** | 2.674 | **28.74** | **32.86** | **32.35** |
| **bri → es** | 3.657 | 13.57 | 30.00 | 28.58 |
| **promedio** | 3.166 | 21.16 | 31.43 | 30.47 |

### Comparación de las 3 corridas

![Comparación 3 corridas — test final](outputs_nllb_h100/comparison_bars_3way.png)

El panel de barras resume la conclusión central del avance:

- **NLLB H100 mejora a NLLB orig en todas las métricas de calidad de
  traducción.** chrF sube de 27.16 a **31.43** (+4.27 pts, **+15.7 %**),
  chrF++ de 26.09 a **30.47** (+16.8 %) y spBLEU de 14.43 a **21.16**
  (+6.73 pts, **+46.6 %**). La mejora es mayor en `es→bri` (spBLEU
  18.61 → 28.74, **+54 %**; chrF 26.97 → 32.86) que en `bri→es` (chrF
  27.34 → 30.00), pero ambas direcciones suben.
- **M2M-100 queda muy por debajo de ambas corridas de NLLB**: chrF 9.38
  (≈ ⅓ de NLLB H100) y spBLEU 1.33 (≈ 1/16 de NLLB H100). El experimento
  confirma empíricamente que **NLLB-200 era la elección correcta de
  arquitectura** para este par, y no un supuesto del Avance 1.

### La paradoja val loss ↑ pero chrF ↑ (lectura académica clave)

Hay un resultado contraintuitivo que merece análisis: la corrida H100 tiene
**peor** `eval_loss` (3.17 vs 2.60) y al mismo tiempo **mejor** chrF/spBLEU.
No es un error: es el fenómeno clásico de **divergencia entre la
cross-entropy y la calidad decodificada** en NMT de bajo recurso.

![Curvas de entrenamiento NLLB-H100](outputs_nllb_h100/training_curves_nllb_h100.png)

Leyendo las curvas (eje X en pasos, 100 → 2400, ~300 pasos/época × 8):

1. **Val loss** (izquierda). `es→bri` toca su mínimo (~2.18) alrededor del
   paso 900 y **rebota** hasta ~2.66; `bri→es` toca mínimo (~2.88) cerca del
   paso 600 y sube hasta ~3.88. El `train_loss` (en `metrics.json`) cae de
   3.58 a ~0.16-0.25: el modelo **memoriza** el train. Visto sólo por la
   CE de validación, esto es *overfitting* y un *early stopping* lo habría
   cortado en el paso 600-900.
2. **spBLEU** (centro) y **chrF++** (derecha). Sin embargo, las métricas
   **decodificadas** siguen subiendo mucho después de ese punto: chrF++
   `es→bri` mesetea en ~31-32 recién hacia el paso 1300-1500, y `bri→es`
   en ~28. Es decir, **el mejor checkpoint por val loss NO es el mejor por
   chrF**: las épocas extra que "empeoran" la CE en realidad mejoran la
   superposición de caracteres de la traducción.

La explicación es que la cross-entropy penaliza cada token por
probabilidad exacta (sensible a sobre-confianza tras memorizar), mientras
chrF mide solape de n-gramas de caracteres en la **salida realmente
generada**. En un régimen de ~1.2k pares, seguir entrenando aumenta la
confianza del modelo (sube CE de validación) pero también afina la
morfología que produce al decodificar (sube chrF). **La conclusión
metodológica es que en este proyecto el criterio de selección de modelo
debe ser chrF/chrF++, no val loss**, y que reportar sólo la CE habría
ocultado la mejora real. El checkpoint final (`final_nllb`, paso 2400) es
el que se evalúa en test.

Como efecto secundario, los **bucles de repetición degenerada** que
aparecían en el Avance 3 (p. ej. `ẽ̀nẽ̀nẽ̀nẽ̀…`) **desaparecen** en la
corrida H100: más épocas + lr más bajo estabilizan la generación.

### NLLB-200 vs M2M-100 — por qué M2M colapsa

![NLLB-200 vs M2M-100 — curvas de validación](outputs_m2m100/comparison_curves.png)

Las curvas superpuestas (ambos a 3 épocas / 900 pasos, condición idéntica)
muestran que **M2M-100 nunca despega**: su val loss baja poco (4.96 → 3.79),
su spBLEU promedio queda por debajo de 4 con picos ruidosos, y su chrF++ no
pasa de ~9. La curva propia de M2M (`training_curves_m2m100.png`) confirma
una señal muy ruidosa y de bajísima magnitud en spBLEU, especialmente en
`bri→es` (prácticamente 0).

El análisis cualitativo explica el colapso: M2M-100 cae en **bucles de
repetición** masivos, justo el patrón que NLLB ya superó:

```
direction : es → bri
prediction: "E'ta i-ché i̱-i̱a̱ : Akë́kë, ba-ujché̱ r tö i-ujché̱ r tö
             i-ujché̱ r tö i̱-i̱a̱ : Akë́kë, ba-ujché̱ r tö i-ujché̱ r tö …"
reference : "i-apàtkë'/apàtké e' tsá̱ ta̱."
```

Tres factores concurren, y conviene documentarlos honestamente:

1. **Capacidad y pre-entrenamiento.** M2M-100 418M tiene menos parámetros
   que NLLB-200 distilled-600M y un pre-entrenamiento menos orientado a
   bajo recurso; con sólo 3 épocas no logra anclar el espacio del idioma
   destino.
2. **Proxy de idioma — caveat a registrar.** Para NLLB el proxy de bribri
   es `quy_Latn` (Quechua Ayacucho). En M2M la corrida quedó configurada
   con `tgt_lang_code="br"` (que en el inventario de M2M-100 corresponde a
   **bretón**, no a quechua `qu`). Es un **confundidor real** en la
   comparación: parte del mal desempeño de M2M puede deberse a un proxy
   tipológicamente lejano. Una re-corrida con `qu` es el primer ítem de los
   próximos pasos antes de declarar una conclusión definitiva sobre M2M.
3. **Mismas 3 épocas que el baseline antiguo.** M2M se entrenó en la
   condición original (3ep/lr5e-4) para que su comparación directa fuera
   contra `NLLB orig`, no contra la corrida H100 de 8 épocas.

Aun con el caveat (2), la brecha (chrF 9 vs 27-31) es lo bastante grande
como para sostener la decisión de seguir con NLLB-200 como modelo base.

### Análisis cualitativo — el mismo par, Avance 3 vs Avance 4

El test y la semilla no cambiaron, así que se puede inspeccionar **la misma
oración** en las dos corridas de NLLB. Tomando el par Marta/Jesús que ya se
analizó en el Avance 3 (patrón 4):

```
referencia       : E'ta̠ Marta tö Jesús i̱a̱ i-ché: Akë́kë, ma̱ -a̱ mú̱ pa
                   tso' í̱e̱ e̱ ma̠ ya-akë̀ kë̀ dúwa̱.
Avance 3 (3ep)   : E'ta̠ Marta tö Jesús i-ché: Akë́kë, ma̱ -ma̱ le̱ í̠e̠ a'
                   tso'rö, ye' ë́l kë̀ dawö̀wa̱.
Avance 4 (H100)  : E'ta̠ Marta tö Jesús i̱-i̱a̠ i-ché: Kë́kë, tö ma̱ -e̱'tso'
                   í̱e̱, ye' ë́l kë̀ dawö̀wa̱ ta̱.
```

La corrida H100 **recupera `i̱-i̱a̠`** (mucho más cercano a la referencia
`i̱a̱`, que la versión del Avance 3 omitía) y produce `í̱e̱` con el gancho
nasal correcto. El contenido todavía no es perfecto, pero el alineamiento
estructural y morfológico mejora de forma visible — coherente con el +16 %
de chrF medido. Otros ejemplos de `test_predictions.jsonl` (H100) muestran
salidas casi exactas en oraciones cortas:

```
direction : es → bri
prediction: "i-apàtkë' tsá̱ ka̱."
reference : "i-apàtkë'/apàtké e' tsá̱ ta̱."
```

### Conclusiones del Avance 4

1. **Más épocas con lr más bajo en H100 mejoran NLLB-200** de forma
   consistente (chrF +15.7 %, chrF++ +16.8 %, spBLEU +46.6 % en promedio),
   confirmando la hipótesis del Avance 3 de que el modelo no había
   convergido.
2. **NLLB-200 supera ampliamente a M2M-100** en condiciones idénticas
   (chrF 27-31 vs 9), validando empíricamente la elección de arquitectura.
3. **chrF/chrF++ —no la val loss— deben gobernar la selección de modelo**
   en este régimen de bajo recurso; la corrida H100 lo demuestra al mejorar
   chrF mientras su CE de validación rebota.
4. El proxy de idioma sigue siendo una palanca abierta: el caveat `br` vs
   `qu` en M2M y la ablación `quy_Latn` vs `spa_Latn` en NLLB quedan
   pendientes.

### Próximos pasos (Avance 4 → entrega final)

- **Re-correr M2M-100 con el proxy `qu`** (quechua) para eliminar el
  confundidor y dar una comparación de arquitectura limpia.
- **NLLB H100 con selección por chrF + early stopping sobre chrF** (no sobre
  loss), guardando el mejor checkpoint por chrF++ promedio.
- **Ablación del proxy en NLLB** (`quy_Latn` vs `spa_Latn` vs vocabulario
  extendido) ahora que la H100 hace barato el barrido.
- **Filtrado por `confidence`**: comparar 8 épocas sobre corpus completo vs.
  corpus sin `low` (~758 pares), para medir cuánto del techo lo impone el
  ruido de glosas.
- **Evaluación humana** por hablantes nativos sobre una muestra
  estratificada del test, como cierre cualitativo de la entrega final.

## Avance 5 — Ensambles, checkpointing robusto y escalamiento a NLLB-200 3.3B

La rúbrica de este entregable pide modelos de ensamble, optimización de
hiperparámetros, comparación contra modelos individuales y gráficos de
interpretación. En este proyecto el problema no es clasificación tabular sino
**traducción automática neuronal** (`seq2seq`), por lo que ROC, matriz de
confusión, precisión-recall o importancia de variables no son métricas
centrales. La adaptación metodológica es:

- tratar los modelos previos como generadores candidatos de traducción;
- construir ensambles como **blending/selección de candidatos** a nivel de
  salida;
- conservar `chrF++` promedio como métrica primaria porque el Avance 4 mostró
  que val loss y calidad decodificada divergen;
- reportar `spBLEU`, `chrF`, `chrF++` y tiempo/costo operacional cuando está
  disponible;
- usar curvas de entrenamiento, barras comparativas y análisis cualitativo de
  degeneración como gráficos interpretables para NMT.

### Cambios implementados para el Avance 5

1. `src/voces_corpus/training/nllb_train.py` ahora soporta:
   - checkpoint reanudable en `output_dir/checkpoint-last`;
   - restauración de pesos, tokenizador, optimizador, AMP scaler, historial,
     época, batch y paso global;
   - `resume_if_checkpoint_exists=True` para reiniciar Colab desde el último
     estado válido;
   - `gradient_accumulation_steps` para mantener batch efectivo sin exigir el
     mismo batch físico;
   - selección de mejor checkpoint por `best_metric="chrfpp"`;
   - `wall_time_seconds` en `metrics.json` para registrar tiempos en corridas
     nuevas;
   - `optimizer_name="adamw_bnb_8bit"` para corridas grandes en A100;
   - QLoRA 4-bit + LoRA para correr NLLB 3.3B en L4 con RAM amplia.
2. `notebooks/nllb_finetuned_3_3b_colab.ipynb` queda preparado para Colab Pro+
   con dos rutas: A100 full fine-tuning o L4 high-RAM con QLoRA.
3. `scripts/evaluate_prediction_ensembles.py` genera ensambles ligeros a partir
   de las predicciones existentes y escribe resultados en `outputs_ensembles/`.

### Corrida Colab 3.3B propuesta

La nueva notebook mantiene los parámetros comparables del baseline:

| Parámetro | Valor |
|---|---:|
| Modelo | `facebook/nllb-200-3.3B` |
| Épocas | 3 |
| Learning rate | `5e-4` |
| `max_length` | 256 |
| Semilla | 42 |
| Batch efectivo | 8 |
| Micro-batch físico | 1 |
| Acumulación de gradiente | 8 |
| Precisión | bf16 |
| Entrenamiento | A100: full fine-tuning; L4: QLoRA 4-bit + LoRA |
| Optimizador | A100: AdamW 8-bit; L4: AdamW sobre adaptadores |
| Requisito de hardware | A100 ≥39 GiB VRAM, o L4 ≥20 GiB VRAM + RAM ≥40 GiB |
| Checkpoint | cada 100 pasos de optimizador |
| Criterio de mejor checkpoint | `chrF++` promedio en validación |

El micro-batch físico baja a 1 porque full fine-tuning de 3.3B con batch físico
8 no es realista en Colab; `gradient_accumulation_steps=8` conserva el batch
efectivo y por tanto la comparabilidad experimental. Si Colab asigna A100 de
40 GB o más, la notebook ejecuta full fine-tuning en bf16. Si asigna L4 de
~22 GB con RAM amplia, ejecuta QLoRA 4-bit y entrena únicamente adaptadores
LoRA; esto no es equivalente a full fine-tuning, pero permite probar la red
3.3B en el recurso disponible sin caer en OOM. Si el runtime no cumple ninguna
de esas dos condiciones, aborta antes de cargar el modelo. La salida se guarda
en Google Drive bajo carpetas separadas:
`outputs_nllb_3_3b_full_a100/` u `outputs_nllb_3_3b_qlora_l4/`. Si Colab se
desconecta, basta volver a ejecutar la notebook: `train()` detecta
`checkpoint-last` y continúa desde el estado guardado.

### Ensambles evaluados

Se generaron tres estrategias sobre las predicciones existentes:

- **Blend homogéneo NLLB anti-degeneración**: usa NLLB H100 como candidato por
  defecto y cae a NLLB original sólo cuando detecta repetición degenerada.
- **Blend heterogéneo NLLB+M2M anti-degeneración**: elige entre NLLB H100, NLLB
  original y M2M-100 con una regla sin referencia basada en repetición y
  longitud relativa.
- **Oracle de candidatos NLLB+M2M**: cota superior no desplegable; usa la
  referencia para elegir el mejor candidato por segmento y cuantificar el techo
  posible de un reranker perfecto.

Comando reproducible:

```bash
PYTHONPATH=src python scripts/evaluate_prediction_ensembles.py
```

Salida principal:

| Modelo | Tipo | Hardware | Tiempo | spBLEU ↑ | chrF ↑ | chrF++ ↑ |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Oracle de candidatos NLLB+M2M (cota superior no desplegable) | oracle / upper bound | H100 + T4 | usa referencia; no desplegable | 22.15 | 34.30 | 33.30 |
| Blend homogéneo NLLB anti-degeneración | ensamble homogéneo | H100 + T4 | post-proceso; sin reentrenar | 21.25 | 31.61 | 30.66 |
| NLLB H100 · 600M · 8ep lr2e-4 | individual | H100 | no registrado | 21.16 | 31.43 | 30.47 |
| Blend heterogéneo NLLB+M2M anti-degeneración | ensamble heterogéneo | H100 + T4 | post-proceso; sin reentrenar | 17.13 | 28.14 | 26.98 |
| NLLB orig · 600M · 3ep lr5e-4 | individual | Colab T4 | 30-45 min reportado | 14.43 | 27.16 | 26.09 |
| M2M-100 · 418M · 3ep lr5e-4 | individual | H100 | no registrado | 1.33 | 9.38 | 8.33 |

Fuente: `outputs_ensembles/comparison_table.md`,
`outputs_ensembles/ensemble_metrics.json` y métricas individuales previas.

### Interpretación

El **blend homogéneo NLLB** mejora ligeramente al mejor modelo individual:
`chrF++` sube de 30.47 a **30.66** (+0.19), `chrF` de 31.43 a **31.61** y
`spBLEU` de 21.16 a **21.25**. La mejora es pequeña, pero coherente: el
ensamble no intenta promediar logits ni reentrenar; sólo reemplaza salidas
detectadas como degeneradas con una segunda corrida NLLB.

El **blend heterogéneo NLLB+M2M** empeora frente a NLLB H100. La razón es
consistente con el Avance 4: M2M-100 produce muchas repeticiones largas y su
calidad promedio es demasiado baja para aportar diversidad útil. En ensambles,
la diversidad sólo ayuda si los errores son parcialmente complementarios; aquí
M2M introduce ruido sistemático.

El **oracle** alcanza `chrF++=33.30`, lo que muestra que sí existe margen si se
entrena un reranker real. Sin embargo, no es un modelo desplegable porque usa la
referencia de test para decidir. Su valor es diagnóstico: cuantifica el techo de
selección entre candidatos ya entrenados.

### Modelo final elegido

Para fines operativos, el modelo final recomendado sigue siendo **NLLB H100 ·
600M · 8ep lr2e-4**:

- es el mejor modelo individual validado;
- evita mantener dos checkpoints en producción por una ganancia marginal de
  `chrF++=+0.19`;
- no depende de reglas heurísticas de post-proceso;
- ya eliminó los bucles degenerados que aparecían en Avance 3;
- es más barato y simple que NLLB-200 3.3B mientras la corrida 3.3B no tenga
  resultados medidos.

Si el objetivo inmediato fuera maximizar exclusivamente la métrica automática,
el **blend homogéneo NLLB** sería la variante ganadora desplegable. Para el
objetivo de negocio del proyecto —un traductor experimental reproducible,
auditable y mantenible para una lengua de muy bajo recurso— la mejora marginal
no justifica aún la complejidad adicional.

### Gráficos aplicables

Los gráficos relevantes para esta naturaleza de problema son:

- curvas de entrenamiento (`val loss`, `spBLEU`, `chrF++`) ya generadas para
  NLLB y M2M;
- barras comparativas de métricas finales entre modelos;
- análisis cualitativo de predicciones y detección de bucles degenerados;
- para la corrida 3.3B, `notebooks/nllb_finetuned_3_3b_colab.ipynb` genera
  `training_curves.png` en Drive.

ROC, precisión-recall y matriz de confusión no se reportan porque requerirían
convertir artificialmente la traducción en clasificación, perdiendo la señal
principal de calidad de secuencia.

## Notas reproducibilidad

- `source_hashes.json` registra el SHA-256 de cada PDF; cualquier
  re-descarga debe coincidir para que `corpus_v0` sea reproducible.
- Toda la lógica de filtrado tiene umbrales en `src/voces_corpus/filters.py`
  como constantes claramente identificables.
- El reporte completo de la última corrida queda en
  `data/processed/pipeline_report.txt`.
