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
│   └── nllb_finetune_colab.ipynb   ← Avance 2: fine-tuning NLLB + métricas
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

## Outputs — resultados experimentales del Avance 2

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

### Próximos pasos (Avance 2 → Avance 3)

- Ablación del token proxy: corrida idéntica con `spa_Latn` y con un
  vocabulario extendido, comparando chrF/chrF++ promedio.
- Filtrado por `confidence`: comparar fine-tuning sobre el corpus
  completo vs. corpus sin `low`.
- Schedulers (warmup lineal de 100 pasos + cosine) y más épocas hasta
  ver rebote en val loss.
- Evaluación humana de 30 oraciones por dirección sobre el split test.

## Notas reproducibilidad

- `source_hashes.json` registra el SHA-256 de cada PDF; cualquier
  re-descarga debe coincidir para que `corpus_v0` sea reproducible.
- Toda la lógica de filtrado tiene umbrales en `src/voces_corpus/filters.py`
  como constantes claramente identificables.
- El reporte completo de la última corrida queda en
  `data/processed/pipeline_report.txt`.
