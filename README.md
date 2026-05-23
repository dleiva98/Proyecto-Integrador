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
└── data/splits/                    ← (generado) train.jsonl, val.jsonl, test.jsonl
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

## Notas reproducibilidad

- `source_hashes.json` registra el SHA-256 de cada PDF; cualquier
  re-descarga debe coincidir para que `corpus_v0` sea reproducible.
- Toda la lógica de filtrado tiene umbrales en `src/voces_corpus/filters.py`
  como constantes claramente identificables.
- El reporte completo de la última corrida queda en
  `data/processed/pipeline_report.txt`.
