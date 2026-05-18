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
└── scripts/
    └── run_pipeline.py             ← orquestador end-to-end
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

## Cómo correr el pipeline v2 (scraping opcional + export corpus legible)

```bash
python scripts/run_pipeline_v2.py
```
Activar scraping web (opcional)
```bash
python scripts/run_pipeline_v2.py --enable-web
```
Exportar corpus legible Bribri:Español
```bash
python scripts/run_pipeline_v2.py --print-corpus
```
Activar scraping + exportar corpus legible + limitar pares
```bash
python scripts/run_pipeline_v2.py --enable-web --print-corpus --print-corpus-max 300
```
Parámetros disponibles en v2
* ```bash --enable-web: habilita extracción web.```
* ```bash --print-corpus: exporta un TXT legible con formato BRI: / ES:.```
* ```bash --print-corpus-path: ruta de salida del TXT (default: data/processed/corpus_v0_bri_es.txt).```
* ```bash --print-corpus-max: máximo de pares a exportar (0 = todos).```


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

## Notas reproducibilidad

- `source_hashes.json` registra el SHA-256 de cada PDF; cualquier
  re-descarga debe coincidir para que `corpus_v0` sea reproducible.
- Toda la lógica de filtrado tiene umbrales en `src/voces_corpus/filters.py`
  como constantes claramente identificables.
- El reporte completo de la última corrida queda en
  `data/processed/pipeline_report.txt`.
