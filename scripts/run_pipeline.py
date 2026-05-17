"""Orquestador end-to-end del pipeline de corpus bribri-español.

Uso:
    python scripts/run_pipeline.py

Recorre todas las fuentes (PDFs + web), aplica extractores específicos,
normaliza, filtra y consolida en data/processed/corpus_v0.{jsonl,parquet}
junto con data/processed/source_hashes.json y un reporte en stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import structlog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voces_corpus.consolidate import (  # noqa: E402
    read_jsonl,
    to_parquet,
    write_jsonl,
    write_source_hashes,
)
from voces_corpus.extractors import (  # noqa: E402
    pdf_dialog,
    pdf_interlinear,
    pdf_trilingual,
    pdf_versicle,
    web_scraper,
)
from voces_corpus.filters import apply_filters  # noqa: E402
from voces_corpus.schema import ParallelPair  # noqa: E402

log = structlog.get_logger()

PDF_DIR = ROOT / "data" / "raw" / "pdfs"
WEB_DIR = ROOT / "data" / "raw" / "web"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"


# Mapeo fuente → (archivo, extractor, dominio)
PDF_PLAN: list[tuple[str, str, str, str]] = [
    # (label, filename, extractor_id, dominio)
    ("gramatica_jara",   "Gramatica_lengua_bribri_Jara2018.pdf", "interlinear", "didactico"),
    ("sevtto_hablemos",  "Sevtto_bribri_ie_Hablemos.pdf",        "dialog",      "didactico"),
    ("i_tte",            "I_tte_Historias_bribris.pdf",          "interlinear", "narrativo"),
    ("ditso_rukuo",      "Ditso_rukuo_Identidad_semillas.pdf",   "trilingual",  "narrativo"),
    ("palabras_garcia",  "Palabras_Francisco_Garcia.pdf",        "interlinear", "etnografico"),
    ("essj_gabb",        "ESSJ_Sanchez_Avendano_Vol1.pdf",       "versicle",    "religioso"),
]

# Algunos extractores producen pares cuyo "español" es en realidad una glosa
# palabra-por-palabra (ej. I ttè) o donde el layout tabular del PDF rompe la
# alineación libre (Palabras de Francisco García). Bajamos la confianza
# a "low" para que el equipo NMT pueda filtrar o ponderar acorde.
CONFIDENCE_OVERRIDE: dict[str, str] = {
    "I_tte_Historias_bribris.pdf": "low",
    "Palabras_Francisco_Garcia.pdf": "low",
}

# PDFs escaneados/no paralelos — explícitamente ignorados (sin OCR):
SKIPPED_PDFS = [
    ("diccionario_mitologia", "Diccionario_mitologia_bribri_UCR.pdf",
     "PDF escaneado, requiere OCR — fuera de alcance del pipeline."),
    ("ko_keska",              "Ko_Keska_Lugar_del_tiempo.pdf",
     "PDF escaneado, mayormente monolingüe español con términos bribri."),
    ("cargos_tradicionales",  "Cargos_tradicionales_pueblo_bribri.pdf",
     "Monolingüe español con terminología bribri en cursiva; sin paralelización."),
    ("se_dor_ste",            "Se_dor_ste.pdf",
     "Bilingüe español-inglés con bribri ocasional; no es par bri-es."),
]

WEB_SOURCES = [
    "https://www.lenguabribri.com/",
]


def run_pdf_extractor(label: str, filename: str, extractor_id: str, domain: str) -> list[ParallelPair]:
    path = PDF_DIR / filename
    if not path.exists():
        log.error("pdf.missing", label=label, path=str(path))
        return []
    src_doc = path.name
    if extractor_id == "interlinear":
        return pdf_interlinear.extract(path, src_doc, domain)
    if extractor_id == "dialog":
        return pdf_dialog.extract(path, src_doc, domain)
    if extractor_id == "trilingual":
        return pdf_trilingual.extract(path, src_doc)
    if extractor_id == "versicle":
        return pdf_versicle.extract(path, src_doc)
    raise ValueError(f"Extractor desconocido: {extractor_id}")


class _Tee:
    """Duplica writes a varios destinos (stdout + archivo de reporte)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def main() -> int:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs: list[ParallelPair] = []
    failures: dict[str, str] = {}
    per_source_raw_counts: dict[str, int] = {}

    # --- PDFs ---
    for label, filename, extractor_id, domain in PDF_PLAN:
        log.info("extract.start", label=label, extractor=extractor_id)
        try:
            pairs = run_pdf_extractor(label, filename, extractor_id, domain)
        except Exception as exc:
            log.exception("extract.failed", label=label, error=str(exc))
            failures[label] = f"{type(exc).__name__}: {exc}"
            continue

        override = CONFIDENCE_OVERRIDE.get(filename)
        if override:
            for p in pairs:
                p.confidence = override  # type: ignore[assignment]

        per_source_raw_counts[filename] = len(pairs)
        write_jsonl(pairs, INTERIM_DIR / f"{label}.jsonl")
        all_pairs.extend(pairs)

    # --- Web ---
    for url in WEB_SOURCES:
        log.info("web.start", url=url)
        try:
            pairs = web_scraper.crawl(url, WEB_DIR)
        except Exception as exc:
            log.exception("web.failed", url=url, error=str(exc))
            failures[url] = f"{type(exc).__name__}: {exc}"
            continue
        if not pairs:
            failures.setdefault(url, "0 pares — sitio inaccesible o sin estructura paralela detectable")
        per_source_raw_counts[url] = len(pairs)
        from urllib.parse import urlparse
        label = "web_" + urlparse(url).netloc.replace(".", "_")
        write_jsonl(pairs, INTERIM_DIR / f"{label}.jsonl")
        all_pairs.extend(pairs)

    # --- Filtrado ---
    kept, stats = apply_filters(all_pairs)

    # --- Salidas consolidadas ---
    corpus_jsonl = PROCESSED_DIR / "corpus_v0.jsonl"
    corpus_parquet = PROCESSED_DIR / "corpus_v0.parquet"
    write_jsonl(kept, corpus_jsonl)
    to_parquet(kept, corpus_parquet)

    # --- Hashes de fuentes ---
    sources_for_hash = {
        filename: PDF_DIR / filename for _, filename, _, _ in PDF_PLAN
    }
    for _, filename, _ in SKIPPED_PDFS:
        sources_for_hash[filename] = PDF_DIR / filename
    write_source_hashes(sources_for_hash, PROCESSED_DIR / "source_hashes.json")

    # --- Reporte final (duplicado a archivo) ---
    report_path = PROCESSED_DIR / "pipeline_report.txt"
    report_file = report_path.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, report_file)  # type: ignore[assignment]

    print("\n" + "=" * 70)
    print("REPORTE FINAL — CORPUS v0")
    print("=" * 70)

    print("\nPares por fuente (antes de filtros):")
    for src, n in sorted(per_source_raw_counts.items()):
        print(f"  {src:55s} {n:6d}")

    print(f"\nTotal antes de filtros: {stats.seen}")
    print(f"Total después de filtros: {stats.kept}")
    print("Filtros aplicados (registros descartados):")
    print(f"  vacíos              : {stats.empty}")
    print(f"  fuera de rango wc   : {stats.word_count}")
    print(f"  ratio longitud      : {stats.ratio}")
    print(f"  bri == es           : {stats.identical}")
    print(f"  duplicados (sha256) : {stats.duplicate}")

    print("\nPares conservados por fuente:")
    for src, n in sorted(stats.per_source_kept.items()):
        print(f"  {src:55s} {n:6d}")

    # distribución por dominio y confidence
    from collections import Counter
    dom_counter = Counter(p.domain for p in kept)
    conf_counter = Counter(p.confidence for p in kept)
    print("\nDistribución por dominio:")
    for k, v in sorted(dom_counter.items()):
        print(f"  {k:15s} {v:6d}")
    print("\nDistribución por confianza:")
    for k, v in sorted(conf_counter.items()):
        print(f"  {k:8s} {v:6d}")

    # Primeros 5 pares por fuente para verificación manual
    print("\n" + "-" * 70)
    print("VERIFICACIÓN MANUAL — primeros 5 pares por fuente conservados")
    print("-" * 70)
    by_src: dict[str, list[ParallelPair]] = {}
    for p in kept:
        by_src.setdefault(p.source_doc, []).append(p)
    for src in sorted(by_src):
        print(f"\n[{src}]  total={len(by_src[src])}")
        for i, p in enumerate(by_src[src][:5]):
            print(f"  {i+1}. bri[{p.confidence}|{p.extraction_method}]: {p.bri}")
            print(f"     es: {p.es}")

    # Fuentes saltadas
    print("\n" + "-" * 70)
    print("FUENTES IGNORADAS (por requerir OCR o no ser paralelas):")
    print("-" * 70)
    for label, filename, reason in SKIPPED_PDFS:
        print(f"  {filename:55s}  — {reason}")

    # Fallos
    if failures:
        print("\n" + "-" * 70)
        print("FUENTES QUE FALLARON:")
        print("-" * 70)
        for src, err in failures.items():
            print(f"  {src}\n     → {err}")

    print("\nArtefactos escritos en data/processed/:")
    print(f"  - {corpus_jsonl}")
    print(f"  - {corpus_parquet}")
    print(f"  - {PROCESSED_DIR / 'source_hashes.json'}")
    print(f"\nInterim JSONL en {INTERIM_DIR}/")

    sys.stdout = original_stdout  # type: ignore[assignment]
    report_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
