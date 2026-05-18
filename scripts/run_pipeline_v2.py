"""Orquestador end-to-end v2 del pipeline de corpus bribri-español.

Uso:
    python scripts/run_pipeline_v2.py
    python scripts/run_pipeline_v2.py --enable-web
    python scripts/run_pipeline_v2.py --print-corpus --print-corpus-max 200

Cambios v2:
- Scraping web opcional por flag (--enable-web).
- Split de salida: strict (high/medium) y aux (low).
- Export legible BRI:ES opcional (--print-corpus).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import structlog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from voces_corpus.consolidate import (  # noqa: E402
    to_parquet,
    write_jsonl,
    write_source_hashes,
)
from voces_corpus.extractors import (  # noqa: E402
    pdf_dialog,
    pdf_interlinear,
    pdf_trilingual,
    pdf_versicle,
    we_scrapper_v2,
)
from voces_corpus.filters import apply_filters  # noqa: E402
from voces_corpus.schema import ParallelPair  # noqa: E402

log = structlog.get_logger()

PDF_DIR = ROOT / "data" / "raw" / "pdfs"
WEB_DIR = ROOT / "data" / "raw" / "web"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"

PDF_PLAN: list[tuple[str, str, str, str]] = [
    ("gramatica_jara", "Gramatica_lengua_bribri_Jara2018.pdf", "interlinear", "didactico"),
    ("sevtto_hablemos", "Sevtto_bribri_ie_Hablemos.pdf", "dialog", "didactico"),
    ("i_tte", "I_tte_Historias_bribris.pdf", "interlinear", "narrativo"),
    ("ditso_rukuo", "Ditso_rukuo_Identidad_semillas.pdf", "trilingual", "narrativo"),
    ("palabras_garcia", "Palabras_Francisco_Garcia.pdf", "interlinear", "etnografico"),
    ("essj_gabb", "ESSJ_Sanchez_Avendano_Vol1.pdf", "versicle", "religioso"),
]

CONFIDENCE_OVERRIDE: dict[str, str] = {
    "I_tte_Historias_bribris.pdf": "low",
    "Palabras_Francisco_Garcia.pdf": "low",
}

SKIPPED_PDFS = [
    (
        "diccionario_mitologia",
        "Diccionario_mitologia_bribri_UCR.pdf",
        "PDF escaneado, requiere OCR - fuera de alcance del pipeline.",
    ),
    (
        "ko_keska",
        "Ko_Keska_Lugar_del_tiempo.pdf",
        "PDF escaneado, mayormente monolingue espanol con terminos bribri.",
    ),
    (
        "cargos_tradicionales",
        "Cargos_tradicionales_pueblo_bribri.pdf",
        "Monolingue espanol con terminologia bribri en cursiva; sin paralelizacion.",
    ),
    (
        "se_dor_ste",
        "Se_dor_ste.pdf",
        "Bilingue espanol-ingles con bribri ocasional; no es par bri-es.",
    ),
]

WEB_SOURCES = [
    "https://www.lenguabribri.com/",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline v2 del corpus bribri-espanol")
    parser.add_argument(
        "--enable-web",
        action="store_true",
        help="Activa scraping web (desactivado por defecto).",
    )
    parser.add_argument(
        "--print-corpus",
        action="store_true",
        help="Exporta un TXT legible con pares BRI:ES.",
    )
    parser.add_argument(
        "--print-corpus-path",
        default=str(PROCESSED_DIR / "corpus_v0_bri_es.txt"),
        help="Ruta del TXT legible.",
    )
    parser.add_argument(
        "--print-corpus-max",
        type=int,
        default=0,
        help="Maximo de pares a exportar (0 = todos).",
    )
    return parser.parse_args()


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


def write_bilingual_txt(pairs: list[ParallelPair], out_path: Path, max_rows: int = 0) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected = pairs if max_rows <= 0 else pairs[:max_rows]
    with out_path.open("w", encoding="utf-8") as f:
        for i, p in enumerate(selected, start=1):
            f.write(f"{i:06d}\n")
            f.write(f"BRI: {p.bri}\n")
            f.write(f"ES : {p.es}\n\n")
    return len(selected)


class _Tee:
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
    args = parse_args()

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs: list[ParallelPair] = []
    failures: dict[str, str] = {}
    per_source_raw_counts: dict[str, int] = {}

    # PDFs
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

    # Web opcional
    if args.enable_web:
        for url in WEB_SOURCES:
            log.info("web.start", url=url)
            try:
                pairs = we_scrapper_v2.crawl(url, WEB_DIR)
            except Exception as exc:
                log.exception("web.failed", url=url, error=str(exc))
                failures[url] = f"{type(exc).__name__}: {exc}"
                continue
            if not pairs:
                failures.setdefault(url, "0 pares - sitio inaccesible o sin estructura paralela detectable")
            per_source_raw_counts[url] = len(pairs)
            from urllib.parse import urlparse
            label = "web_" + urlparse(url).netloc.replace(".", "_")
            write_jsonl(pairs, INTERIM_DIR / f"{label}.jsonl")
            all_pairs.extend(pairs)
    else:
        log.info("web.skipped", reason="disabled_by_flag")

    kept, stats = apply_filters(all_pairs)

    # Split para entrenamiento
    kept_strict = [p for p in kept if p.confidence in ("high", "medium")]
    kept_aux_low = [p for p in kept if p.confidence == "low"]

    # Salidas principales
    corpus_jsonl = PROCESSED_DIR / "corpus_v0.jsonl"
    corpus_parquet = PROCESSED_DIR / "corpus_v0.parquet"
    write_jsonl(kept, corpus_jsonl)
    to_parquet(kept, corpus_parquet)

    # Salidas nuevas v2
    strict_jsonl = PROCESSED_DIR / "corpus_v0_strict.jsonl"
    aux_jsonl = PROCESSED_DIR / "corpus_v0_aux_lowconf.jsonl"
    write_jsonl(kept_strict, strict_jsonl)
    write_jsonl(kept_aux_low, aux_jsonl)

    # Hashes de fuentes
    sources_for_hash = {filename: PDF_DIR / filename for _, filename, _, _ in PDF_PLAN}
    for _, filename, _ in SKIPPED_PDFS:
        sources_for_hash[filename] = PDF_DIR / filename
    write_source_hashes(sources_for_hash, PROCESSED_DIR / "source_hashes.json")

    # Reporte
    report_path = PROCESSED_DIR / "pipeline_report_v2.txt"
    report_file = report_path.open("w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, report_file)  # type: ignore[assignment]

    print("\n" + "=" * 70)
    print("REPORTE FINAL - CORPUS v2")
    print("=" * 70)

    print("\nPares por fuente (antes de filtros):")
    for src, n in sorted(per_source_raw_counts.items()):
        print(f"  {src:55s} {n:6d}")

    print(f"\nTotal antes de filtros: {stats.seen}")
    print(f"Total despues de filtros: {stats.kept}")
    print("Filtros aplicados (registros descartados):")
    print(f"  vacios              : {stats.empty}")
    print(f"  fuera de rango wc   : {stats.word_count}")
    print(f"  ratio longitud      : {stats.ratio}")
    print(f"  bri == es           : {stats.identical}")
    print(f"  duplicados (sha256) : {stats.duplicate}")

    print("\nPares conservados por fuente:")
    for src, n in sorted(stats.per_source_kept.items()):
        print(f"  {src:55s} {n:6d}")

    dom_counter = Counter(p.domain for p in kept)
    conf_counter = Counter(p.confidence for p in kept)
    print("\nDistribucion por dominio:")
    for k, v in sorted(dom_counter.items()):
        print(f"  {k:15s} {v:6d}")
    print("\nDistribucion por confianza:")
    for k, v in sorted(conf_counter.items()):
        print(f"  {k:8s} {v:6d}")

    print("\nSplit de entrenamiento:")
    print(f"  strict (high+medium): {len(kept_strict)}")
    print(f"  aux_low (low)       : {len(kept_aux_low)}")

    if failures:
        print("\n" + "-" * 70)
        print("FUENTES QUE FALLARON:")
        print("-" * 70)
        for src, err in failures.items():
            print(f"  {src}\n     -> {err}")

    if args.print_corpus:
        out_txt = Path(args.print_corpus_path)
        n_export = write_bilingual_txt(kept, out_txt, args.print_corpus_max)
        print(f"\nCorpus legible exportado: {out_txt} (pares={n_export})")

    print("\nArtefactos escritos en data/processed/:")
    print(f"  - {corpus_jsonl}")
    print(f"  - {corpus_parquet}")
    print(f"  - {strict_jsonl}")
    print(f"  - {aux_jsonl}")
    print(f"  - {report_path}")
    print(f"  - {PROCESSED_DIR / 'source_hashes.json'}")
    print(f"\nInterim JSONL en {INTERIM_DIR}/")

    sys.stdout = original_stdout  # type: ignore[assignment]
    report_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())