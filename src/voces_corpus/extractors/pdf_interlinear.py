"""Extractor de ejemplos interlineales numerados (Gramática Jara 2018,
Las palabras de Francisco García, parte interlineal de I ttè).

Patrón objetivo (con líneas blancas tolerables entre ellas):
    1   Ìs   be'  shkẽ̀na?
        cómo 2S  despertar.VM.REC
        ¿Cómo amaneció usted?

Reglas:
- El bloque inicia con una línea que matchea `^\\d{1,3}[\\.\\)]?` (no multi-nivel).
- Las siguientes 2-3 líneas no vacías candidatas se examinan: la primera con
  evidencia de bribri es bri; se acepta hasta UNA línea de glosa intermedia
  (con etiquetas tipo ERG/IMP/COP); la última línea válida en español es es.
- Se descartan candidatos que parezcan entradas de TOC (terminan en número
  de página o tienen relleno de puntos `....`).
"""
from __future__ import annotations

import re
from pathlib import Path

import structlog

from ..normalization import normalize_bribri, normalize_spanish
from ..schema import ParallelPair
from .base import iter_pages, looks_like_bribri, looks_like_spanish

log = structlog.get_logger()

NUM_TRIGGER_RE = re.compile(r"^\s*(\d{1,3})[\.\)]?\s*$")          # solo "12" o "12."
NUM_INLINE_RE = re.compile(r"^\s*(\d{1,3})[\.\)]?\s+(.+)$")        # "12  texto..."
TOC_RE = re.compile(r"\.{3,}|\.{2,}\s*\d{1,4}\s*$|\s\d{1,4}\s*$")  # ...145 / .... 145
MULTILEVEL_RE = re.compile(r"^\s*\d+\.\d+")                        # 4.7.2.1 ...
CAPITULO_RE = re.compile(r"^\s*(CAP[IÍ]TULO|TABLA|FIGURA|ANEXO)\b", re.IGNORECASE)


def _is_toc_like(line: str) -> bool:
    return bool(TOC_RE.search(line)) or bool(MULTILEVEL_RE.match(line)) or bool(CAPITULO_RE.match(line))


def _next_nonempty(lines: list[str], start: int) -> int:
    j = start
    while j < len(lines) and not lines[j].strip():
        j += 1
    return j


def extract(pdf_path: str | Path, source_doc: str, domain: str) -> list[ParallelPair]:
    pairs: list[ParallelPair] = []

    for page in iter_pages(pdf_path):
        lines = page.text.splitlines()
        i = 0
        while i < len(lines):
            ln = lines[i]

            first: str | None = None
            scan_from: int

            m_inline = NUM_INLINE_RE.match(ln)
            m_alone = NUM_TRIGGER_RE.match(ln)

            if m_inline and not MULTILEVEL_RE.match(ln):
                first = m_inline.group(2).strip()
                scan_from = i + 1
            elif m_alone:
                j = _next_nonempty(lines, i + 1)
                if j < len(lines):
                    first = lines[j].strip()
                    scan_from = j + 1
                else:
                    i += 1
                    continue
            else:
                i += 1
                continue

            if not first or _is_toc_like(first):
                i += 1
                continue

            # bri debe ser claramente bribri
            if not looks_like_bribri(first) or looks_like_spanish(first):
                i += 1
                continue

            # buscar hasta 3 líneas más para encontrar la línea española
            es_line: str | None = None
            gloss_line: str | None = None
            k = scan_from
            search_limit = scan_from + 8  # ventana de 8 líneas físicas
            extra_idx = scan_from
            while k < len(lines) and k < search_limit:
                if not lines[k].strip():
                    k += 1
                    continue
                cand = lines[k].strip()
                if _is_toc_like(cand):
                    break
                if looks_like_spanish(cand) and not looks_like_bribri(cand):
                    es_line = cand
                    extra_idx = k + 1
                    break
                # podría ser una línea de glosa; la registramos y seguimos
                if gloss_line is None:
                    gloss_line = cand
                k += 1

            if not es_line:
                i = max(i + 1, scan_from)
                continue

            bri = normalize_bribri(first)
            es = normalize_spanish(es_line)
            if not bri or not es:
                i = extra_idx
                continue

            conf: str = "high" if gloss_line else "medium"
            pairs.append(
                ParallelPair(
                    bri=bri,
                    es=es,
                    source_doc=source_doc,
                    source_page=page.page_number,
                    domain=domain,  # type: ignore[arg-type]
                    extraction_method="pdf_interlinear",
                    confidence=conf,  # type: ignore[arg-type]
                    metadata={"gloss": gloss_line} if gloss_line else {},
                )
            )
            i = extra_idx

    log.info("interlinear.done", source=source_doc, pairs=len(pairs))
    return pairs
