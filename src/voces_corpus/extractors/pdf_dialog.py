"""Extractor de pares marcados explícitamente, estilo diálogo (Sé'ttö' bribri ie).

Patrón 1 — Diálogo con prefijo de hablante:
    Edgar: Ye'    tche.
           Me voy (hasta luego).

Patrón 2 — Listas Q/R simétricas:
    A: bri-pregunta
    B: bri-respuesta
       es-traducción de la respuesta

Solo se emite un par cuando hay marcador de hablante. NO se hace
pareo por mera adyacencia, porque genera demasiados falsos positivos
en columnas de vocabulario o material que cruza columnas.
"""
from __future__ import annotations

import re
from pathlib import Path

import structlog

from ..normalization import normalize_bribri, normalize_spanish
from ..schema import ParallelPair
from .base import iter_pages, looks_like_bribri, looks_like_spanish

log = structlog.get_logger()

SPEAKER_RE = re.compile(r"^\s*([A-ZÁÉÍÓÚÑ][\wáéíóúñ]{0,15})\s*:\s*(.+)$")


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
            m = SPEAKER_RE.match(ln)
            if not m:
                i += 1
                continue
            bri_candidate = m.group(2).strip()

            # Caso 1: bribri en la línea inicial, español en la siguiente
            if looks_like_bribri(bri_candidate) and not looks_like_spanish(bri_candidate):
                k = _next_nonempty(lines, i + 1)
                if k < len(lines):
                    es_candidate = lines[k].strip()
                    m2 = SPEAKER_RE.match(es_candidate)
                    if m2:
                        # siguiente es otro hablante; no es traducción
                        i = k
                        continue
                    if looks_like_spanish(es_candidate) and not looks_like_bribri(es_candidate):
                        bri = normalize_bribri(bri_candidate)
                        es = normalize_spanish(es_candidate)
                        if bri and es:
                            pairs.append(
                                ParallelPair(
                                    bri=bri,
                                    es=es,
                                    source_doc=source_doc,
                                    source_page=page.page_number,
                                    domain=domain,  # type: ignore[arg-type]
                                    extraction_method="pdf_dialog",
                                    confidence="high",
                                    metadata={"speaker": m.group(1)},
                                )
                            )
                        i = k + 1
                        continue
            i += 1

    log.info("dialog.done", source=source_doc, pairs=len(pairs))
    return pairs
