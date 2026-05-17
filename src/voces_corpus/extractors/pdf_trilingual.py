"""Extractor para PDF trilingüe Ditsö̀ rukuö̀ (bribri / español / inglés).

Estrategia robusta independiente del paginado:
- Recorre el documento concatenando todos los párrafos (separados por línea vacía).
- Etiqueta cada párrafo: 'bri' / 'es' / 'en' / 'otro'.
- Para cada párrafo bri, busca el siguiente párrafo 'es' dentro de una ventana
  cercana y los empareja. Ambos deben tener longitudes comparables
  (ratio en chars dentro de [0.4, 2.8]).
"""
from __future__ import annotations

import re
from pathlib import Path

import structlog

from ..normalization import normalize_bribri, normalize_spanish
from ..schema import ParallelPair
from .base import iter_pages, looks_like_bribri, looks_like_spanish

log = structlog.get_logger()


_BOILERPLATE_RE = re.compile(
    r"prior written permission|all rights reserved|ISBN|copyright|©|"
    r"primera edición|segunda edición|Editorial|Editorial UCR|"
    r"impreso en|printed in",
    re.IGNORECASE,
)


def _classify_paragraph(text: str) -> str:
    if _BOILERPLATE_RE.search(text):
        return "otro"
    if looks_like_bribri(text) and not looks_like_spanish(text):
        return "bri"
    if looks_like_spanish(text) and not looks_like_bribri(text):
        return "es"
    return "otro"


def _paragraphs_with_page(pdf_path: str | Path) -> list[tuple[int, str]]:
    items: list[tuple[int, str]] = []
    for page in iter_pages(pdf_path):
        # separar en bloques por línea vacía dentro de la página
        blocks = re.split(r"\n\s*\n", page.text)
        for b in blocks:
            clean = " ".join(line.strip() for line in b.splitlines() if line.strip())
            if clean and len(clean) > 25:  # descartar fragmentos demasiado cortos
                items.append((page.page_number, clean))
    return items


def extract(pdf_path: str | Path, source_doc: str) -> list[ParallelPair]:
    items = _paragraphs_with_page(pdf_path)
    classified = [(p, t, _classify_paragraph(t)) for p, t in items]

    pairs: list[ParallelPair] = []
    used: set[int] = set()
    window = 6  # buscar es dentro de las siguientes 6 unidades

    for i, (page_i, text_i, lbl_i) in enumerate(classified):
        if i in used or lbl_i != "bri":
            continue
        # buscar el primer 'es' siguiente dentro de la ventana
        for j in range(i + 1, min(i + 1 + window, len(classified))):
            page_j, text_j, lbl_j = classified[j]
            if j in used or lbl_j != "es":
                continue
            # ratio de longitud razonable
            ratio = len(text_j) / max(len(text_i), 1)
            if not (0.4 <= ratio <= 2.8):
                continue
            bri = normalize_bribri(text_i)
            es = normalize_spanish(text_j)
            if not bri or not es:
                continue
            pairs.append(
                ParallelPair(
                    bri=bri,
                    es=es,
                    source_doc=source_doc,
                    source_page=page_i,
                    domain="narrativo",
                    extraction_method="pdf_trilingual",
                    confidence="low",
                    metadata={"es_page": page_j, "alignment": "paragraph"},
                )
            )
            used.add(i)
            used.add(j)
            break

    log.info("trilingual.done", source=source_doc, pairs=len(pairs))
    return pairs
