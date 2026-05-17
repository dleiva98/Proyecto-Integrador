"""Extractor de versículos paralelos del corpus ESSJ (Sánchez Avendaño 2025).

Estructura por versículo (X.Y.):
    [TMI]: bribri Gabb antiguo
    [EB]:  glosa morfemática (1-N líneas con etiquetas ERG/IMP/COP/...)
    [RT]:  bribri moderno regularizado
    [RV]:  Reina-Valera (español arcaico)
    [TL]:  retraducción al español moderno

El par útil para entrenamiento es (RT, TL): bribri moderno ↔ español moderno.
"""
from __future__ import annotations

import re
from pathlib import Path

import structlog

from ..normalization import normalize_bribri, normalize_spanish
from ..schema import ParallelPair
from .base import iter_pages, looks_like_bribri, looks_like_spanish

log = structlog.get_logger()

VERSE_RE = re.compile(r"^\s*(\d{1,2})\.(\d{1,3})\.\s*$")
LABEL_RE = re.compile(r"^\s*(TMI|EB|RT|RV|TL)\.?\s*$")
FOOTNOTE_RE = re.compile(r"^\s*\d+\s+[A-ZÁÉÍÓÚÑ]")  # comienzo de nota al pie

# Una línea con muchas etiquetas en mayúsculas suele ser glosa (EB), no RT/TL.
def _is_gloss_line(s: str) -> bool:
    tokens = re.findall(r"\b[\wáéíóúñÁÉÍÓÚÑ]+\b", s)
    if not tokens:
        return True
    upper = [t for t in tokens if t.isupper() and len(t) >= 2 and t.isalpha()]
    return len(upper) / max(len(tokens), 1) >= 0.30


def _accumulate_verses(all_lines: list[tuple[int, str]]) -> list[tuple[int, str, list[str]]]:
    verses: list[tuple[int, str, list[str]]] = []
    current_id: str | None = None
    current_page: int | None = None
    current_body: list[str] = []
    for page_num, ln in all_lines:
        m = VERSE_RE.match(ln)
        if m:
            if current_id is not None:
                verses.append((current_page or page_num, current_id, current_body))
            current_id = f"{m.group(1)}.{m.group(2)}"
            current_page = page_num
            current_body = []
        else:
            if current_id is None:
                continue
            current_body.append(ln)
    if current_id is not None:
        verses.append((current_page or 0, current_id, current_body))
    return verses


def _clean_body(body: list[str]) -> list[str]:
    out: list[str] = []
    for ln in body:
        s = ln.strip()
        if not s:
            continue
        if LABEL_RE.match(s):
            continue
        if FOOTNOTE_RE.match(s):
            # candidato a nota al pie; cortar aquí — el resto suele ser aparato crítico
            break
        if _is_gloss_line(s):
            continue
        out.append(s)
    return out


def _extract_rt_tl(body: list[str]) -> tuple[str | None, str | None]:
    cleaned = _clean_body(body)
    if len(cleaned) < 2:
        return None, None

    # TL = última línea española clara
    tl_idx: int | None = None
    for idx in range(len(cleaned) - 1, -1, -1):
        ln = cleaned[idx]
        if looks_like_spanish(ln) and not looks_like_bribri(ln):
            tl_idx = idx
            break
    if tl_idx is None:
        return None, None

    # RT = última línea bribri anterior a TL
    rt_idx: int | None = None
    for idx in range(tl_idx - 1, -1, -1):
        ln = cleaned[idx]
        if looks_like_bribri(ln) and not looks_like_spanish(ln):
            rt_idx = idx
            break
    if rt_idx is None:
        return None, None

    return cleaned[rt_idx], cleaned[tl_idx]


def extract(pdf_path: str | Path, source_doc: str) -> list[ParallelPair]:
    all_lines: list[tuple[int, str]] = []
    for page in iter_pages(pdf_path):
        for ln in page.text.splitlines():
            all_lines.append((page.page_number, ln))

    pairs: list[ParallelPair] = []
    verses = _accumulate_verses(all_lines)

    for page_num, verse_id, body in verses:
        rt, tl = _extract_rt_tl(body)
        if not rt or not tl:
            continue
        bri = normalize_bribri(rt)
        es = normalize_spanish(tl)
        if not bri or not es:
            continue
        pairs.append(
            ParallelPair(
                bri=bri,
                es=es,
                source_doc=source_doc,
                source_page=page_num,
                domain="religioso",
                extraction_method="pdf_versicle",
                confidence="medium",
                metadata={"verse_id": verse_id},
            )
        )

    log.info("versicle.done", source=source_doc, pairs=len(pairs), verses=len(verses))
    return pairs
