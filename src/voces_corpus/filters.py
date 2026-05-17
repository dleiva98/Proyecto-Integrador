"""Filtrado y deduplicación de pares paralelos.

Aplica los filtros en el orden especificado y registra cuántos pares caen
en cada uno.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

import structlog

from .schema import ParallelPair

log = structlog.get_logger()

MIN_WORDS = 2
MAX_WORDS = 80
MIN_RATIO = 0.3
MAX_RATIO = 3.0


def _wc(s: str) -> int:
    return len(s.split())


def _hash_pair(bri: str, es: str) -> str:
    h = hashlib.sha256()
    h.update(bri.encode("utf-8"))
    h.update(b"\x00")
    h.update(es.encode("utf-8"))
    return h.hexdigest()


@dataclass
class FilterStats:
    seen: int = 0
    empty: int = 0
    word_count: int = 0
    ratio: int = 0
    duplicate: int = 0
    identical: int = 0
    kept: int = 0
    per_source_seen: dict[str, int] = field(default_factory=dict)
    per_source_kept: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "seen": self.seen,
            "empty": self.empty,
            "word_count": self.word_count,
            "ratio": self.ratio,
            "duplicate": self.duplicate,
            "identical": self.identical,
            "kept": self.kept,
            "per_source_seen": self.per_source_seen,
            "per_source_kept": self.per_source_kept,
        }


def apply_filters(pairs: Iterable[ParallelPair]) -> tuple[list[ParallelPair], FilterStats]:
    """Aplica todos los filtros en orden y devuelve los pares conservados."""
    stats = FilterStats()
    seen_hashes: set[str] = set()
    kept: list[ParallelPair] = []

    for pair in pairs:
        stats.seen += 1
        stats.per_source_seen[pair.source_doc] = stats.per_source_seen.get(pair.source_doc, 0) + 1

        bri = pair.bri.strip()
        es = pair.es.strip()

        # 1) vacíos
        if not bri or not es:
            stats.empty += 1
            continue

        # 2) word count en rango
        wc_bri = _wc(bri)
        wc_es = _wc(es)
        if wc_bri < MIN_WORDS or wc_es < MIN_WORDS or wc_bri > MAX_WORDS or wc_es > MAX_WORDS:
            stats.word_count += 1
            continue

        # 3) ratio de longitud en chars
        if len(bri) == 0:
            stats.ratio += 1
            continue
        ratio = len(es) / len(bri)
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            stats.ratio += 1
            continue

        # 5) bri == es (antes del dedup para no contar el dup espurio)
        if bri == es:
            stats.identical += 1
            continue

        # 4) dedup por SHA-256
        h = _hash_pair(bri, es)
        if h in seen_hashes:
            stats.duplicate += 1
            continue
        seen_hashes.add(h)

        kept.append(pair)
        stats.kept += 1
        stats.per_source_kept[pair.source_doc] = stats.per_source_kept.get(pair.source_doc, 0) + 1

    log.info(
        "filters.done",
        seen=stats.seen,
        kept=stats.kept,
        empty=stats.empty,
        word_count=stats.word_count,
        ratio=stats.ratio,
        identical=stats.identical,
        duplicate=stats.duplicate,
    )
    return kept, stats
