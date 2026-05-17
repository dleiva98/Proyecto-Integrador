"""Consolida pares de múltiples fuentes a un corpus único versionado."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import polars as pl
import structlog

from .schema import ParallelPair

log = structlog.get_logger()


def write_jsonl(pairs: Iterable[ParallelPair], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(p.model_dump_json() + "\n")
            n += 1
    log.info("write_jsonl", path=str(path), n=n)
    return n


def read_jsonl(path: Path) -> list[ParallelPair]:
    out: list[ParallelPair] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(ParallelPair.model_validate_json(line))
    return out


def to_parquet(pairs: list[ParallelPair], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in pairs:
        d = p.model_dump()
        # polars no acepta dict heterogéneo bien: serializamos metadata como JSON
        d["metadata"] = json.dumps(d.get("metadata", {}), ensure_ascii=False)
        rows.append(d)
    df = pl.DataFrame(rows)
    df.write_parquet(path)
    log.info("write_parquet", path=str(path), n=len(rows))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_source_hashes(sources: dict[str, Path], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, p in sources.items():
        if p.exists() and p.is_file():
            out[name] = {"path": str(p), "sha256": file_hash(p), "size": p.stat().st_size}
        else:
            out[name] = {"path": str(p), "sha256": None, "size": None}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("write_source_hashes", path=str(path), n=len(out))
