"""Scraper para sitios web bribri-español, principal target: lenguabribri.com.

Diseño:
- Respeta robots.txt (urllib.robotparser).
- Guarda HTML crudo en data/raw/web/<dominio>/<path-hash>.html antes de parsear.
- Recorre BFS con visited set y mismo dominio.
- Extrae pares heurísticamente: tablas con dos celdas (bri, es), o párrafos
  consecutivos donde uno luce bribri y el siguiente español.

Si el entorno de red bloquea el dominio (HTTP 403 host_not_allowed), se
captura la excepción y se reporta como fuente fallida sin abortar el pipeline.
"""
from __future__ import annotations

import hashlib
import re
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import structlog
from bs4 import BeautifulSoup

from ..normalization import normalize_bribri, normalize_spanish
from ..schema import ParallelPair
from .base import looks_like_bribri, looks_like_spanish

log = structlog.get_logger()

USER_AGENT = "voces-corpus-bot/0.1 (academic; contact: programavoces)"
TIMEOUT = 20
DELAY = 1.0  # cortesía entre requests
MAX_PAGES = 200


def _allowed(url: str) -> bool:
    try:
        parsed = urlparse(url)
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True  # si robots no se puede leer, asumimos permitido


def _save_raw(html: str, url: str, raw_dir: Path) -> Path:
    domain = urlparse(url).netloc
    out_dir = raw_dir / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    out_path = out_dir / f"{h}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _extract_pairs_from_html(html: str, url: str) -> list[ParallelPair]:
    soup = BeautifulSoup(html, "html.parser")
    pairs: list[ParallelPair] = []
    domain = urlparse(url).netloc

    # 1) Tablas con dos columnas
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) == 2:
                a, b = cells
                bri, es = _maybe_pair(a, b)
                if bri and es:
                    pairs.append(
                        ParallelPair(
                            bri=bri,
                            es=es,
                            source_doc=url,
                            source_page=None,
                            domain="web",
                            extraction_method="web_scraper:table",
                            confidence="medium",
                            metadata={"domain_host": domain},
                        )
                    )

    # 2) Párrafos consecutivos
    paras = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li"])]
    paras = [p for p in paras if p]
    for i in range(len(paras) - 1):
        bri, es = _maybe_pair(paras[i], paras[i + 1])
        if bri and es:
            pairs.append(
                ParallelPair(
                    bri=bri,
                    es=es,
                    source_doc=url,
                    source_page=None,
                    domain="web",
                    extraction_method="web_scraper:paragraph",
                    confidence="low",
                    metadata={"domain_host": domain},
                )
            )

    return pairs


def _maybe_pair(a: str, b: str) -> tuple[str | None, str | None]:
    if not a or not b:
        return None, None
    if looks_like_bribri(a) and not looks_like_spanish(a) and looks_like_spanish(b) and not looks_like_bribri(b):
        return normalize_bribri(a), normalize_spanish(b)
    if looks_like_bribri(b) and not looks_like_spanish(b) and looks_like_spanish(a) and not looks_like_bribri(a):
        return normalize_bribri(b), normalize_spanish(a)
    return None, None


def crawl(start_url: str, raw_dir: Path, max_pages: int = MAX_PAGES) -> list[ParallelPair]:
    """Recorre el dominio en BFS, guarda HTML crudo y devuelve pares extraídos."""
    base_domain = urlparse(start_url).netloc
    queue = [start_url]
    visited: set[str] = set()
    pairs: list[ParallelPair] = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        if not _allowed(url):
            log.warning("web.robots_blocked", url=url)
            continue

        try:
            resp = session.get(url, timeout=TIMEOUT)
        except Exception as exc:
            log.error("web.fetch_failed", url=url, error=str(exc))
            continue

        if resp.status_code != 200:
            log.warning("web.non200", url=url, status=resp.status_code)
            continue

        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype:
            continue

        _save_raw(resp.text, url, raw_dir)
        pairs.extend(_extract_pairs_from_html(resp.text, url))

        # encolar enlaces del mismo dominio
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"])
            link, _, _ = link.partition("#")
            if urlparse(link).netloc == base_domain and link not in visited:
                queue.append(link)

        time.sleep(DELAY)

    log.info("web.crawl_done", start=start_url, visited=len(visited), pairs=len(pairs))
    return pairs
