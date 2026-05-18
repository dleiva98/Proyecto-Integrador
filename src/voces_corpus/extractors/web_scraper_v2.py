"""Scraper v2 para sitios web bribri-espanol.

Mejoras:
- Retries con backoff para errores transitorios.
- Deteccion de paginas de bloqueo (forbidden/captcha/cloudflare).
- Filtro de enlaces no-HTML.
- Deduplicacion local de pares extraidos.
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..normalization import normalize_bribri, normalize_spanish
from ..schema import ParallelPair
from .base import looks_like_bribri, looks_like_spanish

log = structlog.get_logger()

USER_AGENT = "voces-corpus-bot/0.2 (academic; contact: programavoces)"
TIMEOUT = 20
DELAY = 1.0
MAX_PAGES = 200

BLOCK_PAGE_RE = re.compile(
    r"access denied|forbidden|cloudflare|captcha|host_not_allowed",
    re.IGNORECASE,
)


def _build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _allowed(url: str) -> bool:
    try:
        parsed = urlparse(url)
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        rp.read()
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True


def _save_raw(html: str, url: str, raw_dir: Path) -> Path:
    domain = urlparse(url).netloc
    out_dir = raw_dir / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    out_path = out_dir / f"{h}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _is_block_page(html: str) -> bool:
    return bool(BLOCK_PAGE_RE.search(html or ""))


def _is_navigable(link: str, base_domain: str) -> bool:
    p = urlparse(link)
    if p.netloc != base_domain:
        return False
    if re.search(r"\.(pdf|jpg|jpeg|png|gif|webp|svg|zip|mp3|mp4)$", p.path, re.IGNORECASE):
        return False
    return True


def _maybe_pair(a: str, b: str) -> tuple[str | None, str | None]:
    if not a or not b:
        return None, None
    if len(a.split()) < 2 or len(b.split()) < 2:
        return None, None

    # a -> bri, b -> es
    if looks_like_bribri(a) and not looks_like_spanish(a) and looks_like_spanish(b) and not looks_like_bribri(b):
        ratio = len(b) / max(len(a), 1)
        if 0.25 <= ratio <= 4.0:
            return normalize_bribri(a), normalize_spanish(b)

    # b -> bri, a -> es
    if looks_like_bribri(b) and not looks_like_spanish(b) and looks_like_spanish(a) and not looks_like_bribri(a):
        ratio = len(a) / max(len(b), 1)
        if 0.25 <= ratio <= 4.0:
            return normalize_bribri(b), normalize_spanish(a)

    return None, None


def _extract_pairs_from_html(html: str, url: str) -> list[ParallelPair]:
    soup = BeautifulSoup(html, "html.parser")
    pairs: list[ParallelPair] = []
    domain = urlparse(url).netloc

    # 1) Tablas de dos columnas
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            if len(cells) != 2:
                continue
            bri, es = _maybe_pair(cells[0], cells[1])
            if bri and es:
                pairs.append(
                    ParallelPair(
                        bri=bri,
                        es=es,
                        source_doc=url,
                        source_page=None,
                        domain="web",
                        extraction_method="we_scrapper_v2:table",
                        confidence="medium",
                        metadata={"domain_host": domain},
                    )
                )

    # 2) Parrafos/listas consecutivos
    blocks = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "li"])]
    blocks = [b for b in blocks if b]
    for i in range(len(blocks) - 1):
        bri, es = _maybe_pair(blocks[i], blocks[i + 1])
        if bri and es:
            pairs.append(
                ParallelPair(
                    bri=bri,
                    es=es,
                    source_doc=url,
                    source_page=None,
                    domain="web",
                    extraction_method="we_scrapper_v2:paragraph",
                    confidence="low",
                    metadata={"domain_host": domain},
                )
            )

    return pairs


def crawl(start_url: str, raw_dir: Path, max_pages: int = MAX_PAGES) -> list[ParallelPair]:
    base_domain = urlparse(start_url).netloc
    queue = [start_url.rstrip("/")]
    visited: set[str] = set()
    pairs: list[ParallelPair] = []
    pair_seen: set[tuple[str, str]] = set()
    session = _build_session()

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

        if _is_block_page(resp.text):
            log.warning("web.block_page", url=url)
            continue

        _save_raw(resp.text, url, raw_dir)

        extracted = _extract_pairs_from_html(resp.text, url)
        for p in extracted:
            key = (p.bri, p.es)
            if key in pair_seen:
                continue
            pair_seen.add(key)
            pairs.append(p)

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"]).partition("#")[0].rstrip("/")
            if _is_navigable(link, base_domain) and link not in visited:
                queue.append(link)

        time.sleep(DELAY)

    log.info("web.crawl_done", start=start_url, visited=len(visited), pairs=len(pairs))
    return pairs