"""Normalización ortográfica conservadora.

Reglas:
- NFC para todo texto (Unicode Normalization Form C).
- Bribri: NUNCA lowercase, NUNCA quitar combining marks (tonos, ~, ʼ).
- Español: NFC, colapso de whitespace, normalización de comillas tipográficas.
"""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")

_SPANISH_QUOTE_MAP = {
    "“": '"',  # left double quotation mark
    "”": '"',  # right double quotation mark
    "‘": "'",  # left single quotation mark
    "’": "'",  # right single quotation mark
    "„": '"',
    "‚": "'",
    "«": '"',
    "»": '"',
}

# Hyphens & dashes harmless to collapse to "-".
_DASHES = {"‐", "‑", "‒", "–", "—", "−"}


def _strip_control(s: str) -> str:
    # Mantiene \t y \n; descarta el resto de controles invisibles.
    return "".join(ch for ch in s if ch in ("\t", "\n") or unicodedata.category(ch)[0] != "C")


def normalize_bribri(text: str) -> str:
    """Normaliza un fragmento bribri preservando diacríticos y mayúsculas."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _strip_control(text)
    text = _WS.sub(" ", text).strip()
    return text


def normalize_spanish(text: str) -> str:
    """Normaliza un fragmento español: NFC, comillas rectas, whitespace colapsado."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    for src, dst in _SPANISH_QUOTE_MAP.items():
        text = text.replace(src, dst)
    for d in _DASHES:
        text = text.replace(d, "-")
    text = _strip_control(text)
    text = _WS.sub(" ", text).strip()
    return text
