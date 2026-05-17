"""Base utilitaria para extractores PDF.

`looks_like_bribri` / `looks_like_spanish` son heurísticas de SEPARACIÓN de
idioma: el objetivo es discriminar entre líneas en español (que pueden tener
algunos diacríticos castellanos como á, é, í, ó, ú, ñ) y líneas en bribri,
que usan un repertorio diacrítico mucho más amplio (ö, ë, vocales nasales
con virgulilla, tonos combinantes, corte glotal ʼ/').
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz  # pymupdf

# Glifos exclusivos del bribri estandarizado (no aparecen en español ortográfico).
_BRIBRI_EXCLUSIVE = re.compile(
    r"["
    r"öÖëËẽẼĩĨũŨỹỸ"             # vocales con diéresis o nasales con virgulilla
    r"àÀèÈìÌòÒùÙ"                # acento grave (no usado en español, sí en bribri)
    r"âÂêÊîÎôÔûÛ"                # circunflejo
    r"]|"
    r"[̧̨̣̤̱̀́̂̃̄̆̇̌]"
)

# Glifos que comparte con español (no son evidencia fuerte de bribri).
_SPANISH_DIACRITICS = re.compile(r"[áéíóúÁÉÍÓÚñÑ¿¡]")

# Apóstrofo como corte glotal: marcador bribri si aparece adjunto a vocal.
_BRIBRI_GLOTTAL = re.compile(r"[aeiouAEIOU][ʼ']")

# Palabras españolas frecuentes (función + nexos) — alta especificidad para "es español".
_SPANISH_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "que", "y", "o", "u", "e", "en", "con", "por", "para",
    "se", "es", "son", "su", "sus", "le", "les", "lo", "al",
    "yo", "tú", "él", "ella", "nosotros", "ustedes", "ellos", "ellas",
    "mi", "tu", "no", "sí", "muy", "más", "menos", "como", "cómo",
    "qué", "quien", "donde", "cuando", "porque", "pero", "sino",
    "hay", "está", "están", "ser", "estar", "tener", "hacer",
    "este", "esta", "estos", "estas", "ese", "esa", "aquel", "aquella",
    "esto", "eso", "aquello", "fue", "será", "sería", "había", "habrá",
    "soy", "eres", "somos", "sois", "fueron", "han", "ha", "he",
    "también", "todavía", "ya", "aún", "aquí", "allá", "ahora", "antes",
    "después", "mientras", "hasta", "desde", "entre", "sobre", "tras",
}

# Etiquetas morfológicas / abreviaturas que suelen estar en líneas de glosa.
_GLOSS_LABELS = {
    "ERG", "ABS", "COM", "DAT", "LOC", "GEN", "ACC", "NOM",
    "IMP", "PERF", "REM", "REC", "PROG", "HAB", "REL", "DG", "FM",
    "PL", "SG", "1S", "2S", "3S", "1P", "2P", "3P", "INC", "EXC",
    "COP", "INF", "VM", "AG", "INT", "DIR", "DIM", "TOP", "FOC", "EVID",
}

# Tokenizador simple
_WORD = re.compile(r"\b[\wáéíóúñÁÉÍÓÚÑöÖëËãÃõÕẽẼĩĨũŨ'ʼ\-]+\b", flags=re.UNICODE)


def _word_tokens(s: str) -> list[str]:
    return _WORD.findall(s)


def _spanish_stopword_score(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    low = [t.lower() for t in tokens]
    hits = sum(1 for t in low if t in _SPANISH_STOPWORDS)
    return hits / max(len(tokens), 1)


def looks_like_bribri(s: str) -> bool:
    """True si la línea muestra evidencia ortográfica clara de bribri estandarizado.

    Criterio: al menos uno de
      - contiene glifo exclusivo bribri (ö, ë, vocal nasal, acento grave/circunflejo)
      - contiene corte glotal adjunto a vocal
    Y NO está dominada por stopwords españolas (score < 0.25).
    """
    if not s or len(s.strip()) < 2:
        return False
    has_excl = bool(_BRIBRI_EXCLUSIVE.search(s))
    has_glot = bool(_BRIBRI_GLOTTAL.search(s))
    if not (has_excl or has_glot):
        return False
    tokens = _word_tokens(s)
    if _spanish_stopword_score(tokens) >= 0.25:
        return False
    return True


def looks_like_spanish(s: str) -> bool:
    """True si la línea parece prosa española estándar.

    Criterio: score de stopwords >= 0.10 con al menos 1 hit y al menos 3 tokens,
    AND no contiene glifos exclusivos bribri con frecuencia (< 1 por línea),
    AND no se ve dominada por etiquetas de glosa morfológica.
    """
    if not s or len(s.strip()) < 2:
        return False
    tokens = _word_tokens(s)
    if len(tokens) < 3:
        return False

    # Rechazar líneas de glosa morfológica
    upper_tokens = [t for t in tokens if t.isupper() and len(t) >= 2 and t.isalpha()]
    if upper_tokens and len(upper_tokens) / len(tokens) >= 0.25:
        return False
    if any(t in _GLOSS_LABELS for t in tokens):
        # presencia de etiqueta de glosa, descartar
        return False

    # Demasiados glifos exclusivos bribri → no es prosa española
    bri_marks = len(_BRIBRI_EXCLUSIVE.findall(s))
    if bri_marks >= 2:
        return False

    stop_score = _spanish_stopword_score(tokens)
    if stop_score >= 0.10:
        return True
    # fallback: ratio mínimo
    if any(t.lower() in _SPANISH_STOPWORDS for t in tokens) and stop_score >= 0.05:
        return True
    return False


@dataclass
class PageText:
    page_index: int

    text: str

    @property
    def page_number(self) -> int:
        return self.page_index + 1


def iter_pages(pdf_path: str | Path) -> Iterator[PageText]:
    """Itera páginas de un PDF entregando texto crudo (sin OCR)."""
    doc = fitz.open(str(pdf_path))
    try:
        for i in range(len(doc)):
            yield PageText(page_index=i, text=doc[i].get_text("text"))
    finally:
        doc.close()
