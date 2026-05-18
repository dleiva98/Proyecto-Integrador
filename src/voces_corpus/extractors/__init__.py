"""Paquete de extractores para corpus bribri-español."""

from . import pdf_dialog
from . import pdf_interlinear
from . import pdf_trilingual
from . import pdf_versicle
from . import web_scraper
from . import web_scraper_v2

__all__ = [
    "pdf_dialog",
    "pdf_interlinear",
    "pdf_trilingual",
    "pdf_versicle",
    "web_scraper",
    "web_scraper_v2",
]