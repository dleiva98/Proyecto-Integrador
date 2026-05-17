"""Schema canónico de un par paralelo bribri-español."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Domain = Literal[
    "didactico",
    "narrativo",
    "religioso",
    "etnografico",
    "web",
]

Confidence = Literal["high", "medium", "low"]


class ParallelPair(BaseModel):
    """Un par alineado bribri-español, salida atómica de todos los extractores."""

    bri: str = Field(..., description="Texto bribri normalizado NFC, sin lowercase.")
    es: str = Field(..., description="Texto español normalizado NFC.")
    source_doc: str = Field(..., description="Nombre del archivo o URL de origen.")
    source_page: Optional[int] = Field(
        None, description="Página del PDF; None para fuentes web."
    )
    domain: Domain
    extraction_method: str = Field(
        ..., description="Identificador del extractor que produjo el par."
    )
    confidence: Confidence = "medium"
    metadata: dict[str, Any] = Field(default_factory=dict)
