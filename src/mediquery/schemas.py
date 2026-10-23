
"""Shared data schemas for the MediQuery RAG pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A document chunk returned by a retrieval component."""

    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class SourceReference:
    """A source citation associated with a generated answer."""

    document_name: str
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Results returned by a retrieval strategy."""

    query: str
    chunks: tuple[RetrievedChunk, ...]
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MediQueryResponse:
    """Final response returned by the MediQuery pipeline."""

    question: str
    answer: str
    sources: tuple[SourceReference, ...] = ()
    abstained: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
