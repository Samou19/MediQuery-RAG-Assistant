"""Generate grounded medical answers from retrieved context."""

import unicodedata

from mediquery.generation.llm_client import ClaudeClient
from mediquery.schemas import (
    MediQueryResponse,
    RetrievalResult,
    SourceReference,
)


class MediQueryGenerator:
    """Generate answers using retrieved context and suppress sources on abstention."""

    # Expressions commonly used by the model when refusing an out-of-scope
    # question or indicating that the supplied documents are insufficient.
    ABSTENTION_MARKERS = (
        "je ne peux pas repondre",
        "je ne suis pas en mesure de repondre",
        "je ne peux pas fournir de reponse",
        "ne contiennent pas suffisamment d'informations",
        "ne contient pas suffisamment d'informations",
        "informations insuffisantes",
        "ne permet pas de repondre",
        "ne permettent pas de repondre",
        "hors perimetre",
        "sans lien avec",
        "ne figurent pas dans les documents",
        "n'est pas aborde dans le document",
        "n'est pas abordee dans le document",
        "ne sont pas abordes dans le document",
        "ne sont pas abordees dans le document",
    )

    def __init__(self, client: ClaudeClient | None = None) -> None:
        self.client = client or ClaudeClient()

    def generate(
        self,
        question: str,
        context: RetrievalResult,
    ) -> MediQueryResponse:
        """Generate an answer and omit citations when the model abstains."""
        normalized_question = question.strip()

        if not normalized_question:
            raise ValueError("La question ne peut pas être vide.")

        if not context.chunks:
            return MediQueryResponse(
                question=normalized_question,
                answer=(
                    "Les documents récupérés ne contiennent pas suffisamment "
                    "d'informations pour répondre de manière fiable à cette question."
                ),
                sources=(),
                abstained=True,
                metadata={"strategy": context.strategy, "reason": "no_context"},
            )

        context_text = self._build_context(context)
        answer = self.client.generate(normalized_question, context_text)

        # A retrieval can return superficially similar passages even for an
        # unrelated question. If Claude declines to answer, do not present
        # those passages as supporting sources.
        if self._is_abstention(answer):
            return MediQueryResponse(
                question=normalized_question,
                answer=answer,
                sources=(),
                abstained=True,
                metadata={
                    "strategy": context.strategy,
                    "reason": "model_abstention",
                },
            )

        sources = self._build_sources(context)
        return MediQueryResponse(
            question=normalized_question,
            answer=answer,
            sources=sources,
            abstained=False,
            metadata={"strategy": context.strategy},
        )

    @classmethod
    def _is_abstention(cls, answer: str) -> bool:
        """Detect common French refusal/insufficient-evidence wording."""
        normalized = unicodedata.normalize("NFKD", answer.lower())
        normalized = "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )
        return any(marker in normalized for marker in cls.ABSTENTION_MARKERS)

    @staticmethod
    def _build_context(context: RetrievalResult) -> str:
        """Format retrieved chunks with source and page metadata."""
        sections = []

        for index, chunk in enumerate(context.chunks, start=1):
            page = chunk.metadata.get("page")
            source = chunk.source or chunk.metadata.get("source", "unknown")

            citation = f"Source : {source}"
            if page is not None:
                citation += f" | Page : {page}"

            sections.append(
                f"[Passage {index}]\n{citation}\n{chunk.content}"
            )

        return "\n\n".join(sections)

    @staticmethod
    def _build_sources(
        context: RetrievalResult,
    ) -> tuple[SourceReference, ...]:
        """Build unique source references from retrieved chunks."""
        sources = []
        seen: set[tuple[str, int | None]] = set()

        for chunk in context.chunks:
            raw_page = chunk.metadata.get("page")
            try:
                page = int(raw_page) if raw_page is not None else None
            except (TypeError, ValueError):
                page = None

            document_name = chunk.source or str(
                chunk.metadata.get("source", "unknown")
            )
            key = (document_name, page)

            if key in seen:
                continue

            seen.add(key)
            sources.append(
                SourceReference(
                    document_name=document_name,
                    page=page,
                    chunk_id=chunk.chunk_id,
                )
            )

        return tuple(sources)
