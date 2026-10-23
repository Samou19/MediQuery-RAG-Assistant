
"""Adapter hybrid retrieval results to the MediQuery pipeline schema."""

from langchain_core.documents import Document

from mediquery.retrieval.hybrid import HybridRetriever
from mediquery.schemas import RetrievalResult, RetrievedChunk


class HybridPipelineRetriever:
    """Adapt HybridRetriever output to the pipeline Retriever contract."""

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def retrieve(self, question: str) -> RetrievalResult:
        """Retrieve chunks and convert them to a RetrievalResult."""
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("La question ne peut pas être vide.")

        documents: list[Document] = self.retriever.retrieve(normalized_question)

        chunks = tuple(
            RetrievedChunk(
                chunk_id=self._chunk_id(document),
                content=document.page_content,
                metadata=dict(document.metadata),
                source=str(document.metadata.get("source", "unknown")),
            )
            for document in documents
        )

        return RetrievalResult(
            query=normalized_question,
            chunks=chunks,
            strategy="hybrid",
        )

    @staticmethod
    def _chunk_id(document: Document) -> str:
        """Build a stable identifier from document metadata."""
        source = str(document.metadata.get("source", "unknown"))
        page = str(document.metadata.get("page", "unknown"))
        start_index = str(document.metadata.get("start_index", ""))
        return f"{source}::page={page}::start={start_index}"
