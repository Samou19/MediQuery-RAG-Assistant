
"""Rerank retrieved medical documents with a multilingual cross-encoder."""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """Rerank candidate documents according to their relevance to a query."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k doit être strictement positif.")

        self.top_k = top_k
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:
        """Return the most relevant documents in reranked order."""
        if not query.strip():
            raise ValueError("La requête ne peut pas être vide.")

        if not documents:
            return []

        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [
            document
            for document, _score in ranked[: self.top_k]
        ]

    
    def score_documents(
        self,
        query: str,
        documents: list[Document],
    ) -> list[tuple[Document, float]]:
        """Return candidate documents with their cross-encoder scores."""
        if not query.strip():
            raise ValueError("La requête ne peut pas être vide.")

        if not documents:
            return []

        pairs = [(query, document.page_content) for document in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [(document, float(score)) for document, score in ranked]


def select_diverse_pages(
    ranked_documents: list[Document],
    top_k: int = 5,
) -> list[Document]:
    """Select reranked chunks while keeping at most one chunk per page."""
    if top_k <= 0:
        raise ValueError("top_k doit être strictement positif.")

    selected: list[Document] = []
    seen_pages: set[tuple[str, str]] = set()

    for document in ranked_documents:
        source = str(document.metadata.get("source", "unknown"))
        page = document.metadata.get("page")

        # Si la page est absente, conserver le document sans déduplication.
        if page is None:
            selected.append(document)
        else:
            key = (source, str(page))
            if key in seen_pages:
                continue

            seen_pages.add(key)
            selected.append(document)

        if len(selected) >= top_k:
            break

    return selected
