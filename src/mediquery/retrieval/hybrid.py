
"""Hybrid retrieval combining BM25 and vector search with RRF."""

from langchain_core.documents import Document

from mediquery.config import (
    BM25_SEARCH_K,
    HYBRID_SEARCH_K,
    VECTOR_SEARCH_K,
)
from mediquery.retrieval.bm25 import BM25Retriever
from mediquery.retrieval.reranker import CrossEncoderReranker
from mediquery.retrieval.vector_store import get_vector_store


def _document_key(document: Document) -> tuple[str, str, str]:
    """Build a stable key to identify a chunk across retrievers."""
    return (
        str(document.metadata.get("source", "unknown")),
        str(document.metadata.get("page", "unknown")),
        str(document.metadata.get("start_index", "")),
    )


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    *,
    k: int = 60,
) -> list[Document]:
    """Fuse ranked document lists using Reciprocal Rank Fusion."""
    if k <= 0:
        raise ValueError("Le paramètre k doit être strictement positif.")

    scores: dict[tuple[str, str, str], float] = {}
    documents_by_key: dict[tuple[str, str, str], Document] = {}

    for ranked_documents in ranked_lists:
        seen_in_list: set[tuple[str, str, str]] = set()

        for rank, document in enumerate(ranked_documents, start=1):
            key = _document_key(document)

            if key in seen_in_list:
                continue

            seen_in_list.add(key)
            documents_by_key.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ordered_keys = sorted(scores, key=scores.get, reverse=True)
    return [documents_by_key[key] for key in ordered_keys]



class HybridRetriever:
    """Retrieve medical chunks using lexical and semantic search."""

    def __init__(
        self,
        documents: list[Document],
        *,
        bm25_k: int = BM25_SEARCH_K,
        vector_k: int = VECTOR_SEARCH_K,
        result_k: int = HYBRID_SEARCH_K,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        if not documents:
            raise ValueError("La liste de documents hybrides ne peut pas être vide.")
        if result_k <= 0:
            raise ValueError("result_k doit être strictement positif.")

        self.documents = documents
        self.bm25 = BM25Retriever(documents, k=bm25_k)
        self.vector_store = get_vector_store()
        self.vector_k = vector_k
        self.result_k = result_k
        self.reranker = reranker

    def retrieve(self, query: str) -> list[Document]:
        """Return hybrid results, optionally reranked by a cross-encoder."""
        if not query.strip():
            raise ValueError("La requête ne peut pas être vide.")

        bm25_results = self.bm25.retrieve(query)
        vector_results = self.vector_store.similarity_search(
            query,
            k=self.vector_k,
        )

        fused = reciprocal_rank_fusion(
            [
                [document for document, _score in bm25_results],
                vector_results,
            ]
        )

        # Sans reranker, conserver le comportement historique.
        if self.reranker is None:
            return fused[: self.result_k]

        # Le reranker classe les candidats avant la sélection finale.
        return self.reranker.rerank(
            query,
            fused,
        )[: self.result_k]
