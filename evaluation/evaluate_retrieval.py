
"""Evaluate retrieval strategies using page-level Recall@k and MRR@k."""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_core.documents import Document

from mediquery.config import BM25_SEARCH_K, VECTOR_SEARCH_K
from mediquery.retrieval.bm25 import BM25Retriever
from mediquery.retrieval.hybrid import HybridRetriever
from mediquery.retrieval.reranker import (
    CrossEncoderReranker,
    select_diverse_pages,
)
from mediquery.retrieval.vector_store import get_vector_store

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERIES_FILE = PROJECT_ROOT / "evaluation" / "queries.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

CUTOFF = 5
RERANKER_CANDIDATES_K = 15

logger = logging.getLogger(__name__)

RetrieverFunction = Callable[[str], list[Document]]


def load_queries() -> list[dict[str, Any]]:
    """Load and validate the evaluation questions."""
    if not QUERIES_FILE.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {QUERIES_FILE}")

    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))

    if not isinstance(queries, list) or not queries:
        raise ValueError("Le jeu d'évaluation doit contenir une liste non vide.")

    for query in queries:
        if not isinstance(query, dict):
            raise ValueError("Chaque élément d'évaluation doit être un objet JSON.")

        if not query.get("question", "").strip():
            raise ValueError("Chaque question doit contenir un texte non vide.")

        if not query.get("relevant_pages"):
            raise ValueError(
                f"Pages pertinentes manquantes pour la question {query.get('id')}."
            )

    return queries


def unique_pages(documents: list[Document], k: int = CUTOFF) -> list[int]:
    """Return distinct page numbers in retrieval order, up to k pages."""
    if k <= 0:
        raise ValueError("k doit être strictement positif.")

    pages: list[int] = []
    seen: set[int] = set()

    for document in documents:
        page = document.metadata.get("page")
        if page is None:
            continue

        page_number = int(page)

        if page_number not in seen:
            seen.add(page_number)
            pages.append(page_number)

        if len(pages) >= k:
            break

    return pages


def recall_at_k(retrieved_pages: list[int], relevant_pages: list[int]) -> float:
    """Compute page-level recall over the retrieved distinct pages."""
    relevant = set(relevant_pages)

    if not relevant:
        return 0.0

    return len(set(retrieved_pages) & relevant) / len(relevant)


def reciprocal_rank_at_k(
    retrieved_pages: list[int],
    relevant_pages: list[int],
) -> float:
    """Compute reciprocal rank of the first relevant page."""
    relevant = set(relevant_pages)

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant:
            return 1.0 / rank

    return 0.0


def load_documents_from_chroma() -> list[Document]:
    """Read indexed chunks without modifying the ChromaDB collection."""
    db = get_vector_store()
    result = db._collection.get(include=["documents", "metadatas"])

    contents = result["documents"] or []
    metadatas = result["metadatas"] or []

    documents: list[Document] = []

    for content, metadata in zip(contents, metadatas, strict=True):
        if content is not None:
            documents.append(
                Document(
                    page_content=content,
                    metadata=metadata or {},
                )
            )

    if not documents:
        raise ValueError("Aucun document trouvé dans la collection ChromaDB.")

    return documents


def evaluate_strategy(
    strategy_name: str,
    queries: list[dict[str, Any]],
    retrieve: RetrieverFunction,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Evaluate one strategy and return per-question and aggregate metrics."""
    rows: list[dict[str, Any]] = []

    for item in queries:
        question = item["question"]
        relevant_pages = [int(page) for page in item["relevant_pages"]]

        documents = retrieve(question)
        pages = unique_pages(documents, k=CUTOFF)

        rows.append(
            {
                "strategy": strategy_name,
                "query_id": item["id"],
                "question": question,
                "relevant_pages": relevant_pages,
                "retrieved_pages_at_5": pages,
                "recall_at_5": recall_at_k(pages, relevant_pages),
                "mrr_at_5": reciprocal_rank_at_k(pages, relevant_pages),
            }
        )

    metrics = {
        "recall_at_5": mean(row["recall_at_5"] for row in rows),
        "mrr_at_5": mean(row["mrr_at_5"] for row in rows),
    }

    return rows, metrics


def main() -> None:
    """Run the retrieval evaluation and save its results."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    queries = load_queries()
    logger.info("Questions chargées : %d", len(queries))

    documents = load_documents_from_chroma()
    logger.info("Chunks lus depuis ChromaDB : %d", len(documents))

    db = get_vector_store()

    bm25 = BM25Retriever(
        documents,
        k=max(BM25_SEARCH_K, RERANKER_CANDIDATES_K),
    )

    # Référence hybride : BM25 + recherche vectorielle + fusion RRF.
    hybrid = HybridRetriever(
        documents,
        bm25_k=max(BM25_SEARCH_K, RERANKER_CANDIDATES_K),
        vector_k=max(VECTOR_SEARCH_K, RERANKER_CANDIDATES_K),
        result_k=RERANKER_CANDIDATES_K,
    )

    # Variante avec reranking.
    reranker = CrossEncoderReranker(top_k=RERANKER_CANDIDATES_K)

    hybrid_reranked = HybridRetriever(
        documents,
        bm25_k=max(BM25_SEARCH_K, RERANKER_CANDIDATES_K),
        vector_k=max(VECTOR_SEARCH_K, RERANKER_CANDIDATES_K),
        result_k=RERANKER_CANDIDATES_K,
        reranker=reranker,
    )

    # Variante avec sélection diversifiée après le reranking.
    def retrieve_hybrid_reranked_diverse(
        question: str,
    ) -> list[Document]:
        """Return reranked results with at most one chunk per source page."""
        candidates = hybrid_reranked.retrieve(question)
        return select_diverse_pages(candidates, top_k=CUTOFF)

    strategies: dict[str, RetrieverFunction] = {
        "vector": lambda question: db.similarity_search(
            question,
            k=max(VECTOR_SEARCH_K, RERANKER_CANDIDATES_K),
        ),
        "bm25": lambda question: [
            document
            for document, _score in bm25.retrieve(question)
        ],
        "hybrid": lambda question: hybrid.retrieve(question),
        "hybrid_reranked": lambda question: hybrid_reranked.retrieve(question),
        "hybrid_reranked_diverse": retrieve_hybrid_reranked_diverse,
    }

    all_rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, float]] = {}

    for name, retrieve in strategies.items():
        logger.info("Évaluation de la stratégie : %s", name)
        rows, metrics = evaluate_strategy(name, queries, retrieve)
        all_rows.extend(rows)
        summary[name] = metrics

    # Écarts par rapport à la recherche hybride de référence.
    if "hybrid" in summary:
        for strategy_name in ("hybrid_reranked", "hybrid_reranked_diverse"):
            if strategy_name in summary:
                summary[f"{strategy_name}_delta_vs_hybrid"] = {
                    metric: summary[strategy_name][metric] - summary["hybrid"][metric]
                    for metric in ("recall_at_5", "mrr_at_5")
                }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    details_path = RESULTS_DIR / "retrieval_details.json"
    details_path.write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path = RESULTS_DIR / "retrieval_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Résultats moyens ===")
    print(f"{'Stratégie':<30} {'Recall@5':>12} {'MRR@5':>12}")
    print("-" * 56)

    for name, metrics in summary.items():
        if "_delta_vs_hybrid" in name:
            continue

        print(
            f"{name:<30} "
            f"{metrics['recall_at_5']:>12.3f} "
            f"{metrics['mrr_at_5']:>12.3f}"
        )

    print("\n=== Écarts par rapport à Hybrid ===")
    for name, metrics in summary.items():
        if "_delta_vs_hybrid" in name:
            print(f"\n{name}")
            print(f"  Delta Recall@5 : {metrics['recall_at_5']:+.3f}")
            print(f"  Delta MRR@5    : {metrics['mrr_at_5']:+.3f}")

    print(f"\nDétails : {details_path}")
    print(f"Synthèse : {summary_path}")


if __name__ == "__main__":
    main()
