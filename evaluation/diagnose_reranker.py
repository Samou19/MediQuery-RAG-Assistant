
"""Inspect cross-encoder scores for selected evaluation questions."""

from evaluation.evaluate_retrieval import load_documents_from_chroma, load_queries

from mediquery.config import BM25_SEARCH_K, VECTOR_SEARCH_K
from mediquery.retrieval.hybrid import HybridRetriever
from mediquery.retrieval.reranker import CrossEncoderReranker

DIAGNOSTIC_IDS = {"Q01", "Q05"}
CANDIDATES_K = 15


def main() -> None:
    queries = load_queries()
    documents = load_documents_from_chroma()

    hybrid = HybridRetriever(
        documents,
        bm25_k=max(BM25_SEARCH_K, CANDIDATES_K),
        vector_k=max(VECTOR_SEARCH_K, CANDIDATES_K),
        result_k=CANDIDATES_K,
    )
    reranker = CrossEncoderReranker(top_k=CANDIDATES_K)

    for item in queries:
        if item["id"] not in DIAGNOSTIC_IDS:
            continue

        question = item["question"]
        relevant_pages = {int(page) for page in item["relevant_pages"]}
        candidates = hybrid.retrieve(question)
        ranked = reranker.score_documents(question, candidates)

        print(f"\n{'=' * 90}")
        print(f"{item['id']} — {question}")
        print(f"Pages de référence : {sorted(relevant_pages)}")

        for rank, (document, score) in enumerate(ranked, start=1):
            page = document.metadata.get("page", "?")
            marker = "*" if page != "?" and int(page) in relevant_pages else " "
            excerpt = " ".join(document.page_content.split())[:240]

            print(
                f"{rank:02d}. [{marker}] page={page} "
                f"score={score:.4f} | {excerpt}"
            )


if __name__ == "__main__":
    main()
