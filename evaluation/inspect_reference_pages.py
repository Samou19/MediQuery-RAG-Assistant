
"""Inspect the reference pages used to evaluate MediQuery retrieval."""

import json
from pathlib import Path

from mediquery.retrieval.vector_store import get_vector_store

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERIES_FILE = PROJECT_ROOT / "evaluation" / "queries.json"


def main() -> None:
    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    db = get_vector_store()

    pages = sorted(
        {
            page
            for query in queries
            for page in query["relevant_pages"]
        }
    )

    print(f"Pages à vérifier : {pages}")

    for page_number in pages:
        result = db._collection.get(
            where={"page": page_number},
            include=["documents", "metadatas"],
        )

        print(f"\n{'=' * 80}")
        print(f"PAGE PDF : {page_number} | Chunks trouvés : {len(result['ids'])}")
        print("=" * 80)

        for document, metadata in zip(
            result["documents"],
            result["metadatas"],
            strict=True,
        ):
            print(f"\nChunk {metadata.get('chunk_index')} :")
            print((document or "")[:700].replace("\n", " "))
            print("-" * 40)


if __name__ == "__main__":
    main()
