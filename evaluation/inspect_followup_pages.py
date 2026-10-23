
"""Inspect PDF chunks for pages related to patient follow-up."""

from mediquery.retrieval.vector_store import get_vector_store


def main() -> None:
    db = get_vector_store()
    pages = [35, 36, 37, 38, 39]

    for page_number in pages:
        result = db._collection.get(
            where={"page": page_number},
            include=["documents", "metadatas"],
        )

        print(f"\n{'=' * 80}")
        print(f"PAGE {page_number} — {len(result['ids'])} chunks")
        print("=" * 80)

        for document, metadata in zip(
            result["documents"],
            result["metadatas"],
            strict=True,
        ):
            print(f"\nChunk {metadata.get('chunk_index')}:")
            print((document or "")[:1000].replace("\n", " "))
            print("-" * 40)


if __name__ == "__main__":
    main()
