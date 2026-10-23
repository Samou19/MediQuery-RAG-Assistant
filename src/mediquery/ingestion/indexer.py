
"""Index document chunks in a persistent ChromaDB collection."""

import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from mediquery.config import CHROMA_DIR, EMBEDDING_MODEL

COLLECTION_NAME = "medical_docs"


def create_embeddings() -> HuggingFaceEmbeddings:
    """Create the configured Hugging Face embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_chunk_id(document: Document) -> str:
    """Generate a deterministic ID from the source, page and chunk content."""
    source = str(document.metadata.get("source", "unknown"))
    page = str(document.metadata.get("page", "unknown"))
    content_hash = hashlib.sha256(
        document.page_content.encode("utf-8")
    ).hexdigest()

    identity = f"{source}|{page}|{content_hash}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def index_documents(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """Add documents to the persistent collection using stable IDs."""
    if not documents:
        raise ValueError("La liste de documents à indexer est vide.")

    embedding_function = embeddings or create_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=str(CHROMA_DIR),
    )

    ids = [get_chunk_id(document) for document in documents]

    # upsert avoids duplicates when the same IDs are indexed again.
    vector_store.add_documents(documents=documents, ids=ids)

    return vector_store
