
"""ChromaDB access layer for MediQuery."""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from mediquery.config import CHROMA_DIR
from mediquery.ingestion.indexer import COLLECTION_NAME, create_embeddings


def get_vector_store(
    embeddings: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    """Return the existing MediQuery vector store.

    The collection must already exist. This function must not create
    a new empty collection accidentally.
    """
    embedding_function = embeddings or create_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=str(CHROMA_DIR),
        create_collection_if_not_exists=False,
    )
