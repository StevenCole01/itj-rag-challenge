"""
Embedding configurations for the RAG system.
"""

from typing import Any
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def get_embedding_function() -> Any:
    """
    Get the configured embedding function for ChromaDB.
    
    Uses HuggingFace's all-MiniLM-L6-v2 for local embeddings.

    Returns:
        A ChromaDB compatible embedding function instance.
    """
    # This automatically downloads the model weights on first run
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
