"""
Tests for ChromaDB vector operations.
"""

import pytest
import chromadb

from app.rag.vectorstore import add_chunks_to_vectorstore
from app.rag.embeddings import get_embedding_function


@pytest.fixture
def ephemeral_collection() -> chromadb.Collection:
    """Provides a fresh, temporary in-memory Chroma collection for testing."""
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="test_collection",
        embedding_function=get_embedding_function()
    )
    return collection


def test_add_chunks_upserts_correctly(ephemeral_collection: chromadb.Collection) -> None:
    """Verifies dictionaries can be mapped cleanly into Chroma payloads."""
    chunks = [
        {
            "id": "doc1_p1_c0",
            "text": "This is entirely test content A.",
            "metadata": {"source": "doc1.pdf", "page": 1, "chunk_index": 0}
        },
        {
            "id": "doc1_p1_c1",
            "text": "This is test content B.",
            "metadata": {"source": "doc1.pdf", "page": 1, "chunk_index": 1}
        }
    ]
    
    add_chunks_to_vectorstore(chunks, ephemeral_collection)
    
    # Check item count was incremented
    assert ephemeral_collection.count() == 2
    
    # Query back 
    results = ephemeral_collection.get(ids=["doc1_p1_c0"])
    
    # Validate raw properties
    assert results["ids"][0] == "doc1_p1_c0"
    assert results["documents"][0] == "This is entirely test content A."
    assert results["metadatas"][0]["source"] == "doc1.pdf"
