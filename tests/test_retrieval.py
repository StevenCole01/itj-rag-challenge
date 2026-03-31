"""
Tests for semantic retrieval using ChromaDB.
"""

import pytest
import chromadb
from app.rag.retrieval import retrieve_context
from app.rag.embeddings import get_embedding_function


@pytest.fixture
def mock_collection() -> chromadb.Collection:
    """Provides a fresh in-memory Chroma collection with synthetic content for testing."""
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="test_retrieval",
        embedding_function=get_embedding_function()
    )
    
    # Add dummy documents
    collection.add(
        ids=["id1", "id2", "id3"],
        documents=[
            "Information about machine learning models.",
            "A discussion on deep neural networks.",
            "Historical context of artificial intelligence."
        ],
        metadatas=[
            {"source": "paper1.pdf", "page": 1},
            {"source": "paper2.pdf", "page": 2},
            {"source": "paper3.pdf", "page": 3}
        ]
    )
    return collection


def test_retrieve_context_returns_nearest_neighbor(mock_collection: chromadb.Collection) -> None:
    """Verify semantic retrieval matches the closest context chunk."""
    query = "Tell me about neural networks"
    results = retrieve_context(query, mock_collection, k_results=1)
    
    assert len(results) == 1
    assert "neural networks" in results[0]["text"].lower()
    assert results[0]["metadata"]["source"] == "paper2.pdf"


def test_retrieve_context_handles_empty_collection() -> None:
    """Verify retrieval is safe against empty collections."""
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="empty",
        embedding_function=get_embedding_function()
    )
    results = retrieve_context("Any query", collection)
    assert results == []
