"""
Tests for document chunking utilities.
"""

from app.rag.chunking import clean_text, chunk_documents


def test_clean_text_removes_extra_whitespace() -> None:
    """Verify that clean_text collapses whitespace and removes null bytes."""
    raw_text = "This   text\nhas \t extra spaces \x00and nulls. "
    expected = "This text has extra spaces and nulls."
    assert clean_text(raw_text) == expected


def test_chunk_documents_preserves_metadata() -> None:
    """Verify chunk_documents creates proper chunks with IDs and metadata."""
    # Create a long synthetic document
    long_text = "Word " * 500  # Will be definitely long enough to split
    documents = [
        {
            "text": long_text,
            "metadata": {"source": "test_doc.pdf", "page": 1}
        }
    ]
    
    chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=20)
    
    # Verify more than one chunk was created
    assert len(chunks) > 1
    
    # Pick the first chunk to inspect
    first_chunk = chunks[0]
    
    assert "id" in first_chunk
    assert "text" in first_chunk
    assert "metadata" in first_chunk
    
    assert first_chunk["id"] == "test_doc.pdf_p1_c0"
    assert first_chunk["metadata"]["source"] == "test_doc.pdf"
    assert first_chunk["metadata"]["page"] == 1
    assert first_chunk["metadata"]["chunk_index"] == 0
    
    # Spot-check second chunk
    second_chunk = chunks[1]
    assert second_chunk["id"] == "test_doc.pdf_p1_c1"
    assert second_chunk["metadata"]["chunk_index"] == 1


def test_chunk_documents_skips_empty_text() -> None:
    """Verify chunk_documents skips documents that become empty after cleaning."""
    documents = [
        {
            "text": "   \n \t  \x00  ",
            "metadata": {"source": "empty.pdf", "page": 2} # Should be skipped
        },
        {
            "text": "Valid text.",
            "metadata": {"source": "valid.pdf", "page": 1} # Should be kept
        }
    ]
    
    chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=0)
    
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["source"] == "valid.pdf"
