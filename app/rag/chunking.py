"""
Document chunking utilities for the RAG system.
Handles cleaning and splitting text.
"""

import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """
    Clean text by normalizing whitespace and removing null characters.
    
    Args:
        text: The raw text string to clean.
        
    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
        
    # Remove null characters
    text = text.replace("\x00", "")
    
    # Collapse multiple whitespace characters into a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def get_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """
    Create a configured text splitter for PDF document chunking.
    
    Args:
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


def chunk_documents(
    documents: list[dict[str, Any]], 
    chunk_size: int, 
    chunk_overlap: int
) -> list[dict[str, Any]]:
    """
    Clean and split page documents into smaller chunks.
    
    Args:
        documents: List of page dictionaries from loaders.py.
        chunk_size: Target size for output chunks.
        chunk_overlap: Number of overlapping characters between chunks.
        
    Returns:
        List of chunk dictionaries with unique IDs, text, and expanded metadata.
    """
    splitter = get_text_splitter(chunk_size, chunk_overlap)
    chunks = []
    
    for doc in documents:
        # Clean text
        cleaned_text = clean_text(doc.get("text", ""))
        
        if not cleaned_text:
            continue
            
        # Split text into chunks
        text_chunks = splitter.split_text(cleaned_text)
        
        source = doc["metadata"]["source"]
        page = doc["metadata"]["page"]
        
        for index, text_chunk in enumerate(text_chunks):
            # Create unique chunk ID
            chunk_id = f"{source}_p{page}_c{index}"
            
            # Build chunk metadata
            chunk_metadata = {
                "source": source,
                "page": page,
                "chunk_index": index
            }
            
            chunks.append({
                "id": chunk_id,
                "text": text_chunk,
                "metadata": chunk_metadata
            })
            
    return chunks
