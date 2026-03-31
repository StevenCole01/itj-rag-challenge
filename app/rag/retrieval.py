"""
Retrieval logic for the RAG system using raw ChromaDB queries.
"""

from typing import Any, List, Dict
import chromadb


def retrieve_context(
    query: str, 
    collection: chromadb.Collection, 
    k_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant context chunks for a given query.
    
    Args:
        query: The user's natural language question.
        collection: The ChromaDB collection to search.
        k_results: Number of top relevant results to return.
        
    Returns:
        A list of dictionaries, each containing:
            - "text": The chunk text.
            - "metadata": The chunk's original metadata (source, page, etc.).
    """
    # Chroma handles embedding the query_texts using its configured embedding function
    results = collection.query(
        query_texts=[query],
        n_results=k_results
    )
    
    retrieved_chunks = []
    
    # Chroma returns lists of lists (one list per input query)
    if results["documents"]:
        for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
            retrieved_chunks.append({
                "text": text,
                "metadata": metadata
            })
            
    return retrieved_chunks
