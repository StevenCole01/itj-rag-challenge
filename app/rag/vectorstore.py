"""
Vector database management using raw ChromaDB client.
"""

from typing import Any
import chromadb

from app.rag.embeddings import get_embedding_function


def init_vectorstore(persist_directory: str = "./chroma_db", collection_name: str = "rag_collection") -> tuple[chromadb.ClientAPI, chromadb.Collection]:
    """
    Initialize and return the ChromaDB client and default collection.
    
    Args:
        persist_directory: Path to store SQLite database files.
        collection_name: Name of the vector collection to use.
        
    Returns:
        A tuple of (Chroma Client instance, Chroma Collection instance).
    """
    # Create persistent client targeting the local directory
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Get or create collection with our predefined embedding function
    embedding_func = get_embedding_function()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    
    return client, collection


def add_chunks_to_vectorstore(chunks: list[dict[str, Any]], collection: chromadb.Collection, batch_size: int = 500) -> None:
    """
    Add chunk dictionaries to a ChromaDB collection.
    
    Args:
        chunks: List of chunk dictionaries containing 'id', 'text', and 'metadata'.
        collection: Transformed instances will be added to this Chroma vector store collection.
        batch_size: Number of records to add per single ChromaDB insertion call.
    """
    if not chunks:
        return
        
    ids = []
    documents = []
    metadatas = []
    
    # Pre-process raw chunk dictionaries into flat lists required by Chroma
    for chunk in chunks:
        ids.append(chunk["id"])
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])
        
    # Batch inserts to avoid over-submitting payloads
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_documents = documents[i : i + batch_size]
        batch_metadatas = metadatas[i : i + batch_size]
        
        # We upsert to gracefully handle exact-duplicate inserts
        collection.upsert(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
