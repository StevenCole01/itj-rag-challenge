"""
API routes for the RAG system.
"""

from fastapi import APIRouter, HTTPException
from app.schemas import QueryRequest, QueryResponse, SourceCitation
from app.rag.vectorstore import init_vectorstore
from app.rag.retrieval import retrieve_context
from app.rag.generation import generate_answer

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Handle a RAG query by retrieving context and generating an answer.
    """
    try:
        # Initialize vectorstore
        _, collection = init_vectorstore()
        
        # Retrieve context
        context_chunks = retrieve_context(request.query, collection, k_results=request.top_k)
        
        # Generate answer
        answer = generate_answer(request.query, context_chunks)
        
        # Format sources
        sources = [
            SourceCitation(
                source=chunk["metadata"]["source"],
                page=chunk["metadata"]["page"],
                text=chunk["text"]
            )
            for chunk in context_chunks
        ]
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}
