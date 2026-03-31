"""
Main entrypoint for the FastAPI application.
"""

import uvicorn
from fastapi import FastAPI
from app.api.routes import router as rag_router

# Initialize FastAPI app
app = FastAPI(
    title="arXiv RAG System",
    description="A semantic search and generation API for research papers.",
    version="1.0.0"
)

# Health check at the root
@app.get("/")
def read_root():
    """Simple root welcome message."""
    return {"message": "Welcome to the arXiv RAG API."}


# Include RAG-specific routes
app.include_router(rag_router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
