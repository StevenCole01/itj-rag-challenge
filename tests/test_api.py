"""
Integration tests for the FastAPI RAG endpoints.
"""

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)


def test_health_check():
    """Verify health endpoint is reachable."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_welcome():
    """Verify root endpoint returns welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]


@patch("app.api.routes.init_vectorstore")
@patch("app.api.routes.retrieve_context")
@patch("app.api.routes.generate_answer")
def test_query_endpoint_success(
    mock_generate: MagicMock, 
    mock_retrieve: MagicMock, 
    mock_init: MagicMock
):
    """Verify full retrieval and generation cycle through the API."""
    # Setup mocks
    mock_init.return_value = (MagicMock(), MagicMock())
    mock_retrieve.return_value = [
        {"text": "Sample context", "metadata": {"source": "p1.pdf", "page": 1}}
    ]
    mock_generate.return_value = "Mocked LLM answer"
    
    # Send query
    payload = {"query": "What is AI?", "top_k": 1}
    response = client.post("/api/v1/query", json=payload)
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Mocked LLM answer"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "p1.pdf"
    assert data["sources"][0]["text"] == "Sample context"
