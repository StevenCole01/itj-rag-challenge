# ITJ RAG Challenge — arXiv Research Assistant

A modular Retrieval-Augmented Generation (RAG) system built to ingest local arXiv research papers, store their semantic representations, and answer queries with explicit source citations.

## System Architecture

The project is structured into distinct, decoupled components:

1. **Ingestion Pipeline (`scripts/ingest.py`)**
   - **Load:** Reads local PDF files from the `./data` directory using `PyPDFLoader`.
   - **Chunk:** Splits text into 1000-character chunks with a 200-character overlap using `RecursiveCharacterTextSplitter` to preserve context boundaries.
   - **Embed & Store:** Converts text chunks into dense vector embeddings using a local HuggingFace model (`all-MiniLM-L6-v2`) and stores them persistently in ChromaDB (`./chroma_db`).

2. **Backend API (`app/`)**
   - Built with **FastAPI** to provide a fast, asynchronous interface.
   - Exposes a `/api/v1/query` endpoint that handles semantic retrieval against ChromaDB and natural language generation via the OpenAI API (`gpt-4o-mini`).

3. **Frontend UI (`ui/streamlit_app.py`)**
   - A lightweight **Streamlit** chat interface.
   - Communicates with the FastAPI backend via HTTP requests, displaying generated answers alongside collapsible source citations (PDF name and page number).

4. **Test Suite (`tests/`)**
   - Unit and integration tests covering chunking logic, vector store operations, semantic retrieval, and the API layer.
   - External dependencies (OpenAI, ChromaDB) are mocked where appropriate, so tests run without any live services.

## Project Structure

```
.
├── app/
│   ├── api/
│   │   └── routes.py         # FastAPI route definitions
│   ├── rag/
│   │   ├── chunking.py       # Text cleaning and splitting
│   │   ├── embeddings.py     # HuggingFace embedding function
│   │   ├── generation.py     # OpenAI answer generation
│   │   ├── loaders.py        # PDF ingestion via PyPDFLoader
│   │   ├── retrieval.py      # ChromaDB semantic search
│   │   └── vectorstore.py    # ChromaDB client and upsert logic
│   ├── main.py               # FastAPI app entrypoint
│   └── schemas.py            # Pydantic request/response models
├── scripts/
│   └── ingest.py             # Pipeline orchestration script
├── tests/
│   ├── test_api.py           # API integration tests
│   ├── test_chunking.py      # Chunking unit tests
│   ├── test_retrieval.py     # Retrieval unit tests
│   └── test_vectorstore.py   # Vector store unit tests
├── ui/
│   └── streamlit_app.py      # Streamlit chat interface
├── data/                     # PDF source documents (not tracked)
├── chroma_db/                # Persisted vector database (not tracked)
├── .env.example              # Environment variable template
└── requirements.txt          # Python dependencies
```

## Setup Instructions

### Prerequisites
* Python 3.11+
* An OpenAI API Key

### 1. Installation

Clone the repository and install the dependencies:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` and fill in the OpenAI API key:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your-api-key-here
```

### 3. Running the System

Running the full stack requires three steps across two separate terminal windows.

**Step A: Ingest Documents**

Place arXiv PDFs directly into the `./data` directory (subdirectories are not scanned), then run the ingestion script to populate the vector database:
```bash
python scripts/ingest.py
```
> *(Note: The first run automatically downloads the local embedding model from Hugging Face).*

**Step B: Start the Backend API**

In the first terminal, start the FastAPI server:
```bash
python -m app.main
```
The API becomes available at `http://localhost:8000`.

**Step C: Start the Frontend UI**

In a new terminal (ensuring the virtual environment is activated), run the Streamlit app:
```bash
streamlit run ui/streamlit_app.py
```
The UI should open automatically at `http://localhost:8501`.

## Running Tests

No external services or running API are required, as external dependencies are mocked. The test suite covers chunking, vector store operations, retrieval, and API integration.

```bash
python -m pytest tests/
```

## Testing the API with curl

With the backend running, the `/api/v1/query` endpoint can be tested directly:

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the main findings?", "top_k": 5}'
```

The health check endpoint can also be used to verify the server is up:

```bash
curl http://localhost:8000/api/v1/health
```

## Key Technical Decisions & Trade-offs


* **Raw Database Client vs. High-Level Abstractions:** The project utilizes raw `ChromaDB` client methods rather than relying entirely on LangChain’s vector store wrappers. This provides greater transparency, explicit control over the insertion/retrieval processes, and demonstrates core database indexing concepts without 'magic' abstractions.
* **Hybrid Embedding/Generation Strategy:** A hybrid approach combines local HuggingFace embeddings (`all-MiniLM-L6-v2`) via `sentence-transformers` with OpenAI (`gpt-4o-mini`) for the final text generation.
  * *Trade-off:* This requires downloading a local model initially, but avoids runtime API costs for embeddings and keeps the document vectors private. OpenAI is leveraged for world-class text synthesis to ensure high-quality answers.
* **Separation of Concerns:** The Streamlit user interface connects to the core RAG logic purely over HTTP (calling the FastAPI backend).
  * *Trade-off:* Adds slight network latency even when running locally, but perfectly mimics a modern microservices architecture, allowing the frontend and backend to scale or be deployed independently.
* **Deterministic Chunk IDs:** Chunks are assigned unique, deterministic IDs based on their source file, page, and chunk index (e.g., `paper_p1_c0`). This ensures idempotent database updates—running the ingestion script multiple times updates existing chunks cleanly without duplicating data.
