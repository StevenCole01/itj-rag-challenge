"""
Ingestion script to process PDFs and populate the vector database.
"""

import sys
from pathlib import Path

# Add the project root to the python path so it can find the 'app' module
sys.path.append(str(Path(__file__).parent.parent))

from app.rag.loaders import load_documents_from_directory
from app.rag.chunking import chunk_documents
from app.rag.vectorstore import init_vectorstore, add_chunks_to_vectorstore


def main():
    data_dir = "./data"
    persist_dir = "./chroma_db"
    
    print(f"--- Starting ingestion from {data_dir} ---")
    
    try:
        # 1. Load documents
        print(f"Loading PDFs from {data_dir}...")
        pages = load_documents_from_directory(data_dir)
        if not pages:
            print("No PDF files found in data directory. Please add some papers and try again.")
            return
        print(f"Extracted {len(pages)} pages.")
        
        # 2. Chunk documents
        print("Chunking documents...")
        # Using standard 1000/200 split for research papers
        chunks = chunk_documents(pages, chunk_size=1000, chunk_overlap=200)
        print(f"Generated {len(chunks)} chunks.")
        
        # 3. Initialize vector store
        print(f"Initializing vector store at {persist_dir}...")
        _, collection = init_vectorstore(persist_directory=persist_dir)
        
        # 4. Add chunks to vector store
        print("Adding chunks to ChromaDB (this may take a few moments for embeddings)...")
        add_chunks_to_vectorstore(chunks, collection)
        
        print("--- Ingestion complete! ---")
        print(f"Vector store is now populated and ready at {persist_dir}.")
        
    except Exception as e:
        print(f"Error during ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
