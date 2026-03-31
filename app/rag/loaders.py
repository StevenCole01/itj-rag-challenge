"""
Document loaders for the RAG system.
Handles loading and extracting text from source documents.
"""

from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader


def list_pdf_files(data_dir: str) -> list[Path]:
    """
    List all PDF files in the specified directory.
    
    Args:
        data_dir: Path to the directory containing PDF files.
        
    Returns:
        List of Path objects for each discovered PDF file, sorted by name.
    """
    dir_path = Path(data_dir)
    
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {data_dir}")
        
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {data_dir}")
        
    pdf_files = list(dir_path.glob("*.pdf"))
    return sorted(pdf_files)


def extract_pages_from_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """
    Extract text from a PDF file page by page.
    
    Args:
        pdf_path: Path to the PDF file to load.
        
    Returns:
        A list of dictionaries, each containing:
            - "text": The text content of the page.
            - "metadata": A dictionary with "source" (filename) and "page" (1-indexed).
    """
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    
    extracted_pages = []
    
    # PyPDFLoader usually 0-indexes metadata['page']
    for i, doc in enumerate(docs):
        text = doc.page_content.strip()
        
        if not text:
            # Skip empty pages
            continue
            
        extracted_pages.append({
            "text": text,
            "metadata": {
                "source": pdf_path.name,
                "page": i + 1  # 1-indexed page number
            }
        })
        
    return extracted_pages


def load_documents_from_directory(data_dir: str) -> list[dict[str, Any]]:
    """
    Load all PDF documents from a directory and extract their pages.
    
    Args:
        data_dir: Directory containing the PDF files.
        
    Returns:
        A combined list of dictionaries containing text and metadata for all pages.
    """
    pdf_files = list_pdf_files(data_dir)
    all_pages = []
    
    for pdf_path in pdf_files:
        try:
            pages = extract_pages_from_pdf(pdf_path)
            all_pages.extend(pages)
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF {pdf_path}: {e}") from e
            
    return all_pages
