"""
Generation logic for the RAG system using OpenAI Chat Completion.
"""

import os
from typing import Any, List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()


def generate_answer(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate an answer to a query based on the provided context chunks.
    
    Args:
        query: The user's original question.
        context_chunks: List of retrieved chunks from the vector store.
        
    Returns:
        The generated answer string from the LLM.
    """
    if not context_chunks:
        return "I'm sorry, I couldn't find any relevant information in the documents to answer that question."

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Construct context string
    context_str = "\n\n---\n\n".join([chunk["text"] for chunk in context_chunks])
    
    # Define prompt
    system_prompt = (
        "You are a helpful assistant specialized in answering questions based on research papers. "
        "Use the provided context sections to answer the question. "
        "If the answer is not contained within the context, say that you do not know since the information is not present in the documents. "
        "Do not use outside knowledge. "
        "Keep your answer concise and accurate."
    )
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during generation: {str(e)}"
