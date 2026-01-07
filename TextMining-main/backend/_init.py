"""
Backend module for Legal RAG system.

This module provides:
- Configuration management
- Document loading and processing
- Embedding models
- Vector store management (FAISS)
- LLM provider abstraction
"""

from backend.config import RAGConfig
from backend.document_loader import (
    load_documents_from_folder,
    load_documents_from_folders,
    load_documents_by_country_and_type,
)
from backend.embeddings import get_embedding_model
from backend.vector_store import (
    build_vector_store,
    load_vector_store,
    clear_vector_store_cache,
    list_vector_stores,
)
from backend.llm_provider import LLMBackend

__all__ = [
    "RAGConfig",
    "load_documents_from_folder",
    "load_documents_from_folders",
    "load_documents_by_country_and_type",
    "get_embedding_model",
    "build_vector_store",
    "load_vector_store",
    "clear_vector_store_cache",
    "list_vector_stores",
    "LLMBackend",
]
