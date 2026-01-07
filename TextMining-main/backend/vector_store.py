from __future__ import annotations
from pathlib import Path
from typing import List
import os
import shutil

from langchain_core.documents import Document  
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

# Role of this module:
# This is the vector database layer that builds and loads FAISS vector stores.
#
# - During offline step: called by build scripts to create FAISS DBs
# - During online step: called by RAG pipelines to load the correct DB and create retrievers


# Simple in-memory cache: {path -> FAISS vector store}
_VECTOR_STORE_CACHE: dict[str, FAISS] = {}


def build_vector_store(
    docs: List[Document],
    embedding_model: Embeddings,
    target_dir: str,
) -> None:
    """
    Build a FAISS vector store from documents and save to disk.
    
    Args:
        docs: List of LangChain Document objects
        embedding_model: Embeddings model to use
        target_dir: Directory where to save the FAISS index
    """
    if not docs:
        print(f"[vector_store] No documents provided for {target_dir}")
        return
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"[vector_store] Building FAISS index with {len(docs)} documents...")
    
    # Create FAISS vector store
    vs = FAISS.from_documents(docs, embedding_model)
    
    # Save to disk
    vs.save_local(target_dir)
    
    # Cache in memory
    _VECTOR_STORE_CACHE[target_dir] = vs
    
    print(f"[vector_store] Vector store saved to: {target_dir}")


def load_vector_store(
    path: str,
    embedding_model: Embeddings,
) -> FAISS:
    """
    Load a FAISS vector store from disk.
    Uses in-memory cache to avoid reloading.
    
    Args:
        path: Directory containing the FAISS index
        embedding_model: Embeddings model (must match the one used to build)
    
    Returns:
        FAISS vector store object
    """
    # Check cache first
    cached = _VECTOR_STORE_CACHE.get(path)
    if cached is not None:
        print(f"[vector_store] Using cached vector store: {path}")
        return cached
    
    # Load from disk
    print(f"[vector_store] Loading vector store from: {path}")
    vs = FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    
    # Cache it
    _VECTOR_STORE_CACHE[path] = vs
    
    return vs


def clear_vector_store_cache(path: str = None) -> None:
    """
    Clear vector store from in-memory cache and optionally delete from disk.
    
    Args:
        path: Path to vector store. If None, clears entire cache.
    """
    if path is None:
        # Clear entire cache
        _VECTOR_STORE_CACHE.clear()
        print("[vector_store] Cleared entire cache")
    else:
        # Clear specific path
        if path in _VECTOR_STORE_CACHE:
            del _VECTOR_STORE_CACHE[path]
            print(f"[vector_store] Cleared cache for: {path}")
        
        # Optionally delete from disk
        if os.path.isdir(path):
            response = input(f"Delete {path} from disk? (y/n): ")
            if response.lower() == 'y':
                shutil.rmtree(path)
                print(f"[vector_store] Deleted from disk: {path}")


def get_vector_store_info(path: str) -> dict:
    """
    Get information about a vector store.
    
    Args:
        path: Path to vector store directory
    
    Returns:
        Dictionary with info (exists, num_docs, etc.)
    """
    info = {
        "path": path,
        "exists": os.path.isdir(path),
        "cached": path in _VECTOR_STORE_CACHE,
    }
    
    if info["exists"]:
        # Check if FAISS files exist
        faiss_index = os.path.join(path, "index.faiss")
        faiss_pkl = os.path.join(path, "index.pkl")
        info["valid"] = os.path.exists(faiss_index) and os.path.exists(faiss_pkl)
    else:
        info["valid"] = False
    
    return info


def list_vector_stores(base_dir: str = "vector_store") -> List[str]:
    """
    List all vector stores in the base directory.
    
    Args:
        base_dir: Base directory containing vector stores
    
    Returns:
        List of vector store paths
    """
    if not os.path.exists(base_dir):
        return []
    
    stores = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check if it's a valid FAISS store
            if os.path.exists(os.path.join(item_path, "index.faiss")):
                stores.append(item_path)
    
    return stores
