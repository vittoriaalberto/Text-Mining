from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import RAGConfig


def get_embedding_model(config: RAGConfig) -> Embeddings:
    """
    Returns a LangChain Embeddings object based on config.

    - embedding_provider == "huggingface":
        Uses HuggingFaceEmbeddings with device forced to CPU to avoid
        issues like "Cannot copy out of meta tensor; no data!" on some setups.
        For private/gated models, set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN.
    """
    
    # Default: Hugging Face embeddings on CPU
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model_name,
        model_kwargs={"device": "cpu"},              # Force CPU to avoid tensor issues
        encode_kwargs={"normalize_embeddings": True}, # Normalize for cosine similarity
    )
