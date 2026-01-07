# backend/rag_pipeline.py
from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any

from langchain_core.documents import Document

from .config import RAGConfig
from .rag_single_agent import single_agent_answer_question
from .rag_multiagent import multiagent_answer_question
# RIMOSSO: from .hybrid_rag import hybrid_answer_question  <-- Cancellare questa riga


def answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str], Optional[Dict[str, Any]]]:
    """
    Public entrypoint used by the Chatbot interface.
    """
    
    # Multi-agent mode
    if getattr(config, "use_multiagent", False):
        answer, docs, reasoning = multiagent_answer_question(
            question, config, show_reasoning
        )
        return answer, docs, reasoning, None
    
    # Single-agent mode (ReAct)
    answer, docs, reasoning = single_agent_answer_question(
        question, config, show_reasoning
    )
    return answer, docs, reasoning, None
