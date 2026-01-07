from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from langchain_core.documents import Document


from .config import RAGConfig
from .embeddings import get_embedding_model
from .llm_provider import LLMBackend
from .vector_store import load_vector_store
from .rag_utils import (
    _get_vector_db_dirs,
    _describe_databases,
    _decide_which_dbs,
    _build_agent_config_log,
    _extract_metadata_filters,
)

# =====================================================================
# Context builder + similarity filtering (with logging)
# =====================================================================
def _build_context(docs: List[Document], max_chars: int = 4000) -> str:
    """Costruisce il blocco di contesto formattato per l'LLM, includendo i metadati chiave."""
    chunks = []
    total = 0
    for i, d in enumerate(docs):
        src = d.metadata.get("source", "unknown")
        db_name = d.metadata.get("db_name", "")

        country = d.metadata.get("country", "")
        law = d.metadata.get("law", "")
        
        db_prefix = f"[DB: {db_name}] " if db_name else ""
        country_prefix = f"[Country: {country}] " if country else ""
        law_prefix = f"[Law: {law}] " if law else ""
        
        header = f"[DOC {i+1} | {db_prefix}{country_prefix}{law_prefix}source: {src}]\n"
        text = d.page_content
        piece = header + text + "\n\n"
        if total + len(piece) > max_chars:
            break
        chunks.append(piece)
        total += len(piece)
    return "".join(chunks)


def _similarity_rank_and_filter(
    question: str,
    docs: List[Document],
    embedding_model,
    top_k: int,
    min_sim: float = 0.1,
) -> Tuple[List[Document], str]:
    """
    Rank docs by cosine similarity and filter below min_sim.
    """
    log_lines: List[str] = []
    final_docs: List[Document] = [] 

    if not docs:
        log_lines.append("No documents returned from base retriever.")
        return final_docs, "\n".join(log_lines)

    # Calcolo della similarità coseno
    q_vec = np.array(embedding_model.embed_query(question), dtype="float32")
    doc_texts = [d.page_content for d in docs]
    doc_vecs = np.array(embedding_model.embed_documents(doc_texts), dtype="float32")

    q_norm = np.linalg.norm(q_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    denom = np.maximum(q_norm * doc_norms, 1e-8)
    sims = (doc_vecs @ q_vec) / denom

    num_raw = len(docs)
    sims_min = float(np.min(sims))
    sims_max = float(np.max(sims))
    sims_mean = float(np.mean(sims))

    indices = [i for i, s in enumerate(sims) if s >= min_sim]
    num_after_threshold = len(indices)

    if not indices:
        log_lines.append(
            f"Similarity filtering: {num_raw} raw docs → 0 kept "
            f"(threshold={min_sim:.3f}, "
            f"sim range=[{sims_min:.3f}, {sims_max:.3f}], mean={sims_mean:.3f})."
        )
        return final_docs, "\n".join(log_lines)

    indices_sorted = sorted(indices, key=lambda i: sims[i], reverse=True)[:top_k]
    final_docs = [docs[i] for i in indices_sorted]

    sims_kept = sims[indices_sorted]
    sims_kept_min = float(np.min(sims_kept))
    sims_kept_max = float(np.max(sims_kept))
    sims_kept_mean = float(np.mean(sims_kept))

    log_lines.append(
        "Similarity filtering + reranking:\n"
        f"- Raw docs from retriever: {num_raw}\n"
        f"- Docs above threshold {min_sim:.3f}: {num_after_threshold}\n"
        f"- Final top_k={top_k} docs kept: {len(final_docs)}\n"
        f"- Similarity stats (all raw): min={sims_min:.3f}, max={sims_max:.3f}, "
        f"mean={sims_mean:.3f}\n"
        f"- Similarity stats (kept):   min={sims_kept_min:.3f}, max={sims_kept_max:.3f}, "
        f"mean={sims_kept_mean:.3f}"
    )

    return final_docs, "\n".join(log_lines)


def _retrieve_documents_from_db(
    question: str,
    config: RAGConfig,
    embedding_model,
    db_name: str,
    db_path: str,
    active_filters: Optional[Dict[str, Any]] = None, 
) -> Tuple[List[Document], str]:
    """
    Retrieve docs from a single FAISS DB.
    """
    log_lines: List[str] = [f"[DB {db_name}] path={db_path}"]

    vector_store = load_vector_store(db_path, embedding_model)

    # 1. INITIAL RETRIEVAL
    k_initial = config.top_k 
    
    search_kwargs: Dict[str, Any] = {"k": k_initial}
    
    # Applica filtri metadati se presenti
    if active_filters:
        search_kwargs["filter"] = active_filters
        log_lines.append(f"[DB {db_name}] Filters applied: {active_filters}.")
    
    base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    log_lines.append(f"[DB {db_name}] Initial Retrieval set to k={k_initial}.")

    # Esecuzione ricerca vettoriale base
    raw_docs = base_retriever.invoke(question)
    log_lines.append(f"[DB {db_name}] Raw docs retrieved: {len(raw_docs)}")

    # 2. LOGICA RERANKING E FILTRAGGIO FINALE
    final_docs: List[Document] = []
    
    if getattr(config, "use_rerank", False): # Se la checkbox è SPUNTATA
        log_lines.append(f"[DB {db_name}] Reranking ENABLED. Processing top {len(raw_docs)} docs...")
        
        reranked_docs, sim_log = _similarity_rank_and_filter(
            question=question,
            docs=raw_docs,
            embedding_model=embedding_model,
            top_k=config.top_k_final,
            min_sim=config.similarity_threshold if hasattr(config, 'similarity_threshold') else 0.1, 
        )
        final_docs = reranked_docs
        log_lines.append(sim_log)
        
    else: # Se la checkbox è DISATTIVATA
        log_lines.append(f"[DB {db_name}] Reranking DISABLED. Taking top {config.top_k_final} raw docs.")
        final_docs = raw_docs[:config.top_k_final]
        
        if not final_docs:
            log_lines.append("No docs available even without reranking.")
        else:
            log_lines.append(f"Kept first {len(final_docs)} docs based on vector similarity only.")

    if not final_docs:
        log_lines.append(f"[DB {db_name}] Final Result: 0 docs.")
    else:
        log_lines.append(f"[DB {db_name}] Final Result: {len(final_docs)} doc(s) passed to LLM.")

    # Aggiunge il nome del DB ai metadati per citazioni
    for d in final_docs:
        d.metadata = d.metadata or {}
        d.metadata["db_name"] = db_name

    return final_docs, "\n".join(log_lines)


# =====================================================================
# Agentic decision: do we need retrieval? 
# =====================================================================
def _decide_need_retrieval(
    question: str,
    config: RAGConfig,
    llm_backend: LLMBackend,
) -> Tuple[bool, str]:
    """
    Router Intelligente (ALLINEATO AL MULTI-AGENT):
    - YES: Domande specifiche su Divorzio/Successione in IT/EE/SI che richiedono documenti.
    - NO: Chitchat, Fuori tema, o Definizioni legali generiche (Teoria).
    """
    system_prompt = (
        "You are an expert query router for a Legal RAG system specialized in DIVORCE and INHERITANCE law for ITALY, ESTONIA, and SLOVENIA.\n"
        "Your task is to decide if the user question requires retrieving specific documents from the database.\n"
        "Reply with a single word: YES or NO.\n\n"
        "GUIDELINES:\n"
        "1. REPLY 'YES' (Need Retrieval) IF:\n"
        "   - The question is about specific regulations, procedures, or cases regarding DIVORCE, FAMILY LAW, or INHERITANCE.\n"
        "   - The question implies or mentions the context of ITALY, ESTONIA, or SLOVENIA.\n"
        "   - The question asks for a comparison between these jurisdictions.\n\n"
        "2. REPLY 'NO' (No Retrieval) IF:\n"
        "   - The question is general CHITCHAT (e.g., 'Hi', 'Who are you?').\n"
        "   - The question is about GENERAL LEGAL DEFINITIONS or THEORY without specific country context (e.g., 'What is a void contract?', 'Define tort liability').\n"
        "   - The question is completely OUT OF DOMAIN (e.g., recipes, weather).\n\n"
        "EXAMPLES:\n"
        "User: 'In Estonia, how do I split assets in a divorce?' -> YES\n"
        "User: 'Difference between void and voidable contract?' -> NO\n"
        "User: 'Inheritance taxes in Italy for real estate' -> YES\n"
        "User: 'What is negligence in tort law?' -> NO\n"
        "User: 'Hi, help me please' -> NO"
    )
    user_prompt = f"User Question:\n{question}\n\nDecision (YES/NO):"

    resp = llm_backend.chat(system_prompt, user_prompt).strip().lower()

    if "yes" in resp and "no" not in resp:
        return True, f"Router Decision: SPECIFIC LEGAL query detected ('{resp}'). Proceed to retrieval."
    
    if "no" in resp and "yes" not in resp:
        return False, f"Router Decision: GENERAL KNOWLEDGE/CHITCHAT detected ('{resp}'). Skip retrieval."

    return True, f"Router Decision: Ambiguous ('{resp}'). Defaulting to retrieval."

# =====================================================================
# SINGLE-AGENT CORE (ReAct-style)
# =====================================================================
def _single_agent_answer_question_core(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    
    llm_backend = LLMBackend(config)
    db_map = _get_vector_db_dirs(config)

    # ---- 1. ROUTER: Need retrieval? (Legal vs General) ----
    need_retrieval, decision_log = _decide_need_retrieval(
        question, config, llm_backend
    )

    # Init vars for logging
    refusal_log_geo = "N/A"
    refusal_log_topic = "N/A"

    # -------------------------------------------------------------------

    retrieved_docs: List[Document] = []
    used_db_names: List[str] = []
    context = ""
    filter_log = "No metadata extraction performed."
    active_filters = {}
    db_selection_log = ""
    per_db_logs: Dict[str, str] = {}
    per_db_log_block: str = ""
    
    # ---- 3. ESECUZIONE RETRIEVAL (Solo se need_retrieval è True) ----
    if need_retrieval:
        active_filters, filter_log = _extract_metadata_filters(question)

        embedding_model = get_embedding_model(config)
        
        # USA LA VERSIONE DI UTILS AGGIORNATA (con Random Sample per Estonia/Slovenia)
        db_descriptions = _describe_databases(db_map, embedding_model)

        used_db_names, db_selection_log = _decide_which_dbs(
            question=question,
            db_map=db_map,
            db_descriptions=db_descriptions,
            llm_backend=llm_backend,
        )

        if used_db_names:
            all_docs: List[Document] = []
            for db_name in used_db_names:
                db_path = db_map[db_name]
                docs_db, log_db = _retrieve_documents_from_db(
                    question=question,
                    config=config,
                    embedding_model=embedding_model,
                    db_name=db_name,
                    db_path=db_path,
                    active_filters=active_filters, 
                )
                per_db_logs[db_name] = log_db
                all_docs.extend(docs_db)

            for db_name, log in per_db_logs.items():
                per_db_log_block += f"\n\n[DB {db_name}]\n{log}"
            
            retrieved_docs = all_docs
            context = _build_context(retrieved_docs)
        else:
            filter_log += " (No DBs selected by Supervisor; retrieval aborted)."

    # ---- 4. GENERAZIONE RISPOSTA (Prompt Dinamici) ----
    
    if context:
        # --- CASO A: DOMANDA LEGALE CON DOCUMENTI (Strict Mode) ---
        base_instructions = (
            "You are an agentic legal assistant specialized ONLY in Italian, Estonian, and Slovenian civil law "
            "(focusing on Divorce and Inheritance).\n"
            "IMPORTANT RULES:\n"
            "1. Your knowledge base is STRICTLY limited to the provided context.\n"
            "2. If the user asks about a country NOT in the context, you MUST REFUSE to answer.\n"
            "3. Use the provided context as your ONLY source of truth.\n"
            "4. Cite the source documents where appropriate."
        )
        user_context_part = f"Context from retrieved legal documents (TRUST THIS ONLY):\n{context}"
        fallback_msg = "If the context doesn't contain the answer, state it clearly."
        
    else:
        # --- CASO B: NESSUN CONTESTO DISPONIBILE ---
        if need_retrieval:
            # Domanda legale MA senza documenti (o DB Selection = NONE)
            base_instructions = (
                "You are a legal assistant specialized in Divorce and Inheritance. "
                "The user asked a legal question, but no relevant documents were found in the database. "
                "Politely state that you do not have access to specific documents for this query."
            )
            user_context_part = "Context: No relevant legal documents were retrieved."
            fallback_msg = "Do not hallucinate legal rules."
        else:
            # Vera chitchat / Teoria Generale (Definizioni)
            base_instructions = (
                "You are a helpful and polite assistant. "
                "You can answer general knowledge questions (greetings, definitions of legal terms, theory) "
                "using your internal knowledge.\n"
                "HOWEVER, do not invent specific laws for Italy, Estonia or Slovenia if you don't have them."
            )
            user_context_part = "Context: No external legal documents used (General Knowledge Mode)."
            fallback_msg = "Be helpful, accurate, and concise."

    if config.agentic_mode == "react":
        system_prompt = (
            f"{base_instructions}\n"
            "Do not reveal your internal chain-of-thought; provide only a clear final answer."
        )
    else:
        system_prompt = (
            f"{base_instructions}\n"
            "Provide a concise, accurate answer."
        )

    user_parts = [f"Question:\n{question}", user_context_part, fallback_msg]
    user_prompt = "\n\n".join(user_parts)

    answer = llm_backend.chat(system_prompt, user_prompt)

    # ---- 5. LOGGING / TRACING (Opzionale) ----
    reasoning_trace: Optional[str] = None
    
    if config.agentic_mode == "react" and show_reasoning:
        
        # Thought
        thought_str = (
            f"Question: '{question}'\n"
            f"Router Decision: **{'LEGAL (Retrieve)' if need_retrieval else 'GENERAL (No Retrieve)'}**"
        )

        # Action
        if need_retrieval:
            action_str = (
                f"**CHECKS**: Geo Check={refusal_log_geo} | Topic Check={refusal_log_topic}\n"
                f"**ACTION**: Searching DBs {used_db_names} with filters {active_filters}."
            )
        else:
            action_str = "**ACTION**: Direct Answer (General Knowledge) - Retrieval skipped."

        # Observation
        if retrieved_docs:
            observation_str = f"**OBSERVATION**: Retrieved {len(retrieved_docs)} documents."
        else:
            observation_str = "**OBSERVATION**: No documents retrieved."
        
        # Logs
        per_db_log_block_content = per_db_log_block.strip()
        db_map = _get_vector_db_dirs(config)
        db_descriptions = _describe_databases(db_map, get_embedding_model(config))
        agent_config_log = _build_agent_config_log(config, db_map, db_descriptions)

        retrieval_log_block = (
            f"**Log Router**:\n{decision_log}\n\n"
            f"**Log DB Selection**:\n{db_selection_log}\n\n"
            f"**Log DB Retrieval**:\n{per_db_log_block_content}"
        ).strip()
        
        reasoning_trace = (
            f"**Single Agent Trace**\n"
            f"-----------------------------------\n"
            f"**THOUGHT**: {thought_str}\n\n"
            f"**ACTION**: {action_str}\n\n"
            f"**OBSERVATION**: {observation_str}\n\n"
            f"**SYSTEM LOGS**\n"
            f"-----------------------------------\n"
            f"{retrieval_log_block}\n\n"
            f"**Configuration**:\n```text\n{agent_config_log}\n```"
        )
        
    return answer, retrieved_docs, reasoning_trace

# Public alias
def single_agent_answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    """Public entrypoint for single-agent RAG (alias)"""
    return _single_agent_answer_question_core(question, config, show_reasoning)