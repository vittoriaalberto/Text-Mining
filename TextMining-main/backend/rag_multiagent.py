from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple, Optional, Dict, Any

from langchain_core.documents import Document

from .config import RAGConfig
from .embeddings import get_embedding_model
from .llm_provider import LLMBackend
from .rag_utils import (
    _get_vector_db_dirs,
    _describe_databases,
    _decide_which_dbs,
    _build_agent_config_log,
    _extract_metadata_filters,
)
from .rag_single_agent import (
    single_agent_answer_question, 
    _retrieve_documents_from_db, 
    _build_context
)


# =====================================================================
# HELPER FUNCTIONS (Allineate al Single Agent)
# =====================================================================

def _decide_need_retrieval(
    question: str,
    config: RAGConfig,
    llm_backend: LLMBackend,
) -> Tuple[bool, str]:
    """
    Router Intelligente:
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

    # Chiamata all'LLM
    resp = llm_backend.chat(system_prompt, user_prompt).strip().lower()

    # Logica di interpretazione della risposta
    if "yes" in resp and "no" not in resp:
        return True, f"Router Decision: SPECIFIC LEGAL query detected ('{resp}'). Proceed to retrieval."
    
    if "no" in resp and "yes" not in resp:
        return False, f"Router Decision: GENERAL KNOWLEDGE/CHITCHAT detected ('{resp}'). Skip retrieval."
    
    # Fallback conservativo: nel dubbio, cerca.
    return True, f"Router Decision: Ambiguous ('{resp}'). Defaulting to retrieval."


# =====================================================================
# CORE LOGIC
# =====================================================================

def _run_sub_agent(
    question: str, 
    db_name: str, 
    db_path: str, 
    config: RAGConfig, 
    active_filters: Dict[str, Any],
    show_reasoning: bool
) -> Tuple[str, List[Document], Optional[str]]:
    
    llm_backend = LLMBackend(config)
    embedding_model = get_embedding_model(config)
    
    # 1. Retrieval con filtri
    docs, log_retrieval = _retrieve_documents_from_db(
        question=question,
        config=config,
        embedding_model=embedding_model,
        db_name=db_name,
        db_path=db_path,
        active_filters=active_filters
    )
    
    context = _build_context(docs) 
    
    # 2. Sintesi Sub-Agent
    system_prompt = (
        "You are an assistant specialized in civil law. "
        "Your task is to answer the user's question based ONLY on the provided context "
        f"from the database {db_name}. If the context is insufficient, state this."
    )
    user_prompt = f"Question:\n{question}\n\nContext from retrieved documents:\n{context}\n\nProvide a clear, concise final answer."
    
    answer = llm_backend.chat(system_prompt, user_prompt)
    
    # 3. Trace 
    trace = f"Sub-Agent Retrieval Log:\n```text\n{log_retrieval}\n```"
    
    return answer, docs, trace


def _multiagent_answer_question_core(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    
    supervisor_backend = LLMBackend(config)
    
    # --- STEP 1: ROUTER INTELLIGENTE (Legal vs General) ---
    need_retrieval, decision_log = _decide_need_retrieval(
        question, config, supervisor_backend
    )

    # Inizializza variabili di log
    refusal_log_geo = "N/A"
    refusal_log_topic = "N/A"


    # --- STEP 3: GESTIONE DOMANDE NON LEGALI (Chitchat) ---
    if not need_retrieval:
        # Risposta diretta senza usare sub-agents o DB
        system_prompt = (
            "You are a helpful and polite assistant. "
            "You can answer general knowledge questions (like greetings, recipes, history, legal definitions) using your internal knowledge.\n"
            "HOWEVER, if the user specifically asks for LEGAL ADVICE on a real case and you have no context, "
            "kindly state that you don't have access to those specific legal files."
        )
        answer = supervisor_backend.chat(system_prompt, f"User Question: {question}")
        
        reasoning_trace = None
        if show_reasoning:
            reasoning_trace = (
                f"**Multi-agent Supervisor**: Router decided NO RETRIEVAL needed.\n"
                f"**Log**: {decision_log}\n"
                f"**Action**: Direct answer generated via internal knowledge."
            )
        return answer, [], reasoning_trace

    # =========================================================================
    # DA QUI IN POI: LOGICA MULTI-AGENT CLASSICA (Solo per domande Legal valide)
    # =========================================================================
    
    db_map = _get_vector_db_dirs(config)
    embedding_model = get_embedding_model(config)
    db_descriptions = _describe_databases(db_map, embedding_model)

    final_answer: str = "Error: Answer generation failed." 
    reasoning_trace: Optional[str] = None
    all_docs: List[Document] = []
    
    # Estrazione Filtri Metadati (per i sub-agents)
    active_filters, filter_log = _extract_metadata_filters(question)

    # Decisione Sub-Agents
    chosen_db_names, routing_log = _decide_which_dbs(
        question=question,
        db_map=db_map,
        db_descriptions=db_descriptions,
        llm_backend=supervisor_backend,
    )

    per_agent_answers: List[Tuple[str, str]] = []
    sub_traces: Dict[str, str] = {}

    # Esecuzione Sub-Agents
    for db_name in chosen_db_names:
        db_path = db_map[db_name]

        local_cfg = replace(config)
        local_cfg.vector_store_dirs = [db_path]
        local_cfg.vector_store_dir = db_path
        if hasattr(local_cfg, "use_multiagent"):
            local_cfg.use_multiagent = False

        sub_answer, sub_docs, sub_trace = _run_sub_agent(
            question=question, 
            db_name=db_name, 
            db_path=db_path, 
            config=local_cfg, 
            active_filters=active_filters, 
            show_reasoning=True 
        )
        
        per_agent_answers.append((db_name, sub_answer))
        all_docs.extend(sub_docs)
        if sub_trace:
            sub_traces[db_name] = sub_trace

    # --- FALLBACK SE NESSUN AGENTE RISPONDE ---
    if not per_agent_answers:
        # Se siamo qui, il Router ha detto "YES" (Legal), ma il Supervisor DB ha detto "NONE".
        # Eseguiamo un fallback su Single Agent MA assicurandoci che veda TUTTI i DB.
        
        # Nota: single_agent_answer_question usa config.vector_store_dirs se presente.
        # Dobbiamo assicurarci che il config passato al fallback abbia tutti i path.
        fallback_config = replace(config)
        # Assicuriamoci che i path siano settati correttamente
        all_paths = list(db_map.values())
        fallback_config.vector_store_dirs = all_paths
        
        fallback_answer, fallback_docs, fallback_trace = single_agent_answer_question(
            question, fallback_config, show_reasoning=show_reasoning
        )
        final_answer = fallback_answer
        all_docs = fallback_docs
        reasoning_trace = None
        if show_reasoning and fallback_trace:
            reasoning_trace = (
                "**Multi-agent Supervisor**: No specialized agents were selected (likely due to description mismatch); "
                "falling back to single-agent RAG over ALL databases.\n\n"
                + fallback_trace
            )
        return final_answer, all_docs, reasoning_trace

    # --- SINTESI SUPERVISOR ---
    agents_block_lines = []
    for db_name, ans in per_agent_answers:
        header = f"[Agent: {db_name}]"
        agents_block_lines.append(f"{header}\n{ans}\n")
    agents_block = "\n\n".join(agents_block_lines)

    system_prompt = (
        "You are a supervisor agent coordinating several specialized RAG agents.\n"
        "You are given their partial answers to the user's question about Civil Law (Divorce/Succession).\n"
        "Your job is to synthesize a single, clear, non-redundant answer for the user.\n"
        "If agents disagree, explain the discrepancy briefly, then give your best judgment based on the provided context.\n"
        "Do not mention internal tools or agents; just answer as a single assistant."
    )
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Specialized agent answers:\n{agents_block}\n\n"
        "Now provide a single final answer to the user, in your own words."
    )

    final_answer = supervisor_backend.chat(system_prompt, user_prompt)

    # --- COSTRUZIONE TRACCIA DI RAGIONAMENTO ---
    if show_reasoning:
        routing_info = (
            "Supervisor selected the following specialized agents: "
            + ", ".join(f"`{n}`" for n, _ in per_agent_answers)
            + "."
        )
        per_agent_summary_lines = []
        for db_name, ans in per_agent_answers:
            short_ans = ans[:400] + "..." if len(ans) > 400 else ans
            per_agent_summary_lines.append(
                f"- **Agent `{db_name}`** answer snippet:\n  {short_ans}"
            )
        per_agent_summary = "\n".join(per_agent_summary_lines)

        subagent_log_block = ""
        for db_name, trace in sub_traces.items():
            subagent_log_block += (
                f"\n\n[Sub-agent `{db_name}` detailed trace]\n{trace}"
            )

        agent_config_log = _build_agent_config_log(
            config=config,
            db_map=db_map,
            db_descriptions=db_descriptions,
        )

        reasoning_trace = (
            f"**Multi-agent Supervisor Thought**:\n"
            f"1. **Router**: LEGAL Query Detected ({decision_log})\n"
            f"2. **Routing**: {routing_info}\n"
            f"3. **Metadata Filters**: {filter_log}\n\n"
            f"**Sub-agent outputs (summarized)**:\n{per_agent_summary}\n\n"
            f"**Supervisor Routing Log**:\n```text\n{routing_log}\n```\n\n"
            f"**Agent Configuration**:\n```text\n{agent_config_log}\n```"
        )

        if subagent_log_block:
            reasoning_trace += (
                "\n\n**Sub-agent Retrieval Logs**:\n"
                f"{subagent_log_block}"
            )

    return final_answer, all_docs, reasoning_trace

def multiagent_answer_question(
    question: str,
    config: RAGConfig,
    show_reasoning: bool = False,
) -> Tuple[str, List[Document], Optional[str]]:
    """Public entrypoint for multi-agent RAG with supervisor (alias)"""
    return _multiagent_answer_question_core(question, config, show_reasoning)