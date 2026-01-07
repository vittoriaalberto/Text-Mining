from __future__ import annotations

import os
import random  # <--- IMPORTANTE
from typing import List, Dict, Optional, Tuple, Any

from langchain_core.documents import Document

from .config import RAGConfig
from .llm_provider import LLMBackend
from .vector_store import load_vector_store
import json 


# =====================================================================
# FUNZIONI DI CONTROLLO DOMINIO E FILTRAGGIO METADATI
# =====================================================================

def _extract_metadata_filters(question: str):
    q = question.lower()
    metadata_filter = {}
    log_lines = []

    # -------------------------
    # COUNTRY (MULTI)
    # -------------------------
    countries = []

    if "italia" in q or "italy" in q:
        countries.append("ITALY")
    if "slovenia" in q:
        countries.append("SLOVENIA")
    if "estonia" in q:
        countries.append("ESTONIA")

    if countries:
        metadata_filter["country"] = countries
        log_lines.append(f"Filter: Countries set to {countries}")

    # -------------------------
    # LAW
    # -------------------------
    divorce_kw = [
        "divorz", "divorce", "separazione", "married", "marriage", 
        "matrimonio", "marital", "spouse", "coniuge", "assets", "property regime"
    ]
    # Parole per Inheritance
    inheritance_kw = [
        "eredit", "succession", "inheritance", "death", "morto", 
        "deceduto", "will", "testamento", "assets of the deceased", "compulsory portion"
    ]

    has_divorce = any(k in q for k in divorce_kw)
    has_inheritance = any(k in q for k in inheritance_kw)

    if has_divorce and has_inheritance:
        metadata_filter["law"] = ["Divorce", "Inheritance"]
        log_lines.append("Filter: Law set to Divorce + Inheritance")
    elif has_divorce:
        metadata_filter["law"] = "Divorce"
        log_lines.append("Filter: Law set to Divorce")
    elif has_inheritance:
        metadata_filter["law"] = "Inheritance"
        log_lines.append("Filter: Law set to Inheritance")

    return metadata_filter, " | ".join(log_lines)


# =====================================================================
# DB MAPPING E DESCRIZIONE
# =====================================================================

def _get_vector_db_dirs(config: RAGConfig) -> Dict[str, str]:
    """
    Returns a mapping: {db_name -> folder_path}
    """
    dirs: List[str] = []

    if getattr(config, "vector_store_dirs", None):
        v = config.vector_store_dirs or []
        if isinstance(v, list) and len(v) > 0:
            dirs.extend(v)

    if not dirs:
        dirs.append(config.vector_store_dir)

    db_map: Dict[str, str] = {}
    for path in dirs:
        name = os.path.basename(os.path.normpath(path)) or path
        db_map[name] = path
    return db_map


def _describe_databases(
    db_map: Dict[str, str],
    embedding_model,
) -> Dict[str, str]:
    """
    Descrizione SHORT per i DB estraendo i paesi e le leggi dai metadati FAISS.
    USA CAMPIONAMENTO RANDOM PER EVITARE BIAS SU DATASET ORDINATI.
    """
    descriptions: Dict[str, str] = {}

    for db_name, path in db_map.items():
        try:
            vs = load_vector_store(path, embedding_model)
            
            docs = []
            if hasattr(vs, "docstore") and hasattr(vs, "index_to_docstore_id"):
                all_ids = list(vs.index_to_docstore_id.values())
                
                # --- MODIFICA CRUCIALE: RANDOM SAMPLE ---
                # Se prendiamo solo i primi 50 e il dataset è ordinato per nazione,
                # perdiamo le altre nazioni.
                sample_size = min(len(all_ids), 200) 
                if sample_size > 0:
                    sample_ids = random.sample(all_ids, sample_size)
                    docs = [vs.docstore.search(id_) for id_ in sample_ids]
                else:
                    docs = []
                    
        except Exception:
            descriptions[db_name] = "Database unavailable."
            continue

        laws = set()
        countries = set()
        
        for d in docs:
            if d and hasattr(d, "metadata"):
                meta = d.metadata
                if meta.get("country"):
                    countries.add(str(meta["country"]).upper())
                if meta.get("law"):
                    laws.add(str(meta["law"]))

        # Costruiamo la descrizione specifica per il Supervisor
        parts = []
        
        # 1. Specifichiamo i paesi coperti
        if countries:
            parts.append(f"countries: {', '.join(sorted(countries))}")
        else:
            parts.append("countries: ALL/UNKNOWN") # Fallback se non trova metadata

        # 2. Specifichiamo la materia legale
        if laws:
            parts.append(f"law: {', '.join(sorted(laws))}")
            
        # 3. Specifichiamo la natura del contenuto in base al nome del DB
        if "codes" in db_name.lower():
            parts.append("content: official legal codes and statutes")
        elif "cases" in db_name.lower():
            parts.append("content: past court cases and jurisprudence")

        if parts:
            descriptions[db_name] = " | ".join(parts)
        else:
            descriptions[db_name] = "General legal database."

    return descriptions


def _decide_which_dbs(
    question: str,
    db_map: Dict[str, str],
    db_descriptions: Dict[str, str],
    llm_backend: LLMBackend,
) -> Tuple[List[str], str]:
    """
    LLM Supervisor decide quali DB usare.
    """
    db_names = list(db_map.keys())
    if len(db_names) == 1:
        return db_names, "Only one DB available → using it by default."

    lines = []
    for name in db_names:
        desc = db_descriptions.get(name, "no description")
        lines.append(f"- {name}: {desc}")
    db_descr_block = "\n".join(lines)


    system_prompt = (
        "You are a legal assistant supervisor. Your task is to map a user question to the correct database(s).\n\n"
        "TOPIC MAPPING RULES:\n"
        "- 'Divorce' databases: Include marriage, separation, marital assets, joint property, alimony, and child custody.\n"
        "- 'Inheritance' databases: Include wills, successions, deceased's estates, compulsory portions, and heir rights.\n\n"
        "GUIDELINES:\n"
        "1. Identify the TOPIC (Divorce vs Inheritance).\n"
        "2. Identify the COUNTRY (Italy, Slovenia, Estonia). If the country is not explicitly listed in the DB description but the TOPIC matches and the DB name seems relevant, SELECT IT to be safe.\n"
        "3. Choose 'codes' for law articles (questions about rules/triggers) and 'cases' for court precedents (questions about practice/real disputes).\n"
        "4. If specific country documents appear missing, select the most relevant topic DBs anyway.\n"
        "5. If completely outside legal scope (e.g. cooking, weather), return 'NONE'.\n"
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Available databases:\n{db_descr_block}\n\n"
        "Which database names should be used? Reply with names separated by commas, "
        "or 'NONE'."
    )

    resp = llm_backend.chat(system_prompt, user_prompt).strip()
    resp_lower = resp.lower()

    if "none" in resp_lower:
        return [], f"DB selection: model answered '{resp}' → NONE (Question outside specialized legal domain)."

    chosen = [name.strip() for name in resp.split(",") if name.strip()]
    chosen_valid = [c for c in chosen if c in db_map]

    if not chosen_valid:
        # Se ha risposto qualcosa ma non matcha i nomi, fallback su tutto per sicurezza
        return db_names, f"DB selection: model answered '{resp}' but no valid DB name parsed → Fallback ALL."

    log = (
        f"DB selection: model answered '{resp}' → using DBs: "
        + ", ".join(chosen_valid)
    )
    return chosen_valid, log


def _build_agent_config_log(
    config: RAGConfig,
    db_map: Dict[str, str],
    db_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    lines: List[str] = []

    lines.append(f"LLM provider: {config.llm_provider}")
    lines.append(f"LLM model: {config.llm_model_name}")
    lines.append(f"Embedding provider: {config.embedding_provider}")
    lines.append(f"Embedding model: {config.embedding_model_name}")
    lines.append(f"top_k: {config.top_k}")
    lines.append(f"agentic_mode: {getattr(config, 'agentic_mode', 'N/A')}")
    use_multiagent = getattr(config, "use_multiagent", False)
    lines.append(f"use_multiagent: {use_multiagent}")

    lines.append("Vector DBs:")
    for name, path in db_map.items():
        if db_descriptions and name in db_descriptions:
            desc = db_descriptions[name]
            lines.append(f"  - {name}: path={path} | {desc}")
        else:
            lines.append(f"  - {name}: path={path}")

    return "\n".join(lines)