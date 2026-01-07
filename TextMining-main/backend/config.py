from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from pathlib import Path

import os # Aggiungi l'import di os se non c'è già

# --- DEFINIZIONE RADICE DEL CONFIG (Risolve la portabilità) ---
# Path del file config.py (es: .../TEXTMINING/backend/config.py)
CONFIG_DIR = Path(__file__).parent 

# Radice del progetto (es: .../TEXTMINING)
# Risale di un livello dalla cartella 'backend'
PROJECT_ROOT = CONFIG_DIR.parent 
# -----------------------------------------------------------------

def _find_all_vector_stores(base_dir: str) -> List[str]:
    """
    Scansiona la directory base per trovare tutte le sottodirectory esistenti
    che rappresentano i Vector Store FAISS.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    # Restituisce una lista di percorsi relativi o assoluti delle sottocartelle
    # che sono i tuoi DB (es. 'vector_store/divorce_codes')
    # Usiamo str(p) per mantenere la compatibilità con il tipo List[str]
    return [str(p) for p in base_path.iterdir() if p.is_dir()]


@dataclass
class RAGConfig:
    """
    Global configuration object for the Legal RAG system.
    """
    
    # ---------------- LLM & Embeddings (omessi per brevità) ----------------
    # Cambia da "huggingface" a "openai"
    llm_provider: str = "openai"
    # Usa un modello OpenAI valido (es. "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo")
    llm_model_name: str = "gpt-4o-mini"

    embedding_provider: str = "huggingface"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # ---------------- Data (JSON corpus) (omessi per brevità) ----------------
    #data_base_dir: str = r"C:\Users\vikya\Desktop\TM\Progetto_claude\Contest_Data"
    data_base_dir: str = str(PROJECT_ROOT / "Contest_Data")
    countries: List[str] = field(default_factory=lambda: ["Italy", "Estonia", "slovenia"])
    
    document_types: dict = field(default_factory=lambda: {
        "Italy": {
            "divorce": "Divorce_italy",
            "inheritance": "Inheritance_italy", 
            "cases": "italian_cases_json_processed"
        },
        "Estonia": {
            "divorce": "Divorce_estonia",
            "inheritance": "Inheritance_estonia",
            "cases": "estonian_cases_json_processed"
        },
        "slovenia": {
            "divorce": "Divorce_slovenia",
            "inheritance": "Inheritance_slovenia",
            "cases": "slovenian_cases_json_processed"
        }
    })
    
    # ---------------- Vector stores (paths) ----------------
    # Root folder for all FAISS vector DBs
    vector_store_base_dir: str = "vector_store"
    
    # Default single vector store (per retrocompatibilità, se non usato viene ignorato)
    vector_store_dir: str = "vector_store/divorce"
    
    # Multi-DB: ORA CARICATA DINAMICAMENTE
    vector_store_dirs: List[str] = field(
        default_factory=lambda: _find_all_vector_stores("vector_store")
    )
    
    # ---------------- Retrieval & Agentic behavior (omessi per brevità) ----------------
    top_k: int = 10
    top_k_final: int = 5
    similarity_threshold: float = 0.3
    use_rerank: bool = False
    
    agentic_mode: str = "standard_rag"
    use_multiagent: bool = False
    
    llm_temperature: float = 0.2
    llm_max_tokens: int = 512
    
    logs_dir: str = "logs"
    enable_logging: bool = True
    
    def __post_init__(self):
        """Assicuriamo che i percorsi siano corretti e aggiorniamo la lista dei DB"""
        # Ricalcola la lista dei DB al momento della creazione dell'istanza
        # Questo è ridondante ma garantisce che, se si modifica base_dir in runtime, funzioni
        self.vector_store_dirs = _find_all_vector_stores(self.vector_store_base_dir)


    def get_data_path(self, country: str, doc_type: str) -> Path:
        """Get the full path for a specific country and document type"""
        folder_name = self.document_types[country][doc_type]
        return Path(self.data_base_dir) / country / folder_name
    
    def get_all_data_paths(self) -> List[Path]:
        """Get all data paths for all countries and document types"""
        paths = []
        for country in self.countries:
            for doc_type in ["divorce", "inheritance", "cases"]:
                path = self.get_data_path(country, doc_type)
                if path.exists():
                    paths.append(path)
        return paths