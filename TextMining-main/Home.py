"""
Legal RAG System - Main Entry Point

Landing page che introduce l'applicazione e fornisce navigazione
alle diverse sezioni tramite Streamlit multipage.
"""

import streamlit as st
from pathlib import Path
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend.config import RAGConfig

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================

st.set_page_config(
    page_title="Legal RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# INITIALIZE CONFIG (CHECK VECTOR STORES)
# =====================================================================

@st.cache_resource
def load_config():
    """Load configuration once and cache it"""
    return RAGConfig()

config = load_config()

# =====================================================================
# MAIN PAGE CONTENT
# =====================================================================

st.title("⚖️ Legal RAG System")
st.markdown("### Multi-Agent Retrieval-Augmented Generation for Cross-Border Legal Knowledge")

st.divider()

# Introduction
st.markdown("""
## 👋 Benvenuto

Questo sistema utilizza **Retrieval-Augmented Generation (RAG)** e **architetture agentiche** 
per rispondere a domande legali complesse su:

- 🇮🇹 **Italia** - Divorzio e Successioni
- 🇪🇪 **Estonia** - Divorzio e Successioni  
- 🇸🇮 **Slovenia** - Divorzio e Successioni

Il sistema combina tecniche avanzate di recupero informazioni con modelli di linguaggio 
per fornire risposte accurate e ben documentate su normative e giurisprudenza 
in materia di diritto di famiglia e successioni.
""")

st.divider()

# Navigation Guide
st.markdown("## 🧭 Navigazione")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 💬 Chatbot Interface
    
    Interagisci con il sistema per porre domande legali.
    
    **Caratteristiche:**
    - Configurazione parametri di retrieval
    - Visualizzazione delle fonti citate
    - Tracciamento del reasoning (opzionale)
    - Export della conversazione in JSON
    
    👉 **Vai alla pagina Chatbot nella sidebar** ⬅️
    """)

with col2:
    st.markdown("""
    ### 📊 Evaluation Dashboard
    
    Valuta la qualità delle risposte del sistema.
    
    **Metriche Supportate:**
    - **Faithfulness**: Coerenza e aderenza della risposta rispetto ai soli documenti forniti.
    - **Context Precision**: Rapporto tra i documenti rilevanti recuperati e il totale di quelli recuperati.
    - **Context Recall**: Presenza di tutte le informazioni necessarie all'interno del contesto recuperato.
    - **Answer Relevancy**: Pertinenza diretta della risposta rispetto alla domanda dell'utente.
    - **Answer Correctness**: Esattezza della risposta rispetto alla soluzione di riferimento (ground truth).
    
    👉 **Vai alla pagina Evaluation nella sidebar** ⬅️
    """)

st.divider()

# Quick Start
st.markdown("## 🚀 Quick Start")

st.markdown("""
1. **Naviga** alla pagina **Chatbot** dalla sidebar
2. **Configura** i parametri di retrieval (opzionale)
3. **Poni** la tua domanda legale
4. **Analizza** le fonti e il reasoning forniti
5. **Esporta** la conversazione per analisi future
6. **Valuta** la qualità nella pagina **Evaluation**
""")

# Footer
st.divider()
st.caption("🔬 Legal RAG System | Sviluppato per l'analisi cross-border di normative su divorzio e successioni")