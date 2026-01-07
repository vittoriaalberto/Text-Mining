import streamlit as st
import pandas as pd
import json
import os
from datasets import Dataset
from dotenv import load_dotenv

# RAGAS Imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Carica variabili d'ambiente
load_dotenv()

# =====================================================================
# CONFIGURAZIONE PAGINA
# =====================================================================
st.set_page_config(
    page_title="Evaluation - Legal RAG",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RAG Evaluation (OpenAI Powered)")
st.markdown("""
Carica un file JSON di sessione per visualizzare i turni di conversazione e inserire la *Ground Truth* (risposta ideale) per la valutazione completa.
""")

# =====================================================================
# 1. SETUP OPENAI (LLM & EMBEDDINGS)
# =====================================================================

def get_evaluator_models():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OPENAI_API_KEY non trovata nel file .env!")
        st.stop()
    
    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0,
        openai_api_key=api_key
    )
    
    evaluator_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )
    
    return evaluator_llm, evaluator_embeddings

# =====================================================================
# 2. CARICAMENTO E ESTRAZIONE DATI
# =====================================================================

def extract_turns_from_json(uploaded_file):
    """
    Carica il JSON e restituisce una lista piatta di dizionari 
    contenenti question, answer, e contexts (per l'UI e la modifica GT).
    """
    try:
        json_data = json.load(uploaded_file)
    except json.JSONDecodeError:
        st.error("Il file caricato non è un JSON valido.")
        return None
    
    flat_data = []
    sessions = json_data if isinstance(json_data, list) else [json_data]
        
    for session in sessions:
        history = session.get("history", [])
        
        for i in range(len(history) - 1):
            msg_user = history[i]
            msg_assistant = history[i+1]
            
            if msg_user["role"] == "user" and msg_assistant["role"] == "assistant":
                
                question = msg_user.get("content", "").strip()
                answer = msg_assistant.get("content", "").strip()
                contexts = msg_assistant.get("contexts", [])
                
                if question and answer:
                    flat_data.append({
                        "id": len(flat_data) + 1,
                        "question": question,
                        "answer": answer,
                        "contexts": contexts,
                        "initial_gt": msg_assistant.get("ground_truth", "") # Se c'era già nel JSON
                    })
    
    if not flat_data:
        st.error("Nessuna coppia Domanda/Risposta valida trovata nel file JSON.")
        return None
        
    # Inizializza lo stato della sessione per le Ground Truth modificabili
    if "evaluation_data" not in st.session_state or st.session_state.evaluation_data_id != uploaded_file.file_id:
        st.session_state.evaluation_data = flat_data
        st.session_state.evaluation_data_id = uploaded_file.file_id
    
    return st.session_state.evaluation_data

def prepare_ragas_dataset(evaluation_data_list):
    """
    Prepara il Dataset Ragas finale usando i dati dell'interfaccia.
    """
    ragas_data = []
    has_ground_truth = False
    
    for item in evaluation_data_list:
        # Pulisce i contesti
        contexts = [c.strip() for c in item['contexts'] if c.strip()]
        if not contexts:
            contexts = ["No context retrieved"]
            
        gt = item.get("final_gt", "").strip() # Usa il campo GT finale
        
        if gt:
            has_ground_truth = True
            
        ragas_data.append({
            "question": item['question'],
            "answer": item['answer'],
            "contexts": contexts,
            "ground_truth": gt
        })
        
    df = pd.DataFrame(ragas_data)
    
    # Filtra righe vuote e poi crea il Dataset
    df.dropna(subset=['question', 'answer'], inplace=True)
    if df.empty: return None, False
    
    return Dataset.from_pandas(df), has_ground_truth


# =====================================================================
# 3. INTERFACCIA UTENTE E LOGICA DI ESECUZIONE
# =====================================================================

# Sidebar per info sulle metriche
with st.sidebar:
    st.header("⚙️ Configurazione")
    st.info("Il sistema usa *OpenAI (gpt-4o-mini)* come giudice.")
    st.markdown("---")
    st.markdown("##### Metriche Ragas:")
    st.markdown("Le metriche complete vengono calcolate solo se la *Ground Truth* è fornita per almeno una query.")

uploaded_file = st.file_uploader(
    "1. Carica il file JSON di sessione", 
    type=["json"]
)

if uploaded_file is not None:
    
    # FASE 1: Caricamento e Visualizzazione per l'inserimento GT
    evaluation_data = extract_turns_from_json(uploaded_file)
    
    if evaluation_data:
        st.subheader("2. Inserimento Ground Truth (Risposta Ideale)")
        st.info("Compila il campo 'Ground Truth' con la risposta perfetta per ogni domanda. Se lasci vuoto, le metriche Context Precision, Recall e Answer Correctness non verranno calcolate per quella query.")
        
        # Interfaccia per inserire il GT
        for i, item in enumerate(evaluation_data):
            
            # Usiamo un expander per compattare l'interfaccia
            with st.expander(f"Query {item['id']}: {item['question'][:80]}..."):
                
                st.markdown(f"*Domanda Utente:* {item['question']}")
                st.markdown(f"*Risposta Chatbot:* {item['answer']}")
                
                # Widget per il Ground Truth, legato a st.session_state
                # Usiamo una chiave unica per il widget
                gt_key = f"gt_input_{item['id']}"
                
                # Inizializza il valore in session_state la prima volta
                if gt_key not in st.session_state:
                    st.session_state[gt_key] = item['initial_gt']
                    
                gt_value = st.text_area(
                    f"Ground Truth (Risposta Ideale) per Query {item['id']}",
                    value=st.session_state[gt_key],
                    height=100,
                    key=gt_key
                )
                
                # Aggiorna i dati della sessione con il GT modificato
                st.session_state.evaluation_data[i]['final_gt'] = gt_value
                
                # Visualizza i contesti recuperati (per Context Recall/Precision)
                with st.expander(f"📚 Contesti Recuperati ({len(item['contexts'])} documenti)"):
                    for j, context in enumerate(item['contexts']):
                        st.text(f"Documento {j+1}: {context[:300]}...")


        st.divider()
        
        # FASE 2: Avvio Valutazione
        if st.button("🚀 3. Avvia Valutazione Ragas"):
            
            # Prepara il dataset Ragas finale con i GT inseriti
            dataset_ragas, has_ground_truth = prepare_ragas_dataset(st.session_state.evaluation_data)
            
            if dataset_ragas is None:
                st.error("Nessun dato valido da valutare dopo l'inserimento GT.")
                st.stop()
            
            with st.spinner("Preparazione modelli OpenAI..."):
                llm, embeddings = get_evaluator_models()
                
            st.info(f"Valutazione di {len(dataset_ragas)} interazioni in corso... *Potrebbe richiedere tempo*")
            
            progress_bar = st.progress(0)
            
            try:
                # Definizione dinamica delle metriche
                metrics_to_run = [faithfulness, answer_relevancy]
                if has_ground_truth:
                    metrics_to_run.extend([context_precision, context_recall, answer_correctness])
                
                results = evaluate(
                    dataset=dataset_ragas,
                    metrics=metrics_to_run,
                    llm=llm,
                    embeddings=embeddings,
                    raise_exceptions=False
                )
                
                progress_bar.progress(100)
                
                # --- VISUALIZZAZIONE RISULTATI ---
                st.divider()
                st.subheader("📈 Risultati Valutazione Aggregati")
                
                df_results = results.to_pandas()
                
                # Calcolo Medie e preparazione per Metric Cards
                metrics_display = {
                    'faithfulness': 'Faithfulness',
                    'answer_relevancy': 'Answer Relevancy',
                    'context_precision': 'Context Precision',
                    'context_recall': 'Context Recall',
                    'answer_correctness': 'Answer Correctness',
                }
                
                scores = {}
                for col_name, display_name in metrics_display.items():
                    if col_name in df_results.columns:
                        score = df_results[col_name].mean()
                        scores[display_name] = f"{score:.2f}"
                    else:
                        scores[display_name] = "N/A"

                # A. Metric Cards
                valid_scores = [(dn, s) for dn, s in scores.items() if s != "N/A"]
                cols = st.columns(max(len(valid_scores), 2))
                
                for i, (display_name, score) in enumerate(valid_scores):
                    cols[i].metric(display_name, score)
                
                # B. Tabella Dettagliata
                st.subheader("📝 Tabella Dettagliata per Interazione")
                
                # Colonne da mostrare nella tabella
                table_cols = ['question', 'answer', 'ground_truth', 'faithfulness', 'answer_relevancy', 
                              'context_precision', 'context_recall', 'answer_correctness', 'contexts']
                
                final_cols = [c for c in table_cols if c in df_results.columns]
                
                # Creiamo una copia per la visualizzazione
                df_display = df_results[final_cols].copy()

                # 1. Fai partire l'indice da 1 invece che da 0
                df_display.index = df_display.index + 1

                # 2. Dai il nome "Id" alla colonna dell'indice
                df_display.index.name = "Query ID"
                
                # Visualizza la tabella
                st.dataframe(
                    df_display,
                    width='stretch'
                )

                
            except Exception as e:
                st.error(f"Errore critico Ragas durante la valutazione: {e}")
                import traceback
                traceback.print_exc()