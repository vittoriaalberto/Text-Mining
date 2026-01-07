import os
import sys
from pathlib import Path
import shutil

# Add backend to path (necessario per importare i moduli interni)
sys.path.append(str(Path(__file__).parent))

from backend.config import RAGConfig
from backend.document_loader import load_documents_by_law_type
from backend.embeddings import get_embedding_model
from backend.vector_store import build_vector_store


def build_four_vector_stores(config: RAGConfig):
    """
    Build four granular vector stores:
    1. vector_store/divorce_codes
    2. vector_store/divorce_cases
    3. vector_store/inheritance_codes
    4. vector_store/inheritance_cases
    
    Codes and Cases are separated for better retrieval control.
    """
    print("\n" + "="*80)
    print("BUILDING FOUR GRANULAR VECTOR STORES (Codes & Cases separated)")
    print("Following the enhanced 4-DB approach")
    print("="*80 + "\n")
    
    # Get embedding model (reuse for all stores)
    print(f"Initializing embedding model: {config.embedding_model_name}")
    print("This may take a moment on first run (downloading model)...\n")
    embedding_model = get_embedding_model(config)
    
    law_types = ["Divorce", "Inheritance"]
    vector_store_dirs = []
    
    for law_type in law_types:
        law_type_lower = law_type.lower()
        
        print(f"\n{'='*80}")
        print(f"Loading and processing {law_type.upper()} documents...")
        print('='*80)
        
        # 1. Load ALL documents (Codes + filtered Cases) for this law type
        print(f"\nLoading all {law_type} documents from all countries...")
        # Il tuo document_loader.py è già efficiente nel caricare Codici e filtrare Casi per Law Type
        docs = load_documents_by_law_type(config, law_type)
        
        if not docs:
            print(f"⚠️  No documents found for {law_type}, skipping...")
            continue
            
        # 2. Divide the documents into Codes and Cases using metadata['doc_type']
        codes = [d for d in docs if d.metadata.get('doc_type') == 'code']
        cases = [d for d in docs if d.metadata.get('doc_type') == 'case']
        
        
        # --- A. Build Vector Store for CODES ---
        
        target_dir_codes = os.path.join(config.vector_store_base_dir, f"{law_type_lower}_codes")
        print(f"\n🔨 Building FAISS index for {law_type} CODES at: {target_dir_codes}")
        print(f"   Total Code Documents: {len(codes)}")
        
        if codes:
            build_vector_store(codes, embedding_model, target_dir_codes)
            vector_store_dirs.append(target_dir_codes)
            print(f"✅ {law_type} Codes vector store created successfully!")
        else:
            print(f"⚠️  Skipping {law_type} Codes store: 0 documents found.")
            
            
        # --- B. Build Vector Store for CASES ---
            
        target_dir_cases = os.path.join(config.vector_store_base_dir, f"{law_type_lower}_cases")
        print(f"\n🔨 Building FAISS index for {law_type} CASES at: {target_dir_cases}")
        print(f"   Total Case Documents: {len(cases)}")
        
        if cases:
            build_vector_store(cases, embedding_model, target_dir_cases)
            vector_store_dirs.append(target_dir_cases)
            print(f"✅ {law_type} Cases vector store created successfully!")
        else:
            print(f"⚠️  Skipping {law_type} Cases store: 0 documents found.")

    
    # Update config with created vector store directories
    config.vector_store_dirs = vector_store_dirs
    
    print(f"\n{'='*80}")
    print("✅ ALL 4 VECTOR STORES CREATED SUCCESSFULLY!")
    print('='*80)
    print(f"\nTotal vector stores: {len(vector_store_dirs)}")
    for vsd in vector_store_dirs:
        print(f"   📁 {vsd}")
    
    print("\n💡 These granular vector stores enable more precise routing.")
    print("\n🚀 Next step: Run the chatbot!")
    print("   python app.py")


def verify_data_exists(config: RAGConfig) -> bool:
    """Verify that the data directory and files exist (Same as original)"""
    print("\n🔍 Verifying data directory...")
    
    data_dir = Path(config.data_base_dir)
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Please check the path in backend/config.py")
        return False
    
    print(f"✅ Data directory found: {data_dir}")
    
    # Check each country
    for country in config.countries:
        country_path = data_dir / country
        if not country_path.exists():
            print(f"⚠️  Country directory not found: {country_path}")
        else:
            print(f"   ✅ {country} directory exists")
            
            # Check document types
            for doc_type in ["divorce", "inheritance", "cases"]:
                folder_path = config.get_data_path(country, doc_type)
                json_files = list(folder_path.glob("*.json")) if folder_path.exists() else []
                if json_files:
                    print(f"      └─ {doc_type}: {len(json_files)} JSON files")
                else:
                    print(f"      └─ {doc_type}: ⚠️ no files found")
    
    return True

# Aggiungi 'shutil' all'inizio con gli altri import
import os
import sys
from pathlib import Path
import shutil 

def clean_all_vector_stores(config: RAGConfig):
    """
    Rimuove TUTTE le directory dei Vector Store all'interno della directory base.
    """
    base_dir = Path(config.vector_store_base_dir)
    print("\n" + "="*80)
    print(f"CLEANUP: Rimuovendo TUTTE le directory in {base_dir.name}/")
    
    deleted_count = 0
    
    # Trova tutte le sottodirectory all'interno della directory base
    # Questo include 'divorce', 'inheritance', 'divorce_codes', 'divorce_cases', ecc.
    for item in base_dir.iterdir():
        if item.is_dir():
            print(f"   🗑️ Eliminando la directory: {item.name}/")
            try:
                # Rimuove la directory e tutto il suo contenuto
                shutil.rmtree(item) 
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Errore durante l'eliminazione di {item.name}: {e}")
                
    if deleted_count > 0:
        print(f"✅ Pulizia completata. Rimosse {deleted_count} Vector Store in totale.")
    else:
        print("✅ Nessuna directory Vector Store trovata da rimuovere.")
    print("="*80)


def main():
    """Main function"""
    
    config = RAGConfig()
    
    # Verify data exists
    if not verify_data_exists(config):
        print("\n❌ Data verification failed. Please check your data directory.")
        return
    
    # -----------------------------------------------------------------
    # CHIAMATA PER LA PULIZIA COMPLETA
    # -----------------------------------------------------------------
    clean_all_vector_stores(config) # <--- QUESTA FUNZIONE ELIMINA TUTTO
    # -----------------------------------------------------------------
    
    print("\n" + "="*80)
    input("\nPress ENTER to start building 4 vector stores from scratch (this will take a few minutes)...")
    
    # CHIAMATA PER RICOSTRUIRE I NUOVI 4 DB
    build_four_vector_stores(config)
    
    print("\n" + "="*80)
    print("✅ DONE! Vector stores are ready to use.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()