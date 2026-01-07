import json
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document

# Role of this module:
# This is the ingestion layer: it converts raw JSON legal documents into 
# LangChain Documents that can be embedded and stored in vector databases.


def _extract_docs_from_json_object(obj: Any, source: str) -> List[Document]:
    """
    Accepts:
      - A dict with 'content' and optional 'metadata'
      - A list of such dicts
    Returns a list of LangChain Document objects.
    """
    docs: List[Document] = []

    def normalize_single(item: dict) -> Document | None:
        # Look for content field
        content_key = None
        for k in ["content", "text", "corpus"]:
            if k in item:
                content_key = k
                break

        if content_key is None:
            return None

        content = item[content_key]
        if not isinstance(content, str):
            content = str(content)

        # Get metadata
        meta = item.get("metadata", {})
        if not isinstance(meta, dict):
            meta = {"metadata_raw": str(meta)}

        # Normalize metadata fields
        meta = _normalize_metadata(meta)
        
        # Attach source path
        meta["source"] = source
        
        return Document(page_content=content, metadata=meta)

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                d = normalize_single(item)
                if d is not None:
                    docs.append(d)
    elif isinstance(obj, dict):
        d = normalize_single(obj)
        if d is not None:
            docs.append(d)

    return docs


def _normalize_metadata(meta: dict) -> dict:
    """
    Normalize metadata fields across different document types.
    
    Ensures consistent field names:
    - country: ITALY, ESTONIA, or SLOVENIA
    - law: Divorce, Inheritance
    - doc_type: code or case
    - civil_codes_used: always a list
    """
    normalized = meta.copy()
    
    # Normalize country field (can be 'type' or 'state')
    if 'type' in normalized and 'state' not in normalized:
        normalized['country'] = normalized['type']
    elif 'state' in normalized:
        normalized['country'] = normalized['state']
    
    # Ensure country is uppercase
    if 'country' in normalized:
        normalized['country'] = normalized['country'].upper()
    
    # Normalize law field - IMPORTANT: keep original case
    # Cases have "Divorce" or "Inheritance", codes also have "Divorce" or "Inheritance"
    if 'law' not in normalized:
        normalized['law'] = 'Unknown'
    
    # Normalize civil_codes_used to always be a list
    if 'civil_codes_used' in normalized:
        if isinstance(normalized['civil_codes_used'], str):
            normalized['civil_codes_used'] = [normalized['civil_codes_used']]
    
    # Determine document type (code vs case)
    if 'CASE_ID' in normalized or 'case_id' in normalized:
        normalized['doc_type'] = 'case'
    else:
        normalized['doc_type'] = 'code'
    
    return normalized


def load_documents_from_folder(folder: str | Path, corpus_name: str = None) -> List[Document]:
    """
    Load all JSON documents from a single folder.
    
    Args:
        folder: Path to folder containing JSON files
        corpus_name: Optional name to tag all documents with (stored in metadata['corpus'])
    
    Returns:
        List of LangChain Document objects
    """
    folder_path = Path(folder)
    
    if not folder_path.exists():
        print(f"[document_loader] Folder does not exist: {folder_path}")
        return []
    
    docs: List[Document] = []
    
    for json_file in folder_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            file_docs = _extract_docs_from_json_object(data, source=str(json_file))
            
            # Add corpus name if provided
            if corpus_name:
                for doc in file_docs:
                    doc.metadata = doc.metadata or {}
                    doc.metadata["corpus"] = corpus_name
            
            docs.extend(file_docs)
            
        except Exception as e:
            print(f"[document_loader] Error loading {json_file}: {e}")
    
    return docs


def load_documents_from_folders(folders: List[str | Path], corpus_names: List[str] = None) -> List[Document]:
    """
    Load documents from multiple folders.
    
    Args:
        folders: List of folder paths
        corpus_names: Optional list of corpus names (one per folder)
    
    Returns:
        List of all LangChain Document objects
    """
    all_docs: List[Document] = []
    
    if corpus_names is None:
        corpus_names = [None] * len(folders)
    
    for folder, corpus_name in zip(folders, corpus_names):
        folder_docs = load_documents_from_folder(folder, corpus_name)
        all_docs.extend(folder_docs)
        print(f"[document_loader] Loaded {len(folder_docs)} documents from {folder}")
    
    return all_docs


def load_documents_by_law_type(config, law_type: str) -> List[Document]:
    """
    Load all documents for a specific law type (Divorce or Inheritance).
    
    This loads:
    1. All civil code documents for that law type (from all countries)
    2. All cases filtered by that law type (from all countries)
    
    Args:
        config: RAGConfig object
        law_type: "Divorce" or "Inheritance"
    
    Returns:
        List of LangChain Document objects
    """
    docs: List[Document] = []
    
    law_type_lower = law_type.lower()
    
    for country in config.countries:
        # Load civil codes for this law type
        code_path = config.get_data_path(country, law_type_lower)
        if code_path.exists():
            corpus_name = f"{country}_{law_type_lower}_codes"
            code_docs = load_documents_from_folder(code_path, corpus_name)
            docs.extend(code_docs)
            print(f"[document_loader] Loaded {len(code_docs)} {law_type} codes from {country}")
        
        # Load cases (will be filtered)
        case_path = config.get_data_path(country, "cases")
        if case_path.exists():
            corpus_name = f"{country}_cases"
            case_docs = load_documents_from_folder(case_path, corpus_name)
            
            # Filter cases by law type
            filtered_cases = [
                doc for doc in case_docs 
                if doc.metadata.get('law', '').lower() == law_type_lower
            ]
            
            docs.extend(filtered_cases)
            print(f"[document_loader] Loaded {len(filtered_cases)} {law_type} cases from {country} (filtered from {len(case_docs)} total)")
    
    return docs


def load_documents_by_country_and_type(
    config,
    country: str = None,
    doc_type: str = None
) -> List[Document]:
    """
    Load documents filtered by country and/or document type.
    
    Args:
        config: RAGConfig object
        country: Filter by country (Italy, Estonia, Slovenia) or None for all
        doc_type: Filter by type (divorce, inheritance, cases) or None for all
    
    Returns:
        List of filtered LangChain Document objects
    """
    docs: List[Document] = []
    
    countries = [country] if country else config.countries
    doc_types = [doc_type] if doc_type else ["divorce", "inheritance", "cases"]
    
    for ctry in countries:
        for dtype in doc_types:
            path = config.get_data_path(ctry, dtype)
            if path.exists():
                corpus_name = f"{ctry}_{dtype}"
                folder_docs = load_documents_from_folder(path, corpus_name)
                docs.extend(folder_docs)
                print(f"[document_loader] Loaded {len(folder_docs)} from {corpus_name}")
            else:
                print(f"[document_loader] Path not found: {path}")
    
    return docs
