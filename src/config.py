import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")  # Charge les variables depuis .env

CONFIG = {
    # Embeddings
    "embedding_model": "intfloat/multilingual-e5-small",  # Meilleur pour le français juridique
    "embedding_device": "cpu",
    
    # Découpage (adapté aux articles de loi)
    "chunk_size": 512,
    "chunk_overlap": 100,
    "chunk_separators": ["\n\n", "\n", "Article", "§", ".", " ", ""],
    "chunks_pickle_path": str(BASE_DIR / "chunks.pkl"),
    # Recherche
    "hybrid_search_weights":(0.5, 0.5),
    "top_k_initial": 20,
    "top_k_final": 5,
    
    # LLM
    "llm_model": "gemini-3.0-flash",
    "llm_temperature": 0.2,
    "llm_streaming": True,
    
    # Chemins
    "docs_directory": str(BASE_DIR / "documents"),
    "chroma_persist_directory": str(BASE_DIR / "chroma_db"),
    "collection_name": "togolese_law_collection",
    
    # Agentic
    "enable_agentic_retrieval": True,
    "max_retrieval_iterations": 3,
    }

# Récupération de la clé API depuis .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
