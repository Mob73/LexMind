import os
from dotenv import load_dotenv

load_dotenv()  # Charge les variables depuis .env

CONFIG = {
    # Embeddings
    "embedding_model": "intfloat/multilingual-e5-small",  # Meilleur pour le français juridique
    "embedding_device": "cpu",
    
    # Découpage (adapté aux articles de loi)
    "chunk_size": 512,
    "chunk_overlap": 100,
    "chunk_separators": ["\n\n", "\n", "Article", "§", ".", " ", ""],
    "chunks_pickle_path":"./chunks.pkl",
    # Recherche
    "hybrid_search_weights":(0.5, 0.5),
    "top_k_initial": 20,
    "top_k_final": 5,
    
    # LLM
    "llm_provider": "gemini",  # ou "ollama"
    "llm_model": "gemini-2.0-flash",  # ou "llama3.2" / "mistral"
    "llm_temperature": 0.1,
    "llm_streaming": True,
    
    # Chemins
    "docs_directory": "./documents/",
    "chroma_persist_directory": "./chroma_db/",
    "collection_name": "togolese_law_collection",
    
    # Agentic
    "enable_agentic_retrieval": True,
    "max_retrieval_iterations": 3,
    }

# Récupération des clés API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
