import pickle
from pathlib import Path
from langchain_chroma import Chroma
from src.embeddings import get_embeddings
from src.config import CONFIG

def create_vector_store(chunks):
    """Crée un index vectoriel avec ChromaDB et sauvegarde les chunks."""
    print("🔄 Generating embeddings and creating ChromaDB index...")
    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CONFIG["chroma_persist_directory"],
        collection_name=CONFIG["collection_name"],
    )

    # Sauvegarder les chunks pour pouvoir recréer BM25 plus tard
    with open(CONFIG["chunks_pickle_path"], "wb") as f:
        pickle.dump(chunks, f)
    print(f"✓ Chunks sauvegardés dans {CONFIG['chunks_pickle_path']}")

    print(f"✓ Vector store saved to {CONFIG['chroma_persist_directory']}")
    return vector_store

def load_vector_store():
    """Charge un index vectoriel existant depuis ChromaDB."""
    if Path(CONFIG["chroma_persist_directory"]).exists():
        embeddings = get_embeddings()
        vector_store = Chroma(
            persist_directory=CONFIG["chroma_persist_directory"],
            embedding_function=embeddings,
            collection_name=CONFIG["collection_name"],
        )
        print(f"✓ Vector store loaded from {CONFIG['chroma_persist_directory']}")
        return vector_store
    return None

def load_chunks():
    """Charge les chunks sauvegardés (pour recréer BM25)."""
    try:
        with open(CONFIG["chunks_pickle_path"], "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None