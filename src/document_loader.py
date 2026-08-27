from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
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
    "llm_model": "gemini-3.6-flash",  # ou "llama3.2" / "mistral"
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

def load_documents(directory=None):
    """Charge tous les documents d'un répertoire"""
    directory = directory or CONFIG["docs_directory"]
    documents = []

    if not Path(directory).exists():
        raise FileNotFoundError(f"Directory {directory} not found")

    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    for file_path in Path(directory).rglob("*"):
        if file_path.suffix in supported_extensions:
            try:
                loader_class = supported_extensions[file_path.suffix]
                loader = loader_class(str(file_path))
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["filename"] = file_path.name
                    doc.metadata["file_hash"] = hashlib.md5(
                        str(file_path).encode()
                    ).hexdigest()

                documents.extend(docs)
                print(f"✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {e}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def chunk_documents(documents):
    """Découpe les documents en chunks avec RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        separators=CONFIG["chunk_separators"],
        length_function=len,
    )

    chunks = text_splitter.split_text(documents)



    print(f"✓ Created {len(chunks)} chunks")
    return chunks
