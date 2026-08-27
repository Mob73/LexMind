from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CONFIG
import hashlib

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

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i:06d}"
        chunk.metadata["chunk_length"] = len(chunk.page_content)
        chunk.metadata["chunk_hash"] = hashlib.md5(
            chunk.page_content.encode()
        ).hexdigest()

    print(f"✓ Created {len(chunks)} chunks")
    return chunks
