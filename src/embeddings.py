from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CONFIG

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={"device": CONFIG["embedding_device"]},
        encode_kwargs={"normalize_embeddings": True},
    )