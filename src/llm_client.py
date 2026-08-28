from src.config import CONFIG, GEMINI_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

def create_llm():
    if not GEMINI_API_KEY:
        st.write("GEMINI_API_KEY est absente.")
        raise RuntimeError(
            "GEMINI_API_KEY est absente. Configurez-la dans .env ou dans "
            "les Secrets de Streamlit Cloud."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=CONFIG["llm_temperature"],
    )


def get_llm():
    return create_llm()
