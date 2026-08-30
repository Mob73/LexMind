from src.config import CONFIG, GEMINI_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

def create_llm():

    return ChatGoogleGenerativeAI(
        model=CONFIG["llm_model"],
        google_api_key=GEMINI_API_KEY,
        temperature=CONFIG["llm_temperature"],
    )


def get_llm():
    return create_llm()
