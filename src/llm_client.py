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
    
def translate_to_ewe(text):
    """Traduit une réponse juridique française en éwé avec Gemini."""

    llm = create_llm()

    prompt = f"""
Tu es un traducteur expert en français et en éwé.

Traduis le texte juridique français ci-dessous en éwé.

Règles importantes :
- Préserve exactement le sens juridique.
- Ne résume pas le texte.
- N'ajoute aucune information.
- Conserve les numéros d'articles et les références juridiques.
- Utilise un éwé naturel, clair et compréhensible.
- Retourne uniquement la traduction en éwé.

Texte français à traduire :

{text}
"""

    response = llm.invoke(prompt)

    return response.content
