import streamlit as st
import os
from pathlib import Path
from src.pipeline import initialize_pipeline, index_documents, query_pipeline
from src.config import CONFIG

st.set_page_config(page_title="LexMind", page_icon="⚖️", layout="wide")
st.title("LexMind")
st.subheader("Assistant Juridique Togolais ")
st.markdown("Posez vos questions sur le droit togolais.")

# --- Gestion du pipeline en cache ---
@st.cache_resource
def get_pipeline():
    pipeline = initialize_pipeline()
    return pipeline

pipeline = get_pipeline()
    
# --- Zone de chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.write(f"- {src}")

if prompt := st.chat_input("Posez votre question juridique..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if not pipeline["is_indexed"]:
        with st.chat_message("assistant"):
            st.error("⚠️ Pipeline non indexé. Veuillez indexer dans la barre latérale.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Recherche et génération..."):
                try:
                    answer, docs = query_pipeline(pipeline, prompt)
                    sources = list(set([d.metadata.get("filename", "Inconnu") for d in docs]))
                    st.markdown(answer)
                    with st.expander(" Sources"):
                        for src in sources:
                            st.write(f"- {src}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Erreur : {e}")
