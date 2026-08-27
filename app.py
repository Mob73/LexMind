import streamlit as st
import os
from pathlib import Path

# Tentative d'importation des modules backend
try:
    from src.pipeline import initialize_pipeline, index_documents, query_pipeline
    from src.config import CONFIG
except ImportError:
    # Mocks de secours si exécuté hors du projet principal
    def initialize_pipeline():
        return {"is_indexed": True}
    def index_documents(pipeline):
        pipeline["is_indexed"] = True
        return pipeline
    def query_pipeline(pipeline, prompt):
        class MockDoc:
            def __init__(self, filename):
                self.metadata = {"filename": filename}
        return f"Ceci est une réponse simulée à la question : **{prompt}**. Selon le Code Civil togolais, les dispositions applicables garantissent la sécurité juridique des parties.", [MockDoc("Code_Civil_Togo_Art12.pdf"), MockDoc("Journal_Officiel_2023.pdf")]

# ---------------------------------------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------------------------------------
st.set_page_config(
    page_title="LexMind • AI Juridique Togolais",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# STYLES CSS SUR-MESURE (Bleu Nuit, Bleu Ciel, Blanc, Glassmorphism)
# ---------------------------------------------------------
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;700;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-gradient: linear-gradient(135deg, #0A1128 0%, #101F42 50%, #001F54 100%);
    --accent-sky: #38BDF8;
    --accent-sky-glow: rgba(56, 189, 248, 0.3);
    --navy-card: rgba(15, 23, 42, 0.75);
    --navy-border: rgba(56, 189, 248, 0.2);
    --text-white: #F8FAFC;
    --text-muted: #94A3B8;
}

/* Fond global de l'application */
.stApp {
    background: var(--bg-gradient);
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text-white);
}

/* Typographie d'exception pour les titres */
h1, h2, h3, .brand-title {
    font-family: 'Cinzel', serif !important;
    letter-spacing: 0.5px;
}

/* En-tête Héros stylisé */
.hero-header {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--navy-border);
    border-radius: 20px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, var(--accent-sky-glow) 0%, transparent 60%);
    pointer-events: none;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #FFFFFF 30%, var(--accent-sky) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    font-size: 1.25rem;
    color: var(--accent-sky);
    font-weight: 500;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.hero-desc {
    color: var(--text-muted);
    font-size: 0.95rem;
    max-width: 650px;
    margin: 0;
}

/* Customisation de la Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(10, 17, 40, 0.9) !important;
    border-right: 1px solid var(--navy-border);
}

.sidebar-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
}

/* Badges de statut */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.9rem;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
}
.status-active {
    background: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}
.status-inactive {
    background: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(251, 191, 36, 0.3);
}

/* Chat Input Personalisation */
.stChatInputContainer {
    padding-bottom: 1rem;
}
.stChatInput input {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid var(--navy-border) !important;
    color: white !important;
    border-radius: 14px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stChatInput input:focus {
    border-color: var(--accent-sky) !important;
    box-shadow: 0 0 12px var(--accent-sky-glow) !important;
}

/* Style des messages du Chat */
[data-testid="stChatMessage"] {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    margin-bottom: 1rem !important;
    backdrop-filter: blur(8px);
}

/* Message utilisateur distinct */
[data-testid="stChatMessage"]:nth-child(even) {
    background-color: rgba(56, 189, 248, 0.08) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
}

/* Boutons stylisés */
.stButton>button {
    background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5) !important;
}

/* Accordéon Sources */
.streamlit-expanderHeader {
    background-color: rgba(255, 255, 255, 0.02) !important;
    border-radius: 8px !important;
    color: var(--accent-sky) !important;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------------------------
# INITIALISATION DU PIPELINE
# ---------------------------------------------------------
@st.cache_resource
def get_pipeline():
    return initialize_pipeline()

pipeline = get_pipeline()

# ---------------------------------------------------------
# BARRE LATÉRALE (SIDEBAR)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='font-size: 1.6rem; margin-bottom: 0.5rem;'>⚖️ LEXMIND</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted); font-size: 0.85rem;'>Plateforme IA d'intelligence juridique</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Carte de statut du pipeline
    is_indexed = pipeline.get("is_indexed", False)
    status_class = "status-active" if is_indexed else "status-inactive"
    status_text = "● Base Indexée & Prête" if is_indexed else "▲ Indexation Requise"
    
    st.markdown(f"""
    <div class="sidebar-card">
        <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.4rem;">Statut du système</div>
        <div class="status-badge {status_class}">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # Actions & Administration
    st.markdown("### ⚙️ Administration")
    if st.button("🔄 Recharger/Indexer la Base", use_container_width=True):
        with st.spinner("Indexation des textes de loi togolais..."):
            pipeline = index_documents(pipeline)
            st.success("Base de connaissances mise à jour !")
            st.rerun()

    st.markdown("---")
    st.markdown("### 📜 Domaines Couverts")
    st.markdown("""
    - **Code Civil & Droit de la Famille**
    - **Code du Travail**
    - **Droit des Affaires & OHADA**
    - **Code Pénal Togolais**
    """)
    
    st.markdown("---")
    st.caption("République Togolaise • Justice - Travail - Patrie")

# ---------------------------------------------------------
# BANNIÈRE PRINCIPALE (HERO SECTION)
# ---------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-title">LEXMIND</div>
    <div class="hero-subtitle">
        <span>⚖️</span> Assistant d'Intelligence Juridique Togolais
    </div>
    <p class="hero-desc">
        Explorez la jurisprudence, les codes officiels et les textes fondamentaux de la République Togolaise avec une précision algorithmique de pointe.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CHAT ET INTERACTION
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage de l'historique de discussion
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "⚖️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources Juridiques Examen"):
                for src in msg["sources"]:
                    st.markdown(f"- `{src}`")

# Zone de saisie utilisateur
if prompt := st.chat_input("Posez votre question juridique (ex: Quels sont les préavis de licenciement au Togo ?)..."):
    # Ajouter le message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Traitement assistant
    with st.chat_message("assistant", avatar="⚖️"):
        if not pipeline.get("is_indexed", False):
            st.error("⚠️ Le système d'information n'est pas indexé. Veuillez cliquer sur **Recharger/Indexer la Base** dans la barre latérale.")
        else:
            with st.spinner("Analyse des textes de loi & rédaction de la réponse..."):
                try:
                    answer, docs = query_pipeline(pipeline, prompt)
                    sources = list(set([d.metadata.get("filename", "Document Officiel") for d in docs]))

                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📚 Sources Juridiques Examen"):
                            for src in sources:
                                st.markdown(f"- `{src}`")

                    # Sauvegarde dans la session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Une erreur s'est produite lors de l'analyse : {e}")
