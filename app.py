import os
from pathlib import Path
import streamlit as st

CONFIG = {
    # Embeddings
    "embedding_model": "intfloat/multilingual-e5-small",
    "embedding_device": "cpu",

    # Découpage
    "chunk_size": 512,
    "chunk_overlap": 100,
    "chunk_separators": ["\n\n", "\n", "Article", "§", ".", " ", ""],
    "chunks_pickle_path": "./chunks.pkl",

    # Recherche
    "hybrid_search_weights": (0.5, 0.5),
    "top_k_initial": 20,
    "top_k_final": 5,

    # LLM
    "llm_model": "gemini-3.0-flash",
    "llm_temperature": 0.2,
    "llm_streaming": True,

    # Chemins
    "docs_directory": "./documents/",
    "chroma_persist_directory": "./chroma_db/",
    "collection_name": "lexmind",

    # Agentic
    "enable_agentic_retrieval": False,
    "max_retrieval_iterations": 3,
}

from src.document_loader import load_documents, chunk_documents
from src.vector_store import create_vector_store, load_vector_store, load_chunks
from src.retrievers import (
    create_hybrid_retriever,
    agentic_retrieve,
    hybrid_retrieve,
    get_retriever_from_vector_store,
)
from src.llm_client import create_llm
from src.prompts import create_generation_prompt
from src.config import CONFIG

def initialize_pipeline():
    """Initialise le pipeline en chargeant l'index existant si possible."""
    print("\n" + "=" * 60)
    print("INITIALIZING RAG PIPELINE")
    print("=" * 60)

    llm = create_llm()
    prompt = create_generation_prompt()

    vector_store = load_vector_store()
    hybrid_retriever = None
    is_indexed = False

    if vector_store:
        chunks = load_chunks()

        if chunks:
            print("Chunks trouvés, création du retriever hybride...")
            hybrid_retriever = create_hybrid_retriever(
                chunks,
                vector_store
            )
        else:
            print(
                "Chunks introuvables, utilisation du retriever "
                "vectoriel uniquement."
            )
            hybrid_retriever = get_retriever_from_vector_store(
                vector_store
            )

        is_indexed = True
        print("Index chargé avec succès.")

    else:
        print(
            "Aucun index existant trouvé. "
            "Veuillez indexer des documents."
        )

    return {
        "llm": llm,
        "prompt": prompt,
        "hybrid_retriever": hybrid_retriever,
        "vector_store": vector_store,
        "is_indexed": is_indexed,
    }


def index_documents(pipeline_state):
    """Indexe les documents du dossier documents/ et met à jour le pipeline."""
    print("\n" + "=" * 60)
    print("STARTING INDEXATION")
    print("=" * 60)

    docs = load_documents()

    if not docs:
        raise ValueError("Aucun document chargé.")

    chunks = chunk_documents(docs)

    vector_store = create_vector_store(chunks)

    hybrid_retriever = create_hybrid_retriever(
        chunks,
        vector_store
    )

    pipeline_state["vector_store"] = vector_store
    pipeline_state["hybrid_retriever"] = hybrid_retriever
    pipeline_state["is_indexed"] = True

    print("\nIndexation terminée!")
    print(f"    - {len(docs)} documents")
    print(f"    - {len(chunks)} chunks")
    print("=" * 60)

    return pipeline_state


def query_pipeline(pipeline_state, question):
    """Interroge le pipeline et retourne (réponse, documents)."""

    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("=" * 60)

    llm = pipeline_state["llm"]
    prompt = pipeline_state["prompt"]

    legal_keywords = [
        "loi",
        "article",
        "code",
        "juridique",
        "droit",
        "justice",
        "contrat",
        "travail",
        "licenciement",
        "salarié",
        "employeur",
        "divorce",
        "mariage",
        "succession",
        "héritage",
        "propriété",
        "terrain",
        "foncier",
        "pénal",
        "infraction",
        "crime",
        "délit",
        "tribunal",
        "avocat",
        "plainte",
        "procès",
        "responsabilité",
        "obligation",
        "créance",
        "dette",
        "société",
        "ohada",
        "commerce",
        "civil",
        "famille",
        "togo",
        "togolais",
        "constitution",
        "jurisprudence",
        "peine",
        "amende",
        "prison",
        "bail",
        "location",
        "travailleur",
        "emploi",
        "salaire",
    ]

    question_lower = question.lower()

    is_legal = any(
        keyword in question_lower
        for keyword in legal_keywords
    )

    if not is_legal:
        print("Question non juridique -> réponse directe")

        response = llm.invoke(
            prompt.format_messages(
                context="",
                question=question,
                chat_history=[],
            )
        )

        answer = response.content

        if isinstance(answer, list):
            answer = "".join(
                block.get("text", "")
                for block in answer
                if isinstance(block, dict)
                and block.get("type") == "text"
            )

        return answer, []

    if not pipeline_state["is_indexed"]:
        raise ValueError(
            "La base juridique n'est pas indexée. "
            "Veuillez indexer les documents."
        )

    hybrid_retriever = pipeline_state["hybrid_retriever"]

    if CONFIG["enable_agentic_retrieval"]:
        documents, context_sufficient = agentic_retrieve(
            llm,
            hybrid_retriever,
            question,
        )

        print(
            f"Context sufficient: {context_sufficient}"
        )

    else:
        documents = hybrid_retrieve(
            hybrid_retriever,
            question,
        )

        print(
            f"Retrieved {len(documents)} documents"
        )

    context = "\n\n---\n\n".join(
        [
            (
                f"[Source: {doc.metadata.get('filename', 'Inconnu')}]\n"
                f"{doc.page_content}"
            )
            for doc in documents
        ]
    )

    print("\nGenerating answer...")

    response = (
        prompt | llm
    ).invoke({
        "context": context,
        "question": question,
        "chat_history": [],
    })

    answer = response.content

    if isinstance(answer, list):
        answer = "".join(
            block.get("text", "")
            for block in answer
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    print("\n" + "=" * 60)
    print("Answer generated")
    print("=" * 60)

    return answer, documents


def simple_query(pipeline_state, question):
    """Version simplifiée : ne retourne que la réponse."""
    answer, _ = query_pipeline(
        pipeline_state,
        question
    )
    return answer


# ---------------------------------------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------------------------------------

st.set_page_config(
    page_title="LexMind - Intelligence Juridique Togolaise",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# STYLES CSS (Design Cabinet d'Avocats / Élégant & Moderne)
# ---------------------------------------------------------

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,300&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --primary-navy: #0F172A;
    --secondary-blue: #1E3A8A;
    --accent-gold: #B45309;
    --bg-light: #F8FAFC;
    --card-white: #FFFFFF;
    --border-subtle: #E2E8F0;
    --text-dark: #0F172A;
    --text-muted: #475569;
}

/* Cache complètement la barre latérale */
[data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}

/* Base global */
.stApp {
    background-color: var(--bg-light);
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-dark);
}

.main .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1, h2, h3, h4, .serif-font {
    font-family: 'Merriweather', Georgia, serif !important;
    color: var(--primary-navy);
}

/* Header & Navbar minimaliste */
.site-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.2rem 0;
    border-bottom: 2px solid var(--border-subtle);
    margin-bottom: 2.5rem;
}

.site-logo {
    font-family: 'Merriweather', serif;
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: var(--primary-navy);
    text-transform: uppercase;
}

.site-logo span {
    color: var(--accent-gold);
}

.site-tagline {
    font-size: 0.85rem;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Hero Section */
.hero-container {
    background: var(--card-white);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 3rem 2.5rem;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.04);
    margin-bottom: 3rem;
    text-align: center;
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 700;
    line-height: 1.25;
    margin-bottom: 1rem;
    color: var(--primary-navy);
}

.hero-subtitle {
    font-size: 1.15rem;
    color: var(--text-muted);
    max-width: 780px;
    margin: 0 auto 1.5rem auto;
    line-height: 1.6;
}

/* Grille d'explications / Fonctionnalités */
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-bottom: 3rem;
}

.feature-card {
    background: var(--card-white);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 1.8rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

.feature-number {
    font-family: 'Merriweather', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-gold);
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.feature-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--primary-navy);
    margin-bottom: 0.6rem;
}

.feature-desc {
    font-size: 0.92rem;
    color: var(--text-muted);
    line-height: 1.55;
}

/* Section de recherche / Chat */
.chat-section-header {
    border-top: 1px solid var(--border-subtle);
    padding-top: 2rem;
    margin-bottom: 1.5rem;
}

.chat-section-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

/* Customisation des messages de chat Streamlit */
[data-testid="stChatMessage"] {
    background-color: var(--card-white) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 10px !important;
    padding: 1.25rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02) !important;
    color: var(--text-dark) !important;
}

[data-testid="stChatMessage"]:nth-child(even) {
    background-color: #F1F5F9 !important;
    border-color: #CBD5E1 !important;
}

.stChatInput input {
    background-color: var(--card-white) !important;
    border: 1px solid #CBD5E1 !important;
    color: var(--text-dark) !important;
    border-radius: 8px !important;
    padding: 0.8rem 1rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.98rem !important;
}

.stChatInput input:focus {
    border-color: var(--secondary-blue) !important;
    box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.1) !important;
}

.stButton>button {
    background-color: var(--primary-navy) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    transition: background-color 0.2s ease !important;
}

.stButton>button:hover {
    background-color: var(--secondary-blue) !important;
    color: #FFFFFF !important;
}

/* Styles des expandable & badges status */
.streamlit-expanderHeader {
    background-color: var(--card-white) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 6px !important;
    color: var(--primary-navy) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

.status-indicator {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.status-ok {
    background-color: #DEF7EC;
    color: #03543F;
}

.status-warn {
    background-color: #FDF6B2;
    color: #723B13;
}

footer {
    display: none !important;
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
# BARRE D'EN-TÊTE
# ---------------------------------------------------------

st.markdown(
    """
    <div class="site-navbar">
        <div class="site-logo">LEXMIND <span>JURIDIQUE</span></div>
        <div class="site-tagline">République Togolaise • Base Légale & Jurisprudence</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# BANNIÈRE D'ACCUEIL (HERO)
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Plateforme d'Analyse Juridique et d'Information Légale Togolaise</div>
        <div class="hero-subtitle">
            Accédez aux codes officiels, lois, décrets et textes fondamentaux de la République Togolaise. 
            Obtenez des réponses structurées et motivées en s'appuyant directement sur des sources documentaires certifiées.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SECTION D'EXPLICATIONS (CARTE DE PRÉSENTATION)
# ---------------------------------------------------------

st.markdown(
    """
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-number">MODULE 01</div>
            <div class="feature-title">Recherche Textuelle Hybride</div>
            <div class="feature-desc">
                Exploration simultanée par recherche sémantique et par mots-clés exacts pour identifier les articles de loi et décrets applicables à votre situation.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-number">MODULE 02</div>
            <div class="feature-title">Citations des Sources Officielles</div>
            <div class="feature-desc">
                Chaque réponse est accompagnée de la liste des textes de droit togolais (Code du travail, Code civil, Code pénal, Traités OHADA) ayant servi de référence.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-number">MODULE 03</div>
            <div class="feature-title">Analyse Structurée & Synthèse</div>
            <div class="feature-desc">
                Présentation synthétique des obligations, sanctions, procédures et droits prévus par la réglementation en vigueur au Togo.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# GESTION ADMINISTRATIVE DE LA BASE (ACCORDEON DISCRET)
# ---------------------------------------------------------

is_indexed = pipeline.get("is_indexed", False)
status_class = "status-ok" if is_indexed else "status-warn"
status_label = "Base de données indexée" if is_indexed else "Indexation requise"

with st.expander("Administration du système et indexation des textes"):
    col_stat, col_btn = st.columns([3, 1])
    with col_stat:
        st.markdown(f"**Statut actuel :** <span class='status-indicator {status_class}'>{status_label}</span>", unsafe_allow_html=True)
        st.markdown("La base de connaissances comprend le Code du Travail, le Code Pénal, le Code de la Famille et les actes uniformes OHADA.")
    with col_btn:
        if st.button("Actualiser la base", use_container_width=True):
            with st.spinner("Indexation des documents juridiques togolais en cours..."):
                pipeline = index_documents(pipeline)
                st.success("La base de connaissances a été mise à jour.")
                st.rerun()


# ---------------------------------------------------------
# SECTION CHAT ET INTERACTION JURIDIQUE
# ---------------------------------------------------------

st.markdown(
    """
    <div class="chat-section-header">
        <div class="chat-section-title">Consultation et Requête Juridique</div>
        <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1.5rem;">
            Saisissez votre question ci-dessous (ex. : Quels sont les motifs de licenciement légitimes selon le Code du travail togolais ?).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []


# Affichage des messages enregistrés dans la session
for msg in st.session_state.messages:
    sender_name = "Utilisateur" if msg["role"] == "user" else "LexMind Juridique"
    
    with st.chat_message(msg["role"]):
        st.markdown(f"**{sender_name}**")
        st.markdown(msg["content"])

        if (
            msg["role"] == "assistant"
            and "sources" in msg
            and msg["sources"]
        ):
            with st.expander("Sources légales consultées"):
                for src in msg["sources"]:
                    st.markdown(f"- **Document :** `{src}`")


# Saisie de la question
if prompt := st.chat_input("Saisissez votre question relative au droit togolais..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown("**Utilisateur**")
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.markdown("**LexMind Juridique**")
        
        with st.spinner("Consultation des textes de loi et rédaction du rapport juridique..."):

            try:

                answer, docs = query_pipeline(
                    pipeline,
                    prompt
                )

                sources = list(
                    set(
                        [
                            d.metadata.get(
                                "filename",
                                "Document Officiel"
                            )
                            for d in docs
                        ]
                    )
                )

                st.markdown(answer)

                if sources:
                    with st.expander("Sources légales consultées"):
                        for src in sources:
                            st.markdown(f"- **Document :** `{src}`")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as e:

                st.error(
                    f"Une erreur s'est produite lors de l'analyse : {e}"
                )
